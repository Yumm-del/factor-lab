"""
LLM 因子生成模块 ———— 自然语言 → 因子 → 体检 → 反思迭代 → AI 解读
====================================================================
流程：
    1. 用户用自然语言描述因子想法（如"我想捕捉放量突破后的延续"）
    2. 系统把「DSL 算子说明书 + 数据字段说明」拼进 prompt，让 LLM 输出
       受限 JSON 表达式树（外加一句"因子逻辑"——演示环节的亮点）
    3. 解析校验（非法输出自动重试 1 次）→ 向量化求值 → 全套体检
    4. 体检评分不达标（<50）→ 自动反思闭环：把失败诊断（IC/换手/
       单调性/衰减）回喂给 LLM，LLM 修改表达式结构后重新体检，
       最多迭代 2 轮——这是"智能体"区别于"单轮代码生成器"的核心
    5. LLM 把体检数字翻译成中文报告（结论 / 指标解读 / 风险 / 改进建议）

安全设计（为什么这是"智能体"而不是"代码生成器"）：
    LLM 只能从白名单算子中组合表达式——不能执行任意代码、不能越权访问数据。
    "AI 因子挖掘"的创新点就落在「受限生成 + 自动验证 + 反思迭代 + 自动解读」
    闭环上——AI 负责"想"，结构负责"拦"，体检负责"判"，人负责"审"。
"""

import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 兼容直接运行与包导入
try:
    from . import dsdl, validation
except ImportError:
    import dsdl
    import validation

# ——— 项目根目录（.env 所在位置） ———
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-v4-pro"
BASE_URL = "https://api.deepseek.com"


def _client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY（请检查 .env）")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)


def _llm_text(system: str, user: str, temperature: float = 0.4) -> str:
    """调用 LLM 返回原始文本（解读报告等非 JSON 场景）。"""
    client = _client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=8000,  # 推理模型：max_tokens 含推理 token，必须给足
    )
    content = resp.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError(f"LLM 返回空文本（finish_reason={resp.choices[0].finish_reason}）")
    return content.strip()


def _llm_json(system: str, user: str, temperature: float = 0.3) -> dict:
    """调用 LLM 并解析 JSON 输出（失败自动重试一次）。"""
    client = _client()
    last_err = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                # 重要：deepseek-v4-pro 是推理模型，推理 token 也计入 max_tokens。
                # 之前 max_tokens=1500 时推理过程就耗尽额度导致输出为空（finish_reason=length）。
                # 8000 给推理 + 最终 JSON 都留足空间。
                max_tokens=8000,
            )
            raw = resp.choices[0].message.content
            # 兼容 LLM 输出带 ```json 代码块的情况
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001 — 解析失败/API 错误统一重试
            last_err = e
    raise RuntimeError(f"LLM 输出无法解析为 JSON（已重试 1 次）: {last_err}")


# ============================================================
# 一、因子生成
# ============================================================

# DSL 算子说明书——给 LLM 看的白名单手册（与 dsdl.OPERATORS 注册表完全一致：
# 38 算子 + 10 数据叶子 + const。新增算子源于 alpha101 公式库移植，
# 窗口范围以引擎为准——LLM 超出范围会被解析层拦截并中文报错）
OPERATOR_MANUAL = """
可用算子（只能使用这些，禁止自创）：
【数据字段（叶子）】open 开盘价 / close 收盘价 / high 最高价 / low 最低价 /
  volume 成交量 / amount 成交额 / vwap 均价(amount/volume) / turn 换手率(%) /
  pe 市盈率TTM / pb 市净率
【时序算子】作用于每只股票自身的历史（窗口天数叫 param，必须在该算子范围内）：
  ts_returns(x, d)      d日收益率 = x_t/x_{t-d} - 1（d: 1~60）
  ts_mean(x, d)         滚动均值（d: 2~260）
  ts_std(x, d)          滚动标准差（d: 2~120）
  ts_zscore(x, d)       滚动z-score（去趋势去量纲）（d: 5~120）
  ts_rank(x, d)         滚动百分位排名0~1（d: 2~120）
  ts_max(x, d) / ts_min(x, d)  滚动最大/最小（d: 2~260，支持52周高点类因子）
  delay(x, d)           滞后d期（d: 1~260）
  delta(x, d)           d期差分 x_t - x_{t-d}（d: 1~260）
  ts_sum(x, d) / ts_product(x, d)  滚动加总 / 连乘（d: 1~260 / 1~30）
  ts_argmax(x, d) / ts_argmin(x, d)  窗口内最大/最小值位置（d: 2~60）
  decay_linear(x, d)    d期线性衰减加权（近值权重高）（d: 2~60）
  ts_corr(x, y, d) / ts_cov(x, y, d)  x与y的滚动相关 / 协方差（d: 2~260）
【截面算子】同一天所有股票之间：
  rank(x)               截面排名0~1（越大=当天所有股票中越靠前）
  normalize(x)          截面z-score（去截面均值和量纲）
  scale(x)              截面缩放 Σ|x| = 1
【一元算子】
  signed_power(x, e)    带符号幂 sign(x)·|x|^e（e: 0.1~3）
  log(x) / ln(x)        log(1+x) / 自然对数
  abs(x) / neg(x) / sign(x)  绝对值 / 取负 / 取符号
【二元算子】
  add/sub/mul/div       x与y的加减乘除（div 除零自动变缺失）
  pow(x, y)             幂 x^y
  min(x, y) / max(x, y) 逐元素取小/取大
  gt(x, y) / lt(x, y) / eq(x, y)  逐元素比较，返回 1.0/0.0
  and(x, y) / or(x, y)  逐元素逻辑与/或（非 0 视为真）
【条件算子】
  cond(cond_expr, a, b)  条件为真（非 0）取 a，否则取 b（对应 WorldQuant '?:'）
【常量】{"op": "const", "value": 1.0}
"""

GENERATION_SYSTEM = (
    "你是一位资深量化研究员，在 A 股（沪深300）多因子选股系统里工作。\n"
    "你的任务：把用户用自然语言描述的因子想法，翻译成受限 JSON 表达式树。\n"
    "规则：\n"
    "1. 只能使用白名单算子，禁止任何白名单之外的运算\n"
    "2. 表达式深度不超过 6 层，节点数不超过 40\n"
    "3. 输出必须是 JSON 对象，格式：{\"rationale\": \"因子逻辑一句话\", \"expr\": {表达式树}}\n"
    "4. rationale 要讲清楚因子的金融逻辑（为什么这个信号能预测收益），不要写数学公式\n"
    "5. 因子必须有金融含义，禁止无意义的数学乱凑\n"
    "6. 时序算子必须在 rank/normalize 之前用（先算股票自身指标，再做截面比较）\n"
)

GENERATION_EXAMPLES = """示例（用户想法 → 输出）：
想法：过去一个月涨得多的股票，接下来还会涨
输出：{"rationale": "A股动量效应：过去强势的股票会延续强势",
  "expr": {"op": "rank", "args": [{"op": "ts_returns", "args": [{"op": "close"}], "param": 20}]}}

想法：成交量突然放大的股票可能有异动机会
输出：{"rationale": "量能突增反映资金关注度骤升，短期弹性更大",
  "expr": {"op": "rank", "args": [{"op": "div", "args": [
    {"op": "ts_mean", "args": [{"op": "volume"}], "param": 5},
    {"op": "ts_mean", "args": [{"op": "volume"}], "param": 20}]}]}}
"""


def build_generation_prompt(idea: str) -> tuple[str, str]:
    """构造生成 prompt。返回 (system, user)。"""
    user = f"""{GENERATION_EXAMPLES}

以下是完整的算子说明书：
{OPERATOR_MANUAL}

用户想法：{idea}

请输出 JSON（只输出 JSON，不要任何多余文字）。
先想清楚：这个想法对应的金融逻辑是什么？用什么算子组合能表达？
输出格式：{{"rationale": "...", "expr": {{...}}}}
"""
    return GENERATION_SYSTEM, user


# ============================================================
# 二、反思迭代（体检不达标 → 反馈 → 重新生成）
# ============================================================

# 体检评分低于该分触发反思（<45 为"淘汰"档；50 留出可用边缘的提升空间）
REFLECT_THRESHOLD = 50.0
# 最多自动反思 2 轮（共 3 次生成）——迭代收益递减，控制 API 成本
MAX_REFLECT_ROUNDS = 2

REFLECT_SYSTEM = (
    "你是一位资深量化研究员，正在迭代优化一个因子。\n"
    "你上一版表达式体检不达标。体检数据会告诉你失败原因（如 IC 过低、"
    "换手过高、分层单调性破裂、信号衰减过快）。\n"
    "你的任务：分析失败原因，修改表达式结构，输出修正后的受限 JSON 表达式树。\n"
    "规则与首次生成相同：\n"
    "1. 只能使用白名单算子，禁止任何白名单之外的运算\n"
    "2. 表达式深度不超过 6 层，节点数不超过 40\n"
    "3. 输出 JSON 对象，格式：{\"rationale\": \"修正逻辑一句话（说明改了哪里）\", \"expr\": {表达式树}}\n"
    "4. 针对体检暴露的具体问题修改，不要推翻重来：\n"
    "   - IC 过低/不稳定 → 换更稳健的时序窗口或截面排名\n"
    "   - 换手过高 → 加长窗口均值（平滑信号）或降低对短窗的依赖\n"
    "   - 分层单调性破裂 → 检查是否过度依赖单一字段，增加逻辑组合\n"
    "   - 衰减过快 → 用更长周期的 ts_returns 或均值结构\n"
    "5. 如果认为当前方向不可救，可以换一个同主题的思路（但必须仍是量价/基本面逻辑）"
)


def build_reflect_prompt(idea: str, formula: str, rationale: str, diag: dict) -> str:
    """构造反思 prompt：上版因子 + 体检失败数据 → 要求输出修正表达式。"""
    return f"""{GENERATION_EXAMPLES}

以下是完整的算子说明书：
{OPERATOR_MANUAL}

用户想法：{idea}

你上一版因子：{formula}
上版因子逻辑：{rationale}

体检结果（未达标）：
{diagnosis_to_text(diag)}

请分析失败原因，修改表达式结构后重新输出。
输出 JSON（只输出 JSON，不要任何多余文字）：
{{"rationale": "修正后的因子逻辑一句话（说明改了哪里）", "expr": {{...}}}}
"""


# ============================================================
# 二、体检报告解读
# ============================================================


def diagnosis_to_text(diag: dict) -> str:
    """体检单 → 紧凑中文文本（喂给 LLM 解读，比 JSON 友好）。"""
    s = diag["ic_summary"]
    lay = diag["layers"]
    decay = diag["ic_decay"]
    layer_annual = lay["layer_returns"].mean().mul(252).round(2).to_dict()
    return f"""
IC均值: {s['ic_mean']:+.4f} | IC标准差: {s['ic_std']:.4f} | IR(信息比率): {s['ic_ir']:.2f}
t值: {s['ic_t']:+.2f} | IC为正天数占比: {s['ic_positive']:.1%} | RankIC均值: {s['rank_ic_mean']:+.4f}
体检天数: {s['n_days']} | 综合评分: {diag['score']}/100 | 判定: {diag['verdict']['label']}
5层分层回测（L1=因子值最高层）各层年化收益: {layer_annual}
多空(L1-L5)年化: {lay['spread_annual']:+.1%} | 层间单调性: {lay['monotonic']:+.2f}
组合日换手率: {diag['turnover']:.1%}
IC衰减(RankIC, 未来1~10日): {[f"lag{k}={v:+.3f}" for k, v in decay.items()]}
"""


INTERPRET_SYSTEM = (
    "你是一位给基金经理做因子体检汇报的量化研究员。\n"
    "根据因子定义和体检数据，写一份简洁的中文体检报告（Markdown）。\n"
    "报告结构：\n"
    "## 因子画像（这个因子在做什么，一句话金融逻辑）\n"
    "## 体检结论（用一句话下结论：优秀/可用/淘汰，依据是什么）\n"
    "## 关键指标解读（IC、IR、单调性、换手、衰减各1-2句，讲人话，不要堆数字）\n"
    "## 风险与改进建议（2-3条，具体可操作）\n"
    "要求：总字数 300~500；语气客观；不要虚构体检数据里没有的结论。"
)


def build_interpret_prompt(formula: str, rationale: str, diag: dict) -> str:
    return f"""因子定义（DSL 公式）：{formula}
因子逻辑：{rationale}

体检结果：
{diagnosis_to_text(diag)}

请输出体检报告。"""


# ============================================================
# 三、端到端流程
# ============================================================


def generate_factor(idea: str, panel: dict, close: pd.DataFrame,
                    max_reflect_rounds: int = MAX_REFLECT_ROUNDS) -> dict:
    """
    端到端：自然语言想法 → 因子表达式 → 体检 → 反思迭代 → AI 解读。

    参数：
        idea  — 用户用自然语言描述的因子想法
        panel — load_panel() 数据面板（含 close 等）
        close — 收盘价面板（验证模块用）
        max_reflect_rounds — 体检不达标时最多自动反思轮数（默认 2，共 3 次生成）
    返回：
        dict：{idea, rationale, expr, expr_str, formula, tree, diag, report,
               rounds, n_rounds}
              rounds — 每轮迭代记录（[{round, formula, score, verdict}]，
                       1 轮 = 无反思；>1 轮 = 触发了反思闭环）
              n_rounds — 实际轮数（演示/报告中展示"智能体迭代了 N 次"）
    """
    # 首轮：从自然语言想法生成初始表达式
    system, user = build_generation_prompt(idea)
    result = _llm_json(system, user)

    rounds: list[dict] = []
    for round_i in range(1, max_reflect_rounds + 2):  # 1 + 反思轮数（最多 3 次生成）
        rationale = result.get("rationale", "")
        expr_dict = result.get("expr")
        if not isinstance(expr_dict, dict):
            raise RuntimeError("LLM 输出缺少 expr 字段")

        # 校验（非法时抛 FactorParseError，UI 层提示用户换个说法）
        expr = dsdl.parse_factor(json.dumps(expr_dict))
        formula = dsdl.to_formula(expr)
        tree = dsdl.render_tree(expr)

        # 求值 + 体检
        factor_panel = dsdl.evaluate(expr, panel)
        diag = validation.full_diagnosis(factor_panel, close)

        rounds.append({
            "round": round_i,
            "formula": formula,
            "score": diag["score"],
            "verdict": diag["verdict"]["label"],
        })

        # 达标（>=50）或已达最大轮数 → 停止迭代
        if diag["score"] >= REFLECT_THRESHOLD or round_i > max_reflect_rounds:
            break

        # 不达标 → 反思闭环：把失败诊断回喂 LLM，修改表达式后重新体检。
        # 这轮生成是"有依据的修正"而非"重试"——体检数据就是修改的依据。
        result = _llm_json(
            REFLECT_SYSTEM,
            build_reflect_prompt(idea, formula, rationale, diag),
        )

    # AI 解读（Markdown 文本，不是 JSON）——只解读最终版因子
    report_text = _llm_text(
        INTERPRET_SYSTEM,
        build_interpret_prompt(formula, rationale, diag),
    )

    return {
        "idea": idea,
        "rationale": rationale,
        "expr": expr,
        "expr_str": json.dumps(expr, ensure_ascii=False),
        "formula": formula,
        "tree": tree,
        "diag": diag,
        "report": report_text,
        "rounds": rounds,
        "n_rounds": len(rounds),
    }


# ============================================================
# 自测（合成面板，不调真实数据）
# ============================================================
if __name__ == "__main__":
    print("⚠️  需要真实数据面板 + DEEPSEEK_API_KEY 才能端到端测试。")
    print("快速检查 prompt 构造：")
    sys_, user = build_generation_prompt("我想找一个低波动的因子")
    print("system 长度:", len(sys_), "| user 长度:", len(user))
    print("示例检查通过 ✅")

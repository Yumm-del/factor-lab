"""
因子 DSL ———— 让 LLM 安全地写因子表达式
==========================================
目的：LLM 通过自然语言描述因子想法，输出受限 JSON 表达式树，
      我们把树解析成向量化 pandas 计算，不执行任意代码。

为什么不用「让 LLM 直接写 Python 代码再执行」？
    1. 安全：任意代码执行 = 注入风险。表达式树是纯数据，只能调用白名单算子
    2. 可控：算子、深度、节点数全部有上限，LLM 输出再离谱也不会炸
    3. 可解释：表达式树可以渲染成人类可读公式 + 树状图——这是演示环节的亮点
       （评委一眼看到"因子长什么样"，而不是一团黑盒代码）

表达式格式（JSON）：
    叶子：    {"op": "close"}                        → 取 close 面板
              {"op": "const", "value": 0.02}         → 常量（自动广播）
    一元算子：{"op": "ts_mean", "args": [<expr>], "param": 20}
    二元算子：{"op": "div", "args": [<expr>, <expr>]}
    (param 只出现在需要窗口/天数的算子上)

算子语义分两类：
    ts_*（时序算子）：沿时间轴对每只股票独立滚动计算，输入输出都是 (date × code) 面板
    rank/normalize（截面算子）：在同一个交易日截面上对全部股票排序/标准化

面板约定：所有中间结果都是 DataFrame（index=date, columns=code），
          空参数节点返回原始数据面板，标量自动广播到整个面板。
"""

import json
from functools import partial

import numpy as np
import pandas as pd

# ============================================================
# 一、算子注册表（白名单）
# ============================================================
# 每个算子: kind 决定参数校验方式, fn 接收 (data_or_args, param)
#   "leaf"   — 数据叶子，无 args
#   "ts"     — 时序算子，param 为滚动窗口天数
#   "cross"  — 截面算子，无 param
#   "binop"  — 二元组合算子
#   "unop"   — 一元组合算子


def _ts_returns(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """d 日收益率：x_t / x_{t-d} - 1。窗口语义：间隔 d 期，不是滚动窗口。"""
    return x / x.shift(d) - 1.0


def _ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(2, d // 4)).mean()


def _ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(2, d // 4)).std()


def _ts_zscore(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """滚动 z-score：(x - 滚动均值) / 滚动标准差，去除量纲与趋势。"""
    mean = x.rolling(d, min_periods=max(2, d // 4)).mean()
    std = x.rolling(d, min_periods=max(2, d // 4)).std()
    return (x - mean) / std.replace(0, np.nan)


def _ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """滚动百分位排名（0~1）：股票价格相对自己过去 d 天处于什么位置。"""
    return x.rolling(d, min_periods=max(2, d // 4)).rank(pct=True)


def _ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(2, d // 4)).max()


def _ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(2, d // 4)).min()


def _ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """滚动相关系数（x 与 y 在 d 日窗口内的相关性）。"""
    return x.rolling(d, min_periods=max(2, d // 4)).corr(y)


def _delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """滞后 d 期（避免未来函数泄漏的关键算子）。"""
    return x.shift(d)


def _rank(x: pd.DataFrame) -> pd.DataFrame:
    """截面排名 0~1：同一天所有股票的相对大小。"""
    return x.rank(axis=1, pct=True)


def _normalize(x: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score：同一天减均值、除标准差（截面中性化）。"""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return x.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


def _signed_power(x: pd.DataFrame, e: float) -> pd.DataFrame:
    """带符号幂：sign(x)·|x|^e。可以压缩/放大离群值而不改变符号。"""
    return np.sign(x) * np.abs(x) ** e


def _log(x: pd.DataFrame) -> pd.DataFrame:
    """log(1+x)：压缩右偏分布（常见于成交量类因子）。"""
    return np.log1p(x)


def _abs(x: pd.DataFrame) -> pd.DataFrame:
    """绝对值（如 Amihud 非流动性中的 |日收益率|）。"""
    return x.abs()


# 数据叶子：可直接访问的原始数据面板
DATA_LEAVES = ["close", "high", "low", "volume", "amount", "turn", "pe", "pb"]

OPERATORS: dict[str, dict] = {
    # —— 时序算子（param 为窗口天数）——
    "ts_returns":   {"kind": "ts", "fn": _ts_returns, "param_min": 1, "param_max": 60},
    "ts_mean":      {"kind": "ts", "fn": _ts_mean, "param_min": 2, "param_max": 120},
    "ts_std":       {"kind": "ts", "fn": _ts_std, "param_min": 2, "param_max": 120},
    "ts_zscore":    {"kind": "ts", "fn": _ts_zscore, "param_min": 5, "param_max": 120},
    "ts_rank":      {"kind": "ts", "fn": _ts_rank, "param_min": 5, "param_max": 120},
    "ts_max":       {"kind": "ts", "fn": _ts_max, "param_min": 2, "param_max": 260},
    "ts_min":       {"kind": "ts", "fn": _ts_min, "param_min": 2, "param_max": 260},
    "delay":        {"kind": "ts", "fn": _delay, "param_min": 1, "param_max": 60},
    # —— 截面算子（同一天内所有股票）——
    "rank":         {"kind": "cross", "fn": _rank},
    "normalize":    {"kind": "cross", "fn": _normalize},
    # —— 组合算子 ——
    "add":          {"kind": "binop", "fn": lambda a, b: a + b},
    "sub":          {"kind": "binop", "fn": lambda a, b: a - b},
    "mul":          {"kind": "binop", "fn": lambda a, b: a * b},
    "div":          {"kind": "binop", "fn": lambda a, b: a / b.replace(0, np.nan)},
    "signed_power": {"kind": "unop", "fn": _signed_power, "needs_param": True},
    "log":          {"kind": "unop", "fn": _log},
    "abs":          {"kind": "unop", "fn": _abs},
    "neg":          {"kind": "unop", "fn": lambda a: -a},
}

# 求解限制（防止 LLM 输出把程序拖垮）
MAX_DEPTH = 8          # 表达式树最大深度
MAX_NODES = 64         # 最大节点数


class FactorParseError(ValueError):
    """表达式不合法（LLM 输出问题 / 用户手写错误），带中文提示信息。"""


# ============================================================
# 二、解析与校验
# ============================================================


def parse_factor(expr_str: str) -> dict:
    """
    解析 LLM 输出的 JSON 表达式字符串 → 校验后的表达式树（dict）。

    输入输出：json 字符串 → dict（{"op": ..., "args": [...], "param": ...}）
    原理：解析 + 两遍校验（结构合法 → 递归规则校验），非法时抛 FactorParseError。
    """
    try:
        expr = json.loads(expr_str)
    except json.JSONDecodeError as e:
        raise FactorParseError(f"JSON 解析失败（可能是 LLM 输出的括号不完整）: {e}") from e
    if not isinstance(expr, dict):
        raise FactorParseError("表达式必须是 JSON 对象")
    _validate(expr, depth=0, counter=[0])
    return expr


def _validate(node: dict, depth: int, counter: list[int]) -> None:
    """递归校验一个节点（及子树）。counter 统计总节点数。"""
    if depth > MAX_DEPTH:
        raise FactorParseError(f"表达式深度超过限制（{MAX_DEPTH}），请让因子更简洁")
    counter[0] += 1
    if counter[0] > MAX_NODES:
        raise FactorParseError(f"表达式节点数超过限制（{MAX_NODES}），请让因子更简洁")

    if not isinstance(node, dict) or "op" not in node:
        raise FactorParseError(f"节点必须是 {{'op': ...}} 结构: {node}")

    op = node["op"]
    # —— 叶子：数据字段 ——
    if op in DATA_LEAVES:
        if "args" in node:
            raise FactorParseError(f"数据叶子 {op} 不应带参数")
        return
    # —— 常量 ——
    if op == "const":
        if not isinstance(node.get("value"), (int, float)):
            raise FactorParseError("const 需要数值 value")
        return
    # —— 注册表算子 ——
    if op not in OPERATORS:
        raise FactorParseError(
            f"未知算子 {op!r}。可用算子: {', '.join(list(OPERATORS) + DATA_LEAVES + ['const'])}"
        )
    spec = OPERATORS[op]
    args = node.get("args", [])

    if spec["kind"] in ("binop",):
        if not isinstance(args, list) or len(args) != 2:
            raise FactorParseError(f"{op} 需要恰好 2 个参数")
        if "param" in node:
            raise FactorParseError(f"{op} 不应带 param")
        for a in args:
            _validate(a, depth + 1, counter)
    elif spec["kind"] == "unop":
        if not isinstance(args, list) or len(args) != 1:
            raise FactorParseError(f"{op} 需要恰好 1 个参数")
        if spec.get("needs_param"):  # signed_power 需要 exponent
            if not isinstance(node.get("param"), (int, float)):
                raise FactorParseError(f"{op} 需要数值 param（幂指数）")
        elif "param" in node:
            raise FactorParseError(f"{op} 不应带 param")
        _validate(args[0], depth + 1, counter)
    elif spec["kind"] == "ts":
        if not isinstance(args, list) or len(args) != 1:
            raise FactorParseError(f"{op} 需要恰好 1 个参数")
        p = node.get("param")
        if not isinstance(p, int) or not (spec["param_min"] <= p <= spec["param_max"]):
            raise FactorParseError(f"{op} 的窗口参数必须是 {spec['param_min']}~{spec['param_max']} 的整数")
        _validate(args[0], depth + 1, counter)
    elif spec["kind"] == "cross":
        if not isinstance(args, list) or len(args) != 1:
            raise FactorParseError(f"{op} 需要恰好 1 个参数")
        if "param" in node:
            raise FactorParseError(f"{op} 不应带 param")
        _validate(args[0], depth + 1, counter)


# ============================================================
# 二·五、公式解析器（人类可读公式 → 表达式树）
# ============================================================
# 目的：让"人"和"AI"用同一套 DSL。
#   LLM 输出 JSON 表达式树；人（用户/评委）写数学公式：
#     rank(neg(ts_std(ts_returns(close, 1), 20)))
#   两者解析后是同构的表达式树，走完全相同的求值/体检/策略管线。
# 实现：递归下降解析（tokenize → 语法树 → 复用 _validate 校验）。

import re as _re

# 注意顺序：负数优先（-?\d），否则 '-5' 会被切成 '-' 和 '5'
_TOKEN_RE = _re.compile(r"-?\d+\.?\d*|[a-zA-Z_]\w*|[()]|[,+\-*/]")


def _tokenize(s: str) -> list[str]:
    """把公式字符串切成 token 列表。丢弃空白。
    负数（'-' 紧跟数字）在正则层合并为一个 token，如 ts_returns(close, -5)。
    """
    return _TOKEN_RE.findall(s)


class _FormulaParser:
    """递归下降解析器。每个 parse_* 方法消费 token 并返回表达式树 dict。"""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def pop(self) -> str:
        t = self.peek()
        if t is None:
            raise FactorParseError("公式不完整（意外的结尾）")
        self.pos += 1
        return t

    def expect(self, s: str) -> None:
        t = self.pop()
        if t != s:
            raise FactorParseError(f"语法错误：期望 {s!r}，实际是 {t!r}")

    # expr := sum_expr
    def parse_expr(self) -> dict:
        return self.parse_sum()

    # sum_expr := product_expr (('+'|'-'|'add'|'sub') product_expr)*
    def parse_sum(self) -> dict:
        left = self.parse_product()
        while self.peek() in ("+", "-", "add", "sub"):
            op = {"+": "add", "-": "sub", "add": "add", "sub": "sub"}[self.pop()]
            right = self.parse_product()
            left = {"op": op, "args": [left, right]}
        return left

    # product_expr := atom (('*'|'/'|'mul'|'div') atom)*
    def parse_product(self) -> dict:
        left = self.parse_atom()
        while self.peek() in ("*", "/", "mul", "div"):
            op = {"*": "mul", "/": "div", "mul": "mul", "div": "div"}[self.pop()]
            right = self.parse_atom()
            left = {"op": op, "args": [left, right]}
        return left

    # atom := NUMBER | LEAF | FUNC(args) | '(' expr ')'
    def parse_atom(self) -> dict:
        t = self.peek()
        if t is None:
            raise FactorParseError("公式不完整")
        if t == "(":
            self.pop()
            e = self.parse_expr()
            self.expect(")")
            return e
        if _re.match(r"-?\d+\.?\d*", t):
            self.pop()
            return {"op": "const", "value": float(t)}
        if _re.match(r"[a-zA-Z_]\w*", t):
            name = self.pop()
            if self.peek() == "(":
                return self.parse_call(name)
            # 数据叶子（close 等）或裸常量名
            return {"op": name}
        self.pop()
        raise FactorParseError(f"意外的 token: {t!r}")

    # call := IDENT '(' arg (',' arg)* ')'
    def parse_call(self, name: str) -> dict:
        self.expect("(")
        # 只有时序算子（ts_*）和 signed_power 的第 2 个参数才是数值 param；
        # 其他算子（如 add(close, 5)）的数字参数解析为 const 表达式。
        spec = OPERATORS.get(name, {})
        takes_param = spec.get("kind") == "ts" or spec.get("needs_param")
        args = []
        param = None
        while self.peek() != ")":
            t = self.peek()
            if (takes_param and len(args) >= 1 and param is None
                    and t is not None and _re.match(r"-?\d+\.?\d*", t)):
                self.pop()
                param = float(t)
                if param.is_integer():  # 窗口参数是整数（ts_mean(close, 5)）；5.5 留给校验报错
                    param = int(param)
            else:
                args.append(self.parse_expr())
            if self.peek() == ",":
                self.pop()
        self.expect(")")
        node = {"op": name, "args": args}
        if param is not None:
            node["param"] = param
        return node


def parse_formula(s: str) -> dict:
    """
    人类可读公式 → 表达式树（如 "rank(ts_mean(close, 20))"）。
    与 parse_factor 输出同构：走同一套 _validate 白名单校验。
    支持：
        rank(neg(ts_std(ts_returns(close, 1), 20)))   函数式
        (ts_mean(close, 20) + ts_returns(close, 5))   中缀（+ - * /）
        rank(ts_mean(close, 20) / ts_mean(volume, 20)) 混合
    """
    tokens = _tokenize(s)
    if not tokens:
        raise FactorParseError("公式为空")
    parser = _FormulaParser(tokens)
    expr = parser.parse_expr()
    if parser.peek() is not None:  # 有多余 token（如 "rank(x))" 多余括号）
        raise FactorParseError(f"公式有多余内容: {parser.peek()!r}")
    _validate(expr, depth=0, counter=[0])
    return expr


# ============================================================
# 三、向量化求值
# ============================================================


def evaluate(expr: dict, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    求值表达式树 → (date × code) 因子面板。

    参数：
        expr  — parse_factor 输出的表达式树
        panel — load_panel() 输出的数据面板 dict
    返回：
        DataFrame（index=date, columns=code），NaN 表示数据不足
    """
    return _eval_node(expr, panel)


def _eval_node(node: dict, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """递归求值一个节点。面板缺失时抛清晰的错误（LLM 可能引用不存在的字段）。"""
    op = node["op"]
    if op in DATA_LEAVES:
        if op not in panel:
            raise FactorParseError(f"数据字段 {op} 不在面板中（可用: {', '.join(panel)}）")
        return panel[op]
    if op == "const":
        return pd.DataFrame(node["value"], index=next(iter(panel.values())).index,
                            columns=next(iter(panel.values())).columns)

    spec = OPERATORS[op]
    if spec["kind"] in ("ts", "cross", "unop"):
        x = _eval_node(node["args"][0], panel)
    if spec["kind"] == "ts":
        return spec["fn"](x, node["param"])
    if spec["kind"] == "cross":
        return spec["fn"](x)
    if spec["kind"] == "unop":
        if spec.get("needs_param"):
            return spec["fn"](x, node["param"])
        return spec["fn"](x)
    if spec["kind"] == "binop":
        a = _eval_node(node["args"][0], panel)
        b = _eval_node(node["args"][1], panel)
        return spec["fn"](a, b)


# ============================================================
# 四、人类可读输出（演示/报告用）
# ============================================================


def to_formula(expr: dict) -> str:
    """表达式树 → 数学公式字符串（如 rank(ts_mean(close, 20))）。"""
    op = expr["op"]
    if op in DATA_LEAVES:
        return op
    if op == "const":
        return str(expr["value"])
    if op == "ts_returns":
        return f"ts_returns({to_formula(expr['args'][0])}, {expr['param']})"
    if OPERATORS[op]["kind"] == "ts":
        return f"{op}({to_formula(expr['args'][0])}, {expr['param']})"
    if OPERATORS[op]["kind"] == "cross":
        return f"{op}({to_formula(expr['args'][0])})"
    if OPERATORS[op]["kind"] == "unop":
        if OPERATORS[op].get("needs_param"):
            return f"{op}({to_formula(expr['args'][0])}, {expr['param']})"
        return f"{op}({to_formula(expr['args'][0])})"
    if OPERATORS[op]["kind"] == "binop":
        a, b = (to_formula(x) for x in expr["args"])
        return f"({a} {op} {b})"


def render_tree(expr: dict, indent: int = 0) -> str:
    """
    表达式树 → 缩进树（Streamlit 里 st.code 展示，演示环节一图看懂因子结构）。
    例：
        rank
        └── ts_mean
            ├── close
            └── param: 20
    """
    op = expr["op"]
    pad = "    " * indent
    lines = [pad + op]
    if op == "const":
        lines[-1] += f" = {expr['value']}"
        return "\n".join(lines)
    if op == "ts_returns":
        lines[-1] += f" (period={expr['param']})"
    elif op in OPERATORS and OPERATORS[op]["kind"] == "ts":
        lines[-1] += f" (window={expr['param']})"
    if OPERATORS.get(op, {}).get("needs_param"):
        lines[-1] += f" (exponent={expr['param']})"
    for a in expr.get("args", []):
        lines.append(render_tree(a, indent + 1))
    return "\n".join(lines)


# ============================================================
# 自测（合成面板，不依赖真实数据）
# ============================================================
if __name__ == "__main__":
    # —— 合成 100 天 × 30 只股票的面板 ——
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-06-02", periods=100, freq="B")
    codes = [f"sh.60{i:04d}" for i in range(30)]
    idx = pd.MultiIndex.from_product([dates, codes])
    data = pd.DataFrame(
        {
            "close": rng.normal(10, 2, len(idx)).cumprod(),
            "volume": rng.lognormal(15, 0.5, len(idx)),
            "turn": rng.lognormal(0.5, 0.5, len(idx)),
            "pe": rng.lognormal(2.5, 0.4, len(idx)),
            "pb": rng.lognormal(1, 0.3, len(idx)),
        },
        index=idx,
    ).reset_index()
    data = data.rename(columns={"level_0": "date", "level_1": "code"})
    test_panel = {}
    for name in ["close", "volume", "turn", "pe", "pb"]:
        test_panel[name] = data.pivot_table(index="date", columns="code", values=name)

    print("=== 测试 1: 数据叶子 ===")
    expr1 = parse_factor('{"op": "close"}')
    print(to_formula(expr1), "→", evaluate(expr1, test_panel).shape)

    print("\n=== 测试 2: 组合表达式 ===")
    expr2 = parse_factor(
        '{"op": "rank", "args": [{"op": "ts_mean", "args": [{"op": "close"}], "param": 5}]}'
    )
    print(to_formula(expr2))
    print(render_tree(expr2))
    val2 = evaluate(expr2, test_panel)
    print("值域:", round(float(val2.min().min()), 3), "~", round(float(val2.max().max()), 3))
    assert val2.min().min() >= 0 and val2.max().max() <= 1, "rank 应在 0~1"

    print("\n=== 测试 3: 复杂混合因子（模拟 LLM 输出） ===")
    expr3 = parse_factor(
        '{"op": "div", "args": ['
        '  {"op": "rank", "args": [{"op": "ts_zscore", "args": [{"op": "close"}], "param": 20}]},'
        '  {"op": "rank", "args": [{"op": "ts_std", "args": [{"op": "close"}], "param": 20}]}'
        "]}"
    )
    print(to_formula(expr3))
    val3 = evaluate(expr3, test_panel)
    print("值域:", round(float(val3.min().min()), 3), "~", round(float(val3.max().max()), 3))

    print("\n=== 测试 4: 公式解析器（人类可读公式 → 表达式树） ===")
    formulas = [
        "rank(ts_mean(close, 20))",
        "rank(neg(ts_std(ts_returns(close, 1), 20)))",       # 嵌套 + 数字参数
        "(ts_mean(close, 20) + ts_returns(close, 5))",        # 中缀加法
        "rank(ts_mean(close, 20) / ts_mean(volume, 20))",     # 中缀除法
        "rank((close - ts_mean(close, 20)) / ts_std(close, 20))",  # 布林带式
        "rank(ts_mean(close, 5) * 2 - ts_mean(close, 20))",   # 优先级：* 先于 -
        "delay(close, 5)",                                    # delay 正参数
        "add(close, 5)",                                      # 数字当 const 表达式
    ]
    for f in formulas:
        e = parse_formula(f)
        rt = to_formula(e)
        print(f"  {f}\n    → {rt}  | 求值 {evaluate(e, test_panel).shape}")
        # round-trip：解析后再格式化，应保持算子结构
        assert parse_formula(rt) == e, f"round-trip 失败: {f}"

    print("\n=== 测试 5: 非法公式拦截 ===")
    for bad in [
        "rank(",                       # 缺右括号
        "rank(close))",                # 多余括号
        "ts_mean(close)",              # 缺 param
        "ts_mean(close, 999)",         # param 越界
        "ts_returns(close, -5)",       # 负窗口（未来函数，禁止）
        "eval(close)",                 # 未知算子
        "rank(close, 5)",              # 参数过多
        "close +",                     # 尾部运算符
    ]:
        try:
            parse_formula(bad)
            print("⚠️  未拦截: ", bad)
        except FactorParseError as e:
            print("✅ 拦截:", str(e)[:60])

    print("\n=== 测试 6: 非法输入拦截（JSON 路径） ===")
    for bad in [
        '{"op": "eval", "args": ["__import__(\'os\').system(\'rm -rf /\')"]}',  # 注入尝试
        '{"op": "ts_mean", "args": [{"op": "close"}], "param": 999}',           # 窗口越界
        '{"op": "rank"}',                                                       # 缺参数
        'not json at all',                                                      # 非 JSON
    ]:
        try:
            parse_factor(bad)
            print("⚠️  未拦截: ", bad[:60])
        except FactorParseError as e:
            print("✅ 拦截:", str(e)[:70])

    print("\n全部自测通过 ✅")

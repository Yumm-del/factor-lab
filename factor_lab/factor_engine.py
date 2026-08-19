"""
因子引擎 ———— 经典因子族 + 统一计算入口
==========================================
目的：内置一批业界公认有效的经典因子（动量/反转/波动/流动性/价值/趋势），
      并且——关键设计——这些因子全部用 DSL 表达式树表示。

为什么经典因子也用 DSL 而不是直接写 pandas 代码？
    1. 统一管线：经典因子和 LLM 生成的因子走完全相同的
       【表达式树 → 求值 → 验证 → 可视化】流水线，架构上零特殊分支
    2. 叙事加分：向评委展示"这个工作台里一切因子都是可解释的表达式结构"，
       经典因子只是表达式库里的预置项，LLM 因子是动态生成项
    3. 天然对比：经典因子的验证结果就是 LLM 因子的"对照基线"——
       工作台能告诉你"AI 挖的因子到底有没有跑赢主流经典因子"

每个经典因子 = {name, category, description, expr}，
expr 是 dsdl.parse_factor 可直接消费的表达式树。
"""

try:
    from . import dsdl  # 作为包导入（python -m factor_lab.factor_engine）
except ImportError:
    import dsdl  # 直接运行脚本时的回退（python factor_lab/factor_engine.py）

# ============================================================
# 表达式树构造助手（用 Python 函数生成树，比手写 JSON 好维护）
# ============================================================


def _leaf(name: str) -> dict:
    """数据叶子，如 _leaf("close")。"""
    return {"op": name}


def _ts(op: str, x: dict, param: int) -> dict:
    """时序算子节点，如 _ts("ts_mean", _leaf("close"), 20)。"""
    return {"op": op, "args": [x], "param": param}


def _cross(op: str, x: dict) -> dict:
    """截面算子节点，如 _cross("rank", x)。"""
    return {"op": op, "args": [x]}


def _neg(x: dict) -> dict:
    """负号：rank(-x) 等价于反向排名，用于"低 xx 好"的因子。"""
    return {"op": "neg", "args": [x]}


def _combine(op: str, a: dict, b: dict) -> dict:
    """二元组合节点，如 _combine("div", a, b)。"""
    return {"op": op, "args": [a, b]}


# ============================================================
# 经典因子目录
# ============================================================
# 因子选取原则：A 股实证研究中最稳健、最能代表一类风格异象的因子，
# 每种风格取 1~2 个，避免因子高度相关的冗余。
CLASSIC_FACTORS: dict[str, dict] = {
    # ——— 动量/反转类 ———
    "momentum_20": {
        "name": "20日动量",
        "category": "动量",
        "description": "过去 20 个交易日的累计收益率。动量效应：强者恒强。",
        "expr": _cross("rank", _ts("ts_returns", _leaf("close"), 20)),
    },
    "momentum_60_1": {
        "name": "60日动量(剔1日)",
        "category": "动量",
        "description": "过去 60 日收益减去最近 1 日收益。剔除 1 日反转噪声后的中期动量，A 股最稳健的动量定义。",
        "expr": _cross(
            "rank",
            _combine(
                "sub",
                _ts("ts_returns", _leaf("close"), 60),
                _ts("ts_returns", _leaf("close"), 1),
            ),
        ),
    },
    "reversal_5": {
        "name": "5日反转",
        "category": "反转",
        "description": "过去 5 日收益取负。A 股短期反转效应显著：跌多了会反弹。",
        "expr": _cross("rank", _neg(_ts("ts_returns", _leaf("close"), 5))),
    },
    # ——— 波动类 ———
    "volatility_20": {
        "name": "低波动(20日)",
        "category": "波动",
        "description": "过去 20 日日收益波动率取负。低波动异象：波动率低的股票长期跑赢。",
        "expr": _cross(
            "rank",
            _neg(_ts("ts_std", _ts("ts_returns", _leaf("close"), 1), 20)),
        ),
    },
    # ——— 流动性类 ———
    "liquidity_turn": {
        "name": "低换手(20日)",
        "category": "流动性",
        "description": "过去 20 日平均换手率取负。低换手股票信息不对称程度低，A 股低换手异象显著。",
        "expr": _cross("rank", _neg(_ts("ts_mean", _leaf("turn"), 20))),
    },
    "amihud_illiq": {
        "name": "非流动性(Amihud)",
        "category": "流动性",
        "description": "Amihud 非流动性：|日收益率|/成交额 的 20 日均值取负。流动性差的股票要求更高收益补偿。",
        "expr": _cross(
            "rank",
            _neg(
                _ts(
                    "ts_mean",
                    _combine(
                        "div",
                        {"op": "abs", "args": [_ts("ts_returns", _leaf("close"), 1)]},
                        _leaf("amount"),
                    ),
                    20,
                )
            ),
        ),
    },
    # ——— 价值类 ———
    "value_ep": {
        "name": "盈利收益率(1/PE)",
        "category": "价值",
        "description": "市盈率的倒数（E/P）。盈利收益率越高，估值越便宜。",
        "expr": _cross("rank", _combine("div", {"op": "const", "value": 1.0}, _leaf("pe"))),
    },
    "value_bp": {
        "name": "账面市值比(1/PB)",
        "category": "价值",
        "description": "市净率的倒数（B/P），价值投资的核心度量。",
        "expr": _cross("rank", _combine("div", {"op": "const", "value": 1.0}, _leaf("pb"))),
    },
    # ——— 趋势/质量类 ———
    "trend_52w": {
        "name": "距52周高点",
        "category": "趋势",
        "description": "当前价格占过去 250 日最高价的比例。接近年内高点的股票处于强势趋势。",
        "expr": _cross(
            "rank",
            _combine("div", _leaf("close"), _ts("ts_max", _leaf("close"), 250)),
        ),
    },
    "volume_ratio": {
        "name": "量比(5日/20日)",
        "category": "流动性",
        "description": "近 5 日均量相对 20 日均量的比值。放量信号：关注度骤升。",
        "expr": _cross(
            "rank",
            _combine(
                "div",
                _ts("ts_mean", _leaf("volume"), 5),
                _ts("ts_mean", _leaf("volume"), 20),
            ),
        ),
    },
}


def list_factors() -> list[dict]:
    """因子目录 → 列表（含 DSL 公式字符串，供 UI 展示）。"""
    out = []
    for key, meta in CLASSIC_FACTORS.items():
        out.append({
            "key": key,
            "name": meta["name"],
            "category": meta["category"],
            "description": meta["description"],
            "formula": dsdl.to_formula(meta["expr"]),
        })
    return out


def get_factor(key: str) -> dict:
    """按 key 取因子定义（不存在抛 KeyError，UI 层负责提示）。"""
    if key not in CLASSIC_FACTORS:
        raise KeyError(f"未知因子: {key}，可用: {', '.join(CLASSIC_FACTORS)}")
    return CLASSIC_FACTORS[key]


def compute_factor(expr: dict | str, panel: dict) -> "pd.DataFrame":
    """
    统一因子计算入口：接受表达式树 dict 或 JSON 字符串，返回因子面板。

    参数：
        expr  — DSL 表达式树（dict）或 JSON 字符串（LLM 输出场景）
        panel — load_panel() 数据面板
    返回：
        (date × code) 因子面板
    """
    if isinstance(expr, str):
        expr = dsdl.parse_factor(expr)
    return dsdl.evaluate(expr, panel)


def compute_classic(key: str, panel: dict) -> "pd.DataFrame":
    """快捷入口：按 key 计算某个经典因子。"""
    return dsdl.evaluate(get_factor(key)["expr"], panel)


# ============================================================
# 自测（合成面板）
# ============================================================
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-02", periods=300, freq="B")
    codes = [f"sh.60{i:04d}" for i in range(40)]
    idx = pd.MultiIndex.from_product([dates, codes])
    # 合成带一点动量结构的假数据（便于检查因子方向）
    drift = rng.normal(0.0005, 0.02, (len(dates), len(codes)))
    close = pd.DataFrame(10 * np.exp(np.cumsum(drift, axis=0)), index=dates, columns=codes)
    panel = {
        "close": close,
        "volume": pd.DataFrame(rng.lognormal(15, 0.4, (len(dates), len(codes))), index=dates, columns=codes),
        "amount": pd.DataFrame(rng.lognormal(8, 0.4, (len(dates), len(codes))), index=dates, columns=codes),
        "turn": pd.DataFrame(rng.lognormal(0.4, 0.5, (len(dates), len(codes))), index=dates, columns=codes),
        "pe": pd.DataFrame(rng.lognormal(2.6, 0.35, (len(dates), len(codes))), index=dates, columns=codes),
        "pb": pd.DataFrame(rng.lognormal(1.1, 0.3, (len(dates), len(codes))), index=dates, columns=codes),
    }

    print(f"经典因子库: {len(CLASSIC_FACTORS)} 个因子\n")
    for key, meta in CLASSIC_FACTORS.items():
        fac = dsdl.evaluate(meta["expr"], panel)
        # 每列（股票）应产出数值、形状应和面板一致
        assert fac.shape == close.shape, f"{key} 形状不符"
        nan_ratio = fac.isna().mean().mean()
        print(f"  {key:<18} {meta['name']:<12} NaN占比 {nan_ratio:.1%}  公式: {dsdl.to_formula(meta['expr'])[:48]}")

    # 抽查一个因子的公式与树
    print("\n示例: momentum_60_1")
    print(dsdl.render_tree(CLASSIC_FACTORS["momentum_60_1"]["expr"]))
    print("全部经典因子求值通过 ✅")

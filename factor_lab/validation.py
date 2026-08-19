"""
因子验证模块 ———— 用机构级指标给因子"体检"
=============================================
目的：任何因子（经典或 LLM 生成）进入工作台后，都必须通过同一套体检，
      把"感觉有效"变成"数字证明有效"。

体检项目（每一项回答一个评委最关心的问题）：
    1. IC / RankIC 序列    — 因子值和下期收益有没有稳定的正相关？
    2. IC 统计量            — IC 均值、IR（IC均值/IC标准差）、t 值（均值显著吗）
    3. 分层回测             — 按因子值分 5 层，最高层真能跑赢最低层吗？（单调性）
    4. 换手率               — 组合每周换多少仓？（换手越高，实盘成本侵蚀越严重）
    5. IC 衰减曲线          — 信号能持续几天？（衰减越快，策略调仓越频繁）

核心约定：因子面板是 (date × code)，t 日因子值预测 t+1 日收益（无未来函数——
因子计算只用到 t 日及以前的数据，收益取 t+1，天然错开一天）。
"""

import numpy as np
import pandas as pd

# ============================================================
# 一、IC 计算
# ============================================================


def forward_returns(close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """t 日计算、持有 horizon 天的未来收益：close_{t+horizon}/close_t - 1。"""
    return close.shift(-horizon) / close - 1.0


def compute_ic(factor: pd.DataFrame, close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    每日截面 IC（Pearson）与 RankIC（Spearman）序列。

    原理：每一天 t，取因子面板 t 行（全部股票）与未来收益 t 行的截面相关性。
    截面内先剔 NaN（停牌/新股无数据），当天有效股票 < 10 只则记为 NaN。
    """
    fwd = forward_returns(close, horizon)
    # 对齐因子与收益的行（日期）
    common = factor.index.intersection(fwd.index)
    factor, fwd = factor.loc[common], fwd.loc[common]

    ic_list, rankic_list = [], []
    for date in common:
        x = factor.loc[date]
        y = fwd.loc[date]
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            ic_list.append(np.nan)
            rankic_list.append(np.nan)
            continue
        xv, yv = x[mask].values, y[mask].values
        # Pearson 相关（线性相关）+ Spearman（排名相关，对离群值稳健）
        if np.std(xv) == 0 or np.std(yv) == 0:
            ic_list.append(np.nan)
            rankic_list.append(np.nan)
        else:
            ic_list.append(np.corrcoef(xv, yv)[0, 1])
            rankic_list.append(pd.Series(xv).rank().corr(pd.Series(yv).rank()))

    out = pd.DataFrame({"ic": ic_list, "rank_ic": rankic_list}, index=common)
    return out.dropna()


def ic_summary(ic_table: pd.DataFrame) -> dict:
    """
    IC 统计量汇总：
        ic_mean     — IC 均值（>0.03 在 A 股日频中是很强的因子）
        ic_std      — IC 标准差
        ic_ir       — 信息比率 IC_mean/IC_std（>0.5 算优秀）
        ic_t        — t 值 IC_mean/(IC_std/√n)，|t|>2 统计显著
        ic_positive — IC>0 的天数占比（稳健性的直观度量）
        rank_ic_mean— RankIC 均值（对离群值更稳健）
    """
    n = len(ic_table)
    if n == 0:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan,
                "ic_t": np.nan, "ic_positive": np.nan, "rank_ic_mean": np.nan, "n_days": 0}
    ic = ic_table["ic"].dropna()
    rank_ic = ic_table["rank_ic"].dropna()
    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=1))
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": float(ic_mean / ic_std) if ic_std > 0 else np.nan,
        "ic_t": float(ic_mean / (ic_std / np.sqrt(len(ic)))) if ic_std > 0 else np.nan,
        "ic_positive": float((ic > 0).mean()),
        "rank_ic_mean": float(rank_ic.mean()),
        "n_days": n,
    }


# ============================================================
# 二、分层回测
# ============================================================


def layer_backtest(
    factor: pd.DataFrame, close: pd.DataFrame, n_layers: int = 5
) -> dict:
    """
    分层回测：每天按因子值截面分 n_layers 层，每层等权持有 1 天。

    返回：
        layer_returns — 每层组合的每日收益（DataFrame: date × layer）
        layer_nav     — 每层累计净值
        benchmark     — 全池等权基准收益序列
        spread_nav    — 最高层 - 最低层的多空累计净值
        spread_annual — 多空年化收益（252 交易日）
        monotonic     — 层间单调性：层平均收益与层号（从高到低）的 Spearman 相关
    """
    fwd = forward_returns(close, 1)
    common = factor.index.intersection(fwd.index)
    factor, fwd = factor.loc[common], fwd.loc[common]

    layer_cols = [f"L{i}" for i in range(1, n_layers + 1)]
    layer_returns = pd.DataFrame(index=common, columns=layer_cols)
    bench = pd.Series(index=common, dtype=float)

    for date in common:
        x = factor.loc[date]
        y = fwd.loc[date]
        valid = x.notna() & y.notna() & np.isfinite(x)
        if valid.sum() < n_layers * 5:  # 当天股票太少，跳过（防止分层过稀）
            continue
        xv, yv = x[valid], y[valid]
        # qcut 分 n_layers 层。注意：labels=False 时 0=因子值最低组，
        # 取反让 L1 = 因子值最高组（与全模块的方向约定一致）
        try:
            layers = (n_layers - 1) - pd.qcut(xv.rank(method="first"), n_layers, labels=False)
        except ValueError:
            continue
        for li in range(n_layers):
            mask = layers == li
            if mask.sum() > 0:
                layer_returns.loc[date, f"L{li + 1}"] = yv[mask].mean()
        bench.loc[date] = yv.mean()

    layer_nav = (1 + layer_returns.fillna(0)).cumprod()
    spread = layer_returns["L1"] - layer_returns[f"L{n_layers}"]
    spread_nav = (1 + spread.fillna(0)).cumprod()

    # 层间单调性：L1（因子值最高）应收益最高 → 层收益与层号严格负相关时为 +1.0
    mean_by_layer = layer_returns.mean()
    layer_order = pd.Series(range(1, n_layers + 1), index=mean_by_layer.index)
    monotonic = -mean_by_layer.rank().corr(layer_order)

    return {
        "layer_returns": layer_returns,
        "layer_nav": layer_nav,
        "benchmark": bench,
        "spread_nav": spread_nav,
        "spread_annual": float(spread.mean() * 252) if spread.notna().sum() > 0 else np.nan,
        "monotonic": float(monotonic) if not pd.isna(monotonic) else np.nan,
        "n_layers": n_layers,
    }


def layer_turnover(factor: pd.DataFrame, close: pd.DataFrame, n_layers: int = 5) -> float:
    """
    组合平均换手率：相邻两天最高层（L1）成员变化的比例均值。

    原理：L1 层（因子值最高的 1/n 股票）每天调仓一次，换手率 = 1 - 相邻两天
    成员重叠比例。换手 1.0 = 组合每天整体换一遍；按 A 股双边成本 ~20bps 估算，
    日换手 1.0 意味着每年成本 ≈ 20bps × 2 × 252 ≈ 100%，成本直接吃掉收益。
    因此换手率是「因子是否可实盘」的关键体检项。
    """
    fwd = forward_returns(close, 1)
    common = factor.index.intersection(fwd.index)
    factor = factor.loc[common]

    membership = {}
    for date in common:
        x = factor.loc[date]
        valid = x.notna() & np.isfinite(x)
        if valid.sum() < n_layers * 5:
            continue
        try:
            layers = (n_layers - 1) - pd.qcut(x[valid].rank(method="first"), n_layers, labels=False)
        except ValueError:
            continue
        # 修正方向：layers == n_layers-1 才是因子值最高层（L1）
        membership[date] = set(x[valid].index[layers == n_layers - 1])
    membership = pd.Series(membership)

    dates = membership.index
    turnovers = []
    for i in range(1, len(dates)):
        prev, cur = membership.iloc[i - 1], membership.iloc[i]
        if prev and cur:
            turnovers.append(1 - len(prev & cur) / len(prev))
    return float(np.mean(turnovers)) if turnovers else np.nan


# ============================================================
# 三、IC 衰减
# ============================================================


def ic_decay(factor: pd.DataFrame, close: pd.DataFrame, max_lag: int = 10) -> dict:
    """
    IC 衰减曲线：因子与未来 1~max_lag 天收益的 RankIC。
    解读：衰减快 → 信号是短线信号，需要频繁调仓（成本敏感）；
          衰减慢 → 信号持久，可低频调仓（成本友好）。
    """
    out = {}
    for lag in range(1, max_lag + 1):
        table = compute_ic(factor, close, horizon=lag)
        out[lag] = float(table["rank_ic"].mean()) if len(table) else np.nan
    return out


# ============================================================
# 四、综合体检 + 评分
# ============================================================


def full_diagnosis(factor: pd.DataFrame, close: pd.DataFrame, n_layers: int = 5) -> dict:
    """跑完整体检，输出一张「体检单」（dict），供 UI 与 LLM 解读消费。"""
    ic_table = compute_ic(factor, close, horizon=1)
    summary = ic_summary(ic_table)
    layers = layer_backtest(factor, close, n_layers=n_layers)
    turnover = layer_turnover(factor, close, n_layers=n_layers)
    decay = ic_decay(factor, close, max_lag=10)

    diag = {
        "ic_summary": summary,
        "ic_series": ic_table,
        "layers": layers,
        "turnover": turnover,
        "ic_decay": decay,
        "n_days": summary["n_days"],
    }
    diag["score"] = score_factor(diag)
    diag["verdict"] = verdict_of(diag["score"], ic_mean=summary["ic_mean"])
    return diag


def score_factor(diag: dict) -> float:
    """
    综合评分 0~100。权重与阈值参考 A 股日频因子研究的常见基准：
        IC 均值      — 0.03 已是强因子 → tanh 压缩（30 分）
        IR           — 0.5 以上优秀 → tanh 压缩（25 分）
        多空年化      — 15% 以上优秀 → tanh 压缩（30 分）
        单调性        — 1.0 完美单调（15 分）
    用 tanh 的好处：超强值不会无限加分（防过拟合因子的虚荣分数），
    弱因子的分数曲线线性起步（低分区分度保留）。
    """
    s = diag["ic_summary"]
    lay = diag["layers"]
    ic = abs(s["ic_mean"]) if np.isfinite(s["ic_mean"]) else 0.0
    ir = abs(s["ic_ir"]) if np.isfinite(s["ic_ir"]) else 0.0
    spread = abs(lay["spread_annual"]) if np.isfinite(lay["spread_annual"]) else 0.0
    mono = lay["monotonic"]
    mono_score = 15 * max(0.0, mono) if np.isfinite(mono) else 0.0

    return round(
        30 * np.tanh(ic / 0.03)
        + 25 * np.tanh(ir / 0.5)
        + 30 * np.tanh(spread / 0.15)
        + mono_score,
        1,
    )


def verdict_of(score: float, ic_mean: float | None = None) -> dict:
    """
    评分 → 结论标签（给 LLM 解读报告做骨架，也给 UI 直接展示）。
    带 ic_mean 时在文案中标明因子方向（正/反），避免"高分但方向没讲"的误导。
    """
    direction = ""
    if ic_mean is not None and np.isfinite(ic_mean) and ic_mean != 0:
        direction = f"（方向：{'正向' if ic_mean > 0 else '反向'}）"
    if score >= 70:
        return {"label": "优秀", "color": "green",
                "text": f"因子具备统计显著的选股能力{direction}，值得进入策略组合构建阶段。"}
    if score >= 45:
        return {"label": "可用", "color": "orange",
                "text": f"因子有一定区分度，但强度或稳健性有限{direction}，建议与经典因子组合使用。"}
    return {"label": "淘汰", "color": "red",
            "text": "因子区分度不足或方向不稳定，建议修改表达式后重新体检。"}


# ============================================================
# 自测（合成面板：一个注入真信号的因子 + 一个纯噪声因子）
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n_days, n_stocks = 250, 80
    dates = pd.date_range("2025-06-02", periods=n_days, freq="B")
    codes = [f"sh.60{i:04d}" for i in range(n_stocks)]

    # 真实收益：含一个隐藏的"动量因子"结构
    hidden_factor = rng.normal(0, 1, (n_days, n_stocks))
    returns = 0.02 * np.roll(hidden_factor, 1, axis=0) + rng.normal(0, 0.02, (n_days, n_stocks))
    returns[0] = 0
    close = pd.DataFrame(10 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=codes)
    # 好因子：已知隐藏因子的滞后版（应检出强 IC）
    good_factor = pd.DataFrame(hidden_factor, index=dates, columns=codes)
    # 坏因子：纯噪声（IC 应接近 0）
    bad_factor = pd.DataFrame(rng.normal(0, 1, (n_days, n_stocks)), index=dates, columns=codes)

    for name, fac in [("注入信号因子", good_factor), ("纯噪声因子", bad_factor)]:
        diag = full_diagnosis(fac, close)
        s = diag["ic_summary"]
        print(f"\n===== {name} =====")
        print(f"  IC均值 {s['ic_mean']:+.4f} | IR {s['ic_ir']:.2f} | t值 {s['ic_t']:+.1f} | RankIC {s['rank_ic_mean']:+.4f}")
        print(f"  多空年化 {diag['layers']['spread_annual']:+.1%} | 单调性 {diag['layers']['monotonic']:.2f} | 换手 {diag['turnover']:.1%}")
        print(f"  评分 {diag['score']} → {diag['verdict']['label']}")
        decay = diag["ic_decay"]
        print("  IC衰减:", " ".join(f"lag{k}={v:+.3f}" for k, v in list(decay.items())[:5]))

    # 好因子应显著优于坏因子
    good = full_diagnosis(good_factor, close)
    bad = full_diagnosis(bad_factor, close)
    assert good["score"] > 60 > bad["score"], "体检应能区分好坏因子"
    print("\n✅ 验证模块自测通过：能区分真信号与噪声")

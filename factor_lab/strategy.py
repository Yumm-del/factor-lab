"""
策略构建模块 ———— 因子的"最后一公里"：从因子到组合净值
==========================================================
目的：把"体检通过"的因子变成一条可以展示的净值曲线，回答评委最关心的问题：
      "这个因子真拿去选股，能赚多少？"

设计（简单、可解释、不过拟合）：
    1. 调仓：每周（5 个交易日）按因子值截面选 Top N 只股票，等权持有
    2. 收益：调仓后的每个交易日按持仓股票的当日收益等权计算组合收益
    3. 成本：调仓日按实际换手率 × 双边成本（默认 20bps）扣减——
       这是 A 股可实盘性的关键约束，也展示"扣除成本后还赚不赚"
    4. 基准：真实沪深300指数（baostock sh.000300），不是自己造的对标

指标（metrics）：年化收益/年化波动/Sharpe/最大回撤/超额年化/信息比率/胜率
"""

import numpy as np
import pandas as pd

try:
    from . import validation  # 作为包导入（python -m factor_lab.strategy）
except ImportError:
    import validation  # 直接运行脚本时的回退


def build_portfolio(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    index_close: pd.Series | None = None,
    n_stocks: int = 30,
    rebalance_days: int = 5,
    cost_bps: float = 20.0,
) -> dict:
    """
    用因子构建 Top-N 等权组合（周频调仓）。

    参数：
        factor         — (date × code) 因子面板
        close          — 收盘价面板（t 日因子 → t+1 日起生效）
        index_close    — 沪深300指数收盘序列（基准，可为 None 则跳过对比）
        n_stocks       — 组合持仓数量
        rebalance_days — 调仓间隔（交易日），5 = 周频
        cost_bps       — 双边交易成本（基点，1bp=0.01%）
    返回：
        dict：{nav, returns, benchmark_nav, metrics, turnover, rebalance_days}
    """
    fwd = validation.forward_returns(close, 1)
    common = factor.index.intersection(fwd.index)
    factor, fwd = factor.loc[common], fwd.loc[common]

    # ——— 1. 确定调仓日：从第一个有效数据日起，每隔 rebalance_days 天调一次 ———
    reb_dates = factor.index[::rebalance_days].tolist()

    # ——— 2. 逐调仓周期计算组合收益 ———
    portfolio_returns = pd.Series(index=common, dtype=float)
    holdings: set[str] = set()
    turnovers: list[float] = []

    for i, reb_date in enumerate(reb_dates):
        # 选股：调仓日因子值最高的 n_stocks 只
        x = factor.loc[reb_date].dropna()
        if len(x) < n_stocks:
            continue
        new_holdings = set(x.sort_values(ascending=False).head(n_stocks).index)

        # 换手率与成本（首次建仓不算成本——避免初始资本影响指标）
        if holdings and i > 0:
            turnover = 1 - len(holdings & new_holdings) / len(holdings)
            turnovers.append(turnover)
            cost = turnover * (cost_bps / 10000)
        else:
            cost = 0.0
        holdings = new_holdings

        # 该调仓周期内的每个交易日：持仓股票当日收益等权
        period_end = reb_dates[i + 1] if i + 1 < len(reb_dates) else common[-1]
        period_days = common[(common >= reb_date) & (common <= period_end)]
        for day in period_days:
            daily = fwd.loc[day].loc[list(holdings)].dropna()
            if len(daily) > 0:
                ret = float(daily.mean()) - (cost if day == period_days[0] else 0.0)
                portfolio_returns.loc[day] = ret

    # ——— 3. 净值与基准 ———
    nav = (1 + portfolio_returns.dropna()).cumprod()
    nav = nav.reindex(common).ffill().fillna(1.0)
    nav.name = "strategy"

    benchmark_nav = None
    if index_close is not None:
        bench = index_close.reindex(common).ffill()
        benchmark_nav = (bench / bench.iloc[0]).rename("hs300")

    # ——— 4. 绩效指标 ———
    rets = portfolio_returns.dropna()
    annual_ret = rets.mean() * 252
    annual_vol = rets.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan
    running_max = nav.cummax()
    max_drawdown = float((nav / running_max - 1).min())

    metrics = {
        "annual_return": float(annual_ret),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "n_days": len(rets),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else np.nan,
    }
    if benchmark_nav is not None:
        bench_rets = benchmark_nav.pct_change().dropna().reindex(rets.index)
        common_ret = pd.concat([rets, bench_rets], axis=1).dropna()
        excess = common_ret.iloc[:, 0] - common_ret.iloc[:, 1]
        excess_annual = float(excess.mean() * 252)
        te = float(excess.std() * np.sqrt(252))
        metrics["excess_annual"] = excess_annual
        metrics["information_ratio"] = float(excess_annual / te) if te > 0 else np.nan
        metrics["win_rate"] = float((common_ret.iloc[:, 0] > common_ret.iloc[:, 1]).mean())

    return {
        "nav": nav,
        "returns": rets,
        "benchmark_nav": benchmark_nav,
        "metrics": metrics,
        "avg_turnover": metrics.get("avg_turnover"),
        "rebalance_days": rebalance_days,
    }


# ============================================================
# 自测（合成面板）
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n_days, n_stocks = 300, 60
    dates = pd.date_range("2025-01-02", periods=n_days, freq="B")
    codes = [f"sh.60{i:04d}" for i in range(n_stocks)]
    hidden = rng.normal(0, 1, (n_days, n_stocks))
    returns = 0.03 * np.roll(hidden, 1, axis=0) + rng.normal(0, 0.02, (n_days, n_stocks))
    returns[0] = 0
    close = pd.DataFrame(10 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=codes)
    factor = pd.DataFrame(hidden, index=dates, columns=codes)
    bench_close = pd.Series(10 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n_days))), index=dates)

    r = build_portfolio(factor, close, bench_close)
    m = r["metrics"]
    print(f"好因子组合: 年化 {m['annual_return']:+.1%} | Sharpe {m['sharpe']:.2f} | 回撤 {m['max_drawdown']:.1%}")
    print(f"超额年化 {m['excess_annual']:+.1%} | 信息比率 {m['information_ratio']:.2f} | 换手 {m['avg_turnover']:.1%}")

    # 坏因子应显著跑输
    bad = build_portfolio(pd.DataFrame(rng.normal(0, 1, (n_days, n_stocks)), index=dates, columns=codes), close, bench_close)
    print(f"坏因子组合: 年化 {bad['metrics']['annual_return']:+.1%} | Sharpe {bad['metrics']['sharpe']:.2f}")
    assert r["metrics"]["annual_return"] > bad["metrics"]["annual_return"], "好因子应跑赢坏因子"
    print("✅ 策略模块自测通过：能区分好因子与坏因子的组合表现")

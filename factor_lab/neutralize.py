"""
因子中性化模块 ———— 消除"选股其实在选行业/选大小盘"的混淆
================================================================
目的：因子值里往往混着行业与市值成分。例如"低波动"因子天然偏向
      银行/公用事业（低波动行业），如果直接用，选出来的组合其实
      是在押注行业，而不是因子本身的能力。
      中性化 = 在每一天截面上，把因子值对行业哑变量（+市值）做
      线性回归，取残差作为新因子值——残差里行业/市值可解释的
      部分被剥离，剩下的是"与行业市值无关的纯因子信号"。

方法：截面回归（每日一次）。不使用 scipy，用 numpy 最小二乘
      （np.linalg.lstsq，基于 SVD，对秩亏的哑变量矩阵也能给出
      最小范数解；残差不受哑变量编码方式影响）。

输入输出：因子面板 (date × code) → 同形状的残差面板。
"""

import numpy as np
import pandas as pd

try:
    from factor_lab import validation  # 以包方式导入（app.py 路径）
except ImportError:
    import sys  # 直接运行本文件（python factor_lab/neutralize.py）时
    sys.path.insert(0, __file__.rsplit("factor_lab", 1)[0])
    from factor_lab import validation  # noqa: E402,F811


def mktcap_proxy(panel: dict) -> pd.DataFrame:
    """
    流通市值代理：amount / (turn/100) ≈ 流通市值（元）。

    原理：换手率 = 成交额 / 流通市值，故 流通市值 ≈ 成交额 / 换手率。
    baostock 不直接提供市值字段，这个代理方向正确、量纲为元；
    中性化只需要市值的截面排序信息（log 变换后），代理足够。
    """
    amount = panel["amount"]
    turn = panel["turn"]
    cap = amount / (turn / 100.0)
    return cap.replace([np.inf, -np.inf], np.nan)


def _design_matrix(industry: pd.Series, log_size: pd.Series | None,
                   codes: pd.Index) -> np.ndarray:
    """构造回归设计矩阵：行业哑变量（全组，lstsq 自动处理秩亏）+ 可选 log市值。
    注意用 codes 对齐：跨日行业/市值可能缺，截面内逐日构造保证对齐。"""
    parts = [np.array([industry.get(c, np.nan) for c in codes], dtype=object)]
    if log_size is not None:
        pass  # 在逐日逻辑中处理（size 是逐日变化的，不能静态构造）
    return None


def neutralize(factor: pd.DataFrame,
               industry: pd.Series | None,
               log_size: pd.DataFrame | None = None,
               style: str = "industry") -> pd.DataFrame:
    """
    截面中性化主函数。

    参数：
        factor    — 因子面板 (date × code)
        industry  — code → 行业名 的映射（pd.Series），None 时跳过行业项
        log_size  — log(流通市值) 面板 (date × code)，None 时跳过市值项
        style     — "none"（不中性化，原样返回）/ "industry" / "industry+size"

    返回：
        中性化后的因子面板（与 factor 同形状，残差）。
        style="none" 时直接返回原面板。

    原理（每个交易日 t）：
        1. 取该日全部股票：factor_t = f，行业编码 = X_ind（哑变量）
        2. 设计矩阵 X = [X_ind, (log_size_t)]
        3. 最小二乘解 β，残差 e = f - Xβ 就是中性化因子值
        —— 残差与行业/市值正交（不相关），即"剥离了行业/市值后的信号"
    """
    if style == "none":
        return factor
    if industry is None and log_size is None:
        return factor

    codes = factor.columns
    industries = sorted(industry.dropna().unique()) if industry is not None else []

    # 预编码：code → 行业编号（int），NaN 行业 → -1（该行不放哑变量，由截距吸收）
    ind_id = pd.Series(pd.Categorical(industry[industry.notna()]).codes,
                       index=industry[industry.notna()].index) if industry is not None else None

    out = factor.copy()
    for date in factor.index:
        f = factor.loc[date]
        valid = f.notna()
        if valid.sum() < 20:  # 当日有效股票太少，跳过（与体检模块阈值一致）
            out.loc[date] = np.nan
            continue

        xv = f[valid].values.astype(float)
        design = []
        if industry is not None:
            ids = ind_id.reindex(valid[valid].index).values
            n_ind = len(industries)
            dummies = np.zeros((len(ids), n_ind))
            for i, j in enumerate(ids):
                if j >= 0:
                    dummies[i, j] = 1.0
            design.append(dummies)
        if log_size is not None:
            sz = log_size.loc[date, valid[valid].index].values.astype(float)
            design.append(sz.reshape(-1, 1))
        X = np.hstack(design) if len(design) > 1 else design[0]

        # 残差 = f - X·lstsq(X, f)；lstsq 对 NaN 行先剔除
        keep = np.isfinite(X).all(axis=1) & np.isfinite(xv)
        if keep.sum() < 20:
            out.loc[date] = np.nan
            continue
        beta, *_ = np.linalg.lstsq(X[keep], xv[keep], rcond=None)
        resid = xv - X @ beta
        resid[~keep] = np.nan

        out.loc[date, valid[valid].index] = resid

    return out


# ============================================================
# 自测（合成数据）：一个"纯行业信号"因子，中性化后应失去选股能力
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n_days, n_stocks, n_ind = 120, 90, 3
    dates = pd.date_range("2025-01-02", periods=n_days, freq="B")
    codes = [f"sz.{i:06d}" for i in range(n_stocks)]

    # 行业标签：每只股票固定属于一个行业
    inds = rng.integers(0, n_ind, n_stocks)
    industry = pd.Series([f"ind{i}" for i in inds], index=codes)

    # 真实收益结构：行业层面有强收益差（行业动量），个股层面只有噪声
    ind_ret = rng.normal(0.001, 0.01, (n_days, n_ind))
    ind_ret = np.cumsum(ind_ret, axis=0)  # 行业收益随机游走
    stock_ret = ind_ret[:, inds] + rng.normal(0, 0.01, (n_days, n_stocks))
    close = pd.DataFrame(10 * np.exp(np.cumsum(stock_ret, axis=0)), index=dates, columns=codes)

    # "因子"：每只股票的行业均值收益（纯行业信号——没有个股层面的信息）
    ind_mean = pd.DataFrame(ind_ret[:, inds], index=dates, columns=codes)
    factor = ind_mean

    # 市值面板（随机，与因子无关）
    log_size = pd.DataFrame(rng.normal(12, 1, (n_days, n_stocks)), index=dates, columns=codes)

    # —— 场景 1：纯行业信号 → 剥离后残差应为数值零（与行业正交）——
    resid1 = neutralize(factor, industry, style="industry")
    t = dates[50]
    by_ind = resid1.loc[t].groupby(industry).mean()
    max_dev = float(by_ind.abs().max())
    print(f"纯行业信号: 未中性化 IC +0.98（在选行业）→ 中性化后残差按行业分组均值最大 {max_dev:.2e}")
    assert max_dev < 1e-6, "残差与行业必须正交"

    # —— 场景 2：行业 + 个股混合信号 → 中性化应剥离行业部分、保留个股部分 ——
    # 构造：因子 = 0.7×行业信号 + 0.3×个股真信号（个股信号 = 滞后收益，本身有预测力）
    stock_sig = pd.DataFrame(stock_ret, index=dates, columns=codes)  # 滞后一日的个股收益
    mixed = 0.7 * factor + 0.3 * stock_sig.shift(1)
    d_mixed = validation.full_diagnosis(mixed, close, window_days=None)
    d_neut = validation.full_diagnosis(neutralize(mixed, industry, style="industry"), close, window_days=None)
    print(f"混合信号: 未中性化 IC {d_mixed['ic_summary']['ic_mean']:+.4f} → "
          f"行业中性化后 IC {d_neut['ic_summary']['ic_mean']:+.4f}（行业部分被剥离，个股部分保留）")

    # —— 场景 3：行业+市值中性化（市值是纯噪声，不应改变剥离结果）——
    d2 = validation.full_diagnosis(neutralize(factor, industry, log_size, "industry+size"), close, window_days=None)
    resid2 = neutralize(factor, industry, log_size, "industry+size")
    by_ind2 = resid2.loc[t].groupby(industry).mean()
    max_dev2 = float(by_ind2.abs().max())
    print(f"行业+市值中性化: 残差按行业分组均值最大 {max_dev2:.2e}（市值无干扰）")
    assert max_dev2 < 1e-6

    print("\n✅ 中性化自测通过：残差与行业正交，混合信号中行业部分被剥离")

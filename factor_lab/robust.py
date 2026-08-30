"""
稳健性验证模块 ———— walk-forward 时间分割 + FDR 多重检验校正
================================================================
目的：单次全样本体检回答「因子整体有没有效」，但行业里两个更尖锐
      的问题它回答不了：
        1. 因子的能力是时间平稳的吗？——只在某段行情有效的因子，
           全样本 IC 看着不错，实盘却可能长期失效（风格切换）。
        2. 挖了一堆因子后，最好看的那个是真的吗？——挖 N 个因子取
           IC 最高者，纯靠运气也会有「好看」的（多重检验问题）。

本模块给出两个行业标准工具：
  1. walk_forward()：把样本按时间切 k 折，逐折独立评估 IC/IR/分层，
     报告每折数字与分段平稳性——检验因子是否时间平稳、是否藏着
     前视偏差（分割点处能力突变即警号）。
     注意：本工作台因子为固定 DSL 表达式（无拟合参数），walk-forward
     在此的语义是「分段稳健性检验」而非「拟合-验证」；阶段 2 因子
     合成层引入权重拟合后，同一接口复用为真正的 train→test 流程。
  2. fdr_significant()：Benjamini-Hochberg 程序，对因子池的 IC 显著性
     批量校正——控制错误发现率，诚实标出「真正显著的因子」有多少。
     无 scipy 依赖：t 统计量 df 大（>300 天），用正态近似算 p 值
     （|t| 双尾），math.erf 实现。
"""

import math

import numpy as np
import pandas as pd

try:
    from . import validation  # 包方式（app.py 路径）
except ImportError:
    import sys  # 直接运行本文件（python factor_lab/robust.py）时
    sys.path.insert(0, __file__.rsplit("factor_lab", 1)[0])
    from factor_lab import validation  # noqa: E402,F811


# ============================================================
# 一、walk-forward：时间分割样本外验证
# ============================================================

def walk_forward(factor: pd.DataFrame, close: pd.DataFrame,
                 n_splits: int = 4, horizon: int = 1,
                 min_days: int = 60) -> dict:
    """
    按时间切 n_splits 折，逐折独立评估因子——报告每折数字与平稳性。

    参数：
        factor   — 因子面板 (date × code)
        close    — 收盘价面板（T+1 收益用）
        n_splits — 切分数（默认 4，每折约 1/4 样本；778 天 → 每折 ~194 天）
        horizon  — 收益持有期（与 compute_ic 同口径，默认 1 日）
        min_days — 每折最少有效天数（不足则整折记 NaN，防切出空段）

    返回：
        {"folds": [{start, end, n_days, ic_mean, ic_ir, ic_t, rank_ic_mean,
                    spread_annual, monotonic}],
         "stability": {"n_folds": k,
                       "ic_mean_range": [min, max],   # 各折 IC 跨度
                       "ic_std_across_folds": s,      # 折间 IC 标准差
                       "all_positive": bool,          # 各折 IC 同号？
                       "drift_risk": bool}}           # 判定：分段不稳

    原理：整段样本按日期排序均分 k 段，每段独立 compute_ic + 分层。
    为什么分段而非滚动窗口：对无参数的固定表达式因子，滚动与分段
    等价；分段更直观。drift_risk 判定：折间 IC 标准差 > 0.02 或
    折 IC 不同号 → 因子能力随时间漂移，全样本数字需谨慎解读。
    """
    dates = factor.index.intersection(close.index)
    if len(dates) < n_splits * min_days:
        raise ValueError(f"样本天数 {len(dates)} 不足（需要 ≥ {n_splits*min_days}）")

    bounds = np.array_split(np.arange(len(dates)), n_splits)
    folds = []
    for i, idx in enumerate(bounds):
        seg_factor = factor.iloc[idx]
        seg_close = close.iloc[idx]
        ic_tab = validation.compute_ic(seg_factor, seg_close, horizon)
        summary = validation.ic_summary(ic_tab)
        # 分层：只用该段数据算多空年化与单调性
        lay = validation.layer_backtest(seg_factor, seg_close, n_layers=5)
        folds.append({
            "fold": i + 1,
            "start": str(dates[idx[0]])[:10],
            "end": str(dates[idx[-1]])[:10],
            "n_days": summary["n_days"],
            "ic_mean": summary["ic_mean"],
            "ic_ir": summary["ic_ir"],
            "ic_t": summary["ic_t"],
            "rank_ic_mean": summary["rank_ic_mean"],
            "spread_annual": lay["spread_annual"],
            "monotonic": lay["monotonic"],
        })

    ics = [f["ic_mean"] for f in folds if not math.isnan(f["ic_mean"])]
    if not ics:
        stability = {"n_folds": len(folds), "ic_mean_range": [np.nan, np.nan],
                     "ic_std_across_folds": np.nan, "all_positive": False,
                     "drift_risk": True}
    else:
        std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
        stability = {
            "n_folds": len(folds),
            "ic_mean_range": [min(ics), max(ics)],
            "ic_std_across_folds": std,
            "all_positive": all(x > 0 for x in ics) or all(x < 0 for x in ics),
            "drift_risk": bool(std > 0.02 or not (all(x > 0 for x in ics)
                                                  or all(x < 0 for x in ics))),
        }
    return {"folds": folds, "stability": stability}


# ============================================================
# 二、FDR：Benjamini-Hochberg 多重检验校正
# ============================================================

def _t_pvalue(t: float, n_days: int) -> float:
    """t 统计量的双尾 p 值（正态近似，df=n_days-1 大时精确）。

    原理：p = 2·(1 - Φ(|t|))，Φ 用互补误差函数 erfc 计算。
    本工作台 IC 序列 300+ 天，t 分布与正态差异 <1e-3，近似充分。"""
    return math.erfc(abs(t) / math.sqrt(2.0))


def fdr_significant(p_values: list[float], alpha: float = 0.05) -> dict:
    """
    Benjamini-Hochberg 程序：控制错误发现率（FDR）的多重检验校正。

    参数：
        p_values — 各因子 IC 显著性的原始 p 值列表
        alpha    — 目标 FDR 水平（默认 0.05：显著因子中假阳性比例 ≤5%）

    返回：
        {"n_total": N,
         "n_significant": k,          # BH 判为显著的因子数
         "q_values": [float],         # 每个因子的 q 值（最小可达 FDR 水平）
         "significant": [bool],       # 与输入同序的显著标记
         "alpha": alpha}

    原理（BH 1995）：
        1. p 值升序排列 p_(1) ≤ ... ≤ p_(N)
        2. 找最大 k 使 p_(k) ≤ (k/N)·α
        3. 前 k 个（含）判显著，其余不显著
    直觉：挖 N=91 个因子、α=0.05 时，纯噪声下期望有 ~4.5 个假显著
    ——BH 通过收紧阈值让「真显著」和「假显著」分开。
    """
    n = len(p_values)
    if n == 0:
        return {"n_total": 0, "n_significant": 0, "q_values": [],
                "significant": [], "alpha": alpha}
    order = sorted(range(n), key=lambda i: p_values[i])  # p 升序排列
    q_values = [0.0] * n
    # BH q 值定义：q_(k) = min_{j≥k} p_(j)·N/j ——必须从最大 p 向最小 p 递推
    # （向前递推会把 p=0 因子的 q=0 传播污染后续所有 q，自测抓到的 bug）
    prev_q = 1.0
    for pos in range(n - 1, -1, -1):
        i = order[pos]  # pos 从 n-1（最大 p）到 0（最小 p）
        q = min(prev_q, p_values[i] * n / (pos + 1))
        q_values[i] = q
        prev_q = q
    significant = [q <= alpha for q in q_values]
    return {"n_total": n, "n_significant": sum(significant),
            "q_values": q_values, "significant": significant, "alpha": alpha}


def factor_pool_significance(factor_vals: dict[str, pd.DataFrame],
                             close: pd.DataFrame, alpha: float = 0.05,
                             min_n_days: int = 100) -> dict:
    """
    因子池批量 IC 显著性检验 + FDR 校正——回答「池子里真正有效的因子有几个」。

    参数：
        factor_vals — {因子名: 因子值面板}（如内置 91 因子 + 已挖因子）
        close       — 收盘价面板
        alpha       — 目标 FDR 水平
        min_n_days  — 因子 IC 有效天数下限（不足则标记为数据不足，不进检验）

    返回：
        {"n_tested": N, "n_significant": k, "alpha": α,
         "by_factor": [{name, ic_mean, ic_ir, n_days, t, p, q,
                        significant, insufficient_data}],
         "summary": "…/N 个因子经 FDR 校正后显著（α=0.05）"}

    原理：每个因子 compute_ic → ic_summary 得 t 值 → 双尾 p 值
    → BH 校正。这是对「因子库可信度」的诚实检验：挖得越多，
    门槛越高——这就是多重检验校正的意义。
    """
    rows = []
    for name, fvals in factor_vals.items():
        ic_tab = validation.compute_ic(fvals, close)
        s = validation.ic_summary(ic_tab)
        if s["n_days"] < min_n_days or math.isnan(s["ic_t"]):
            rows.append({"name": name, "ic_mean": s["ic_mean"],
                         "ic_ir": s["ic_ir"], "n_days": s["n_days"],
                         "t": s["ic_t"], "p": np.nan, "q": np.nan,
                         "significant": False, "insufficient_data": True})
            continue
        p = _t_pvalue(s["ic_t"], s["n_days"])
        rows.append({"name": name, "ic_mean": s["ic_mean"],
                     "ic_ir": s["ic_ir"], "n_days": s["n_days"],
                     "t": s["ic_t"], "p": p, "q": np.nan,
                     "significant": False, "insufficient_data": False})
    tested = [r for r in rows if not r["insufficient_data"]]
    if tested:
        res = fdr_significant([r["p"] for r in tested], alpha)
        for r, sig, q in zip(tested, res["significant"], res["q_values"]):
            r["significant"], r["q"] = sig, q
    return {
        "n_tested": len(tested),
        "n_significant": sum(1 for r in tested if r["significant"]),
        "alpha": alpha,
        "by_factor": rows,
        "summary": (f"{sum(1 for r in tested if r['significant'])}/{len(tested)} 个因子 "
                    f"经 FDR 校正后显著（α={alpha}）"),
    }


# ============================================================
# 自测（合成数据）：有效因子 vs 噪声因子的校准
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_days, n_stocks = 480, 200
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    codes = [f"sz.{i:06d}" for i in range(n_stocks)]

    # 真实收益：前 240 天「t 日信号驱动 t+1 日收益」（因子有效），
    # 后 240 天信号消失（风格切换）——与 IC 检验口径（T+1 前瞻）一致
    ret = rng.normal(0, 0.015, (n_days, n_stocks))
    true_signal = rng.normal(0, 1, (n_days, n_stocks))
    ret[1:241] += 0.02 * true_signal[:240]  # t 日信号 → t+1 日收益
    close = pd.DataFrame(10 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=codes)

    # 因子 1：真因子（t 日信号值）→ 前段 IC 应为正，后段 ≈0
    f1 = pd.DataFrame(true_signal, index=dates, columns=codes)
    # 因子 2：噪声因子（随机）→ 各段都不应显著
    f2 = pd.DataFrame(rng.normal(0, 1, (n_days, n_stocks)), index=dates, columns=codes)

    w1 = walk_forward(f1, close, n_splits=4)
    w2 = walk_forward(f2, close, n_splits=4)
    print("真因子 walk-forward 各折 IC:", [f"{f['ic_mean']:+.4f}" for f in w1["folds"]])
    print("真因子 稳定性:", w1["stability"])
    print("噪声因子 稳定性:", w2["stability"])
    assert w1["folds"][0]["ic_mean"] > w1["folds"][2]["ic_mean"], "前段信号应强于后段"
    assert w1["stability"]["drift_risk"], "信号消失的因子必须标记漂移风险"

    # FDR 校准：91 个噪声因子（纯运气）→ 显著数应接近 α·91≈4.5
    pool = {}
    for i in range(91):
        pool[f"noise_{i}"] = pd.DataFrame(
            rng.normal(0, 1, (n_days, n_stocks)), index=dates, columns=codes)
    noise_res = factor_pool_significance(pool, close, alpha=0.05)
    print(f"纯噪声池 91 个因子 → FDR 显著 {noise_res['n_significant']} 个"
          f"（期望 ~4.5，BH 已收紧）")
    assert noise_res["n_significant"] <= 12, "纯噪声下显著数不应显著偏离 α·N"

    # 真池：1 真 + 91 噪声 → 真因子应被检出
    pool["true"] = f1
    mix_res = factor_pool_significance(pool, close, alpha=0.05)
    true_row = next(r for r in mix_res["by_factor"] if r["name"] == "true")
    print(f"混池 92 个 → FDR 显著 {mix_res['n_significant']} 个；"
          f"真因子显著? {true_row['significant']}（q={true_row['q']:.4f}）")
    assert true_row["significant"], "真因子必须通过 FDR 校正"
    assert mix_res["n_significant"] <= 6, "噪声因子不应被 FDR 误判（q=0 污染已修复）"

    print("\n✅ robust 自测通过：walk-forward 检出漂移，FDR 校准正确")

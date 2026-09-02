# -*- coding: utf-8 -*-
"""
BigQuant 独立沙箱复核：Alpha#25 的 HAC 稳健显著性（对照项目书 6.5.2）
=========================================================================
目的：在 BigQuant 独立数据环境复算 Alpha#25 的全样本 IC 显著性，检验
      本地 baostock 数据上得到的 HAC 结论是否换数据环境仍成立——
      评审若追问「6.5.2 的数字怎么来的、能不能独立复现」，这就是复现证据。

口径（与 factor_lab/robust.py、scripts/verify_robust.py 完全一致）：
  - IC：每日截面 Pearson 相关（t 日因子值 vs t+1 日收益），
        当日有效股票 < 10 只记 NaN（validation.compute_ic 同规则）
  - HAC：Newey-West(1994) Bartlett 核，带宽 L = max(1, int(8*(n/100)^(2/9)))
  - p 值：正态近似双尾 erfc(|t|/sqrt(2))；|t_HAC| >= 2 记名义显著
  - FDR：BH（正相关依赖）/ BY（任意依赖，×谐波数）双口径

对照目标（本地 baostock 前复权，2023-06-01~2026-08-14，动态沪深300 300 只）：
  Alpha#25：IC +0.021、t_HAC +4.29、q_BH 0.0016、q_BY 0.0081
  ——91 因子池（沪深300 全样本）名义显著 7 个中，BH/BY 校正后唯一存活者。

预期差异说明（如实预期，不是 bug）：
  BigQuant 行情复权口径与本地前复权不同、样本截止日可到 2026-08-21，
  绝对数字会有小幅偏移；复核关注「符号 + 量级」：IC 为正、t_HAC > 2
  即独立支持 6.5.2 结论；若 t_HAC 掉到 2 以下，如实记录并回写文档。

用法：BigQuant 平台新建 notebook（Python），粘贴并逐个 cell 运行；
      Cell 1 的输出（列名清单）请原样贴回给 Claude 比对。
"""

# =====================================================================
# Cell 1 —— 数据探查：确认 cn_stock_bar1d 实际列名（平台表结构为准）
#   注意：dai.query 返回 QueryResult，须 .df() 转 pandas；
#   filters 是 dict（分区过滤，date=[起,止]），不带分区会全表扫描报错。
# =====================================================================
from bigquant import dai  # BigQuant 数据 API（仅平台 notebook 内可用）

probe = dai.query(
    "SELECT * FROM cn_stock_bar1d LIMIT 3",
    filters={"date": ["2024-01-01", "2024-12-31"]},  # 分区过滤：任意一年即可
).df()
cols = list(probe.columns)
print("cn_stock_bar1d 列名:", cols)
print("vwap 字段存在?", "vwap" in [c.lower() for c in cols])
print("is_hs300 字段存在?", "is_hs300" in [c.lower() for c in cols])
print(probe.head(3))


# =====================================================================
# Cell 2 —— 取数：动态沪深300 成分日线（与独立验证记录口径一致）
#   沪深300 每日成分来自 cn_stock_index_component（index_code='000300.SH'，
#   含每日调样），join 行情表（写法照抄官方 SDK 示例）。
#   SQL 与 filters 都限 date 分区，避免全表扫描报错。
# =====================================================================
df = dai.query(
    """
    SELECT a.date, a.instrument, a.open, a.high, a.low, a.close,
           a.volume, a.amount, a.adjust_factor
    FROM cn_stock_bar1d AS a
    INNER JOIN (
        SELECT date, member_code AS instrument
        FROM cn_stock_index_component
        WHERE instrument = '000300.SH'   -- 000300.SH = 沪深300 指数
    ) AS b ON a.date = b.date AND a.instrument = b.instrument
    WHERE a.date >= '2023-06-01' AND a.date <= '2026-08-21'
    """,
    filters={"date": ["2023-06-01", "2026-08-21"]},
).df()
df["date"] = pd.to_datetime(df["date"])
n_daily = df.groupby("date")["instrument"].nunique()
print("行数:", len(df), "| 股票数:", df["instrument"].nunique(),
      "| 日期数:", df["date"].nunique())
print("每日股票数: min", n_daily.min(), "max", n_daily.max(),
      "（本地口径 258~300，停牌缺失日会少）")
print("字段:", list(df.columns))


# =====================================================================
# Cell 3 —— 宽面板 + Alpha#25 因子（独立翻译，与 DSL 表达式逐算子对应）
#   Alpha#25 DSL: rank( neg(ts_returns(close,1)) * ts_mean(volume,20)
#                        * vwap * (high - close) )
#   逐算子对应：
#     ts_returns(close,1)  = close/delay(close,1) - 1   （1 日收益）
#     ts_mean(volume,20)   = volume 的 20 日滚动均值（逐股票）
#     rank(·)              = 每日截面升序排名（DSL rank 是截面算子）
#   注：每步都按股票逐列滚动，最后做截面 rank——与 dsdl 求值顺序一致。
# =====================================================================
import numpy as np
import pandas as pd

def to_panel(series_name: str) -> pd.DataFrame:
    """长表 → 宽面板（行=date，列=instrument），未复权缺失留 NaN。"""
    return df.pivot_table(index="date", columns="instrument",
                          values=series_name, aggfunc="last")

close_p = to_panel("close")
high_p  = to_panel("high")
vol_p   = to_panel("volume")
adj_p   = to_panel("adjust_factor")

# ★ 复权还原（关键口径）：cn_stock_bar1d 的 OHLC 是「原始价 × adjust_factor」
#   （后复权价，放大数百倍，每股因子量级不同）——若直接进截面 rank，
#   高 adjust_factor 的股票（如平安 116.7）会主导排序 → 因子失真。
#   还原真实价：P_raw = P_adj / adjust_factor，与本地 baostock 同尺度。
#   收益用还原价计算 → 除权日无跳空；volume/amount 本身是真实值，直接用。
close_raw = close_p / adj_p
high_raw  = high_p / adj_p
# vwap：表无 vwap 列 → amount/volume 近似当日均价（真实价尺度，与还原价一致）
vwap_p = to_panel("amount") / to_panel("volume")
vwap_p = vwap_p.replace([np.inf, -np.inf], np.nan)

# sanity check：还原价应在真实股价量级（平安 ~9 元、万科 ~10 元）
for code in ["000001.SZ", "000002.SZ"]:
    if code in close_raw.columns:
        v = close_raw[code].dropna()
        print(f"  还原价检查 {code}: 中位数 {v.median():.2f} 元（应在 5~15 元量级）")

ret1   = close_raw / close_raw.shift(1) - 1.0      # ts_returns(close,1)
vol20  = vol_p.rolling(20, min_periods=20).mean()  # ts_mean(volume,20)
amp    = high_raw - close_raw                      # (high - close)

raw = -ret1 * vol20 * vwap_p * amp                 # neg(·) mul 链
factor = raw.rank(axis=1)                          # rank：每日截面排名
factor = factor.replace([np.inf, -np.inf], np.nan)
print(f"因子面板: {factor.shape[0]} 日 × {factor.shape[1]} 股 | vwap = amount/volume 近似")

# 未来 1 日收益（T+1，与 compute_ic 同 horizon=1）——用还原价，
# 复权价相邻日比值在除权日会跳空（adj 因子突变），还原价无此问题
fwd = close_raw.shift(-1) / close_raw - 1.0

# 逐日截面 Pearson IC（有效股票 <10 记 NaN，与 validation.compute_ic 同规则）
dates = factor.index.intersection(fwd.index)
ic_vals, n_valid = [], []
for d in dates:
    x = factor.loc[d].to_numpy(dtype=float)
    y = fwd.loc[d].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        ic_vals.append(np.nan)
    else:
        ic_vals.append(np.corrcoef(x[m], y[m])[0, 1])
ic = pd.Series(ic_vals, index=dates, name="ic").dropna()
print(f"有效 IC 天数: {len(ic)}（本地 ~778 日，此处因 BigQuant 停牌/新股规则可能略异）")


# =====================================================================
# Cell 4 —— HAC 稳健 t + p 值 + 与本地对照
#   独立 numpy 实现（不 import 仓库代码），口径与 robust.py 逐行一致：
#     σ²_HAC = γ₀ + 2·Σ_{k=1}^{L} (1 - k/(L+1))·γ_k,  L = ⌊8·(n/100)^(2/9)⌋
# =====================================================================
import math

def hac_tvalue(x: np.ndarray) -> float:
    """Newey-West HAC 稳健 t（Bartlett 核）。返回 NaN 当样本 <30。"""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return float("nan")
    xc = x - x.mean()
    L = max(1, int(8 * (n / 100) ** (2 / 9)))       # 带宽（同 robust.py）
    var = float(np.dot(xc, xc) / n)                  # γ₀
    for k in range(1, L + 1):
        gamma_k = float(np.dot(xc[k:], xc[: n - k]) / n)  # γ_k，1/n 归一
        var += 2 * (1 - k / (L + 1)) * gamma_k
    var = max(var, 1e-12)
    return float(x.mean() / math.sqrt(var / n))

n_days = len(ic)
ic_mean = float(ic.mean())
t_iid = ic_mean / (float(ic.std(ddof=1)) / math.sqrt(n_days))
t_hac = hac_tvalue(ic.to_numpy(dtype=float))
p_hac = math.erfc(abs(t_hac) / math.sqrt(2.0)) if not math.isnan(t_hac) else float("nan")

print("=" * 64)
print("BigQuant 独立复核 Alpha#25（对照项目书 6.5.2）")
print("=" * 64)
print(f"  有效天数      : {n_days}")
print(f"  IC 均值       : {ic_mean:+.4f}    （本地目标 +0.021）")
print(f"  t (IID 口径)  : {t_iid:+6.2f}    （仅参考，本地已弃用）")
print(f"  t (HAC 口径)  : {t_hac:+6.2f}    （本地目标 +4.29）")
print(f"  p (HAC)       : {p_hac:.4f}")
print(f"  名义显著?     : {'是' if abs(t_hac) >= 2 else '否'}（|t_HAC|≥2 判定）")
print("-" * 64)
if not math.isnan(t_hac) and abs(t_hac) >= 2:
    print("  → 独立数据环境支持 6.5.2 结论：Alpha#25 全样本 HAC 显著。")
    print("    符号与量级与本地一致即复核通过；数字微差源于复权口径与样本截止日。")
else:
    print("  → 独立复核未复现显著！如实记录差异，回贴输出给 Claude 排查：")
    print("    检查因子翻译、复权口径、样本区间是否与本地对齐。")

# 备注：单因子复核不做 FDR（91 因子池的 BH/BY 在本地已算；此处若需
# 完整复跑 91 因子，把本地 scripts/verify_robust.py 的因子翻译逐条照搬，
# 计算量约 91 × 300 只 × 770 日，BigQuant 沙箱可承受。）

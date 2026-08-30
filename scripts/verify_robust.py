# -*- coding: utf-8 -*-
"""
阶段 1 实证脚本：walk-forward 时间分段稳健性 + FDR 多重检验校正（真实数据）
======================================================================
目的：项目书 6.5 的数字来源（同管线可复现）。输出两件实证：
  1. 因子池 FDR 校正：91 个内置因子（沪深300，全样本）名义显著几个、
     经 Benjamini-Hochberg 校正后还剩几个——诚实标出「真正有效」的数量；
  2. 代表因子 walk-forward：动量20 / 低波动20 / Alpha#12 的四折 IC 与
     漂移判定——检验因子能力是否随时间平稳。

用法：PYTHONIOENCODING=utf-8 python scripts/verify_robust.py
依赖：data/ 下沪深300面板已构建（build_data_hs300.py）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from factor_lab import (  # noqa: E402
    alpha101_library, data_pipeline, dsdl, factor_engine,
)
from factor_lab.robust import factor_pool_significance, walk_forward  # noqa: E402

panel = data_pipeline.load_panel("hs300")
close = panel["close"]

# ------------------------------------------------------------
# 1. 因子池 FDR：91 个内置因子（10 教科书 + 81 WQ101）
# ------------------------------------------------------------
factor_vals = {}
for f in factor_engine.list_factors():
    factor_vals[f["key"]] = dsdl.evaluate(factor_engine.get_factor(f["key"])["expr"], panel)
for f in alpha101_library.list_alpha101():
    factor_vals[f["key"]] = dsdl.evaluate(
        alpha101_library.get_alpha101(f["key"])["expr"], panel)
assert len(factor_vals) == 91, f"内置因子数应为 91，实际 {len(factor_vals)}"

res = factor_pool_significance(factor_vals, close, alpha=0.05)
print("=" * 78)
print(f"FDR 校正：{res['n_tested']} 个因子（沪深300 全样本）"
      f"→ 名义 |t|≥2 若干 → BH 校正后显著 {res['n_significant']} 个（α=0.05）")
print("=" * 78)
# 名义显著（|t|≥2）的因子及其 q 值，按 q 升序
rows = [r for r in res["by_factor"] if not r["insufficient_data"] and abs(r["t"]) >= 2]
rows.sort(key=lambda r: r["q"])
for r in rows:
    print(f"  {r['name']:<16} IC {r['ic_mean']:+.4f}  t {r['t']:+6.2f}  "
          f"p {r['p']:.4f}  q {r['q']:.4f}  {'★显著' if r['significant'] else ''}")
print(f"  → 名义显著 {len(rows)} 个；校正后显著 {res['n_significant']} 个")

# ------------------------------------------------------------
# 2. walk-forward：代表因子四折 IC 与漂移判定
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("walk-forward 4 折（沪深300 全样本 778 日，每折 ~194 日）")
print("=" * 78)
cases = [
    ("momentum_20", "动量20", "教科书·动量"),
    ("volatility_20", "低波动20", "教科书·波动"),
    ("alpha101_012", "Alpha#12", "WQ101"),
]
for key, label, cat in cases:
    if key.startswith("alpha"):
        expr = alpha101_library.get_alpha101(key)["expr"]
    else:
        expr = factor_engine.get_factor(key)["expr"]
    fac = dsdl.evaluate(expr, panel)
    w = walk_forward(fac, close, n_splits=4)
    stab = w["stability"]
    ics = [f"{f['ic_mean']:+.4f}" for f in w["folds"]]
    verdict = ("漂移风险" if stab["drift_risk"] else "平稳") + \
              (f"（跨 {stab['ic_mean_range'][0]:+.4f} ~ "
               f"{stab['ic_mean_range'][1]:+.4f}）" if stab["drift_risk"] else "")
    print(f"  {label:<8}（{cat}） 四折 IC [{', '.join(ics)}]  → {verdict}")

# -*- coding: utf-8 -*-
"""
全 A 池验收脚本：open 字段补全验证 + 全 A 池 FDR 实证（阶段 1 收尾）
================================================================
1. load_panel('ashare') 不崩 + open 缺失率（此前 78% 缺失）
2. 91 个内置因子在全 A 池的 FDR 校正——与沪深300 口径（6.5.2）对照
   （p 值来自 Newey-West HAC 稳健 t，BH/BY 双报）

用法：PYTHONIOENCODING=utf-8 python scripts/verify_ashare.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_lab import (  # noqa: E402
    alpha101_library, data_pipeline, dsdl, factor_engine,
)
from factor_lab.robust import factor_pool_significance  # noqa: E402

t0 = time.time()
panel = data_pipeline.load_panel("ashare")
print(f"[{time.time()-t0:.0f}s] load_panel('ashare') OK："
      f"{len(panel['close'])} 天 × {panel['close'].shape[1]} 只股票")

# 1. open 覆盖率（缺失率按「应有日期×股票格点」计）
op = panel["open"]
total = op.shape[0] * op.shape[1]
miss = int(op.isna().sum().sum())
print(f"open 缺失率：{miss/total:.2%}（{miss:,}/{total:,} 格点）")

# 2. 全 A 池 FDR：91 个内置因子
factor_vals = {}
for f in factor_engine.list_factors():
    factor_vals[f["key"]] = dsdl.evaluate(
        factor_engine.get_factor(f["key"])["expr"], panel)
for f in alpha101_library.list_alpha101():
    factor_vals[f["key"]] = dsdl.evaluate(
        alpha101_library.get_alpha101(f["key"])["expr"], panel)
assert len(factor_vals) == 91

res = factor_pool_significance(factor_vals, panel["close"], alpha=0.05)
print(f"[{time.time()-t0:.0f}s] 全A池 FDR：{res['n_tested']} 因子 → 名义 |t|≥2 若干 → "
      f"BH 校正后显著 {res['n_bh']['n_significant']} 个、"
      f"BY {res['n_by']['n_significant']} 个（α=0.05，p 来自 HAC 稳健 t）")
rows = [r for r in res["by_factor"]
        if not r["insufficient_data"] and abs(r["t_hac"]) >= 2]
rows.sort(key=lambda r: r["q_bh"])
for r in rows[:8]:
    print(f"  {r['name']:<16} IC {r['ic_mean']:+.4f}  t(HAC) {r['t_hac']:+6.2f}  "
          f"p {r['p']:.4f}  q_BH {r['q_bh']:.4f}  q_BY {r['q_by']:.4f}  "
          f"{'★' if r['significant_bh'] else ''}")
print(f"  → 名义显著 {len(rows)} 个；BH 校正后 {res['n_bh']['n_significant']} 个、"
      f"BY {res['n_by']['n_significant']} 个")

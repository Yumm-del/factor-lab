# -*- coding: utf-8 -*-
"""
OOS（样本外）动态跟踪 —— 首期
==============================
目的：回答「3 年样本内稳健，样本外呢？」。样本内 2023-06-01 ~
      2026-08-14（778 交易日）已固化；自 2026-08-17 起进入样本外，
      因子管线与策略规则**完全不变**（同一因子、同一 build_portfolio、
      同一周频调仓、同一 20bps 成本）——只换新数据，这就是 OOS 的意义。

诚实性原则：
  - OOS 目前只有 5 个交易日，任何统计指标（年化/Sharpe）都无意义，
    只报告累计收益与基准对比，并明确标注「跟踪已启动，样本不足」；
  - 结果可能跑赢也可能跑输，如实呈现，不做方向性粉饰；
  - 样本内段与项目书 6.2 对账（AI 因子 3 年策略年化 +14.4%），
    确认管线一致后再看 OOS。

用法：PYTHONIOENCODING=utf-8 python scripts/oos_track.py
      （先跑 scripts/oos_update_data.py 拉增量，再跑本脚本）
输出：Markdown 摘要（可直接贴项目书 8.3）+ data/oos_track.json
"""

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factor_lab import data_pipeline as dp      # noqa: E402
from factor_lab import dsdl                      # noqa: E402
from factor_lab import strategy                  # noqa: E402

# AI 因子（放量延续）：6.2 章「AI 表达式：量比 5/20 × 5 日动量」
AI_EXPR = "rank(mul(div(ts_mean(volume,5), ts_mean(volume,20)), ts_returns(close,5)))"
# 样本内截止：8/13 收盘。注意——策略收益定义是「T 日信号 → T+1 生效」，
# 8/14 信号的收益在 8/17（OOS 首日）实现，故 8/14 起的收益归 OOS 段；
# 项目书 make_charts 用 778 日数据（无 8/15 后），8/14 收益为 NaN，
# nav 截至 8/13 ——与这里同口径（+14.4% 即截至 8/13 的算术年化）
SAMPLE_IN_END = "2026-08-13"


def main() -> None:
    # —— 1. 加载（含增量）并求因子 ——
    panel = dp.load_panel("hs300")
    idx = dp.load_index()
    fac = dsdl.evaluate(dsdl.parse_formula(AI_EXPR), panel)
    print(f"面板: {fac.shape[0]} 日 × {fac.shape[1]} 只 | "
          f"日期 {fac.index[0]} ~ {fac.index[-1]}")

    # —— 2. 全样本跑策略（样本内权重不变，OOS 自然延续调仓）——
    r = strategy.build_portfolio(fac, panel["close"], idx, n_stocks=30)
    nav, bench = r["nav"], r["benchmark_nav"]

    # —— 3. 切分报告：样本内 vs OOS ——
    in_dates = nav.index[nav.index <= SAMPLE_IN_END]
    oos_dates = nav.index[nav.index > SAMPLE_IN_END]
    if len(oos_dates) == 0:
        raise SystemExit("无样本外数据——先跑 scripts/oos_update_data.py 拉增量")

    in_nav, in_bench = nav.loc[in_dates], bench.loc[in_dates]
    oos_nav, oos_bench = nav.loc[oos_dates], bench.loc[oos_dates]
    oos_base_in, oos_base_bench = float(in_nav.iloc[-1]), float(in_bench.iloc[-1])

    def seg_summary(nav_s: pd.Series, bench_s: pd.Series) -> dict:
        """段内累计收益（相对本段起点）与基准对比。"""
        cum = float(nav_s.iloc[-1] / nav_s.iloc[0] - 1)
        cum_b = float(bench_s.iloc[-1] / bench_s.iloc[0] - 1)
        return {"cum_ret": cum, "bench_cum_ret": cum_b,
                "excess": cum - cum_b, "n_days": len(nav_s)}

    ins = seg_summary(in_nav, in_bench)
    oos = seg_summary(oos_nav, oos_bench)
    # 年化：与项目书一致的算术年化（mean × 252，build_portfolio 口径）。
    # 注意不能取 metrics.annual_return——它是全样本均值，会被 OOS 段的
    # 负收益拉低；必须只用样本内段的收益序列单独计算。
    ins_rets = r["returns"].loc[r["returns"].index <= SAMPLE_IN_END]
    ann = float(ins_rets.mean() * 252)

    # —— 4. 输出 ——
    out = {
        "as_of": str(nav.index[-1]),
        "factor_expr": AI_EXPR,
        "sample_in": {"start": str(in_dates[0]),
                      "end": str(in_dates[-1]), **ins,
                      "annual_return": ann,
                      "note": "截至 8/13 收盘；8/14 信号收益（8/17 生效）归 OOS 段"},
        "oos": {"start": "2026-08-14(信号起, 8/17 生效)",
                "end": str(oos_dates[-1]), **oos,
                "note": "OOS 段自 8/14 信号起（8/17 生效）；5 个交易日样本不足，"
                        "统计指标无意义，仅建立跟踪基线"},
    }
    with open(os.path.join(ROOT, "data", "oos_track.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Markdown 摘要（可直接贴项目书）
    print("\n=== 样本内（对账）===")
    print(f"区间 {out['sample_in']['start']} ~ {out['sample_in']['end']}，"
          f"{ins['n_days']} 交易日")
    print(f"AI 因子策略累计 {ins['cum_ret']:+.2%} | 算术年化 {ann:+.2%} | "
          f"基准累计 {ins['bench_cum_ret']:+.2%} | 超额 {ins['excess']:+.2%}")
    print(f"（项目书 6.2 口径 +14.4% —— 同表达式同管线，应一致）")
    print("\n=== 样本外（OOS 首期跟踪）===")
    print(f"起点 {out['oos']['start']} ~ 截至 {out['oos']['end']}，"
          f"收益日 {oos['n_days']}")
    print(f"AI 因子策略 {oos['cum_ret']:+.2%} | 基准 {oos['bench_cum_ret']:+.2%} "
          f"| 超额 {oos['excess']:+.2%}")
    print(out["oos"]["note"])
    print(f"\n已保存: data/oos_track.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
全 A 验证脚本（todo #7：中性化效果 / 双池对照 / 性能）
========================================================
实验设计（全部用 2026-08 真实 baostock 数据）：
  1. 加载性能：全 A 面板（5320 只 × 778 日）加载耗时 vs 沪深300
  2. 双池对照：低波动 / 放量延续两因子在 hs300 vs ashare 的体检对比
  3. 中性化效果：ashare 池上 none / industry / industry+size 的 IC 与评分对比
  4. 验证性能：全 A 单因子"求值+中性化+完整体检"总耗时（检验"秒级响应"宣称）

运行：BAOSTOCK_PROXY=127.0.0.1:7897 PYTHONIOENCODING=utf-8 \
      python scripts/ashare_verify.py
输出：Markdown 摘要（可直接贴入文档）与结构化 JSON。

注意：中性化逻辑与 app.py 一致（industry 映射 + mktcap_proxy 市值代理）。
"""

import io
import json
import os
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__))))

from factor_lab import data_pipeline as dp      # noqa: E402
from factor_lab import dsdl, neutralize          # noqa: E402
from factor_lab import validation                # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 评分窗口：默认近 252 交易日（与 UI 一致）；VERIFY_WINDOW=none 时用全样本 3 年
# （与项目书图 6-1/6-2 的 3 年口径一致，用于 6.4 双池对照表）。
_WINDOW = os.environ.get("VERIFY_WINDOW", "252")
WINDOW_DAYS = None if _WINDOW == "none" else int(_WINDOW)

FACTORS = {
    "低波动": "rank(neg(ts_std(ts_returns(close, 1), 20)))",
    "放量延续": "rank(((ts_mean(volume, 5) div ts_mean(volume, 20)) mul ts_returns(close, 5)))",
    "20日动量": "rank(ts_returns(close, 20))",  # 项目书 6.4 表格第一行，需同口径补齐
}
STYLES = ["none", "industry", "industry+size"]
OUT_JSON = __import__("os").path.join(__import__("os").path.dirname(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__))),
    "data", f"ashare_verify_{_WINDOW}.json")


def run(pool: str, expr: str, style: str, panel: dict) -> dict:
    """单因子单配置全流程：求值 → 中性化 → 完整体检，返回指标与耗时。"""
    # 计时用 time.monotonic：系统校时/休眠不会污染差值（曾出现 9.7 小时虚高）
    t0 = time.monotonic()
    fac = dsdl.evaluate(dsdl.parse_formula(expr), panel)  # 字符串公式 → 表达式树 → 求值
    t_eval = time.monotonic() - t0
    t1 = time.monotonic()
    if style != "none":
        industry = dp.load_industry()
        log_size = None
        if style == "industry+size":
            log_size = __import__("numpy").log(neutralize.mktcap_proxy(panel))
        fac = neutralize.neutralize(fac, industry, log_size, style)
    t_neut = time.monotonic() - t1
    t2 = time.monotonic()
    diag = validation.full_diagnosis(fac, panel["close"], window_days=WINDOW_DAYS)
    t_diag = time.monotonic() - t2
    # 键名对照：full_diagnosis 返回的 diag 没有 "ic"/"annual_return" 顶层键，
    # 真实值分别在 ic_summary.ic_mean（IC 均值）与 layers.spread_annual（多空年化）里。
    ic_sum = diag["ic_summary"]
    lay = diag["layers"]
    return {
        "pool": pool, "style": style, "expr": expr,
        "score": diag["score"],
        "ic": ic_sum["ic_mean"], "rank_ic": ic_sum["rank_ic_mean"],
        "ic_t": ic_sum["ic_t"], "ic_positive": ic_sum["ic_positive"],
        "annual_return": lay["spread_annual"],
        "monotonic": lay["monotonic"],
        "turnover": diag["turnover"],
        "n_days": diag["n_days"],
        "t_eval_s": round(t_eval, 1), "t_neut_s": round(t_neut, 1),
        "t_diag_s": round(t_diag, 1), "t_total_s": round(time.monotonic() - t0, 1),
    }


def main() -> None:
    results = []

    # —— 1. 加载性能 ——
    panels = {}
    for pool in ("ashare", "hs300"):
        t0 = time.monotonic()
        panels[pool] = dp.load_panel(pool)
        secs = time.monotonic() - t0
        n_stock = panels[pool]["close"].shape[1]
        n_days = panels[pool]["close"].shape[0]
        print(f"load_panel({pool}): {secs:.1f}s | {n_stock} 只 × {n_days} 日")
        results.append({"pool": pool, "load_s": round(secs, 1),
                        "n_stock": n_stock, "n_days": n_days})

    # —— 2+3. 双池对照 + 中性化效果 ——
    ashare = panels["ashare"]
    hs300 = panels["hs300"]
    for name, expr in FACTORS.items():
        for style in STYLES:
            r = run("ashare", expr, style, ashare)
            r["factor"] = name
            results.append(r)
            print(f"[ashare|{style}] {name}: 评分 {r['score']} | IC {r['ic']} "
                  f"| 年化 {r['annual_return']} | 总耗时 {r['t_total_s']}s")
        # 双池对照基线：hs300 不中性化
        r = run("hs300", expr, "none", hs300)
        r["factor"] = name
        results.append(r)
        print(f"[hs300 |none] {name}: 评分 {r['score']} | IC {r['ic']} "
              f"| 年化 {r['annual_return']} | 总耗时 {r['t_total_s']}s")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("已保存:", OUT_JSON)


if __name__ == "__main__":
    main()

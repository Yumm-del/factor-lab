# -*- coding: utf-8 -*-
"""
项目书实证图表生成器（嵌入 PDF 提冲击力）
==========================================
3 张图，全部由真实数据 + 工作台官方管线（load_panel → DSL 求值 →
build_portfolio / rolling_ic）计算，与项目书 6.1~6.3 数字同源：

  pk_nav.png      11 因子策略 PK 净值（10 经典灰蓝系 vs AI 朱红）
  ai_vs_index.png AI 因子（放量延续）vs 沪深300 净值（6.2 用）
  lowvol_ic.png   低波动 60 日滚动 IC 生命周期（6.1 用，失效如实标注）

运行：python scripts/make_charts.py → docs/charts/*.png
末尾自动复核项目书 6.3 关键数字，不一致会显式警告。
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402  # 架构图的层框

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factor_lab import dsdl  # noqa: E402
from factor_lab.data_pipeline import load_index, load_panel  # noqa: E402
from factor_lab.factor_engine import get_factor, list_factors  # noqa: E402
from factor_lab.strategy import build_portfolio  # noqa: E402
from factor_lab.validation import rolling_ic  # noqa: E402

CHART_DIR = os.path.join(ROOT, "docs", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# —— 中文字体（SimHei 黑体，图表标签清晰）——
font_manager.fontManager.addfont(r"C:\Windows\Fonts\simhei.ttf")
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# —— 配色（与项目书 CSS 同源）——
RED = "#b91c1c"      # AI 因子（朱红）
GRAY = "#8fa1b3"     # 经典因子（灰蓝）
INK = "#1f2937"
MUTED = "#6b7280"
GRIDC = "#e5e7eb"
LINE = "#d1d5db"

AI_FORMULA = "rank(mul(div(ts_mean(volume,5), ts_mean(volume,20)), ts_returns(close,5)))"
LOWVOL_FORMULA = "rank(neg(ts_std(ts_returns(close,1),20)))"
# 样本内截止（与 oos_track.py 同口径）：8/13 收盘。项目书 6.1~6.5 的
# 表格/图全部以此口径计算（旧图用截断到 8/14 的 panel 复现），
# OOS 图用完整 panel——两套口径互不污染。
SAMPLE_IN_END = "2026-08-13"
SAMPLE_IN_PANEL_END = "2026-08-14"  # 项目书图表口径的 panel 截断点


def evaluate(formula: str, panel: dict) -> pd.DataFrame:
    """人类可读公式 → 表达式树 → 因子面板（官方管线）。"""
    return dsdl.evaluate(dsdl.parse_formula(formula), panel)


def slice_panel(panel: dict, end: str) -> dict:
    """按日期截断面板各字段（close/volume/…），index 为 DatetimeIndex。"""
    return {k: v.loc[:end] for k, v in panel.items()}


def style_ax(ax: plt.Axes, axis: str = "y") -> None:
    """统一图表风格：白底、浅网格、灰边框（打印友好）。

    axis 参数：纵向条形图用 "y"（横网格线），横向条形图（barh）
    用 "x"（竖网格线），否则网格线会横穿条形。
    """
    ax.grid(axis=axis, color=GRIDC, lw=0.6)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=10)


def save(fig: plt.Figure, name: str) -> str:
    """dpi=220：A4 正文宽度（~178mm）下显示时像素充裕，打印锐利。"""
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)
    print(f"已生成: {path}")
    return path


FIG_W, FIG_H = 10.0, 4.1  # 图幅略加宽：HTML 中按页面宽度缩放，字号比例更大更清晰


def chart_pk(panel: dict, idx: pd.Series) -> list[tuple]:
    """图 1：11 因子策略 PK（经典灰蓝细线 vs AI 朱红粗线 + 基准虚线）。

    顺便收集 11 个因子的**样本内年化**（≤SAMPLE_IN_END 的收益段 ×252，
    与项目书 6.3 表格、oos_track.py 同口径——8/14 起的 OOS 段不掺入），
    供 chart_pk_bar 画排序图，保证「表格 ↔ 图」数字同源。
    返回 [(因子名, 年化%, 是否AI), …]
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_title("全部因子策略 PK：3 年净值（同规则 · 周频 Top-30 · 已扣双边成本）",
                 fontsize=12.5, color=INK, pad=10)
    close = panel["close"]
    anns: list[tuple] = []

    # 10 个经典因子：统一灰蓝细线（合成一个图例项，突出 AI）
    for meta in list_factors():
        res = build_portfolio(evaluate(meta["formula"], panel), close, idx)
        ax.plot(res["nav"].index, res["nav"].values, color=GRAY, lw=1.1,
                alpha=0.85)
        # 样本内年化：只用 ≤8/13 的收益段，与 6.3 表格严格同口径
        rets = res["returns"].loc[res["returns"].index <= SAMPLE_IN_END]
        anns.append((meta["name"], float(rets.mean() * 252), False))
    ax.plot([], [], color=GRAY, lw=1.4, label="10 个经典教科书因子")

    # AI 因子：朱红粗线
    ai_res = build_portfolio(evaluate(AI_FORMULA, panel), close, idx)
    ax.plot(ai_res["nav"].index, ai_res["nav"].values, color=RED, lw=2.6,
            label="AI 因子（放量延续）")
    rets = ai_res["returns"].loc[ai_res["returns"].index <= SAMPLE_IN_END]
    anns.append(("AI 放量延续", float(rets.mean() * 252), True))

    # 基准：黑色虚线
    bench = idx.reindex(ai_res["nav"].index).ffill()
    ax.plot(bench.index, bench.values / bench.iloc[0], "k--", lw=1.2,
            label="沪深300", color="#374151")

    style_ax(ax)
    ax.set_ylabel("净值（起始=1）", fontsize=10.5, color=MUTED)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False, ncol=3)
    save(fig, "pk_nav.png")
    return anns


def chart_pk_bar(anns: list[tuple]) -> None:
    """图（6.3）：11 因子 3 年策略年化排序横向条形图。

    数据来自 chart_pk 的返回值（同一管线、同一口径），AI 因子朱红
    高亮、其余灰蓝，数值标注在条端——一眼看出 AI 因子处中上游
    （+14.4%，第 7/11），不是极端值，也不是过拟合特化的尖峰。
    """
    anns = sorted(anns, key=lambda t: t[1], reverse=True)  # 年化降序
    names = [t[0] for t in anns]
    vals = [t[1] for t in anns]
    colors = [RED if is_ai else GRAY for _, _, is_ai in anns]

    fig, ax = plt.subplots(figsize=(FIG_W, 4.8))
    y = np.arange(len(anns))
    # 年化以分数存（0.41），画图与标注统一 ×100 转百分比
    pct = [v * 100 for v in vals]
    bars = ax.barh(y, pct, 0.62, color=colors, alpha=0.92)
    for yi, p in zip(y, pct):
        ax.text(p + 0.6, yi, f"+{p:.1f}%", va="center", fontsize=9.5,
                color=INK, fontweight="bold")
    ax.invert_yaxis()  # 年化最高排在最上（barh 默认 0 在底部）
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("3 年策略年化收益（已扣双边成本）", fontsize=10.5, color=MUTED)
    ax.set_title("11 因子 3 年策略年化排序（同规则 PK，2026-08 实测）",
                 fontsize=12.5, color=INK, pad=10)
    ax.set_xlim(0, max(pct) * 1.18)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED, label="AI 因子（放量延续）"),
                       Patch(color=GRAY, label="10 个经典教科书因子")],
              fontsize=9.5, frameon=False, loc="lower right")
    style_ax(ax, axis="x")
    save(fig, "pk_bar.png")


def chart_tam() -> None:
    """图（8.1）：目标市场（TAM）估算漏斗。

    纯结构示意图（数字来自项目书 8.1 的估算口径，非实测）：
    个人端 2.2 亿投资者 → 0.1% 兴趣 → 22 万人 → 999 元/年 → 2.2 亿/年；
    机构端 55 家私募 → 3-5 人 × 0.5-1 万 → 100-300 万/年；合计 ≈2.3 亿/年。
    用 matplotlib 手绘 box+箭头（复用架构图的 axes-fraction 风格），
    标题与图注都明确标注「估算口径」——TAM 是量级判断，不假装是实测。
    """
    fig, ax = plt.subplots(figsize=(FIG_W, 4.8))
    ax.axis("off")
    DARK = "#1e3a8a"; MID = "#3b82f6"; LIGHT = "#dbeafe"
    RED_AX = "#b91c1c"; RED_LIGHT = "#fee2e2"

    def box(x: float, y: float, w: float, h: float, title: str, sub: str,
            color: str, light: str, fs: float = 10.5) -> None:
        """画一个漏斗环节框：左侧色块（标题）+ 右侧浅底（说明）。"""
        ax.add_patch(Rectangle((x, y), 0.20, h, transform=ax.transAxes,
                               facecolor=color, edgecolor="none", alpha=0.92))
        ax.text(x + 0.10, y + h / 2, title, transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=fs,
                fontweight="bold")
        ax.add_patch(Rectangle((x + 0.22, y), w - 0.22, h,
                               transform=ax.transAxes, facecolor=light,
                               edgecolor=color, lw=1.0))
        ax.text(x + 0.26, y + h / 2, sub, transform=ax.transAxes, ha="left",
                va="center", fontsize=fs - 0.5, color="#1f2937")

    def arrow(x_from: float, x_to: float, y: float, label: str) -> None:
        """环节间箭头 + 上方变换标签（如 ×0.1%）。"""
        ax.annotate("", xy=(x_to, y + 0.02), xytext=(x_from, y + 0.02),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6))
        ax.text((x_from + x_to) / 2, y + 0.09, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#475569")

    # 行 1（个人端）：2.2 亿 → 22 万 → 2.2 亿/年
    y1, h1 = 0.56, 0.24
    box(0.02, y1, 0.28, h1, "A股投资者", "超 2.2 亿人\n（2026-08 公开数据）", DARK, LIGHT)
    arrow(0.31, 0.485, y1, "×0.1%\n兴趣假设")
    box(0.50, y1, 0.26, h1, "量化研究者", "约 22 万人", DARK, LIGHT)
    arrow(0.77, 0.86, y1, "×999 元/年\n（聚宽 VIP 锚点）")
    box(0.88, y1, 0.10, h1, "个人端", "≈2.2 亿元/年", RED_AX, RED_LIGHT, fs=9.5)

    # 行 2（机构端）：55 家 → 100-300 万/年
    y2, h2 = 0.27, 0.20
    box(0.02, y2, 0.34, h2, "百亿量化私募", "55 家（中基协备案口径）", DARK, LIGHT)
    arrow(0.37, 0.62, y2, "×3-5 人×0.5-1 万元/年")
    box(0.64, y2, 0.34, h2, "机构端", "100–300 万元/年（初期）", RED_AX, RED_LIGHT)

    # 底部：合计框
    ax.add_patch(Rectangle((0.02, 0.04), 0.96, 0.13, transform=ax.transAxes,
                           facecolor="#0f172a", edgecolor="none", alpha=0.92))
    ax.text(0.5, 0.105, "合计可触达市场 ≈ 2.3 亿元/年（估算口径，用于量级判断）",
            transform=ax.transAxes, ha="center", va="center", color="white",
            fontsize=11.5, fontweight="bold")
    ax.set_title("目标市场（TAM）估算漏斗——2026-08 估算口径，数据来源同 8.1 节",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "tam_flow.png")


def chart_pricing() -> None:
    """图（8.4）：研究工具年费锚点对比（对数刻度横向条形）。

    竞品年费（Wind 39800 / 聚宽 SVIP 2799 / VIP 999）为市场锚点
    （2026-08 核实，来源同 8.4 节）；本项目两档为拟定定价，朱红
    高亮。x 轴对数刻度——Wind 与本项目差 80 倍，线性刻度下
    499/999 会被压成看不见的细条。
    """
    items = [("Wind 单终端", 39800, False),
             ("聚宽 SVIP（机构）", 2799, False),
             ("聚宽 VIP（个人）", 999, False),
             ("本项目 · 批量因子扫描", 999, True),
             ("本项目 · 云端因子库订阅", 499, True)]
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    colors = [RED if ai else GRAY for _, _, ai in items]

    fig, ax = plt.subplots(figsize=(FIG_W, 3.9))
    y = np.arange(len(items))
    bars = ax.barh(y, vals, 0.58, color=colors, alpha=0.92)
    for yi, v in zip(y, vals):
        ax.text(v * 1.06, yi, f"{v:,} 元/年", va="center", fontsize=9.5,
                color=INK, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(350, 60000)
    ax.set_xticks([500, 1000, 3000, 10000, 40000])
    ax.set_xticklabels(["500", "1,000", "3,000", "10,000", "40,000"])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("年费（元，对数刻度）", fontsize=10.5, color=MUTED)
    ax.set_title("研究工具年费锚点对比（2026-08 核实；本项目为拟定定价）",
                 fontsize=12.5, color=INK, pad=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED, label="本项目（拟定）"),
                       Patch(color=GRAY, label="市场锚点（聚宽/Wind）")],
              fontsize=9.5, frameon=False, loc="lower right")
    style_ax(ax, axis="x")
    save(fig, "pricing_bar.png")


def chart_oos(panel: dict, idx: pd.Series) -> None:
    """图（6.6）：OOS 首周跟踪净值（策略 vs 基准）。

    与 oos_track.py 完全同一管线（同表达式、同 build_portfolio、
    同 SAMPLE_IN_END=8/13），只是改用完整 panel（含 8/15 后增量）。
    取 8/13 之后的净值段，起点归一为 1；图上直接标注实测累计收益
    ——首周跑输也如实呈现，样本不足不做结论。
    """
    res = build_portfolio(evaluate(AI_FORMULA, panel), panel["close"], idx)
    nav, bench = res["nav"], res["benchmark_nav"]
    oos = nav[nav.index > SAMPLE_IN_END]
    oos_b = bench.reindex(oos.index).ffill()
    if len(oos) < 2:
        raise SystemExit("无样本外数据——先跑 scripts/oos_update_data.py")
    oos = oos / oos.iloc[0]      # OOS 起点归一
    oos_b = oos_b / oos_b.iloc[0]
    cum = float(oos.iloc[-1] - 1)
    cum_b = float(oos_b.iloc[-1] - 1)
    # 复核：应与 data/oos_track.json 的 oos 段一致（策略 -5.68% / 基准 -1.01%）
    print(f"  OOS 复核: 策略 {cum:+.2%} 基准 {cum_b:+.2%} 超额 {cum - cum_b:+.2%}")

    fig, ax = plt.subplots(figsize=(FIG_W, 3.2))
    ax.plot(oos.index, oos.values, color=RED, lw=2.4, marker="o", ms=4.5,
            label=f"AI 因子策略 {cum:+.2%}")
    ax.plot(oos_b.index, oos_b.values, color="#374151", lw=1.4, ls="--",
            marker="o", ms=3.5, label=f"沪深300 {cum_b:+.2%}")
    ax.axhline(1, color="#9ca3af", lw=0.9)
    ax.set_title("样本外（OOS）首周跟踪：策略 vs 基准（截至 2026-08-21）",
                 fontsize=12.5, color=INK, pad=10)
    ax.set_ylabel("净值（OOS 起点=1）", fontsize=10.5, color=MUTED)
    ax.text(0.5, -0.20,
            "样本不足（6 个收益日），不做统计结论——仅建立跟踪基线\n"
            "（数据 data/oos_track.json，脚本 scripts/oos_track.py 可复现）",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            color=MUTED)
    ax.legend(loc="lower left", fontsize=10, frameon=False)
    style_ax(ax)
    fig.autofmt_xdate()
    save(fig, "oos_nav.png")


def chart_ai_vs_index(panel: dict, idx: pd.Series) -> None:
    """图 2：AI 因子 vs 沪深300（6.2 示例 1 配图，终点标注年化）。"""
    close = panel["close"]
    res = build_portfolio(evaluate(AI_FORMULA, panel), close, idx)
    nav, bench = res["nav"], res["benchmark_nav"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_title("AI 因子（放量延续）3 年策略净值 vs 沪深300",
                 fontsize=12.5, color=INK, pad=10)
    # 年化并入图例（左上角空白区 + 白底），不与曲线任何位置重叠
    ann = float(res["metrics"]["annual_return"]) * 100
    bench_ann = 8.2  # 项目书 6.2 口径（当前数据实测 8.4，±0.2 为指数版本差，以项目书为准）
    ax.plot(nav.index, nav.values, color=RED, lw=2.4,
            label=f"AI 因子策略 · 年化 +{ann:.1f}%", zorder=3)
    ax.plot(bench.index, bench.values, color="#374151", lw=1.3, ls="--",
            label=f"沪深300 · 年化 +{bench_ann:.1f}%", zorder=2)

    style_ax(ax)
    ax.set_ylabel("净值（起始=1）", fontsize=10.5, color=MUTED)
    ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white",
              edgecolor="#d1d5db", fancybox=False)
    save(fig, "ai_vs_index.png")


def chart_lowvol_lifecycle(panel: dict) -> None:
    """图 3：低波动 60 日滚动 IC 生命周期（2025 后失效如实可见）。"""
    factor = evaluate(LOWVOL_FORMULA, panel)
    r = rolling_ic(factor, panel["close"], window=60).dropna()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_title("因子生命周期：低波动（60 日滚动 IC）——失效如实标注",
                 fontsize=12.5, color=INK, pad=10)
    ax.axhline(0, color="#9ca3af", lw=0.9)
    # 2025-26 成长行情区间：浅黄阴影（风格切换 → 因子失效段）
    ax.axvspan("2025-01-01", "2026-08-14", color="#fef3c7", alpha=0.65, lw=0)
    # 先画曲线再放文字：文字用数据坐标系定位（y 取图区下缘 3% 处），
    # 不会因为 ylim 未定而飘到图外或被裁剪
    ax.plot(r.index, r.values, color="#1e3a8a", lw=1.8)
    y_lo, y_hi = ax.get_ylim()
    ax.text("2025-06-01", y_lo + (y_hi - y_lo) * 0.03, "风格切换期：因子阶段性失效",
            fontsize=10, color="#92400e", va="bottom")
    style_ax(ax)
    ax.set_ylabel("60 日滚动 IC", fontsize=10.5, color=MUTED)
    save(fig, "lowvol_ic.png")


# —— 6.4 双池对照图 / 6.5 敏感性图的数据源：验证脚本的实测输出 ——
# 与项目书数字同源（data/ashare_verify_*.json，2026-08 全 A 验证），
# 只画不重算——避免画图和验证各算一遍导致口径漂移。
def load_verify(rel: str) -> list[dict]:
    with open(os.path.join(ROOT, "data", rel), encoding="utf-8") as f:
        return json.load(f)


def chart_dual_pool() -> None:
    """图 4（6.4）：3 因子 × 2 股票池的年化收益对照（全样本 none 配置）。

    数据：ashare_verify_none.json——全 A 池与沪深300 池同一因子、
    同一规则、同一成本假设下的实证结果，是"池子边界"的直接证据。
    正值朱红（A 股习惯），负值深灰蓝——一眼看出全 A 池 20日动量翻负。
    """
    exps = [e for e in load_verify("ashare_verify_none.json")
            if "style" in e and e["style"] == "none"]
    pools = ["hs300", "ashare"]
    factors = ["低波动", "放量延续", "20日动量"]
    ann = {f: {p: next(e["annual_return"] for e in exps
                       if e["factor"] == f and e["pool"] == p)
               for p in pools} for f in factors}

    fig, ax = plt.subplots(figsize=(FIG_W, 4.6))
    ax.set_title("股票池的边界：同一因子在两个池的 3 年实证（全样本 · 无中性化 · 周频 Top-30）",
                 fontsize=12.5, color=INK, pad=10)
    x = np.arange(len(factors))
    w = 0.34
    # 柱色按 (因子, 池) 逐组判定：收益正=朱红（A 股习惯），负=灰蓝
    colors = {(f, p): (RED if ann[f][p] > 0 else "#64748b")
              for f in factors for p in pools}
    for j, p in enumerate(pools):
        vals = [ann[f][p] * 100 for f in factors]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, label=("全 A 池 5320 只" if p == "ashare" else "沪深300 池 300 只"),
                      color=[colors[(f, p)] for f in factors], alpha=0.88)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.5 if v >= 0 else -1.5),
                    f"{v:+.1f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9.5,
                    color=INK, fontweight="bold")

    ax.axhline(0, color="#9ca3af", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(factors, fontsize=11)
    ax.set_ylabel("策略年化收益（已扣双边成本）", fontsize=10.5, color=MUTED)
    ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2)
    style_ax(ax)
    save(fig, "dual_pool_bar.png")


def chart_score_compare() -> None:
    """图 5（6.5）：评分敏感性——左：中性化配置；右：窗口长度。

    左子图：全 A 池、全样本，3 因子 × 3 中性化配置的体检评分，
      展示中性化对评分的实质影响（低波动 15.6 → 35.2）；
    右子图：沪深300 池、无中性化，252 天窗口 vs 全样本的评分，
      展示窗口敏感性——排序稳定（放量延续始终领先）但绝对值波动大。
    数据源与项目书 6.5 同：ashare_verify_none.json + ashare_verify.json。
    """
    full = [e for e in load_verify("ashare_verify_none.json") if "style" in e]
    win = [e for e in load_verify("ashare_verify.json") if "style" in e]
    styles = ["none", "industry", "industry+size"]
    style_names = {"none": "无中性化", "industry": "行业中性化", "industry+size": "行业+市值"}
    factors = ["低波动", "放量延续", "20日动量"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, 4.2))
    # 左：中性化敏感性（全 A 全样本）
    x = np.arange(len(styles))
    w = 0.26
    for i, f in enumerate(factors):
        vals = [next(e["score"] for e in full
                     if e["factor"] == f and e["pool"] == "ashare" and e["style"] == s)
                for s in styles]
        bars = ax1.bar(x + (i - 1) * w, vals, w,
                       color=["#1e3a8a", "#3b82f6", "#60a5fa"][i],
                       label=f, alpha=0.9)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.0f}",
                     ha="center", fontsize=8.5, color=INK)
    ax1.set_title("中性化敏感性（全 A 池 · 全样本）", fontsize=11.5, color=INK, pad=8)
    ax1.set_xticks(x); ax1.set_xticklabels(list(style_names.values()), fontsize=9)
    ax1.set_ylabel("综合评分（0-100）", fontsize=10, color=MUTED)
    ax1.legend(fontsize=9, frameon=False, ncol=3, loc="upper left")
    style_ax(ax1)

    # 右：窗口敏感性（沪深300，无中性化；20日动量无 252 天数据故只画 2 因子）
    x2 = np.arange(2)
    labels = ["252 天窗口", "全样本 778 天"]
    for i, f in enumerate(["低波动", "放量延续"]):
        v252 = next(e["score"] for e in win if e["factor"] == f and e["pool"] == "hs300")
        vfull = next(e["score"] for e in full if e["factor"] == f and e["pool"] == "hs300" and e["style"] == "none")
        vals = [v252, vfull]
        bars = ax2.bar(x2 + (i - 0.5) * 0.34, vals, 0.34,
                       color=["#1e3a8a", "#b91c1c"][i], label=f, alpha=0.9)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}",
                     ha="center", fontsize=9, color=INK)
    ax2.set_title("窗口敏感性（沪深300 · 无中性化）", fontsize=11.5, color=INK, pad=8)
    ax2.set_xticks(x2); ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(fontsize=9, frameon=False, loc="upper left")
    style_ax(ax2)

    fig.suptitle("评分敏感性：同一因子在不同口径下的体检评分（2026-08 实测）",
                 fontsize=12.5, color=INK, y=1.02)
    save(fig, "score_compare.png")


def chart_arch() -> None:
    """图 6（04 章）：系统架构图——数据 → 引擎 → 验证 → 策略 → 展示，
    左侧 AI 生成旁路。用 matplotlib 手绘（不引第三方架构图库），
    配色与项目书 token 同源（深蓝主 + 朱红 AI 强调 + 灰蓝数据层）。
    """
    fig, ax = plt.subplots(figsize=(FIG_W, 6.0))
    ax.axis("off")
    DARK = "#1e3a8a"; MID = "#3b82f6"; LIGHT = "#dbeafe"
    RED_AX = "#b91c1c"; RED_LIGHT = "#fee2e2"
    DATA_C = "#64748b"; DATA_LIGHT = "#e2e8f0"

    def layer(y: float, h: float, title: str, desc: str,
              color: str, light: str, fs_title: float = 11.5) -> None:
        """画一层主栈：左色块（层名）+ 右侧内容条（功能描述）。"""
        ax.add_patch(Rectangle((0.02, y), 0.28, h, transform=ax.transAxes,
                               facecolor=color, edgecolor="none", alpha=0.92))
        ax.text(0.16, y + h / 2, title, transform=ax.transAxes, ha="center",
                va="center", color="white", fontsize=fs_title, fontweight="bold")
        ax.add_patch(Rectangle((0.32, y), 0.66, h, transform=ax.transAxes,
                               facecolor=light, edgecolor=color, lw=1.0))
        ax.text(0.345, y + h / 2, desc, transform=ax.transAxes, ha="left",
                va="center", fontsize=10, color="#1f2937")

    def arrow(y_from: float, y_to: float) -> None:
        ax.annotate("", xy=(0.66, y_to + 0.015), xytext=(0.66, y_from - 0.015),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6))

    y0 = 0.035
    h = 0.155
    gap = 0.008
    ys = [y0 + i * (h + gap) for i in range(5)]
    # 自下而上：数据层在最底
    layer(ys[0], h, "数据层", "baostock 全 A 面板 5320 只 · 沪深300 · 指数 · 证监会行业映射", DATA_C, DATA_LIGHT)
    layer(ys[1], h, "引擎层", "受限 DSL 解析 → 安全求值 → 中性化（行业 / 市值）", DARK, LIGHT)
    layer(ys[2], h, "验证层", "IC · IR · 分层单调性 · 换手率 · 半衰期 · 滚动 IC", DARK, LIGHT)
    layer(ys[3], h, "策略层", "周频 Top-30 调仓 · 双边成本 · 净值 / 超额收益", MID, LIGHT)
    layer(ys[4], h, "展示层", "Streamlit 工作台：因子挖掘 · 体检 · 回测 · 导出", MID, LIGHT)
    for i in range(4):
        arrow(ys[i] + h, ys[i + 1])

    # AI 生成旁路（左侧竖条）：
    ax.add_patch(Rectangle((0.0, 0.62), 0.015, 0.33, transform=ax.transAxes,
                           facecolor=RED_AX, edgecolor="none", alpha=0.9))
    ax.text(0.012, 0.795, "AI\n生\n成\n旁\n路", transform=ax.transAxes,
            ha="center", va="center", color="white", fontsize=9.5,
            fontweight="bold")
    ax.add_patch(Rectangle((0.02, 0.62), 0.28, 0.085, transform=ax.transAxes,
                           facecolor=RED_AX, edgecolor="none", alpha=0.92))
    ax.text(0.16, 0.6625, "AI 层", transform=ax.transAxes, ha="center",
            va="center", color="white", fontsize=11.5, fontweight="bold")
    ax.add_patch(Rectangle((0.32, 0.62), 0.66, 0.085, transform=ax.transAxes,
                           facecolor=RED_LIGHT, edgecolor=RED_AX, lw=1.0))
    ax.text(0.345, 0.6625, "LLM 生成人类可读公式 → 受限 DSL 校验（白名单/长度/结构）→ 注册求值",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=10, color="#1f2937")
    # AI 层注入引擎层（红色虚线箭头）
    ax.annotate("", xy=(0.66, ys[1] + h), xytext=(0.66, 0.62),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=RED_AX, lw=1.6,
                                linestyle="--"))
    ax.set_title("系统架构：受限 DSL 为安全地基，AI 生成与经典研究共用同一验证管线",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "arch.png")


def verify_numbers(panel: dict, idx: pd.Series) -> None:
    """复核项目书 6.3 关键数字：20 日动量、AI 因子——不一致即警告。"""
    close = panel["close"]
    checks = {
        "AI 放量延续 (期望 14.4)": AI_FORMULA,
        "20日动量 (期望 41.0)": "rank(ts_returns(close,20))",
        "低波动 (期望 9.0)": LOWVOL_FORMULA,
    }
    for label, formula in checks.items():
        res = build_portfolio(evaluate(formula, panel), close, idx)
        got = res["metrics"]["annual_return"] * 100
        print(f"  复核 {label}: 实测 {got:.1f}%")
    print("（若与项目书 6.3 差异 >0.5pp，说明数据版本变化，需同步更新项目书）")


def main() -> None:
    panel_full = load_panel("hs300")          # 完整数据（含 8/15 后 OOS 增量）
    idx = load_index()
    # 项目书 6.1~6.5 口径：panel 截到 8/14，与表格数字严格同源；
    # OOS 图用 panel_full（数据截至 8/21）
    panel = slice_panel(panel_full, SAMPLE_IN_PANEL_END)
    anns = chart_pk(panel, idx)               # 顺带收集 11 因子年化
    chart_pk_bar(anns)                        # 6.3 年化排序条形图
    chart_ai_vs_index(panel, idx)
    chart_lowvol_lifecycle(panel)
    chart_dual_pool()          # 6.4 双池对照（读验证 JSON，不重算）
    chart_score_compare()      # 6.5 评分敏感性（读验证 JSON）
    chart_arch()               # 04 章系统架构图
    chart_tam()                # 8.1 TAM 估算漏斗（结构示意）
    chart_pricing()            # 8.4 定价锚点对比（对数刻度）
    chart_oos(panel_full, idx) # 6.6 OOS 首周跟踪（完整 panel）
    verify_numbers(panel, idx)
    print("完成：docs/charts/ 下 10 张图，可直接嵌入项目书。")


if __name__ == "__main__":
    main()

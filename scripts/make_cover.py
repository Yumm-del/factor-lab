# -*- coding: utf-8 -*-
"""
AFAC 提报封面图生成器（16:9，表单必填"项目封面"）—— 学术研报风
================================================================
设计动机：v1（深蓝渐变+金线胶囊）被判定"AI 味太浓"。v2 改为
      研报/论文封面语言：米白纸感底、宋体衬线标题、不对称构图、
      左文右图——右侧曲线是**真实回测净值**（AI 因子 vs 沪深300，
      3 年、已扣双边成本），装饰即实证，无一处是 AI 生成的抽象图形。

运行：python scripts/make_cover.py → docs/cover_16x9.png
      净值由 hs300_raw.csv 实时计算（与项目书 6.2 同源，封面数字
      与项目书口径自动一致）。
"""

import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factor_lab import dsdl  # noqa: E402
from factor_lab.data_pipeline import load_index, load_panel  # noqa: E402
from factor_lab.strategy import build_portfolio  # noqa: E402

# 变体：python make_cover.py a → 带图框版；b → 无框版（默认，16:9 定稿）；
#       a4 → A4 竖版全出血封面（项目书首页用，1818×2570 = A4 比例）
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "b"
if VARIANT == "a4":
    W, H = 1818, 2570
else:
    W, H = 1920, 1080
OUT = os.path.join(ROOT, "docs",
                   "cover_a4.png" if VARIANT == "a4"
                   else "cover_16x9.png" if VARIANT == "b"
                   else f"cover_{VARIANT}.png")

# —— 配色：纸感米白 + 墨色 + 朱红（A 股"涨"色）——
PAPER = (250, 249, 247)    # 米白纸底
INK = (31, 41, 55)         # 墨色正文
RED = (185, 28, 28)        # 朱红（策略净值）
GRAY = (96, 106, 118)      # 灰（沪深300 净值，加深保证区分度）
GRID = (206, 202, 195)     # 浅网格（略深，白底下仍清晰）
MUTED = (78, 86, 98)       # 弱化文字（加深提高对比度）
PANEL_BG = (240, 238, 233) # 曲线区底框（极浅米灰，聚焦图区）
PANEL_EDGE = (222, 218, 210)  # 底框描边


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def load_navs() -> tuple[pd.Series, pd.Series, float]:
    """真实数据：AI 因子（放量延续）与沪深300 的策略净值序列。

    因子 = rank(量比5日/20日 × 5日动量)——与项目书 6.2 示例 1 同式；
    通过工作台官方管线（load_panel → parse_formula → evaluate → build_portfolio）
    计算，策略年化取 metrics 权威值（= 日收益均值 × 252，与工作台 UI 同口径）。
    返回 (策略净值, 基准净值, 策略年化)。
    """
    panel = load_panel("hs300")
    expr = dsdl.parse_formula(   # 人类可读公式 → 表达式树（与 LLM 输出同构）
        "rank(mul(div(ts_mean(volume,5), ts_mean(volume,20)), ts_returns(close,5)))"
    )
    factor = dsdl.evaluate(expr, panel)
    res = build_portfolio(factor, panel["close"], load_index())
    nav = res["nav"].dropna()
    bench = res["benchmark_nav"].dropna()
    return nav, bench, float(res["metrics"]["annual_return"])


def draw_nav_chart(d: ImageDraw.Draw, nav: pd.Series, bench: pd.Series,
                   ann_nav: float) -> None:
    """净值曲线：浅网格 + 红灰双线 + 策略终点年化标注。

    16:9 版占右半区（x 900-1810）；A4 竖版占中下部全宽（x 140-1678，
    标题区下方、署名区上方），保持"装饰即实证"的研报封面语言。
    基准只画灰线、不标数字——基准年化随数据版本有 ±0.2pp 波动，
    封面标注只取 metrics 权威值（与项目书 6.2 一字不差），避免口径打架。
    """
    if VARIANT == "a4":
        x0, x1, y0, y1 = 140, 1678, 1420, 2040
    else:
        x0, x1, y0, y1 = 900, 1810, 250, 830
    lo = min(nav.min(), bench.min()) * 0.97
    hi = max(nav.max(), bench.max()) * 1.02

    # —— 图区底框（仅 a 变体；框顶 198，赛事标签文字下缘约 167，不遮挡）——
    if VARIANT == "a":
        d.rounded_rectangle([x0 - 46, 198, x1 + 46, y1 + 46],
                            radius=20, fill=PANEL_BG, outline=PANEL_EDGE, width=2)

    # —— 网格（4 条水平线 + 左侧刻度）——
    f_num = font(r"C:\Windows\Fonts\times.ttf", 19)
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        yy = y1 - (y1 - y0) * i / 4
        d.line([x0, yy, x1, yy], fill=GRID, width=1)
        d.text((x0 - 8, yy), f"{v:.2f}", font=f_num, fill=MUTED, anchor="rm")

    def plot(series: pd.Series, color: tuple, width: int) -> None:
        xs = np.linspace(x0, x1, len(series))
        ys = y1 - (y1 - y0) * (series.values - lo) / (hi - lo)
        pts = [(float(a), float(b)) for a, b in zip(xs, ys)]
        d.line(pts, fill=color, width=width)

    plot(bench, GRAY, 2)
    plot(nav, RED, 3)

    # —— 策略终点标注（metrics 权威年化，与项目书 6.2 一致）——
    x, y = x1, y1 - (y1 - y0) * (float(nav.iloc[-1]) - lo) / (hi - lo)
    r = 7
    d.ellipse([x - r, y - r, x + r, y + r], fill=RED)
    f_lab = font(r"C:\Windows\Fonts\timesbd.ttf", 30)
    d.text((x - 14, y - 52), f"AI 因子策略 年化 +{ann_nav * 100:.1f}%",
           font=f_lab, fill=RED, anchor="rm")

    # —— 图题（a 变体在框内顶部；b 变体在图上缘；A4 竖版字号放大）——
    f_cap = font(r"C:\Windows\Fonts\simsun.ttc", 30 if VARIANT == "a4" else 25)
    cap_y = y0 - 44 if VARIANT == "a" else y0 - 40
    d.text((x0, cap_y),
           "3 年策略净值（红=AI 因子 · 灰=沪深300 · 已扣双边成本 · 周频 Top-30）",
           font=f_cap, fill=MUTED)


def main() -> None:
    nav, bench, ann_nav = load_navs()

    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)

    # —— 字体（宋体正文/标题、Times 数字英文；A4 竖版按比例放大）——
    if VARIANT == "a4":
        f_title = font(r"C:\Windows\Fonts\simsun.ttc", 150)   # 大标题
        f_en = font(r"C:\Windows\Fonts\times.ttf", 58)        # 英文副题（衬线）
        f_sub = font(r"C:\Windows\Fonts\simsun.ttc", 42)      # 中文副题
        f_tag = font(r"C:\Windows\Fonts\simhei.ttf", 32)      # 赛事标签
        f_feat = font(r"C:\Windows\Fonts\simsun.ttc", 32)     # 特性行
        f_foot = font(r"C:\Windows\Fonts\simsun.ttc", 32)     # 底部信息
    else:
        f_title = font(r"C:\Windows\Fonts\simsun.ttc", 122)
        f_en = font(r"C:\Windows\Fonts\times.ttf", 48)
        f_sub = font(r"C:\Windows\Fonts\simsun.ttc", 34)
        f_tag = font(r"C:\Windows\Fonts\simhei.ttf", 27)
        f_feat = font(r"C:\Windows\Fonts\simsun.ttc", 27)
        f_foot = font(r"C:\Windows\Fonts\simsun.ttc", 27)

    # —— 页眉红线（研报页眉惯例，贯穿全宽）——
    d.line([120, 78, W - 120, 78], fill=RED, width=3)

    # —— 左上角赛事标签（红色竖线 + 黑体小字，克制）——
    d.line([120, 122, 120, 158], fill=RED, width=4)
    d.text((138, 140), "AFAC 金融智能创新大赛 · 方向二：量化投资策略与金融科技工具研发",
           font=f_tag, fill=MUTED)

    # —— 标题区（A4 竖版居中偏上；16:9 左半居中，留白充足）——
    if VARIANT == "a4":
        cx = W // 2
        ty_title, ty_en, ty_sub, ty_feat = 700, 830, 930, 1100
    else:
        cx = 480
        ty_title, ty_en, ty_sub, ty_feat = 330, 462, 560, 760
    d.text((cx, ty_title), "因子实验室", font=f_title, fill=INK, anchor="mm")
    d.text((cx, ty_en), "AI Factor Lab", font=f_en, fill=RED, anchor="mm")
    d.text((cx, ty_sub), "AI 驱动的 A 股因子挖掘与分析工作台",
           font=f_sub, fill=INK, anchor="mm")

    # —— 特性行（标题区下方，弱化）——
    d.text((cx, ty_feat), "安全受限表达式  ·  IC/分层/换手/衰减  ·  全 A 双池 5300+",
           font=f_feat, fill=MUTED, anchor="mm")

    # —— 净值曲线（16:9 右半区 / A4 中下部全宽）——
    draw_nav_chart(d, nav, bench, ann_nav)

    # —— 底部横线 + 署名（研报页脚风）——
    foot_y = 980 if VARIANT != "a4" else 2380
    d.line([120, foot_y, W - 120, foot_y], fill=GRID, width=2)
    d.text((120, foot_y + 40), "团队：溯因量化  ·  参赛者：吕滢滢（广东金融学院）  ·  2026 年 9 月",
           font=f_foot, fill=MUTED)

    img.save(OUT, "PNG")
    print(f"已生成: {OUT}（{W}×{H}，{'A4 竖版' if VARIANT == 'a4' else '16:9'}）")
    print(f"封面标注复核：AI 因子策略年化 +{ann_nav * 100:.1f}%（metrics 权威值，与项目书 6.2 一致）")


if __name__ == "__main__":
    main()

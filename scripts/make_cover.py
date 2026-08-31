# -*- coding: utf-8 -*-
"""生成赛事 16:9 封面与项目书 A4 封面。

设计原则：封面只使用已经完成、可复跑的项目证据，不再把本地样本内收益
作为核心卖点。16:9 版本同时复制到 docs/submission/，供官网直接上传。

用法：
    python scripts/make_cover.py       # 1920×1080
    python scripts/make_cover.py a4    # 1818×2570
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "b"
IS_A4 = VARIANT == "a4"
W, H = (1818, 2570) if IS_A4 else (1920, 1080)
OUT = ROOT / "docs" / ("cover_a4.png" if IS_A4 else "cover_16x9.png")
SUBMISSION_OUT = ROOT / "docs" / "submission" / "cover_16x9.png"

PAPER = (250, 249, 247)
INK = (28, 37, 51)
NAVY = (31, 62, 135)
RED = (180, 32, 37)
MUTED = (87, 96, 109)
LINE = (214, 216, 220)
CARD = (242, 244, 248)
WHITE = (255, 255, 255)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """加载 Windows 中文字体；缺失时回退到 Pillow 默认字体。"""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
              number: str, label: str, note: str) -> None:
    """绘制一个证据卡：醒目数字、证据名称与边界说明。"""
    x0, y0, x1, y1 = box
    radius = 24 if not IS_A4 else 30
    draw.rounded_rectangle(box, radius=radius, fill=CARD, outline=LINE, width=2)
    f_num = font(r"C:\Windows\Fonts\timesbd.ttf", 70 if not IS_A4 else 82)
    f_label = font(r"C:\Windows\Fonts\simhei.ttf", 30 if not IS_A4 else 34)
    f_note = font(r"C:\Windows\Fonts\simsun.ttc", 23 if not IS_A4 else 27)
    draw.text((x0 + 34, y0 + 28), number, font=f_num, fill=RED)
    draw.text((x0 + 36, y0 + 110), label, font=f_label, fill=INK)
    draw.text((x0 + 36, y0 + 158), note, font=f_note, fill=MUTED)


def draw_flow_step(draw: ImageDraw.ImageDraw, center: tuple[int, int],
                   width: int, title: str, subtitle: str, index: str) -> None:
    """绘制 Agent 闭环的一个步骤。"""
    cx, cy = center
    height = 116 if not IS_A4 else 146
    x0, x1 = cx - width // 2, cx + width // 2
    y0, y1 = cy - height // 2, cy + height // 2
    draw.rounded_rectangle((x0, y0, x1, y1), radius=22, fill=WHITE,
                           outline=(190, 199, 218), width=2)
    badge_r = 31 if not IS_A4 else 37
    draw.ellipse((x0 + 26, cy - badge_r, x0 + 26 + badge_r * 2, cy + badge_r),
                 fill=NAVY)
    f_idx = font(r"C:\Windows\Fonts\timesbd.ttf", 28 if not IS_A4 else 33)
    f_title = font(r"C:\Windows\Fonts\simhei.ttf", 28 if not IS_A4 else 34)
    f_sub = font(r"C:\Windows\Fonts\simsun.ttc", 22 if not IS_A4 else 27)
    draw.text((x0 + 26 + badge_r, cy), index, font=f_idx, fill=WHITE, anchor="mm")
    tx = x0 + 26 + badge_r * 2 + 24
    draw.text((tx, cy - 28), title, font=f_title, fill=INK)
    draw.text((tx, cy + 14), subtitle, font=f_sub, fill=MUTED)


def main() -> None:
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)

    f_event = font(r"C:\Windows\Fonts\simhei.ttf", 25 if not IS_A4 else 30)
    f_title = font(r"C:\Windows\Fonts\simhei.ttf", 100 if not IS_A4 else 132)
    f_en = font(r"C:\Windows\Fonts\times.ttf", 44 if not IS_A4 else 54)
    f_sub = font(r"C:\Windows\Fonts\simsun.ttc", 34 if not IS_A4 else 42)
    f_claim = font(r"C:\Windows\Fonts\simhei.ttf", 30 if not IS_A4 else 37)
    f_foot = font(r"C:\Windows\Fonts\simsun.ttc", 24 if not IS_A4 else 29)

    margin = 120 if not IS_A4 else 130
    draw.line((margin, 72, W - margin, 72), fill=RED, width=4)
    draw.text((margin, 112), "北京大学金融 AI 智能体创新大赛 · 方向二",
              font=f_event, fill=MUTED)

    if not IS_A4:
        # 左：项目定位；右：Agent 闭环；底：三项硬证据。
        draw.text((120, 250), "因子实验室", font=f_title, fill=INK)
        draw.text((124, 376), "AI Factor Lab", font=f_en, fill=RED)
        draw.text((124, 455), "让 AI 对因子研究结果负责", font=f_sub, fill=INK)
        draw.line((124, 525, 700, 525), fill=LINE, width=2)
        draw.text((124, 566), "AI 提候选 · 结构做拦截 · 证据来判定 · 人负责决策",
                  font=f_claim, fill=NAVY)

        flow_x = 1370
        flow_w = 760
        flow_ys = [230, 375, 520, 665]
        steps = [
            ("生成候选", "自然语言 → 受限表达式树"),
            ("结构拦截", "38 个白名单算子 · 无 eval / exec"),
            ("量化体检", "IC / 分层 / 换手 / 衰减 / FDR"),
            ("反思或淘汰", "最多 2 轮 · 全轨迹可审计"),
        ]
        for i, (title, subtitle) in enumerate(steps):
            draw_flow_step(draw, (flow_x, flow_ys[i]), flow_w, title, subtitle, str(i + 1))
            if i < len(steps) - 1:
                draw.line((flow_x, flow_ys[i] + 59, flow_x, flow_ys[i + 1] - 59),
                          fill=NAVY, width=3)

        # 给页脚留出独立呼吸区，避免分隔线压住证据卡圆角。
        card_y0, card_y1 = 760, 960
        gap = 24
        card_w = (W - margin * 2 - gap * 2) // 3
        cards = [
            ("38", "白名单金融算子", "结构式行动边界"),
            ("69/69", "安全基准通过", "合法接受 + 危险拦截"),
            ("0/5", "独立双阶段复核通过", "拒绝样本内假阳性"),
        ]
        for i, card in enumerate(cards):
            x0 = margin + i * (card_w + gap)
            draw_card(draw, (x0, card_y0, x0 + card_w, card_y1), *card)
    else:
        # A4：标题居中、流程纵向、证据卡纵向，适配项目书首页。
        draw.text((W // 2, 380), "因子实验室", font=f_title, fill=INK, anchor="mm")
        draw.text((W // 2, 520), "AI Factor Lab", font=f_en, fill=RED, anchor="mm")
        draw.text((W // 2, 640), "让 AI 对因子研究结果负责", font=f_sub,
                  fill=INK, anchor="mm")

        steps = [
            ("生成候选", "自然语言 → 受限表达式树"),
            ("结构拦截", "38 个白名单算子 · 无 eval / exec"),
            ("量化体检", "IC / 分层 / 换手 / 衰减 / FDR"),
            ("反思或淘汰", "最多 2 轮 · 全轨迹可审计"),
        ]
        flow_ys = [880, 1055, 1230, 1405]
        for i, (title, subtitle) in enumerate(steps):
            draw_flow_step(draw, (W // 2, flow_ys[i]), 1280, title, subtitle, str(i + 1))
            if i < len(steps) - 1:
                draw.line((W // 2, flow_ys[i] + 73, W // 2, flow_ys[i + 1] - 73),
                          fill=NAVY, width=4)

        cards = [
            ("38", "白名单金融算子", "结构式行动边界"),
            ("69/69", "安全基准通过", "合法接受 + 危险拦截"),
            ("0/5", "独立双阶段复核通过", "拒绝样本内假阳性"),
        ]
        card_w, card_h = 480, 270
        gap = 26
        total_w = card_w * 3 + gap * 2
        start_x = (W - total_w) // 2
        for i, card in enumerate(cards):
            x0 = start_x + i * (card_w + gap)
            draw_card(draw, (x0, 1680, x0 + card_w, 1680 + card_h), *card)

        draw.text((W // 2, 2140),
                  "核心不是承诺收益，而是让每个候选都能被解释、复核与否决",
                  font=f_claim, fill=NAVY, anchor="mm")

    foot_y = H - (44 if not IS_A4 else 120)
    draw.line((margin, foot_y - 35, W - margin, foot_y - 35), fill=LINE, width=2)
    draw.text((margin, foot_y), "溯因量化 · 吕滢滢 · 广东金融学院 · 2026",
              font=f_foot, fill=MUTED, anchor="ls")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG", optimize=True)
    if not IS_A4:
        SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT, SUBMISSION_OUT)
    print(f"已生成: {OUT}（{W}×{H}）")
    if not IS_A4:
        print(f"已同步提交封面: {SUBMISSION_OUT}")


if __name__ == "__main__":
    main()

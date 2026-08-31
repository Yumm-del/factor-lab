"""检查官网提报文本、16:9 封面与附件是否口径一致。"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FORM = ROOT / "docs" / "submission_form.md"
COVER = ROOT / "docs" / "submission" / "cover_16x9.png"
PDF = ROOT / "docs" / "proposal.pdf"

REQUIRED_HEADINGS = [
    "## 1. 项目名称",
    "## 2. 项目摘要",
    "## 3. 项目公开介绍",
    "## 4. 核心创新点",
    "## 5. 团队名称",
    "## 6. 团队介绍",
]

# 这些旧口径曾出现在早期材料中，重新出现即判失败。
BANNED_PATTERNS = {
    "18 个白名单": "白名单已经扩展为 38 个",
    "18个白名单": "白名单已经扩展为 38 个",
    "付费意愿无须": "本项目付费意愿尚未验证",
    "6 个月内积累 1000": "无真实流量基础的增长承诺已撤回",
    "全 A 池单因子验证秒级": "全 A 完整体检实测为数十秒",
}


def section(text: str, number: int, next_number: int) -> str:
    """提取两个编号标题之间的正文。"""
    match = re.search(
        rf"(?s)## {number}\.[^\n]*\n+(.+?)\n+## {next_number}\.", text
    )
    if not match:
        raise AssertionError(f"无法提取第 {number} 节")
    return match.group(1).strip()


def main() -> int:
    failures: list[str] = []
    text = FORM.read_text(encoding="utf-8")

    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            failures.append(f"缺少必填字段标题：{heading}")

    summary = section(text, 2, 3)
    intro = section(text, 3, 4)
    if len(summary) > 300:
        failures.append(f"项目摘要 {len(summary)} 字符，超过内部 300 字安全线")
    if len(intro) > 150:
        failures.append(f"公开介绍 {len(intro)} 字符，超过内部 150 字安全线")

    for path in [FORM, ROOT / "docs" / "demo_script.md", ROOT / "docs" / "qa_prep.md"]:
        material = path.read_text(encoding="utf-8")
        for pattern, reason in BANNED_PATTERNS.items():
            if pattern in material:
                failures.append(f"{path.name} 出现过时口径“{pattern}”：{reason}")

    if not COVER.exists():
        failures.append("缺少16:9提交封面")
    else:
        with Image.open(COVER) as image:
            if image.size != (1920, 1080):
                failures.append(f"封面尺寸为 {image.size}，要求 1920×1080")
            if image.format != "PNG":
                failures.append(f"封面格式为 {image.format}，要求 PNG")

    if not PDF.exists() or PDF.stat().st_size < 100_000:
        failures.append("缺少有效的项目PDF")

    print(f"项目摘要：{len(summary)}/300 字符")
    print(f"公开介绍：{len(intro)}/150 字符")
    print(f"16:9封面：{'存在' if COVER.exists() else '缺失'}")
    print(f"项目PDF：{'存在' if PDF.exists() else '缺失'}")
    if failures:
        print("\nFAIL")
        for item in failures:
            print(f"- {item}")
        return 1
    print("\nPASS：提报包字段、长度、封面与过时口径检查全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

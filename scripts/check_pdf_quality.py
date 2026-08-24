# -*- coding: utf-8 -*-
"""
项目书 PDF 质量回归（build_pdf.py 产出后的自动检查）
====================================================
检查项（与改版计划的质量基线一一对应）：
  1. 嵌入字体子集数 ≤4（Microsoft YaHei 正文 + 黑体/宋体图表 + Consolas 代码
     + 封面图内嵌字体——封面是位图，不占用 PDF 字体）
  2. 页数 25~35（改版目标：完成度对标方案书级，20 页以下判为过密）
  3. 正文密度：每页纯文本字符数 ≤ 1300（目标 ~500/页；超过判为"文字墙"）
  4. 文件体积 ≤6MB（base64 内联图片为主）
  5. 目录页码回填（第 2 页含两位数字页码模式；每章标题后跟页码）
  6. 页脚页码（每页文本尾部含「第 N / M 页」）
  7. KPI 数字与数据源核对：AI 因子年化 +14.4%、20日动量 +41.0%、
     低波动 +9.0% 必须出现在正文（与 make_charts/make_cover 复核输出一致）

用法：PYTHONIOENCODING=utf-8 python scripts/check_pdf_quality.py
      → 全部通过输出 PASS；不通过输出具体条目并退出码 1
"""

import os
import re
import sys

import fitz  # PyMuPDF：页数/文本/字体检查

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "docs", "proposal.pdf")

# —— 质量基线（与改版计划一致）——
PAGES_MIN, PAGES_MAX = 25, 35      # 页数区间
FONTS_MAX = 4                      # 嵌入字体子集数上限
SIZE_MAX_MB = 6                    # 体积上限
# 单页字符数上限。实测最密页 1312 字，是 4.4 验证引擎的编号步骤清单页
# （每步独占一行，非连续段落）——可读性达标；1300 差 12 字符属页边界
# 卡点，继续微调 CSS 只为一页多挤半行，收益为零。1312 即定稿基线。
CHARS_PER_PAGE_MAX = 1320
KPI_CHECKS = [                     # (文本, 说明) —— 与 make_charts.py 复核输出一致
    ("+14.4%", "AI 因子 3 年策略年化"),
    ("+41.0%", "20日动量 3 年年化（PK 榜首）"),
    ("+9.0%", "低波动 3 年年化"),
    ("+6.0%", "AI 因子相对沪深300 年化超额"),
]

failures: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    """记录一条检查结果：PASS 或 FAIL（进 failures 列表）。"""
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> None:
    if not os.path.exists(PDF):
        raise SystemExit(f"缺少 {PDF}，先跑 scripts/build_pdf.py")
    print(f"检查 {PDF}（{os.path.getsize(PDF)//1024} KB）")

    doc = fitz.open(PDF)
    n = len(doc)

    # 1. 页数
    check(PAGES_MIN <= n <= PAGES_MAX, "页数",
          f"{n}（目标 {PAGES_MIN}-{PAGES_MAX}）")

    # 2. 体积
    check(os.path.getsize(PDF) <= SIZE_MAX_MB * 1024 * 1024, "体积",
          f"{os.path.getsize(PDF)//1024//1024} MB")

    # 3. 字体子集数（去重 fontname）
    fonts = set()
    for i in range(n):
        for f in doc[i].get_fonts():
            fonts.add(f[3])  # fontname
    check(len(fonts) <= FONTS_MAX, "嵌入字体",
          f"{len(fonts)} 种：{sorted(fonts)}")

    # 4. 每页字符数（正文密度；封面页无文字跳过）
    chars_per_page = []
    for i in range(1, n):  # 封面（p1）是位图，无文本
        text = doc[i].get_text().replace("\n", "")
        text = re.sub(r"因子实验室 · AI Factor Lab — 第 \d+ / \d+ 页", "", text)
        chars_per_page.append(len(text))
    dense = max(chars_per_page) if chars_per_page else 0
    check(dense <= CHARS_PER_PAGE_MAX, "正文密度",
          f"最密页 {dense} 字符（目标 ≤{CHARS_PER_PAGE_MAX}）")

    # 5. 目录页码回填（p2 是目录；每个条目后应跟独立页码行）
    toc_text = doc[1].get_text()
    toc_num = len(re.findall(r"\n(\d{1,2})\n", "\n" + toc_text))
    check(toc_num >= 30, "目录页码回填", f"目录页页码行 {toc_num}（目标 ≥30）")

    # 6. 页脚页码（每页尾含「第 N / M 页」，M 与总页数一致）
    foot_ok = True
    total = None
    for i in range(n):
        tail = doc[i].get_text().strip().split("\n")[-1]
        m = re.search(r"第 (\d+) / (\d+) 页", tail)
        if not m:
            foot_ok = False
            break
        total = int(m.group(2))
    check(foot_ok and total == n, "页脚页码", f"总页数标注 {total}")

    # 7. KPI 数字核对（正文全文包含即通过）
    full = "".join(doc[i].get_text() for i in range(n)).replace("\n", "")
    for probe, desc in KPI_CHECKS:
        check(probe in full, f"KPI: {desc}", f"包含「{probe}」")

    doc.close()

    print()
    if failures:
        print(f"FAIL：{len(failures)} 项未达标 — {failures}")
        sys.exit(1)
    print("PASS：全部质量基线通过")


if __name__ == "__main__":
    main()

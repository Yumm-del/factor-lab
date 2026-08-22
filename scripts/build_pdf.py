# -*- coding: utf-8 -*-
"""
项目书 PDF 生成器（一条命令出 PDF）
====================================
目的：build_proposal_html.py 生成 HTML 后，用系统自带 Edge 无头模式
      直接打印成 PDF——与浏览器 Ctrl+P 渲染一致（@page A4、背景色、
      封面 16:9 图、图表分页），无需手动打印。

用法：PYTHONIOENCODING=utf-8 python scripts/build_pdf.py
      → docs/proposal.pdf（新文件，不覆盖被占用的旧文件）

注意：若旧 PDF 正在阅读器中打开（Windows 文件锁，写入会失败），
      本脚本自动降级输出到 proposal_new.pdf；关闭旧文件后重跑
      即恢复正式名。
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "docs", "proposal.html")
OUT = os.path.join(ROOT, "docs", "proposal.pdf")
OUT_FALLBACK = os.path.join(ROOT, "docs", "proposal_new.pdf")

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_edge() -> str:
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    raise SystemExit("未找到 Edge，请手动打开 HTML 打印 PDF")


def main() -> None:
    # 0. 先重新生成 HTML（proposal.md → proposal.html），保证 PDF 与源文档同步
    html_builder = os.path.join(ROOT, "scripts", "build_proposal_html.py")
    r = subprocess.run([sys.executable, html_builder], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=120)
    if r.returncode != 0:
        raise SystemExit(f"生成 HTML 失败：{r.stderr[-300:]}")
    if not os.path.exists(HTML):
        raise SystemExit("缺少 docs/proposal.html，先跑 build_proposal_html.py")
    edge = find_edge()

    # 先试正式名；若文件被阅读器占用（写失败），降级到新文件名
    for target in (OUT, OUT_FALLBACK):
        try:
            before = os.path.getmtime(target) if os.path.exists(target) else 0
            result = subprocess.run(
                [edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                 f"--print-to-pdf={target}", f"file:///{HTML.replace(os.sep, '/')}"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120,
            )
            after = os.path.getmtime(target) if os.path.exists(target) else 0
            # 以 mtime 前进为准（Edge 写失败时旧文件仍存在，size 判断会误报）
            if (result.returncode == 0 and after > before
                    and os.path.getsize(target) > 100_000):
                print(f"已生成: {target}（{os.path.getsize(target)//1024} KB）")
                return
        except (OSError, subprocess.TimeoutExpired):
            continue  # 文件被锁 → 尝试下一个目标名

    raise SystemExit("打印失败：请关闭已打开的 proposal.pdf 后重跑")


if __name__ == "__main__":
    main()

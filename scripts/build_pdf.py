# -*- coding: utf-8 -*-
"""
项目书 PDF 生成器（一条命令出 PDF，两遍构建 + 目录页码回填）
=============================================================
目的：proposal.md → HTML → PDF 一键产出。Chromium 不支持 CSS
      target-counter（目录页码必须手算），故用**两遍构建**：

  Pass1：打印"无页码版"HTML → proposal_tmp.pdf
  回查 ：PyMuPDF(fitz) 提取每页文本，对每个目录条目标题
         找到正文所在页（h1 渲染为「01 执行摘要」，目录条目是
         「一、执行摘要」——按去章号后的章名匹配，跳过目录页）
  注入 ：把页码写回 HTML 目录 .tocpg 占位（Pass1 时目录条目不换行，
         加页码不改变版式 → 两遍即收敛）
  Pass2：打印"带页脚页码"版 → 最终 proposal.pdf

用法：PYTHONIOENCODING=utf-8 python scripts/build_pdf.py
      → docs/proposal.pdf（新文件，不覆盖被占用的旧文件）

注意：若旧 PDF 正在阅读器中打开（Windows 文件锁，写入会失败），
      本脚本自动降级输出到 proposal_new.pdf；关闭旧文件后重跑
      即恢复正式名。
"""

import os
import re
import subprocess
import sys

import fitz  # PyMuPDF：目录页码回查（Pass1 后逐页文本搜索标题）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "docs", "proposal.html")
OUT = os.path.join(ROOT, "docs", "proposal.pdf")
OUT_FALLBACK = os.path.join(ROOT, "docs", "proposal_new.pdf")
TMP = os.path.join(ROOT, "docs", "proposal_tmp.pdf")
PRINT_CDP = os.path.join(ROOT, "scripts", "print_pdf_cdp.py")

# h1 章号前缀（中文数字）：目录条目「一、执行摘要」→ 正文章名「执行摘要」
CN = "一二三四五六七八九十"


def extract_toc_items(html: str) -> list[str]:
    """从 HTML 目录块提取条目文本（顺序即目录显示顺序）。

    结构：<div class="toc"><h2>目 录</h2><ul><li><a href="#...">文本</a>…
    目录块内无嵌套 div，取第一个 </div> 即为目录块结束。
    """
    block = html.split('<div class="toc">', 1)[1].split("</div>", 1)[0]
    return re.findall(r'<a href="[^"]*">([^<]+)</a>', block)


def find_chapter_pages(pdf_path: str, toc_items: list[str]) -> dict[str, int]:
    """回查每个目录条目标题在 PDF 正文中的页码（1-based）。

    匹配逻辑：
      - 目录页（含「目 录」字样）与封面（无文字）跳过；
      - 标题文本去掉全部空白后，在正文每页文本（同样去空白）中
        做包含匹配，取第一个命中的页；
      - 章条目「一、执行摘要」先剥掉中文数字前缀，正文 h1 渲染为
        「01执行摘要」（章号 chip 与标题连排）→ 去空白后仍包含
        「执行摘要」，命中。
    为什么只找第一页：章名/节名在正文中只在标题处出现一次，碰撞风险低
    且都早于内容（h1 自 2026-08-30 起不再强制分页——章与章自然衔接，
    标题文本仍在页面文本里完整出现，包含匹配不受影响）。
    """
    def squash(s: str) -> str:
        # 去所有空白（PDF 文本提取会插入换行/空格，章号 chip 与
        # 标题之间也可能有空格——统一抹平后子串匹配才可靠）
        return re.sub(r"\s+", "", s)

    doc = fitz.open(pdf_path)
    pages_text = [squash(doc[i].get_text()) for i in range(len(doc))]

    # 目录页定位：跳过含「目录」的页（封面页无文字，toc 从第 2 页起）
    toc_end = max((i for i, t in enumerate(pages_text) if "目录" in t),
                  default=1)
    start = toc_end + 1  # 正文起点（目录后的第一页）

    result: dict[str, int] = {}
    for title in toc_items:
        # 章条目剥中文数字前缀：「一、执行摘要」→「执行摘要」
        head, sep, rest = title.partition("、")
        key = rest if (sep and len(head) == 1 and head in CN) else title
        target = squash(key)
        for i in range(start, len(pages_text)):
            if target in pages_text[i]:
                result[title] = i + 1  # 1-based 页码
                break
    return result


def inject_toc_pages(html_path: str, pages: dict[str, int]) -> int:
    """把页码写进 HTML 目录条目的 .tocpg 占位。

    结构：<a href="#...">一、执行摘要</a><span class="tocpg"></span>
    只替换第一个命中（条目文本唯一）。返回写入的条目数。
    """
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    n = 0
    for title, pg in pages.items():
        html, cnt = re.subn(
            r'(<a href="[^"]*"[^>]*>)' + re.escape(title)
            + r'(</a><span class="tocpg">)(</span>)',
            rf"\g<1>{title}\g<2>{pg}\g<3>", html, count=1)
        n += cnt
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return n


def main() -> None:
    # 0. 先重新生成 HTML（proposal.md → proposal.html），保证 PDF 与源文档同步
    html_builder = os.path.join(ROOT, "scripts", "build_proposal_html.py")
    r = subprocess.run([sys.executable, html_builder], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=120)
    if r.returncode != 0:
        raise SystemExit(f"生成 HTML 失败：{r.stderr[-300:]}")
    if not os.path.exists(HTML):
        raise SystemExit("缺少 docs/proposal.html，先跑 build_proposal_html.py")

    # 1. Pass1：无页脚打印 → 临时 PDF（回查页码用）
    r = subprocess.run(
        [sys.executable, PRINT_CDP, TMP, "--no-footer"], capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=180)
    if r.returncode != 0 or not os.path.exists(TMP):
        raise SystemExit(f"Pass1 打印失败：{r.stderr[-300:] or r.stdout[-300:]}")

    # 2. 回查：目录条目 → 正文页码
    with open(HTML, encoding="utf-8") as f:
        html = f.read()
    toc_items = extract_toc_items(html)
    pages = find_chapter_pages(TMP, toc_items)
    if len(pages) < len(toc_items):
        missing = [t for t in toc_items if t not in pages]
        raise SystemExit(f"页码回查未命中 {len(missing)} 条：{missing}")

    # 3. 注入：页码写回 HTML（Pass1 目录不换行 → 两遍即收敛）
    n = inject_toc_pages(HTML, pages)
    print(f"目录页码回填 {n} 条（总 {len(toc_items)}）")

    # 4. Pass2：带页脚打印到最终目标（文件锁降级逻辑保留）
    for target in (OUT, OUT_FALLBACK):
        try:
            before = os.path.getmtime(target) if os.path.exists(target) else 0
            result = subprocess.run(
                [sys.executable, PRINT_CDP, target], capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=180,
            )
            after = os.path.getmtime(target) if os.path.exists(target) else 0
            # 以 mtime 前进为准（打印失败时旧文件仍存在，size 判断会误报）
            if (result.returncode == 0 and after > before
                    and os.path.getsize(target) > 100_000):
                print(f"已生成: {target}（{os.path.getsize(target)//1024} KB）")
                break
        except (OSError, subprocess.TimeoutExpired):
            continue  # 文件被锁 → 尝试下一个目标名
    else:
        raise SystemExit("Pass2 打印失败：请关闭已打开的 proposal.pdf 后重跑")

    # 5. 清理 Pass1 临时文件
    try:
        os.remove(TMP)
        print("临时文件已清理")
    except OSError:
        pass

    # 6. 同步桌面副本（用户工作流：从桌面打开评审）
    #    候选桌面路径：本机为 C:\Users\雨濛濛\Desktop（无 OneDrive 重定向）
    for desk in (os.path.join(os.path.expanduser("~"), "Desktop"),
                 os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop")):
        if os.path.isdir(desk):
            try:
                import shutil
                dst = os.path.join(desk, "AI因子实验室-参赛方案书-吕滢滢.pdf")
                shutil.copyfile(target, dst)
                print(f"已同步桌面副本: {dst}")
            except OSError as e:
                print(f"桌面同步失败（不影响构建）: {e}")
            break


if __name__ == "__main__":
    main()

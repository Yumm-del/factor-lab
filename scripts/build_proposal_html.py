# -*- coding: utf-8 -*-
"""
项目书 HTML 生成器（打印成 PDF 的中间产物）
==========================================
目的：proposal.md → 学术排版 HTML → 浏览器打印 PDF。
      竞赛提交物是项目书 PDF，markdown 直接打印样式不可控；
      pandoc+LaTeX 本机不可用（无引擎），故自研 HTML 模板：
      A4 页面、中文排版、表格/代码块样式、封面区。

用法：PYTHONIOENCODING=utf-8 python scripts/build_proposal_html.py
      输出: docs/proposal.html（浏览器打开 → Ctrl+P → 另存 PDF，
      纸张 A4、边距默认、勾选"背景图形"以保留配色）
"""

import os
import re

import markdown as md

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "proposal.md")
OUT = os.path.join(ROOT, "docs", "proposal.html")

# A4 打印样式：@page 定义页面尺寸，中文用系统字体回退链
CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body {
  font-family: "Source Han Serif SC", "Noto Serif CJK SC", "SimSun",
               "Songti SC", serif;
  font-size: 10.5pt; line-height: 1.75; color: #1f2937;
  max-width: 800px; margin: 0 auto;
}
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
/* ↑ 无头浏览器打印 PDF 时保留背景色（封面标签/表头/代码块底色），
   否则背景默认丢弃，白字会直接消失 */
h1 { font-size: 19pt; color: #1e3a8a; margin: 0 0 4pt; text-align: center;
     font-family: "Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
h2 { font-size: 13.5pt; color: #1e3a8a; border-bottom: 1.5pt solid #bfdbfe;
     padding-bottom: 3pt; margin: 20pt 0 8pt; page-break-after: avoid;
     font-family: "Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
h3 { font-size: 11.5pt; color: #2563eb; margin: 14pt 0 6pt; page-break-after: avoid;
     font-family: "Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
p { margin: 6pt 0; text-align: justify; }
strong { color: #111827; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9.5pt;
        page-break-inside: avoid; }
th { background: #eff6ff; color: #1e3a8a; border: 0.5pt solid #93c5fd;
     padding: 5pt 7pt; text-align: left; font-family: "Microsoft YaHei", sans-serif; }
td { border: 0.5pt solid #d1d5db; padding: 4.5pt 7pt; }
tr:nth-child(even) td { background: #f9fafb; }
code { background: #f3f4f6; border-radius: 3pt; padding: 0 4pt;
       font-family: Consolas, monospace; font-size: 9pt; color: #b91c1c; }
pre { background: #f8fafc; border: 0.5pt solid #e2e8f0; border-radius: 6pt;
      padding: 10pt 12pt; overflow-x: auto; font-size: 9pt; line-height: 1.5;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; color: #334155; }
blockquote { border-left: 3pt solid #93c5fd; margin: 8pt 0; padding: 4pt 12pt;
             color: #475569; background: #f8fafc; }
hr { border: none; border-top: 1pt solid #e5e7eb; margin: 16pt 0; }
img { max-width: 100%; display: block; }
/* 图容器：图片+图注同页不拆散；图注居中、与图片拉开间距 */
figure { margin: 12pt auto; text-align: center; page-break-inside: avoid; }
figure img { margin: 0 auto; border: 0.5pt solid #e2e8f0; border-radius: 4pt; }
figcaption { margin-top: 7pt; font-size: 9pt; color: #64748b;
             line-height: 1.6; text-align: center; }
/* 封面区（文件第一个 H1 之前手动插入的 HTML）：
   文字封面——16:9 横图放 A4 竖版只占 1/3 高度、留白大观感弱，
   故改为纯文字排版：标签 → 主标题 → 副标题 → 细线 → 署名，
   内容整体偏上居中分布，满页不留白 */
.cover { text-align: center; padding-top: 95pt;
         page-break-after: always; }  /* 封面独占一页，目录从下一页开始 */
.cover .tag { display: inline-block; background: #1e3a8a; color: #fff;
              border-radius: 4pt; padding: 4pt 16pt; font-size: 10.5pt;
              letter-spacing: 1pt; }
.cover h1 { font-size: 30pt; margin: 36pt 0 8pt; border-bottom: none; }
.cover .sub { color: #4b5563; font-size: 12.5pt; margin: 0; }
.cover-line { width: 64pt; height: 2.5pt; background: #93c5fd;
              margin: 34pt auto; }
.cover .author { color: #374151; font-size: 12pt; margin: 3pt 0; }
.cover .date { color: #94a3b8; margin-top: 40pt; font-size: 11pt; }
/* 摘要页（目录后、正文前，独立一页——评审 30 秒抓核心） */
.summary-page { page-break-after: always; }
.summary-page h2 { margin-top: 0; }
.summary-page blockquote { border-left-color: #b91c1c; }
/* 目录页（toc 扩展生成，置于封面后、正文前，独立一页） */
.toc { page-break-after: always; margin: 4pt 0 12pt; }
.toc-title { font-size: 14pt; text-align: center; border: none; }
.toc ul { list-style: none; margin: 0; padding-left: 0; }
.toc ul ul { padding-left: 12pt; }
.toc li { margin: 1.5pt 0; }
.toc a { color: #1f2937; text-decoration: none; line-height: 1.3; }
/* ↑ 目录收紧行高：toc 链接继承 body 的 1.75 行距，33 行累计超出一页 */
/* 一级标题（一、二、三…章节）加粗大号突出，二级小节常规小字，
   形成"章节 > 小节"的层级对比（默认全平级、看不出结构） */
.toc > ul > li > a { font-weight: 700; color: #111827; font-size: 12pt; }
.toc > ul ul a { font-weight: 400; color: #4b5563; font-size: 10.5pt; }
.toc a:hover { color: #2563eb; }
/* 打印分页控制 */
h1 { page-break-before: always; }
.cover h1 { page-break-before: avoid; }
"""

def my_slugify(value: str, separator: str) -> str:
    """中文标题锚点：空白转分隔符，保留中文（HTML id 允许任意非空文本，
    浏览器点击跳转时自动做 URL 编码，无需拼音/英文 id）。"""
    return re.sub(r"\s+", separator, value.strip())


# 封面区：纯文字排版（标签 + 主标题 + 副标题 + 分隔线 + 署名）。
# 主标题取标题的短名（"因子实验室"），"——"之后的长标题放副标题位。
def build_cover(title: str) -> str:
    main, _, sub = title.partition("——")
    return f"""
<div class="cover">
  <span class="tag">AFAC 金融智能创新大赛</span>
  <h1>{main.strip()}</h1>
  <p class="sub">{sub.strip()}</p>
  <div class="cover-line"></div>
  <p class="author">参赛者：吕滢滢　|　广东金融学院 · 金融科技专业</p>
  <p class="date">2026 年 9 月</p>
</div>
"""


def wrap_figures(html: str) -> str:
    """把「图片段落 + 紧随的斜体图注段」打包成 <figure>/<figcaption>。

    目的：图注与图片在视觉上是一个整体——间距可控、居中排版，
    且打印分页时同页不拆散（markdown 原生输出只是两个相邻 <p>）。
    匹配的段落形如 <p><img ... /></p> 后跟 <p><em>图注</em></p>。
    """
    def repl(m: re.Match) -> str:
        img = m.group(1)
        # 图注取 alt 文本；若后面紧跟着斜体段落，则以段落为准（更完整）
        cap = m.group(2)
        if cap is None:
            alt_m = re.search(r'alt="([^"]*)"', img)
            cap = alt_m.group(1) if alt_m else ""
        if not cap:
            return m.group(0)
        return f"<figure>\n{img}\n<figcaption>{cap}</figcaption>\n</figure>"

    return re.sub(
        r"<p>(<img[^>]*>)\s*</p>\s*(?:<p><em>([^<]*)</em></p>)?",
        repl,
        html,
    )


def isolate_summary(html: str) -> str:
    """把「摘要」章节（从 `<h2 id="摘要">` 到下一个 h2 之前）包成独占一页的 div。

    目的：评审快速浏览时封面 → 目录 → 摘要 → 正文，摘要单独成页，
    30 秒内抓住全部核心；同时保证摘要页的宽度与正文一致（页边距相同）。
    """
    m = re.search(
        r'(<h2 id="摘要">.*?)(?=<h2 |\Z)',
        html,
        flags=re.S,
    )
    if not m:
        return html
    block = m.group(1)
    return html.replace(block, f'<div class="summary-page">\n{block}\n</div>')


def inline_images(html: str) -> str:
    """把 <img src="相对路径"> 转成 base64 data URI。

    目的：项目书是"单文件 HTML → 打印 PDF"的管线，图片若走相对路径，
    换机器打开/转发时会丢图；内嵌后单文件自包含，任何方式打开都稳定。
    相对路径以 docs/ 为基准（与 markdown 源同目录）。
    """
    import base64

    def repl(m: re.Match) -> str:
        tag = m.group(0)
        src_m = re.search(r'\bsrc="([^"]+)"', tag)
        if not src_m:
            return tag
        src = src_m.group(1)
        path = os.path.normpath(os.path.join(ROOT, "docs", src))
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return tag.replace(src, f"data:image/png;base64,{b64}", 1)
        return tag  # 文件缺失：保留原引用并让浏览器自行处理

    return re.sub(r"<img[^>]*>", repl, html)


def main() -> None:
    with open(SRC, encoding="utf-8") as f:
        text = f.read()

    # 抽出第一个 H1 作为封面标题，正文里保留其余标题
    m = re.search(r"^# (.+)$", text, flags=re.M)
    title = m.group(1).strip() if m else "因子实验室"
    body = re.sub(r"^# .+$", "", text, count=1, flags=re.M)

    # 实例化方式（便捷函数 md.markdown 不暴露 toc）：toc 扩展同时做两件事——
    # 给每个 h2/h3 标题生成 id 锚点；产出目录 HTML（存于 mdx.toc）插到封面后
    mdx = md.Markdown(
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        extension_configs={"toc": {"toc_depth": "2-3", "slugify": my_slugify}},
    )
    html_body = mdx.convert(body)
    toc_html = f'<h2 class="toc-title">目录</h2>\n{mdx.toc}'

    html_body = wrap_figures(html_body)  # 图注与图片打包（markdown 转换后、装配前）
    html_body = isolate_summary(html_body)  # 摘要独占一页（独立分页控制）
    html = inline_images(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
{build_cover(title)}
{toc_html}
{html_body}
</body>
</html>
""")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成: {OUT}（{len(html)//1024} KB）")
    print("打开后 Ctrl+P → 目标打印机选 '另存为 PDF' → 纸张 A4 → 勾选背景图形")


if __name__ == "__main__":
    main()

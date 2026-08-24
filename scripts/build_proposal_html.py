# -*- coding: utf-8 -*-
"""
项目书 HTML 生成器 v2（方案书级视觉，打印成 PDF 的中间产物）
============================================================
v1（markdown 直出学术排版）被判定"完成度不够"，v2 按方案书标准重写：
  - 统一字体 Microsoft YaHei（本机 msyh.ttc），pre/code 用 Consolas
    → 嵌入字体 ≤4 个子集（PDF 检查脚本自动回归）
  - 配色 token 化：深蓝主强调 + 朱红 AI accent（保留研报/实验室风格，
    不套用商业方案书的浅色卡模板）
  - 封面：cover_a4.png 全出血铺满 A4（负边距溢出到页边距区）
  - 章首色带：每个 h1 前自动注入深蓝渐变横幅 + 两位数字章号 chip
  - 卡片组件（md 内嵌 HTML 透传 + attr_list 挂类）：kpi-grid 大数字卡 /
    feature-card / badge / callout / pipeline / timeline / card-table
  - 目录页码：toc 每个条目带 .tocpg 占位，由 build_pdf.py 两遍构建回填
    （Chromium 不支持 target-counter，必须回查后注入）

用法：PYTHONIOENCODING=utf-8 python scripts/build_proposal_html.py
      输出: docs/proposal.html（由 build_pdf.py 一键打印为 PDF）
"""

import os
import re

import markdown as md

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "proposal.md")
OUT = os.path.join(ROOT, "docs", "proposal.html")

# 中文数字 → 两位章号（章首色带 chip 用）
CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

CSS = r"""
/* ================= 设计 token（与 make_cover / make_charts 同源） ================= */
:root {
  --ink: #1e293b;      /* 正文墨色 */
  --strong: #0f172a;   /* 加粗 */
  --accent: #1e3a8a;   /* 主强调（深蓝） */
  --accent2: #2563eb;  /* 次强调 */
  --muted: #64748b;    /* 辅助文字 */
  --card-bg: #f1f5f9;  /* 卡片底（浅灰蓝，区别于商业方案书的纯白卡） */
  --line: #e2e8f0;     /* 分隔线 */
  --red: #b91c1c;      /* AI / 朱红 accent */
  --green: #16a34a;    /* 正向徽章 */
  --amber: #d97706;    /* 警示徽章 */
  --grad: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #1e40af 100%);
}

@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
/* 封面页专用页规则：margin 0 让封面图真正全出血铺满 A4。
   负 margin 方案实测无效（Chromium 打印把内容钳制在 @page 盒内，
   图片被截成 18mm 上白边 + 左右 16mm 白边的"带边框封面"），
   named pages（page: cover）是 Chromium 111+ 的打印特性。
   margin 0 同时让页脚页码在封面页自动不渲染——封面保持干净 */
@page cover { margin: 0; }
* { -webkit-print-color-adjust: exact; print-color-adjust: exact;
    box-sizing: border-box; }
/* ↑ 无头浏览器打印必须保留背景色（色带/表头/卡片），否则白字直接消失 */

html, body { margin: 0; padding: 0; }
body {
  font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
  font-size: 10.5pt; line-height: 1.9; color: var(--ink);
}
/* ↑ 10.5pt + 1.9 行距：方案书常规阅读尺寸。9.5pt/1.72 实测 22 页、
   最密页 1572 字符——过密；10pt/1.8 为 23 页、1405 字符——仍贴线。
   目标 25-35 页、≤1300 字符/页（质量回归基线） */

/* ================= 封面：A4 全出血 ================= */
.cover-page { page: cover; page-break-after: always; }
.cover-page img { width: 210mm; height: 297mm; display: block; }
/* 整页尺寸 A4 图（210×297mm），margin 0 的 cover 页规则下正好满页铺满 */

/* ================= 章首色带（自动注入，见 decorate_chapters） ================= */
h1 {
  page-break-before: always;
  background: var(--grad);
  color: #fff; font-size: 17pt; font-weight: 700;
  margin: 0 0 14pt; padding: 22pt 20pt 18pt;
  border-radius: 8pt;
  page-break-after: avoid;
}
h1 .chno {
  display: inline-block; min-width: 34pt; text-align: center;
  background: rgba(255,255,255,.18); color: #fff;
  border-radius: 6pt; padding: 3pt 8pt; margin-right: 12pt;
  font-size: 15pt; font-weight: 700;
}
/* 章导语（h1 后第一段，弱化处理） */
h1 + p { color: var(--muted); font-size: 10pt; margin-top: -6pt; }

/* ================= 小节标题 ================= */
h2 {
  font-size: 13pt; color: var(--accent);
  border-bottom: 1.2pt solid var(--accent2); padding-bottom: 3pt;
  margin: 20pt 0 9pt; page-break-after: avoid;
}
h3 { font-size: 11pt; color: var(--accent2); margin: 14pt 0 7pt;
     page-break-after: avoid; }
p { margin: 6.5pt 0; text-align: justify; }
/* 列表项行距：步骤清单页（4.4 验证引擎等）实测 1343 字/页贴线超限，
   给 li 加 2.5pt 间距让内容自然溢出到下一页（toc 的 .toc li 优先级
   更高保持 2pt，目录版式不受影响） */
li { margin: 2.5pt 0; }
strong { color: var(--strong); }
hr { border: none; border-top: 1pt solid var(--line); margin: 12pt 0; }

/* ================= 目录 ================= */
.toc { page-break-after: always; margin: 6pt 0 12pt; }
.toc-title {
  font-size: 16pt; text-align: center; color: var(--accent);
  border: none; margin: 10pt 0 18pt;
}
.toc ul { list-style: none; margin: 0; padding-left: 0; }
.toc ul ul { padding-left: 14pt; }
.toc li { margin: 0.25pt 0; }
.toc a { color: var(--ink); text-decoration: none; line-height: 1.4; }
/* 目录 39 条必须压进 1 页。踩坑记录（2026-08-25）：
   a 上的 font-size/line-height 对 flex 容器 li 无效——li 的行盒由
   li 继承的 body 字号/行距决定（body 10.5pt/1.9 → 行盒 18~20pt，
   39 条必然溢出到第 2 页）。必须把字号/行距设到 li 本身：
   二级 li 8pt/1.25（行盒 10pt）、一级 li 11pt/1.25（行盒 13.8pt），
   39 条总高 ≈550pt < 内容区 ≈750pt，单页放下。
   页码回填后条目不换行，两遍构建仍收敛 */
.toc > ul > li { margin: 0.7pt 0; font-size: 11pt; line-height: 1.25; }
.toc > ul > li > a { font-weight: 700; color: var(--strong); font-size: 11pt; }
.toc > ul ul { margin: 0; }
.toc > ul ul li { margin: 0.1pt 0; font-size: 8pt; line-height: 1.25; }
.toc > ul ul a { font-weight: 400; color: var(--muted); font-size: 8pt; }
/* 页码：条目右对齐（flex 撑开，页码贴右侧）。
   注意 flex-wrap：嵌套 ul（h2 子列表）width:100% 若不换行会压缩
   一级标题的 a 导致断行（实测「四、技术方|案」），wrap 后 a+页码
   排第一行、子列表独占第二行 */
.toc li { display: flex; flex-wrap: wrap; justify-content: space-between;
          align-items: baseline; }
.toc li a { flex: 0 1 auto; }
.toc li .tocpg { color: var(--muted); font-size: 9.5pt; margin-left: 8pt; }
.toc li ul { flex-basis: 100%; }
.toc li li { display: flex; flex-wrap: wrap; justify-content: space-between; }

/* ================= 表格 ================= */
table { border-collapse: collapse; width: 100%; margin: 9pt 0; font-size: 9.2pt;
        page-break-inside: avoid; }
th { background: var(--accent); color: #fff; border: 0.5pt solid #1e40af;
     padding: 5.5pt 7pt; text-align: left; font-weight: 600; }
td { border: 0.5pt solid var(--line); padding: 5pt 7pt; }
tr:nth-child(even) td { background: #f8fafc; }
/* 表头深蓝白字（默认即 card-table 风格；挂 .card-table 可整表包圆角） */
.card-table { border-radius: 6pt; overflow: hidden; }

/* ================= 代码 =================
   注意：Consolas 无中文字形（公式注释/因子名里的中文会回退到 NSimSun，
   导致嵌入字体超标）——回退链显式加 Microsoft YaHei，中文用雅黑正体 */
code { background: #eef2f7; border-radius: 3pt; padding: 0 3pt;
       font-family: Consolas, "Microsoft YaHei", monospace; font-size: 8.5pt;
       color: var(--red); }
pre { background: #0f172a; color: #dbeafe; border-radius: 6pt;
      padding: 9pt 11pt; margin: 10pt 0; overflow-x: auto;
      font-size: 8.5pt; line-height: 1.5; page-break-inside: avoid;
      font-family: Consolas, "Microsoft YaHei", monospace; }
pre code { background: none; padding: 0; color: #dbeafe; }

/* ================= 引用 / 图片 ================= */
blockquote { border-left: 3pt solid var(--accent2); margin: 7pt 0;
             padding: 3pt 10pt; color: var(--muted); background: #f8fafc; }
img { max-width: 100%; display: block; }
figure { margin: 10pt auto; text-align: center; page-break-inside: avoid; }
figure img { margin: 0 auto; border: 0.5pt solid var(--line); border-radius: 4pt; }
/* 中文没有斜体字形：markdown 图注 *…* 生成的 <em> 若保持 italic，
   Chromium 会回退到 NSimSun 等宋体（嵌入字体超标 + 字形生硬）。
   中文出版惯例图注即正体——统一关闭 italic */
em { font-style: normal; }
figcaption { margin-top: 5pt; font-size: 8.5pt; color: var(--muted);
             font-style: normal; line-height: 1.55; text-align: center; }

/* ================= 卡片组件（md 内嵌 HTML 使用） ================= */
/* —— KPI 大数字卡（执行摘要/实证结果）——
   <div class="kpi-grid">
     <div class="kpi"><div class="kpi-num">+14.4%</div><div class="kpi-label">3 年策略年化（AI 因子）</div></div>
     ...
   </div> */
.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr);
            gap: 6pt; margin: 10pt 0; }
.kpi { background: var(--card-bg); border: 0.5pt solid var(--line);
       border-top: 3pt solid var(--accent); border-radius: 6pt;
       padding: 9pt 8pt; text-align: center; page-break-inside: avoid; }
.kpi-num { font-size: 16.5pt; font-weight: 700; color: var(--accent);
           line-height: 1.25; }
.kpi-label { font-size: 8.2pt; color: var(--muted); margin-top: 3pt; line-height: 1.45; }

/* —— 功能/特性卡片 ——
   <div class="cards">
     <div class="feature-card"><h4>标题</h4><p>正文</p></div>
     ...
   </div> */
.cards { display: grid; grid-template-columns: repeat(2, 1fr); gap: 7pt;
         margin: 10pt 0; }
.feature-card { background: var(--card-bg); border: 0.5pt solid var(--line);
                border-radius: 6pt; padding: 10pt 12pt; page-break-inside: avoid; }
.feature-card h4 { margin: 0 0 5pt; font-size: 10.5pt; color: var(--accent); }
.feature-card p { margin: 0; font-size: 9.2pt; }

/* —— 徽章 ——
   <span class="badge ai">AI 生成</span> / <span class="badge classic">经典</span>
   / <span class="badge good">优秀</span> 等 */
.badge { display: inline-block; border-radius: 3pt; padding: 1pt 7pt;
         font-size: 8pt; font-weight: 600; margin: 0 2pt; }
.badge.ai { background: #fef2f2; color: var(--red); border: 0.5pt solid #fecaca; }
.badge.classic { background: #eff6ff; color: var(--accent);
                 border: 0.5pt solid #bfdbfe; }
.badge.good { background: #f0fdf4; color: var(--green);
              border: 0.5pt solid #bbf7d0; }
.badge.warn { background: #fffbeb; color: var(--amber);
              border: 0.5pt solid #fde68a; }

/* —— 结论/风险框 ——
   <div class="callout info">…</div> / .warn / .risk */
.callout { border-radius: 6pt; padding: 8pt 12pt; margin: 9pt 0;
           border-left: 4pt solid var(--accent2); background: #eff6ff;
           page-break-inside: avoid; font-size: 9pt; }
.callout.warn { border-left-color: var(--amber); background: #fffbeb; }
.callout.risk { border-left-color: var(--red); background: #fef2f2; }
.callout h4 { margin: 0 0 3pt; font-size: 9.5pt; }

/* —— 链路条（一句话 → 因子 → 体检 → 策略）——
   <div class="pipeline"><span>…</span><i>→</i><span>…</span>…</div> */
.pipeline { display: flex; align-items: stretch; gap: 0; margin: 10pt 0;
            page-break-inside: avoid; }
.pipeline span { flex: 1; background: var(--card-bg); border: 0.5pt solid var(--line);
                 border-radius: 5pt; padding: 7pt 8pt; text-align: center;
                 font-size: 8.8pt; color: var(--ink); }
.pipeline span b { display: block; font-size: 9.5pt; color: var(--accent); }
.pipeline i { font-style: normal; align-self: center; padding: 0 4pt;
              color: var(--accent2); font-weight: 700; }

/* —— 双图并排（截图对比用）——
   <div class="fig-grid"><figure>…<img …><figcaption>…</figcaption></figure>…</div>
   注意：fig-grid 内必须写原始 HTML——Python-Markdown 把 <div> 当 HTML 块
   原样透传，内部 markdown 图片语法不会解析。 */
.fig-grid { display: flex; gap: 8pt; margin: 10pt 0;
            page-break-inside: avoid; }
.fig-grid figure { flex: 1; margin: 0; min-width: 0; }
.fig-grid img { border: 0.5pt solid var(--line); border-radius: 4pt;
                width: 100%; height: auto; }
.fig-grid figcaption { font-size: 8pt; line-height: 1.5; }

/* —— 时间线（里程碑）——
   <div class="timeline"><div class="tl-item"><b>2026-08</b><span>…</span></div>…</div> */
.timeline { margin: 10pt 0; }
.tl-item { display: flex; gap: 10pt; padding: 5pt 0 5pt 12pt;
           border-left: 2pt solid var(--accent2); margin-left: 4pt; }
.tl-item b { flex: 0 0 90pt; color: var(--accent); font-size: 9pt; }
.tl-item span { font-size: 9pt; }
"""


def my_slugify(value: str, separator: str) -> str:
    """中文标题锚点：空白转分隔符，保留中文（HTML id 允许任意非空文本）。"""
    return re.sub(r"\s+", separator, value.strip())


def build_cover() -> str:
    """封面：cover_a4.png 全出血铺满整页（图片自带全部封面信息）。

    注意路径：make_cover.py 的 a4 变体输出到 docs/cover_a4.png（不在 charts/），
    而 inline_images 以 docs/ 为基准解析相对路径——这里用 cover_a4.png。
    """
    return '<div class="cover-page"><img src="cover_a4.png" alt="封面"></div>'


def decorate_chapters(html: str) -> str:
    """给每个一级标题注入章号 chip：`一、执行摘要` → `01 执行摘要`。

    目的：色带里用两位数字章号（01/02/…）强化方案书结构感，
    toc 跳转锚点（id）不受影响——只改 h1 内部结构。
    """
    def repl(m: re.Match) -> str:
        attrs, text = m.group(1), m.group(2)
        num_cn, _, name = text.partition("、")
        n = CN_NUM.get(num_cn.strip())
        if n is None or not name:
            return m.group(0)
        return (f'<h1{attrs}><span class="chno">{n:02d}</span>{name.strip()}</h1>')
    return re.sub(r"<h1([^>]*)>([^<]+)</h1>", repl, html)


def add_toc_pages(toc_html: str) -> str:
    """给目录每个条目的链接后加 .tocpg 占位（页码由 build_pdf.py 两遍构建回填）。

    结构：<li><a href="#...">一、执行摘要</a><ul>…</ul></li>
    只在 a 文本后插空 span——Pass2 时按条目文本回查 Pass1 PDF 页号写入。
    """
    def repl(m: re.Match) -> str:
        return f'<a href="{m.group(1)}">{m.group(2)}</a><span class="tocpg"></span>'
    return re.sub(r'<a href="([^"]+)">([^<]+)</a>', repl, toc_html)


def wrap_figures(html: str) -> str:
    """把「图片段落 + 紧随的斜体图注段」打包成 <figure>/<figcaption>。"""
    def repl(m: re.Match) -> str:
        img, cap = m.group(1), m.group(2)
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


def inline_images(html: str) -> str:
    """把 <img src="相对路径"> 转成 base64 data URI（单文件自包含）。"""
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
        return tag
    return re.sub(r"<img[^>]*>", repl, html)


def main() -> None:
    with open(SRC, encoding="utf-8") as f:
        text = f.read()

    mdx = md.Markdown(
        extensions=["tables", "fenced_code", "sane_lists", "toc", "attr_list"],
        extension_configs={"toc": {"toc_depth": "1-2", "slugify": my_slugify}},
    )
    html_body = mdx.convert(text)
    toc_html = add_toc_pages(mdx.toc)

    html_body = wrap_figures(html_body)      # 图注与图片打包
    html_body = decorate_chapters(html_body)  # 章首色带注入章号 chip

    html = inline_images(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>因子实验室（AI Factor Lab）· AFAC 参赛项目书</title>
<style>{CSS}</style>
</head>
<body>
{build_cover()}
<div class="toc-shell"><h2 class="toc-title">目 录</h2>
{toc_html}
</div>
{html_body}
</body>
</html>
""")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成: {OUT}（{len(html)//1024} KB，toc 条目 {toc_html.count('tocpg')} 个）")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
终端运行图生成器（附录「一键复现实录」配图）
============================================
目的：把 scripts/ashare_verify.py 的**真实运行输出**渲染成终端窗口样式的
      PNG 插图——评委看到「真的能跑」比「只是说能跑」直观得多。

诚实性原则：图片内容 100% 来自真实重跑 stdout（本机 2026-08-24 实测），
      HTML 只负责「终端外观」（深色窗口 + 标题栏 + 等宽字体），不改动
      任何一行输出文本；标题栏标注脚本名与运行日期。

渲染方式：HTML → Edge headless CDP captureScreenshot（复用
      print_pdf_cdp.py 的 Edge 启动/等待逻辑，不新增依赖）。

用法：PYTHONIOENCODING=utf-8 python scripts/make_terminal_shot.py \
          <输出文本文件> [输出PNG]
      输出默认 docs/charts/terminal_run.png
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

import websocket

# 复用 print_pdf_cdp.py 的 Edge 定位与 DevTools 等待函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from print_pdf_cdp import find_edge, wait_devtools  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(ROOT, "docs", "charts", "terminal_run.png")
PORT = 9224  # 与打印（9223）/截图工具（9222）错开，避免端口冲突
PROFILE = os.path.join(os.environ.get("TEMP", "."), "edge_cdp_term_shot")

# 终端窗口配色：深色底 + 等宽字体（Consolas 数字/英文，中文回退雅黑）
WIN_BG = "#0b1220"        # 窗口底色（近似真实终端深色主题）
TITLE_BG = "#1e293b"      # 标题栏底色
TITLE_FG = "#94a3b8"      # 标题栏文字
TEXT_FG = "#e2e8f0"       # 正文文字
ACCENT_FG = "#38bdf8"     # 命令行高亮（仅用于「$ 命令」行，真实输出不变）


def render_html(rows: list[str], script_name: str) -> str:
    """把真实输出文本包进终端窗口外观的 HTML。

    参数：
      rows        — 真实 stdout 按行拆分（含空行，原样保留）
      script_name — 标题栏显示的脚本名
    原理：每个 <pre> 块一个 <div> 行；命令提示符行（$ 开头）着蓝色
          （仅提示符与命令文本，输出内容颜色不动）。
    """
    lines_html = []
    for line in rows:
        safe = (line.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace(" ", "&nbsp;"))
        if line.startswith("$ "):
            # 提示符行：$ 与命令着色（这是「输入」，非「输出」，着色不违背诚实）
            lines_html.append(
                f'<div class="ln"><span class="ps1">$&nbsp;</span>'
                f'<span class="cmd">{safe[2:]}</span></div>')
        else:
            lines_html.append(f'<div class="ln">{safe}</div>')
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
      /* 固定宽度：截图尺寸可预测（640 内容宽 → 1280px@2x），
         终端 15 行输出压缩到 ~0.4 高宽比，A4 页内 150mm 宽即可容纳 */
      body {{ margin:0; padding:8px; background:#334155; width:624px; }}
      .win {{ background:{WIN_BG}; border-radius:6px; overflow:hidden;
             box-shadow:0 8px 24px rgba(0,0,0,.4); font-family:Consolas,
             "Microsoft YaHei",monospace; }}
      .titlebar {{ background:{TITLE_BG}; padding:6px 10px; display:flex;
                  align-items:center; gap:6px; font-size:9.5px;
                  color:{TITLE_FG}; }}
      .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
      .dot.r {{ background:#ef4444; }} .dot.y {{ background:#eab308; }}
      .dot.g {{ background:#22c55e; }}
      .title {{ margin-left:6px; }}
      .body {{ padding:8px 10px 10px; font-size:9.5px; line-height:1.4;
              color:{TEXT_FG}; }}
      .ln {{ white-space:pre; }}
      .ps1 {{ color:{ACCENT_FG}; font-weight:bold; }}
      .cmd {{ color:#a5b4fc; }}
    </style></head><body>
    <div class="win">
      <div class="titlebar">
        <span class="dot r"></span><span class="dot y"></span>
        <span class="dot g"></span>
        <span class="title">bash — python {script_name}</span>
      </div>
      <div class="body">{''.join(lines_html)}</div>
    </div>
    </body></html>"""


def shot(html_path: str, png_path: str) -> None:
    """CDP 截图：HTML 文件 → PNG（2x 像素比保证印刷清晰）。

    注意：必须带 --user-data-dir 独立 profile——否则 Edge 会复用已开的
    实例（僵尸进程占住端口时，新实例退出、旧实例继续服务 → 截到旧内容；
    2026-08-24 实测踩坑：11 个僵尸 headless Edge 占满 9222~9225）。
    """
    edge = find_edge()
    proc = subprocess.Popen(
        [edge, "--headless=new", "--disable-gpu",
         f"--remote-debugging-port={PORT}", "--remote-allow-origins=*",
         f"--user-data-dir={PROFILE}",
         "--window-size=1400,2400", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)  # 给进程启动留时间；端口被占时 wait_devtools 会连到旧实例
    # 注意：不能检查 proc.poll()——Edge 151 有 relaunch 机制，Popen 的原始
    # 进程会先退出、子进程继续服务（wmic 可见 --edge-skip-compat-layer-relaunch），
    # poll() 非 None 不代表 Edge 挂了；wait_devtools 轮询兜底即可。
    try:
        ws = websocket.create_connection(wait_devtools(port=PORT), timeout=30)
        cmd_id = 0

        def send(method, params=None):
            nonlocal cmd_id
            cmd_id += 1
            ws.send(json.dumps({"id": cmd_id, "method": method,
                                "params": params or {}}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == cmd_id:
                    if "error" in msg:
                        raise RuntimeError(f"CDP {method}: {msg['error']}")
                    return msg.get("result", {})

        send("Page.enable")
        send("Page.navigate", {"url": "file:///" + html_path.replace("\\", "/")})
        for _ in range(300):
            r = send("Runtime.evaluate",
                     {"expression": "document.readyState",
                      "returnByValue": True})
            if r.get("result", {}).get("value") == "complete":
                break
            time.sleep(0.2)
        time.sleep(1.0)  # 等字体渲染

        # 固定 CSS 尺寸截图（640 宽 = body 624 + padding 8×2；高 320 容纳
        # 15 行输出 + 标题栏）。不要用 scrollWidth——本机 Edge 的 dpr
        # (1.25~2.15) 会把它放大到 1374，再叠 clip scale=2 → 4x 超大图
        # （2026-08-24 实测：5496×9228，直接撑破 A4 页面）。
        send("Emulation.setDeviceMetricsOverride",
             {"width": 640, "height": 320, "deviceScaleFactor": 2,
              "mobile": False})
        img = send("Page.captureScreenshot",
                   {"format": "png", "captureBeyondViewport": True,
                    "clip": {"x": 0, "y": 0, "width": 640, "height": 320,
                             "scale": 1}})
        with open(png_path, "wb") as f:
            import base64
            f.write(base64.b64decode(img["data"]))
        print(f"已生成: {png_path}")
    finally:
        ws.close()
        proc.terminate()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit("用法: make_terminal_shot.py <真实输出.txt> [输出PNG]")
    text_path = args[0]
    png_path = args[1] if len(args) > 1 else OUT_PNG
    with open(text_path, encoding="utf-8") as f:
        rows = f.read().splitlines()
    # 脚本名从路径推断（ashare_verify.py）
    script_name = os.path.basename(text_path).replace(".txt", "").replace(
        "_out", "") + ".py"
    html = render_html(rows, script_name)
    tmp = os.path.join(ROOT, "docs", "terminal_shot_tmp.html")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    try:
        shot(tmp, png_path)
    finally:
        os.remove(tmp)


if __name__ == "__main__":
    main()

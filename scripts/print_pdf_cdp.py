# -*- coding: utf-8 -*-
"""
用 Edge CDP 打印 PDF（绕开 --print-to-pdf CLI 失效）
====================================================
背景：Edge 151 起，`msedge --headless --print-to-pdf=...` 返回 0 但不产出文件
      （本机 2026-08-23 实测：最小 HTML 同样失败）。CLI 打印路径不可靠。
      但 CDP（Chrome DevTools Protocol）路径实测正常——截图工具已验证。

方案：启动 Edge headless + --remote-debugging-port，通过 websocket 发
      Page.navigate → 等 readyState=complete → Page.printToPDF（返回 base64）。

页脚页码：printToPDF 的 displayHeaderFooter + footerTemplate（Chromium 只认
      内联样式）。footer 出现在封面属已知限制——封面 A4 图全出血铺满，
      页脚直接印在页面底部边距区（封面底部信息条下方，淡灰小字，视觉上
      与封面底部信息承接）。Pass1（回查页码用）可传 footer=False 省去。

用法：PYTHONIOENCODING=utf-8 python scripts/print_pdf_cdp.py [输出名] [--no-footer]
      → 默认输出 docs/proposal.pdf（带页脚）；--no-footer 用于 Pass1 回查
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request

import websocket  # websocket-client（pip 包名），CDP 同步客户端

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "docs", "proposal.html")
PORT = 9223  # 固定端口；与截图工具（9222）错开，避免冲突
PROFILE = os.path.join(os.environ.get("TEMP", "."), "edge_cdp_print")
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# 页脚模板：淡灰小字 + 页码/总页数（Chromium 页脚只认内联样式；
# 标签必须写在 <span class="pageNumber"> / <span class="totalPages">）
# 注意：必须显式 font-family——footer 是独立片段，不继承页面 body 字体，
# 缺省会用系统中文字体（NSimSun 等）导致嵌入字体超标（质量回归项）
FOOTER_HTML = (
    '<div style="font-family:Microsoft YaHei,微软雅黑,sans-serif; '
    'font-size:8px; color:#94a3b8; width:100%; text-align:center;">'
    '因子实验室 · AI Factor Lab — 第 <span class="pageNumber"></span> / '
    '<span class="totalPages"></span> 页</div>'
)

_ws = None  # 模块级 ws 连接，退出时关闭
_edge_proc = None


def find_edge() -> str:
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    raise SystemExit("未找到 Edge")


def wait_devtools(timeout: float = 30.0, port: int | None = None) -> dict:
    """轮询 /json/list 直到拿到 page target 的 webSocketDebuggerUrl。

    port 可覆盖（截图工具用独立端口 9224，与打印 9223 错开避免冲突）。
    """
    port = port or PORT
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list",
                                        timeout=2) as r:
                targets = json.loads(r.read().decode("utf-8"))
            for t in targets:
                if t.get("type") == "page" and not t.get("url", "").startswith(
                        "devtools://"):
                    return t["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001 — 端口还没就绪，继续轮询
            pass
        time.sleep(0.3)
    raise SystemExit("DevTools 端口未就绪")


def print_pdf(html_path: str, out_path: str, footer: bool = True,
              timeout: float = 180.0) -> None:
    """CDP 打印：HTML 文件 → PDF。

    参数：
      html_path — 源 HTML 绝对路径（file:// 加载，base64 图片无网络依赖）
      out_path  — 输出 PDF 路径
      footer    — 是否带页脚页码（Pass1 回查传 False，最终稿传 True）
    原理：Chromium 渲染引擎 + Page.printToPDF（printBackground 保留色带/
      表头/卡片背景色；preferCSSPageSize 尊重 @page A4 规则）。
    """
    global _ws, _edge_proc
    if not os.path.exists(html_path):
        raise SystemExit(f"缺少 {html_path}，先跑 build_proposal_html.py")

    edge = find_edge()
    # 1. 启动 Edge headless（独立 profile，不干扰用户正在用的 Edge）
    _edge_proc = subprocess.Popen(
        [edge, "--headless=new", "--disable-gpu",
         f"--remote-debugging-port={PORT}", "--remote-allow-origins=*",
         f"--user-data-dir={PROFILE}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        ws_url = wait_devtools()
        _ws = websocket.create_connection(ws_url, timeout=30)
        cmd_id = 0

        def send(method, params=None):
            nonlocal cmd_id
            cmd_id += 1
            _ws.settimeout(timeout)
            _ws.send(json.dumps(
                {"id": cmd_id, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(_ws.recv())
                if msg.get("id") == cmd_id:
                    if "error" in msg:
                        raise RuntimeError(f"CDP {method} 失败: {msg['error']}")
                    return msg.get("result", {})
                # 其他消息（事件）忽略

        send("Page.enable")
        send("Runtime.enable")

        # 2. 导航到本地 HTML（file:// 路径正斜杠转义）
        url = "file:///" + html_path.replace("\\", "/")
        send("Page.navigate", {"url": url})

        # 3. 等页面加载完成（base64 内联图片无网络依赖，加载即渲染）
        for _ in range(300):  # 最多 ~60s
            r = send("Runtime.evaluate",
                     {"expression": "document.readyState", "returnByValue": True})
            if r.get("result", {}).get("value") == "complete":
                break
            time.sleep(0.2)
        time.sleep(2.0)  # 渲染收尾（图片/字体）

        # 4. printToPDF：preferCSSPageSize 尊重 HTML 里的 @page 规则（A4）
        params: dict = {"printBackground": True, "preferCSSPageSize": True}
        if footer:
            # 页脚页码显示在 @page 底部边距区（20mm，足够放下 8px 小字）
            params.update({
                "displayHeaderFooter": True,
                "footerTemplate": FOOTER_HTML,
                "headerTemplate": "<div></div>",
            })
        result = send("Page.printToPDF", params)
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(result["data"]))
        print(f"已打印: {out_path}（{os.path.getsize(out_path)//1024} KB，"
              f"{'带页脚' if footer else '无页脚'}）")
    finally:
        if _ws is not None:
            try:
                _ws.close()
            except Exception:  # noqa: BLE001
                pass
        if _edge_proc is not None:
            _edge_proc.terminate()


def main() -> None:
    args = sys.argv[1:]
    out = args[0] if args and not args[0].startswith("--") else (
        os.path.join(ROOT, "docs", "proposal.pdf"))
    footer = "--no-footer" not in args
    print_pdf(HTML, out, footer=footer)


if __name__ == "__main__":
    main()

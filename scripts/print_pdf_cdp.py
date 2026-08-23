# -*- coding: utf-8 -*-
"""
用 Edge CDP 打印 PDF（绕开 --print-to-pdf CLI 失效）
====================================================
背景：Edge 151 起，`msedge --headless --print-to-pdf=...` 返回 0 但不产出文件
      （本机 2026-08-23 实测：最小 HTML 同样失败）。CLI 打印路径不可靠。
      但 CDP（Chrome DevTools Protocol）路径实测正常——截图工具已验证。

方案：启动 Edge headless + --remote-debugging-port，通过 websocket 发
      Page.navigate → 等 readyState=complete → Page.printToPDF（返回 base64）。

用法：PYTHONIOENCODING=utf-8 python scripts/print_pdf_cdp.py [输出名]
      → 默认输出 docs/proposal.pdf
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
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "proposal.pdf")
PORT = 9223  # 固定端口；与截图工具（9222）错开，避免冲突
PROFILE = os.path.join(os.environ.get("TEMP", "."), "edge_cdp_print")
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

_ws = None  # 模块级 ws 连接，退出时关闭
_edge_proc = None


def find_edge() -> str:
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    raise SystemExit("未找到 Edge")


def wait_devtools(timeout: float = 30.0) -> dict:
    """轮询 /json/list 直到拿到 page target 的 webSocketDebuggerUrl。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list",
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


def cmd(method: str, params: dict | None = None, timeout: float = 120.0):
    """发一条 CDP 命令并等待对应 id 的响应（同步阻塞）。"""
    global _ws
    _ws.settimeout(timeout)
    _ws.send(json.dumps({"id": _cmd_id, "method": method, "params": params or {}}))
    while True:
        msg = json.loads(_ws.recv())
        if msg.get("id") == _cmd_id:
            if "error" in msg:
                raise RuntimeError(f"CDP {method} 失败: {msg['error']}")
            return msg.get("result", {})
        # 其他消息（事件）忽略


def js(expression: str) -> object:
    """Runtime.evaluate 求值并返回 result.value。"""
    r = cmd("Runtime.evaluate", {"expression": expression, "returnByValue": True})
    return r.get("result", {}).get("value")


def main() -> None:
    global _ws, _cmd_id, _edge_proc
    if not os.path.exists(HTML):
        raise SystemExit("缺少 docs/proposal.html，先跑 build_proposal_html.py")

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
        _cmd_id = 0  # 每次命令前自增，保证 id 唯一（模块级变量）

        def send(method, params=None):
            global _cmd_id
            _cmd_id += 1
            return cmd(method, params)

        send("Page.enable")
        send("Runtime.enable")

        # 2. 导航到本地 HTML（file:// 路径正斜杠转义）
        url = "file:///" + HTML.replace("\\", "/")
        send("Page.navigate", {"url": url})

        # 3. 等页面加载完成（base64 内联图片无网络依赖，加载即渲染）
        for _ in range(300):  # 最多 ~60s
            state = js("document.readyState")
            if state == "complete":
                break
            time.sleep(0.2)
        time.sleep(2.0)  # 渲染收尾（图片/字体）

        # 4. printToPDF：preferCSSPageSize 尊重 HTML 里的 @page 规则（A4）
        result = send("Page.printToPDF", {
            "printBackground": True,
            "preferCSSPageSize": True,
        })
        data = result["data"]
        with open(OUT, "wb") as f:
            f.write(base64.b64decode(data))
        print(f"已生成: {OUT}（{os.path.getsize(OUT)//1024} KB）")
    finally:
        if _ws is not None:
            try:
                _ws.close()
            except Exception:  # noqa: BLE001
                pass
        if _edge_proc is not None:
            _edge_proc.terminate()


if __name__ == "__main__":
    main()

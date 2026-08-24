# -*- coding: utf-8 -*-
"""
工作台 UI 截图器（项目书 03 章配图，CDP 方案，无 playwright 依赖）
==================================================================
Streamlit 是交互式应用，URL 打开只是首屏。本脚本用 CDP 在真实浏览器里
模拟点击（tab 切换 / 预置示例注入 / 体检按钮），把 4-5 个关键画面截下来：

  ui_home.png        首屏：hero 横幅 + AI 因子工场输入框
  ui_diagnosis.png   AI 因子体检结果（载入预置示例，离线不调 API）
  ui_classic.png     经典因子库体检（对照组，证明工作台严谨）
  ui_compare.png     因子对比 PK（10 经典 + AI 全量体检对比表）
  ui_strategy.png    策略构建结果（Top30 周频回测，扣双边成本）

原理：Edge headless + remote-debugging-port，websocket 走 CDP。
Streamlit 正文渲染在页面 iframe 里，故先 Page.getFrameTree 找到 iframe，
再用 Page.createIsolatedWorld 拿到它的执行上下文，所有点击 JS 都发到
iframe 上下文内执行（跨文档 DOM 直接访问，主文档 JS 摸不到 iframe 内容）。

用法：PYTHONIOENCODING=utf-8 python scripts/shot_ui.py
      → docs/charts/ui_*.png（5 张）
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

import websocket  # websocket-client（CDP 同步客户端）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "charts")
os.makedirs(OUT_DIR, exist_ok=True)

PORT_S = 8501          # Streamlit 端口
PORT_E = 9222          # Edge CDP 端口（与打印工具 9223 错开）
PROFILE = os.path.join(os.environ.get("TEMP", "."), "edge_cdp_shot")
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
VIEW_W, VIEW_H = 1440, 900

_ws, _ctx = None, None   # CDP 连接 + iframe 执行上下文
_edge_proc = None
_st_proc = None


def find_edge() -> str:
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    raise SystemExit("未找到 Edge")


# ————————————————————————————————————————————
# Streamlit 生命周期
# ————————————————————————————————————————————
def start_streamlit() -> None:
    global _st_proc
    _st_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         os.path.join(ROOT, "app.py"),
         "--server.port", str(PORT_S), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # 健康检查：/_stcore/health 返回 "ok" 即就绪（进程起来后轮询）
    t0 = time.time()
    while time.time() - t0 < 60:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT_S}/_stcore/health",
                                        timeout=2) as r:
                if r.read().decode("utf-8").strip() == "ok":
                    print("Streamlit 就绪")
                    return
        except Exception:  # noqa: BLE001 — 还没起来，继续轮询
            pass
        time.sleep(0.5)
    raise SystemExit("Streamlit 启动超时")


# ————————————————————————————————————————————
# CDP 基础（与 print_pdf_cdp.py 同构）
# ————————————————————————————————————————————
_cmd_id = 0


def wait_devtools(timeout: float = 30.0) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT_E}/json/list",
                                        timeout=2) as r:
                targets = json.loads(r.read().decode("utf-8"))
            for t in targets:
                if t.get("type") == "page" and not t.get("url", "").startswith("devtools://"):
                    return t["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    raise SystemExit("DevTools 端口未就绪")


def cmd(method: str, params: dict | None = None, timeout: float = 120.0) -> dict:
    global _cmd_id
    _cmd_id += 1
    _ws.settimeout(timeout)
    _ws.send(json.dumps({"id": _cmd_id, "method": method, "params": params or {}}))
    while True:
        msg = json.loads(_ws.recv())
        if msg.get("id") == _cmd_id:
            if "error" in msg:
                raise RuntimeError(f"CDP {method} 失败: {msg['error']}")
            return msg.get("result", {})
        # 事件消息忽略（截图流程用不到事件回调）


def js(expr: str) -> object:
    """在主文档上下文求值 JS（Streamlit 正文在 shadow DOM 内，选择器自行穿透）。"""
    r = cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
    return r.get("result", {}).get("value")


# 带 shadow DOM 穿透的查询（Streamlit 1.38+ 应用正文在 shadow root 内，
# 普通 querySelector 摸不到；递归遍历所有元素检查 shadowRoot）
SHADOW_SELECTOR = r"""
window.__all = (sel) => {
  const found = [];
  const walk = (root) => {
    try { root.querySelectorAll(sel).forEach(e => found.push(e)); } catch (e) {}
    root.querySelectorAll('*').forEach(e => {
      if (e.shadowRoot) walk(e.shadowRoot);
    });
  };
  walk(document);
  return found;
};
true
"""


def wait_streamlit(t: float = 60.0) -> None:
    """等 shadow DOM 里出现 stApp（Streamlit 渲染完成信号）。"""
    js(SHADOW_SELECTOR)
    t0 = time.time()
    while time.time() - t0 < t:
        if js('window.__all(\'[data-testid="stApp"]\').length > 0'):
            return
        time.sleep(0.5)
    raise SystemExit("Streamlit 页面渲染超时")


def shot(name: str) -> None:
    r = cmd("Page.captureScreenshot", {"format": "png"})
    path = os.path.join(OUT_DIR, name)
    with open(path, "wb") as f:
        f.write(__import__("base64").b64decode(r["data"]))
    print(f"已截: {path}")


def scroll_to_text(needle: str, dy: int = -100) -> None:
    """滚动到包含指定文本的可见元素（结果区定位）。

    Streamlit rerun 后滚动位置不变，结果区通常在视口下方——
    先把目标内容 scrollIntoView 到视口顶部，再微调 dy 像素留出页眉。
    注意：Streamlit 所有 tab 的内容都在 DOM 里（隐藏 tab 是 display:none），
    必须过滤 offsetParent===null 的隐藏元素，否则会命中不可见 tab 的内容。
    """
    js(SHADOW_SELECTOR)
    js("""
    (() => {
      const els = window.__all(
        'h1, h2, h3, h4, [data-testid="stMarkdown"], [data-testid="stMetric"]')
        .filter(e => e.offsetParent !== null);
      const hit = els.find(e => e.textContent.includes(%s));
      if (hit) { hit.scrollIntoView({block: "start"}); window.scrollBy(0, %d); return true; }
      return false;
    })()
    """ % (json.dumps(needle), dy))
    time.sleep(2.0)


def click_by_text(needle: str, wait: float = 3.0) -> bool:
    """按文本找可点击元素并点击（shadow DOM 穿透）。

    Streamlit 元素结构：tab 按钮 [data-testid=stTab] button、
    普通按钮 [data-testid=stButton] button、expander 头 summary——
    都在 shadow root 内，用 window.__all 递归穿透查找。
    点击后 Streamlit 重新渲染，等待 wait 秒。
    """
    js(SHADOW_SELECTOR)
    js("""
    (() => {
      // stTab 是 div[role=tab]（非 button）；stButton 的 testid 挂在 button 上；
      // expander 头是 summary——三类元素统一按 textContent 匹配后 click()
      const els = window.__all(
        '[data-testid="stTab"], [data-testid="stButton"] button, summary');
      const hit = els.find(e => e.textContent.includes(%s));
      if (hit) { hit.click(); return true; }
      return false;
    })()
    """ % json.dumps(needle))
    time.sleep(wait)
    return True


# ————————————————————————————————————————————
# 主流程：5 个画面
# ————————————————————————————————————————————
def main() -> None:
    global _ws, _edge_proc

    start_streamlit()
    _edge_proc = subprocess.Popen(
        [find_edge(), "--headless=new", "--disable-gpu",
         f"--remote-debugging-port={PORT_E}", "--remote-allow-origins=*",
         f"--user-data-dir={PROFILE}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        ws_url = wait_devtools()
        _ws = websocket.create_connection(ws_url, timeout=30)
        cmd("Page.enable")
        cmd("Runtime.enable")
        cmd("Emulation.setDeviceMetricsOverride",
            {"width": VIEW_W, "height": VIEW_H, "deviceScaleFactor": 1,
             "mobile": False})

        cmd("Page.navigate", {"url": f"http://127.0.0.1:{PORT_S}/"})
        time.sleep(4)                     # 首帧
        wait_streamlit()                  # 等 shadow DOM 里出现 stApp
        time.sleep(2)

        # 1. 首屏（AI 因子工场 + hero 横幅）
        shot("ui_home.png")

        # 2. AI 体检结果：展开"演示模式" → 载入第一个预置示例（离线）
        click_by_text("演示模式")
        click_by_text("载入：低波动率的股票未来表现更好", wait=6)
        scroll_to_text("多空年化", dy=-150)     # 指标卡上方即徽章横幅
        shot("ui_diagnosis.png")

        # 3. 经典因子库体检
        click_by_text("经典因子库", wait=2)
        click_by_text("开始体检", wait=8)
        scroll_to_text("DSL 表达式", dy=-40)    # 结果区在视口下方，滚动定位
        shot("ui_classic.png")

        # 4. 因子对比（自动体检 10 个经典因子，首次约 10s）
        click_by_text("因子对比", wait=16)
        scroll_to_text("因子综合评分", dy=-60)  # 评分 PK 表 + 条形图
        shot("ui_compare.png")

        # 5. 策略构建（默认经典因子 Top30 周频回测，约 15s）
        click_by_text("策略构建", wait=3)
        click_by_text("构建策略", wait=20)
        scroll_to_text("绩效明细", dy=-60)      # 指标卡 + 净值图
        shot("ui_strategy.png")

        print("完成：docs/charts/ui_*.png 共 5 张")
    finally:
        if _ws is not None:
            try:
                _ws.close()
            except Exception:  # noqa: BLE001
                pass
        if _edge_proc is not None:
            _edge_proc.terminate()
        if _st_proc is not None:
            _st_proc.terminate()


if __name__ == "__main__":
    main()

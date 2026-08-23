# -*- coding: utf-8 -*-
"""
baostock 代理隧道支持（绕开 IP 风控黑名单）
=============================================
背景：baostock 为免费公共服务器，按出口 IP 限流——下载太频繁时返回
      错误码 10001011（黑名单）。服务器地址写死（public-api.baostock.com:10030，
      TCP 直连），没有备用服务器；但黑名单只认 IP，所以「换出口 IP」即可绕开。

原理：把 baostock 的 TCP 直连改经本地 HTTP 代理做 CONNECT 隧道转发，
      流量从代理节点 IP 出去——黑名单认的是代理节点 IP，而不是本机 IP。

用法：
    from baostock_proxy import install_proxy
    install_proxy()   # 读取环境变量 BAOSTOCK_PROXY（如 127.0.0.1:7897），未设置时 no-op

    import baostock as bs   # 之后再 import/登录即可
    lg = bs.login()

实现：monkeypatch socket.socket → 自定义 ProxyConn，connect() 时先与代理
      完成 CONNECT 握手，之后 send/recv 原样转发（代理对隧道流量透明）。
      仅 patch socket.socket 一个符号，不影响项目其他模块。
"""

import os
import socket

_orig_socket = socket.socket  # 原始构造器（隧道自身也要用它建连）

_PROXY_ENV = "BAOSTOCK_PROXY"  # 环境变量：host:port，如 127.0.0.1:7897


class _ProxyConn:
    """包装 socket：connect 时先经 HTTP 代理做 CONNECT 隧道（全双工转发）。"""

    def __init__(self, *args, **kwargs):
        self._s = None

    def connect(self, addr):
        proxy_host, proxy_port = self._proxy.split(":")
        self._s = _orig_socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.settimeout(20)
        self._s.connect((proxy_host, int(proxy_port)))
        host, port = addr
        # HTTP CONNECT 握手：之后代理对该连接做透明 TCP 转发
        self._s.sendall(
            f"CONNECT {host}:{port} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n\r\n".encode()
        )
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += self._s.recv(4096)
        if not resp.startswith(b"HTTP/1.1 200"):
            raise ConnectionError(
                "proxy CONNECT 被拒: " + resp.split(b"\r\n")[0].decode(errors="replace")
            )

    def send(self, data):
        return self._s.send(data)

    def recv(self, n):
        return self._s.recv(n)

    def settimeout(self, t):
        return self._s.settimeout(t)

    def close(self):
        self._s.close()

    def __getattr__(self, name):  # 其余属性透传（如 fileno 等）
        return getattr(self._s, name)


def install_proxy() -> bool:
    """读取 BAOSTOCK_PROXY 环境变量；设置时 patch socket 走代理隧道。

    返回是否已启用（未设置环境变量时 no-op 返回 False）。
    """
    proxy = os.environ.get(_PROXY_ENV, "").strip()
    if not proxy:
        return False

    _ProxyConn._proxy = proxy  # 实例共享同一个代理地址
    socket.socket = _ProxyConn
    return True

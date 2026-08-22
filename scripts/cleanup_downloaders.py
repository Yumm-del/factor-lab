# -*- coding: utf-8 -*-
"""
清理所有全 A 下载进程（防并发写同一个 CSV）
=============================================
背景：baostock 服务器不可达时，手动启动 + watcher 重启的下载脚本
      会同时存活并 append 同一个 ashare_raw.csv，造成 (code,date)
      重复行污染（去重前实测 485118 行里只有 422751 行是干净的）。
      本脚本按命令行精确匹配 build_data_ashare 进程并杀光，
      只保留唯一下载实例。

用法：PYTHONIOENCODING=utf-8 python scripts/cleanup_downloaders.py
"""

import subprocess

# PowerShell 查询：匹配命令行里含 build_data_ashare 的进程
# 注意：关键字必须在 PowerShell 侧拆开拼接——否则查询命令本身
# 含完整关键字，会匹配到查询进程自己（自匹配幻影，永远"剩余 1"）
PS_QUERY = ("Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -like ('*build_data_' + 'ashare*') }")


def run_ps(cmd: str) -> str:
    """执行一段 PowerShell，返回 stdout（失败时返回 stderr）。"""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True, text=True, errors="replace",
    )
    return (r.stdout or r.stderr).strip()


if __name__ == "__main__":
    # 逐个杀掉（杀完再验，ErrorAction SilentlyContinue 容忍进程已自行退出）
    out = run_ps(
        PS_QUERY + " | ForEach-Object { Write-Host ('kill ' + $_.ProcessId); "
        "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    print("清理:", out or "(没有找到下载进程)")

    # 验证清零（等待 2 秒让进程真正结束）
    import time
    time.sleep(2)
    n = run_ps(PS_QUERY + " | Measure-Object | Select-Object -ExpandProperty Count")
    print(f"剩余下载进程: {n}")
    if n not in ("0", ""):
        raise SystemExit("仍有下载进程存活，请手动检查")
    print("OK：下载进程已清零")

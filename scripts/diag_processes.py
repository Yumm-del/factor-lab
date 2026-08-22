# -*- coding: utf-8 -*-
"""诊断：列出所有与下载/watcher 相关的进程（命令行含关键字的）。"""
import subprocess

patterns = ["build_data_ashare", "ashare_watcher", "while true"]
for pat in patterns:
    cmd = (
        f"Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.CommandLine -like '*{pat}*' }} | "
        f"ForEach-Object {{ Write-Host ($_.ProcessId.ToString() + ' | ' + "
        f"$_.CreationDate.ToString('HH:mm:ss') + ' | ' + $_.Name) }}"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                       capture_output=True, text=True, errors="replace")
    out = (r.stdout or r.stderr).strip()
    print(f"[{pat}]")
    print(out or "(无)")
    print()

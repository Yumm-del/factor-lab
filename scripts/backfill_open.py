# -*- coding: utf-8 -*-
"""
全 A 数据 open 字段增量补数脚本
=================================
背景：data/ashare_raw.csv（全 A 日线）由 build_data_ashare.py 生成时
      FIELDS 未含 open 列，而 alpha101 因子库大量公式使用 open。
      全量重下 434MB 不划算——本脚本只对每只股票补查 open 一列，
      输出 data/ashare_open.csv（code,date,open 长表），
      load_panel() 在面板缺 open 时自动 merge 进数据。

断点续传：读 ashare_open.csv 中已完成的 code 集合，跳过已完成股票。
加固：复用 build_data_ashare.py 的超时子线程 / 登录重试 / 主动重连 /
      服务器波动等待恢复 模式（baostock 断连会无限阻塞，必须有超时保护）。

运行：PYTHONIOENCODING=utf-8 python scripts/backfill_open.py
      （建议后台跑，日志看 data/backfill_open.log）
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_data_ashare import (  # noqa: E402  — 复用主下载器的加固模式
    DATA_DIR, ensure_login, logout_timeout, _reconnect, _wait_for_server,
)

# build_data_ashare 模块级已执行 install_proxy()（设了 BAOSTOCK_PROXY 时走代理）
import baostock as bs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CSV = os.path.join(ROOT, "data", "ashare_raw.csv")   # 主数据（无 open 列）
OUT_CSV = os.path.join(ROOT, "data", "ashare_open.csv")  # 补数结果（code,date,open）
START, END = "2023-06-01", "2026-08-15"                  # 与主下载器同区间
FIELDS = "code,date,open"                                # 只查 open 一列，最小化请求体
# 前复权参数与主下载器一致（adjustflag="2"）——保证与 close 等列同口径


def _done_codes() -> set[str]:
    """读回已补数的 code 集合（断点续传）。"""
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, encoding="utf-8") as f:
            done = {line.split(",")[0] for line in f if line.startswith(("sh.", "sz."))}
    return done


def _src_codes() -> list[str]:
    """从主数据文件提取全部 code（流式读，不整文件进内存）。"""
    codes = []
    with open(SRC_CSV, encoding="utf-8") as f:
        seen = set()
        for line in f:
            if not line.startswith(("sh.", "sz.")):
                continue
            c = line.split(",", 1)[0]
            if c not in seen:
                seen.add(c)
                codes.append(c)
    return codes


def _download_one(code: str, retries: int = 3) -> list[str]:
    """单只股票的 (code,date,open) 行列表，失败重试。"""
    for _ in range(retries):
        rs = bs.query_history_k_data_plus(
            code, FIELDS, start_date=START, end_date=END,
            frequency="d", adjustflag="2",
        )
        if rs.error_code != "0":
            time.sleep(1)
            continue
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        return rows
    return []


def _download_timeout(code: str, timeout: float = 30.0) -> list | None:
    """带超时的下载（baostock socket 无超时，断连会无限阻塞——必须限时）。"""
    box: list = []

    def _worker():
        box.extend(_download_one(code))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    return box if not t.is_alive() else None


def main() -> None:
    if not os.path.exists(SRC_CSV):
        raise SystemExit(f"找不到主数据 {SRC_CSV}，请先运行 build_data_ashare.py")
    done = _done_codes()
    codes = _src_codes()
    todo = [c for c in codes if c not in done]
    print(f"主数据 {len(codes)} 只，已补数 {len(done)} 只，待补 {len(todo)} 只", flush=True)
    if not todo:
        print("全部已完成，无需补数")
        return

    if not ensure_login():
        _wait_for_server()

    ok, fail = 0, 0
    fail_codes: list[str] = []
    t0 = time.time()
    try:
        for i, code in enumerate(todo):
            time.sleep(0.15)  # 请求间隔，避免 baostock 限流（与主下载器同参）
            rows = _download_timeout(code)
            if rows is None:
                print(f"  [超时] {code}: 30s 无响应，重建连接", flush=True)
                if not _reconnect():
                    _wait_for_server()
                fail += 1
                fail_codes.append(code)
            elif rows:
                with open(OUT_CSV, "a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(",".join(r) + "\n")
                ok += 1
            else:
                print(f"  [失败] {code}: 重试后仍无数据", flush=True)
                fail += 1
                fail_codes.append(code)
            if (i + 1) % 200 == 0:
                if not _reconnect():
                    _wait_for_server()
                print("  [维护] 已重连（每 200 只主动换连接）", flush=True)
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(todo)} 完成 | 成功 {ok} 失败 {fail} | "
                      f"速度 {ok / el:.1f} 只/秒", flush=True)

        # 补录 2 轮（与主下载器一致：服务器抽风多为暂时性）
        for rnd in range(1, 3):
            if not fail_codes:
                break
            print(f"  补录第 {rnd} 轮：{len(fail_codes)} 只待重试", flush=True)
            still_fail = []
            for j, code in enumerate(fail_codes):
                time.sleep(0.15)
                rows = _download_timeout(code)
                if rows:
                    with open(OUT_CSV, "a", encoding="utf-8") as f:
                        for r in rows:
                            f.write(",".join(r) + "\n")
                    ok += 1
                else:
                    still_fail.append(code)
                if (j + 1) % 200 == 0 and not _reconnect():
                    _wait_for_server()
            fail_codes = still_fail
            print(f"  补录第 {rnd} 轮结束，仍失败 {len(fail_codes)} 只", flush=True)

        print(f"完成：成功 {ok} 只，失败 {fail} 只，总耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
        print(f"输出: {OUT_CSV}（load_panel 自动 merge）", flush=True)
    finally:
        logout_timeout()


if __name__ == "__main__":
    main()

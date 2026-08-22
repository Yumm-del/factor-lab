# -*- coding: utf-8 -*-
"""
全 A 股数据下载脚本（断点续传版）
=================================
目的：把股票池从沪深300（300 只）扩展到全 A 股（5000+ 只），
      外加申万行业映射表，为行业/市值中性化铺路。

数据源：baostock 日线（前复权，与沪深300 数据同源同参）。
字段：code/date/close/high/low/volume/amount/turn/peTTM/pbMRQ（与 hs300 一致）。

断点续传：已下载的股票直接跳过——baostock 偶发断连（WinError 10054），
          断掉后重跑本脚本即可续传，不必从头再来。

运行：PYTHONIOENCODING=utf-8 python scripts/build_data_ashare.py
      （建议后台跑：nohup ... &，日志看 data/ashare_download.log）
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baostock as bs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_CSV = os.path.join(DATA_DIR, "ashare_raw.csv")
IND_CSV = os.path.join(DATA_DIR, "ashare_industry.csv")
START, END = "2023-06-01", "2026-08-15"

FIELDS = "code,date,close,high,low,volume,amount,turn,peTTM,pbMRQ"

# 已下载股票集合（断点续传：逐行读 code 列即可，CSV 按 code 分块追加写）
_done = set()


def _load_done() -> None:
    """从已存在的输出文件读回已下载的 code，实现断点续传。"""
    global _done
    if not os.path.exists(OUT_CSV):
        return
    with open(OUT_CSV, encoding="utf-8") as f:
        _done = {line.split(",")[0] for line in f if line.startswith(("sh.", "sz."))}


def get_all_stocks() -> list[str]:
    """全 A 证券列表：剔除北交所（8/4 开头）、非 A 股、退市/未上市。
    返回格式：['sh.600000', ...]。前提：调用方已 bs.login()。"""
    rs = bs.query_all_stock(day=START)
    codes = []
    while rs.next():
        r = rs.get_row_data()
        code = r[0]
        # 只保留沪深主板/创业板/科创板（sh.6, sz.0, sz.3）；剔除北交所 sh.8/sz.4 与基金债券
        if code.startswith(("sh.6", "sz.0", "sz.3")):
            codes.append(code)
    return codes


def get_industry_map() -> dict[str, str]:
    """申万一级行业映射：code → 行业名（中性化用）。前提：调用方已 bs.login()。"""
    mapping = {}
    rs = bs.query_stock_industry()
    while rs.next():
        r = rs.get_row_data()  # code, code_name, industry, industryClassification
        if r and r[0] and r[0].startswith(("sh.6", "sz.0", "sz.3")):
            mapping[r[0]] = r[2]
    return mapping


def download_one(code: str, retries: int = 3) -> list[str]:
    """下载单只股票日线，返回 CSV 行列表。

    重试逻辑：baostock 偶发"接收数据异常"（连接被掐/限流），
    失败后等 1 秒重试，最多 retries 次；彻底失败返回 []（由调用方记录）。
    """
    for attempt in range(retries):
        rs = bs.query_history_k_data_plus(
            code, FIELDS,
            start_date=START, end_date=END,
            frequency="d", adjustflag="2",  # 前复权，与沪深300 数据一致
        )
        if rs.error_code != "0":
            time.sleep(1)
            continue
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        return rows
    return []


def download_one_timeout(code: str, timeout: float = 30.0):
    """
    带超时保护的下载：baostock 的 socket 没有超时设置，
    服务器断连时客户端会无限阻塞（实测卡死一夜）。

    方案：下载放子线程，主线程 join(timeout) 限时。
    超时 → 放弃该只返回 None，调用方负责 logout+login 重建连接
    （旧线程卡在旧 socket 上无害，daemon 线程不阻止退出）。
    """
    result: list = []

    def _worker():
        result.extend(download_one(code))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    return result if not t.is_alive() else None


def try_download(code: str) -> bool:
    """
    单只下载并追加写入 CSV。

    返回值：True = 成功写盘；False = 该只失败（超时或服务器返回空）。
    副作用：连接超时后自动重建（_reconnect）；服务器持续不可达时
            raise SystemExit —— 此时数据已断点保存，重跑续传即可。
    设计动机：失败必须返回 False 而非抛异常，调用方才能收集
              fail_codes 做收尾补录（否则失败的股票被永久跳过）。
    """
    try:
        rows = download_one_timeout(code, timeout=30.0)
        if rows is None:
            # 超时：连接已死，重建连接（波动期可能长时间连不上 → 原地等待恢复）
            print(f"  [超时] {code}: 30s 无响应，重建连接")
            if not _reconnect():
                _wait_for_server()
            return False
        if rows:
            with open(OUT_CSV, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(",".join(r) + "\n")
            return True
        print(f"  [失败] {code}: 重试后仍无数据")
        return False
    except Exception as e:  # noqa: BLE001 — 单只失败不中断整体
        print(f"  [失败] {code}: {e}")
        return False


def _call_timeout(fn, timeout: float, box: list):
    """在子线程执行任意 baostock 调用（login/logout 都可能无限阻塞）。"""
    def _worker():
        try:
            box.append(fn())
        except Exception:  # noqa: BLE001
            box.append(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    return box[0] if not t.is_alive() and box else None


def login_timeout(timeout: float = 20.0):
    """带超时的 bs.login()：服务器不可达时返回 None（不阻塞主流程）。"""
    return _call_timeout(bs.login, timeout, [])


def logout_timeout(timeout: float = 10.0):
    """带超时的 bs.logout()。"""
    return _call_timeout(bs.logout, timeout, [])


def ensure_login(max_attempts: int = 10) -> bool:
    """重试式登录：登录失败/超时等 3 秒重试，最多 max_attempts 次。
    返回是否成功——失败时由调用方决定继续等待还是退出。"""
    for i in range(max_attempts):
        lg = login_timeout()
        if lg is not None and lg.error_code == "0":
            return True
        print(f"  [登录] 失败或超时（第 {i + 1} 次），3 秒后重试", flush=True)
        time.sleep(3)
    return False


def _reconnect(max_attempts: int = 10) -> bool:
    """断开旧连接并重新登录（均带超时）。返回是否成功。"""
    logout_timeout()
    return ensure_login(max_attempts)


def _wait_for_server(max_wait_min: float = 90.0) -> None:
    """
    服务器波动期专用：重连失败后不退出，原地等待服务器恢复。

    背景：baostock 深夜会进入"恢复几分钟→断连→再恢复"的波动模式，
    若每次断连都退出，等外部（cron/人工）重启会浪费大量可达窗口。

    做法：登录失败后打印提示，等 60 秒再试，直到连上；
          超过 max_wait_min 分钟仍连不上才退出（数据已断点保存）。
    注意：等待期间脚本仍存活，外部 watcher 不会重复拉起新进程。
    """
    t0 = time.time()
    while time.time() - t0 < max_wait_min * 60:
        print(f"  [等待] 服务器暂不可达，60 秒后重试"
              f"（已等 {(time.time() - t0) / 60:.0f} 分钟，上限 {max_wait_min:.0f} 分钟）", flush=True)
        time.sleep(60)
        if _reconnect(10):
            print("  [等待] 服务器已恢复，继续下载", flush=True)
            return
    raise SystemExit(f"等待 {max_wait_min:.0f} 分钟仍无法连接，退出（数据已断点保存，重跑续传）")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    _load_done()
    print(f"已下载 {len(_done)} 只，断点续传模式")

    if not ensure_login():
        _wait_for_server()  # 启动时服务器不可达 → 原地等待恢复，不退出

    try:
        codes = get_all_stocks()
        print(f"全 A 证券共 {len(codes)} 只（剔除北交所/退市后）")

        todo = [c for c in codes if c not in _done]
        print(f"待下载 {len(todo)} 只（已跳过 {len(codes) - len(todo)} 只）")

        ok, fail = 0, 0
        fail_codes: list[str] = []  # 收集失败的股票，主循环后统一补录
        t0 = time.time()
        for i, code in enumerate(todo):
            time.sleep(0.15)  # 请求间隔，避免 baostock 限流（与沪深300 脚本同参）
            if try_download(code):
                ok += 1
            else:
                fail += 1
                fail_codes.append(code)
            _done.add(code)
            # 每 200 只强制重连一次：长连接可能被服务器静默掐断
            # （本次事故根因——无超时的 socket 阻塞一整夜），主动换连接更稳
            if (i + 1) % 200 == 0:
                if not _reconnect():
                    _wait_for_server()
                print("  [维护] 已重连（每 200 只主动换连接）")
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(todo)} 完成 | 成功 {ok} 失败 {fail} | "
                      f"速度 {ok / el:.1f} 只/秒 | 预计剩余 {el / max(ok, 1) * (len(todo) - i - 1) / 60:.0f} 分钟")

        # —— 补录轮：主循环里失败的股票再试 2 轮（服务器抽风多为暂时性，
        #    失败的股票若不补录会被 _done 永久跳过，全 A 覆盖率打折扣）——
        for rnd in range(1, 3):
            if not fail_codes:
                break
            print(f"  补录第 {rnd} 轮：{len(fail_codes)} 只待重试")
            still_fail = []
            for j, code in enumerate(fail_codes):
                time.sleep(0.15)
                if try_download(code):
                    ok += 1
                    fail -= 1
                else:
                    still_fail.append(code)
                if (j + 1) % 200 == 0 and not _reconnect():
                    _wait_for_server()
            fail_codes = still_fail
            print(f"  补录第 {rnd} 轮结束，仍失败 {len(fail_codes)} 只")

        # —— 行业映射表 ——
        ind = get_industry_map()
        with open(IND_CSV, "w", encoding="utf-8") as f:
            f.write("code,industry\n")
            for c, i in ind.items():
                f.write(f"{c},{i}\n")
        print(f"行业映射 {len(ind)} 只 → {IND_CSV}")

        logout_timeout()
        print(f"完成：成功 {ok} 只，失败 {fail} 只，总耗时 {(time.time() - t0) / 60:.1f} 分钟")
        print(f"输出: {OUT_CSV}")
    finally:
        logout_timeout()


if __name__ == "__main__":
    main()

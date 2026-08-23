# -*- coding: utf-8 -*-
"""
行业映射表拉取脚本（独立于全 A 下载）
=====================================
目的：行业表只需要一次 query_stock_industry 查询（几秒），
      不依赖逐只下载 5320 只股票。把它从下载脚本尾部拆出来，
      可以在下载进行中（或服务器波动期）单独拉取，
      让「行业中性化验证」不用等全量下载完成。

运行：PYTHONIOENCODING=utf-8 python scripts/fetch_industry.py
      （成功写入 data/ashare_industry.csv 后自动退出；
        服务器不可达时原地等待，每 60 秒重试，最多 90 分钟）
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baostock as bs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IND_CSV = os.path.join(ROOT, "data", "ashare_industry.csv")


def _call_timeout(fn, timeout: float, box: list):
    """在子线程执行 baostock 调用（login/query 都可能无限阻塞）。"""
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
    return _call_timeout(bs.login, timeout, [])


def logout_timeout(timeout: float = 10.0):
    return _call_timeout(bs.logout, timeout, [])


def get_industry_map() -> dict[str, str]:
    """证监会行业分类映射：code → 行业名（带超时保护）。"""
    mapping = {}
    rs = bs.query_stock_industry()
    while rs.next():
        r = rs.get_row_data()  # code, code_name, industry, industryClassification
        if r and r[0] and r[0].startswith(("sh.6", "sz.0", "sz.3")):
            mapping[r[0]] = r[2]
    return mapping


def fetch_with_timeout(timeout: float = 60.0):
    """带超时的行业表拉取；超时返回 None（连接已死，需重登录）。"""
    box: list = []

    def _worker():
        try:
            box.append(get_industry_map())
        except Exception:  # noqa: BLE001
            box.append(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    return box[0] if not t.is_alive() and box else None


def main() -> None:
    if os.path.exists(IND_CSV):
        print(f"行业表已存在: {IND_CSV}，跳过")
        return

    t0 = time.time()
    while True:
        # 登录（20s 超时）+ 拉取（60s 超时），失败则整体重试
        lg = login_timeout()
        if lg is not None and lg.error_code == "0":
            mapping = fetch_with_timeout()
            logout_timeout()
            if mapping is not None and mapping:
                with open(IND_CSV, "w", encoding="utf-8") as f:
                    f.write("code,industry\n")
                    for c, i in mapping.items():
                        f.write(f"{c},{i}\n")
                print(f"行业表 {len(mapping)} 只 → {IND_CSV}")
                return
        print(f"  [等待] 服务器不可达，60 秒后重试"
              f"（已等 {(time.time() - t0) / 60:.0f} 分钟，上限 90 分钟）", flush=True)
        time.sleep(60)
        if time.time() - t0 > 90 * 60:
            print("等待 90 分钟仍失败，退出（重跑即可续拉）")
            return


if __name__ == "__main__":
    main()

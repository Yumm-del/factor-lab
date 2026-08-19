"""
数据补全 ———— baostock 连接中断导致缺失股票时，只补下载缺失部分
用法：python scripts/repair_data.py
原理：对比成分股全量列表与已有 CSV 的 code 集合，只拉差集并追加。
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_lab import data_pipeline  # noqa: E402


def main():
    import baostock as bs

    # 已有数据
    raw_path = data_pipeline.RAW_PATH
    if not os.path.exists(raw_path):
        print("没有已有数据，请先运行 python scripts/build_data.py")
        return
    existing = set(pd.read_csv(raw_path)["code"].unique())
    print(f"已有 {len(existing)} 只股票")

    # 全量成分股
    bs.login()
    try:
        rs = bs.query_hs300_stocks()
        all_codes = []
        while rs.error_code == "0" and rs.next():
            all_codes.append(rs.get_row_data()[1])
    finally:
        bs.logout()

    missing = [c for c in all_codes if c not in existing]
    if not missing:
        print("✅ 无缺失，数据完整")
        return
    print(f"缺失 {len(missing)} 只，开始补下载…")

    frames = [pd.read_csv(raw_path)]
    for i, code in enumerate(missing):
        bs.login()
        try:
            rs = bs.query_history_k_data_plus(
                code, data_pipeline.K_FIELDS,
                start_date="2023-06-01", end_date="2026-08-15",
                frequency="d", adjustflag="2",
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                frames.append(pd.DataFrame(rows, columns=data_pipeline.K_FIELDS.split(",")))
            if (i + 1) % 20 == 0:
                print(f"  进度: {i + 1}/{len(missing)}")
        finally:
            bs.logout()
        time.sleep(0.3)

    df = pd.concat(frames, ignore_index=True)
    num_cols = ["close", "high", "low", "volume", "amount", "turn", "peTTM", "pbMRQ"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"✅ 补全完成: 共 {df['code'].nunique()} 只股票, {len(df)} 行")


if __name__ == "__main__":
    main()

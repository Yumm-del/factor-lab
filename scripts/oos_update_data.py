# -*- coding: utf-8 -*-
"""
OOS（样本外）数据增量更新
=========================
目的：样本内数据截止 2026-08-14（778 交易日）。OOS 动态跟踪需要
      8/15 之后的真实行情——从 baostock 拉增量并合并进本地缓存，
      不重新下载历史全量（增量下载，几分钟内完成）。

合并口径：
  - hs300_raw.csv   ：追加 8/15 起沪深300 成分股日线，按 (date, code) 去重
  - hs300_index.csv ：追加 8/15 起沪深300 指数日线（策略基准 sh.000300）

用法：BAOSTOCK_PROXY=127.0.0.1:7897 PYTHONIOENCODING=utf-8 \
      python scripts/oos_update_data.py [start] [end]
      start/end 默认 2026-08-15 ~ 2026-08-22（已收盘交易日）
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factor_lab import data_pipeline as dp  # noqa: E402


def merge_append(existing: pd.DataFrame, new: pd.DataFrame,
                 subset: list[str] | None = None) -> pd.DataFrame:
    """长表合并：按指定列去重，保序。

    subset=None 时自动取两表共有列（指数文件无 code 列，只有 date,
    按 date 去重即可；成分股表按 [date, code]）。
    """
    subset = subset or [c for c in ("date", "code") if c in existing.columns]
    merged = pd.concat([existing, new], ignore_index=True)
    return (merged.drop_duplicates(subset=subset, keep="last")
            .sort_values(["date", "code"] if "code" in merged.columns
                         else ["date"]).reset_index(drop=True))


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-08-15"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-22"

    # —— 1. 沪深300 成分股日线增量 ——
    raw_path = os.path.join(ROOT, "data", "hs300_raw.csv")
    old = pd.read_csv(raw_path, dtype={"code": str})
    print(f"合并前 hs300_raw: {len(old)} 行，最新日期 {old['date'].max()}")
    new = dp.download_hs300_data(start_date=start, end_date=end)
    merged = merge_append(old, new)
    merged.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"合并后 hs300_raw: {len(merged)} 行，最新日期 {merged['date'].max()}"
          f"（新增 {len(merged) - len(old)} 行）")

    # —— 2. 沪深300 指数日线增量（策略基准）——
    idx_path = os.path.join(ROOT, "data", "hs300_index.csv")
    old_idx = pd.read_csv(idx_path, dtype={"code": str})
    print(f"合并前 hs300_index: {len(old_idx)} 行，最新日期 {old_idx['date'].max()}")
    new_idx = dp.download_index(start_date=start, end_date=end)
    merged_idx = merge_append(old_idx, new_idx)
    merged_idx.to_csv(idx_path, index=False, encoding="utf-8")
    print(f"合并后 hs300_index: {len(merged_idx)} 行，最新日期 "
          f"{merged_idx['date'].max()}（新增 {len(merged_idx) - len(old_idx)} 行）")


if __name__ == "__main__":
    main()

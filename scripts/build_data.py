"""
数据构建脚本 ———— 一键下载/更新沪深300数据
用法：
    python scripts/build_data.py            # 全量下载（300 只）
    python scripts/build_data.py --test     # 小样（5 只，验证接口）
数据缓存在 ../data/hs300_raw.csv（已存在则跳过）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_lab import data_pipeline  # noqa: E402

if __name__ == "__main__":
    if "--test" in sys.argv:
        df = data_pipeline.download_hs300_data(limit=5)
        print(df.head())
    else:
        df = data_pipeline.download_hs300_data()
        data_pipeline.save_raw(df)

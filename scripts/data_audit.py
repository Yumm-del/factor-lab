# -*- coding: utf-8 -*-
"""
数据质量审计（供项目书"数据来源与可靠性"自证）
================================================
逐项检查 data/ 下全部数据文件，输出审计报告：
  1. 文件结构：行数 / 列数 / 列名与文档一致性
  2. 时间覆盖：日期范围 / 交易日数与 778 核对 / 每股缺失天数
  3. 价格有效性：close<=0、high<low、单日收益极端值占比
  4. 量价一致性：volume>0 但 amount==0、turn 缺失
  5. 基本面字段：peTTM/pbMRQ 缺失率与负值比例（亏损企业应存在）
  6. 指数对照：hs300_index 与成分股面板的日期对齐、日收益范围
  7. 行业映射：覆盖率（5320 只中有行业者）、行业分布
  8. 股票池构成：主板/创业板/科创板计数；上市初期样本占比（<60 交易日）

用法：PYTHONIOENCODING=utf-8 python scripts/data_audit.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

ASHARE_COLS = ["code", "date", "close", "high", "low", "volume",
               "amount", "turn", "peTTM", "pbMRQ"]
# 注意：两个下载脚本输出的列序/表头不一致——ashare 无表头且 code 在前；
# hs300 有表头且 date 在前。data_pipeline.load_raw 分别兼容，审计也分别处理。
HS300_COLS = ["date", "code", "close", "high", "low", "volume",
              "amount", "turn", "peTTM", "pbMRQ"]


def load(fp: str, names=None, header=None) -> pd.DataFrame:
    return pd.read_csv(fp, header=header, names=names,
                       dtype={"code": str, "date": str})


def audit_ashare() -> None:
    print("=" * 70)
    print("1. 全 A 面板 ashare_raw.csv")
    print("=" * 70)
    raw = load(os.path.join(DATA, "ashare_raw.csv"), ASHARE_COLS)
    print(f"总行数 {len(raw):,}；列数 {raw.shape[1]}（应与 10 一致）")

    n_stock = raw["code"].nunique()
    d0, d1 = raw["date"].min(), raw["date"].max()
    n_days = raw["date"].nunique()
    print(f"股票数 {n_stock}；日期范围 {d0} ~ {d1}；交易日数 {n_days}"
          f"（项目书声明 778，期望一致）")

    # 重复 (code, date)
    dup = raw.duplicated(subset=["code", "date"]).sum()
    print(f"重复 (code,date) 对：{dup}")

    # 每只股票行数分布
    cnt = raw.groupby("code").size()
    print(f"每股行数：min {cnt.min()} / median {cnt.median():.0f} / max {cnt.max()}")
    few = cnt[cnt < 60]
    print(f"上市初期样本（<60 交易日）：{len(few)} 只（{len(few)/n_stock:.1%}）"
          f"——对应项目书 6.5 局限声明")
    print(f"  最少 10 只：{few.sort_values().head(10).to_dict()}")

    # 价格有效性
    close = raw["close"].astype(float)
    bad_close = (close <= 0).sum()
    bad_hl = (raw["high"].astype(float) < raw["low"].astype(float)).sum()
    print(f"close<=0：{bad_close}；high<low：{bad_hl}")

    # 单日收益极端值（按股票内排序后差分；前复权数据跳变点多来自除权/新股首日）
    raw["ret"] = raw.groupby("code")["close"].pct_change()
    big = raw["ret"].abs() > 0.30
    print(f"单日收益 |ret|>30%：{big.sum()} 条（{big.mean():.4%}）")
    print(f"  |ret|>50%：{(raw['ret'].abs() > 0.5).sum()} 条")

    # 量价一致性
    vol0 = ((raw["volume"].astype(float) == 0) & (raw["amount"].astype(float) > 0)).sum()
    amt0 = ((raw["volume"].astype(float) > 0) & (raw["amount"].astype(float) == 0)).sum()
    print(f"volume=0 但 amount>0：{vol0}；volume>0 但 amount=0：{amt0}")

    # 基本面字段
    for col in ("turn", "peTTM", "pbMRQ"):
        s = raw[col]
        if col in ("turn", "peTTM", "pbMRQ"):
            f = s.astype(float)
            miss = f.isna().sum()
            neg = (f < 0).sum()
            note = f"缺失 {miss}（{miss/len(raw):.2%}），负值 {neg}（{neg/len(raw):.2%}）"
            if col == "peTTM" and neg > 0:
                note += " ← peTTM 负值=亏损企业，应存在"
            if col == "pbMRQ":
                note += " ← pbMRQ 负值=资不抵债，少量正常"
            print(f"  {col}: {note}")


def audit_hs300() -> None:
    print()
    print("=" * 70)
    print("2. 沪深300 面板 hs300_raw.csv")
    print("=" * 70)
    raw = load(os.path.join(DATA, "hs300_raw.csv"), HS300_COLS, header=0)
    print(f"总行数 {len(raw):,}；股票数 {raw['code'].nunique()}（应 300）")
    d0, d1 = raw["date"].min(), raw["date"].max()
    print(f"日期范围 {d0} ~ {d1}；交易日数 {raw['date'].nunique()}（应 778）")

    # 与全 A 面板日期集合对比
    full = pd.read_csv(os.path.join(DATA, "ashare_raw.csv"), header=None,
                       names=ASHARE_COLS, usecols=[1], dtype=str)
    missing_days = set(full["date"]) - set(raw["date"])
    print(f"全 A 有而 hs300 缺的日期：{len(missing_days)} 天"
          f"{list(missing_days)[:5] if missing_days else ''}")

    # 每股行数分布（宽面板里的缺失值）
    cnt = raw.groupby("code").size()
    print(f"每股行数：min {cnt.min()} / median {cnt.median():.0f} / max {cnt.max()}"
          f"（完整应为 778）")
    incomplete = cnt[cnt < 778]
    if len(incomplete):
        print(f"行数不足 778 的股票：{len(incomplete)} 只"
              f"（最多缺 {778 - incomplete.min()} 天），清单："
              f"{incomplete.sort_values().head(8).to_dict()}")

    # 指数对齐检查
    print()
    print("-" * 70)
    print("3. 沪深300 指数 hs300_index.csv")
    print("-" * 70)
    idx = pd.read_csv(os.path.join(DATA, "hs300_index.csv"), dtype={"date": str})
    print(f"列名：{list(idx.columns)}；行数 {len(idx):,}")
    r = idx["close"].astype(float).pct_change()
    print(f"指数日收益：min {r.min():.2%} / max {r.max():.2%}"
          f" / |r|>10% 的条数 {(r.abs() > 0.10).sum()}"
          f"（A 股指数单日 ±10% 封顶，超过即异常）")
    # 指数日期与成分股日期对齐
    common = len(set(raw["date"]) & set(idx["date"]))
    print(f"指数与成分股面板共有日期 {common} / {raw['date'].nunique()}")


def audit_industry() -> None:
    print()
    print("=" * 70)
    print("4. 行业映射 ashare_industry.csv")
    print("=" * 70)
    ind = pd.read_csv(os.path.join(DATA, "ashare_industry.csv"),
                      dtype={"code": str})
    print(f"行数 {len(ind)}（下载 5542 只）；列名 {list(ind.columns)}")
    full = pd.read_csv(os.path.join(DATA, "ashare_raw.csv"), header=None,
                       names=ASHARE_COLS, usecols=[0], dtype={"code": str})
    pool = set(full["code"])
    covered = sum(1 for c in ind["code"] if c in pool)
    print(f"全 A 5320 只中已覆盖行业：{covered}（{covered/len(pool):.1%}）")
    print(f"行业数：{ind['industry'].nunique()}；前 5 大："
          f"{ind['industry'].value_counts().head(5).to_dict()}")


def audit_pool() -> None:
    print()
    print("=" * 70)
    print("5. 股票池构成（代码前缀）")
    print("=" * 70)
    raw = pd.read_csv(os.path.join(DATA, "ashare_raw.csv"), header=None,
                      names=ASHARE_COLS, usecols=[0], dtype={"code": str})
    codes = raw["code"].drop_duplicates()
    mkt = codes.str[3:4].value_counts().to_dict()
    print(f"全 A 池：{codes.nunique()} 只；前缀分布 {mkt}"
          f"（6=沪主板/科创 0,3=深主板/创业）")


def main() -> None:
    audit_ashare()
    audit_hs300()
    audit_industry()
    audit_pool()
    # Python 版本核对（项目书 4.1 声称 Python 3.14）
    print()
    print(f"运行环境 Python {sys.version.split()[0]}")


if __name__ == "__main__":
    main()

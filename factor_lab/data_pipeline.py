"""
因子实验室 ———— 真实 A 股数据管道
=====================================
目的：把「真实 A 股数据」变成因子计算需要的面板结构。

设计原则（为什么自建数据管道而不是用现成库）：
    1. 数据是参赛作品差异化的地基——评委看到的是"真实数据驱动的因子工作台"，
       不是"加载一个示例 CSV"的 demo
    2. baostock 免费、无需 token、国内直连稳定（已在 stock-analyzer 验证过）
    3. 数据面板统一为 (date × code) 的 wide 格式，因子 DSL 求值时按截面/时序
       两个轴都能向量化计算

数据规模控制（性能决策）：
    全 A 股 5000+ 只 × 因子计算太慢，4 天项目不冒险。
    用沪深 300 成分股（300 只 × ~250 个交易日）——
    「沪深 300 是市场主流机构关注的核心池」，叙事上也站得住。

接口约定：
    download_hs300_data()  -> DataFrame（长表：date, code, close, ...）
    load_panel()           -> dict[str, pd.DataFrame]（wide 面板：close/volume/turn/pe/pb）
"""

import os
import time

import numpy as np
import pandas as pd

# 数据目录（本文件所在包的上一级 /data）
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# 缓存的原始长表路径（双股票池：沪深300 与全 A）
RAW_PATH = os.path.join(DATA_DIR, "hs300_raw.csv")
A_SHARE_RAW_PATH = os.path.join(DATA_DIR, "ashare_raw.csv")

# 沪深300 指数日线（策略模块的基准，单独文件）
INDEX_PATH = os.path.join(DATA_DIR, "hs300_index.csv")

# 行业映射（全 A 池中性化用）：code → 证监会行业分类
INDUSTRY_PATH = os.path.join(DATA_DIR, "ashare_industry.csv")

# 全 A 池 open 字段补数缓存（scripts/backfill_open.py 生成）：code,date,open 长表。
# ashare_raw.csv 由 build_data_ashare.py 生成时 FIELDS 未含 open，
# 而 alpha101 因子库大量公式使用 open —— load_panel 检测缺列时自动 merge 此表。
OPEN_CACHE_PATH = os.path.join(DATA_DIR, "ashare_open.csv")

POOLS = {"hs300": RAW_PATH, "ashare": A_SHARE_RAW_PATH}

# baostock 日线字段说明（字段名与 baostock 文档一致）：
#   date    交易日期
#   code    证券代码（sh.600000 格式）
#   open    开盘价（前复权）——alpha101 因子库大量公式使用
#   close   收盘价（前复权）
#   high    最高价
#   low     最低价
#   volume  成交量（股）
#   amount  成交额（元）
#   turn    换手率（%）
#   peTTM   市盈率 TTM
#   pbMRQ   市净率
K_FIELDS = "date,code,open,close,high,low,volume,amount,turn,peTTM,pbMRQ"


def _ensure_baostock():
    """按需导入 baostock（避免模块导入时就必须装好）。"""
    try:
        import baostock as bs
        return bs
    except ImportError:
        raise RuntimeError("baostock 未安装，请运行: pip install baostock")


def download_hs300_data(
    start_date: str = "2023-06-01",
    end_date: str = "2026-08-15",
    limit: int | None = None,
    pause: float = 0.15,
) -> pd.DataFrame:
    """
    下载沪深 300 成分股的日线数据（长表格式）。

    参数：
        start_date / end_date — 日期区间（字符串 YYYY-MM-DD）
        limit — 只下载前 N 只股票（小样测试用，正式跑传 None）
        pause — 每只股票之间的停顿秒数（礼貌限速，防止被断开）

    返回：
        DataFrame：date, code, close, high, low, volume, amount, turn, peTTM, pbMRQ
    """
    bs = _ensure_baostock()
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

    try:
        # ——— 1. 取沪深 300 成分股 ———
        rs = bs.query_hs300_stocks()
        codes = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            # 行格式: [updateDate, code, code_name]
            codes.append(row[1])
        if not codes:
            raise RuntimeError("未能获取沪深 300 成分股列表")
        if limit:
            codes = codes[:limit]

        print(f"📥 准备下载 {len(codes)} 只股票的日线数据 ({start_date} ~ {end_date})")

        # ——— 2. 逐只拉取日线 ———
        frames = []
        for i, code in enumerate(codes):
            rs = bs.query_history_k_data_plus(
                code,
                K_FIELDS,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",  # 前复权：保证收益率计算准确
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                frames.append(pd.DataFrame(rows, columns=K_FIELDS.split(",")))
            if (i + 1) % 30 == 0:
                print(f"   进度: {i + 1}/{len(codes)}")
            time.sleep(pause)

        if not frames:
            raise RuntimeError("未获取到任何数据")

        df = pd.concat(frames, ignore_index=True)

        # ——— 3. 类型转换（baostock 全部返回字符串） ———
        num_cols = ["open", "close", "high", "low", "volume", "amount", "turn", "peTTM", "pbMRQ"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values(["date", "code"]).reset_index(drop=True)

        print(f"✅ 下载完成: {len(df)} 行, {df['code'].nunique()} 只股票")
        return df
    finally:
        bs.logout()


def save_raw(df: pd.DataFrame) -> str:
    """保存原始长表到 data/（存在则不覆盖——避免重复下载）。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_PATH):
        print(f"ℹ️  数据已存在: {RAW_PATH}，跳过保存")
        return RAW_PATH
    df.to_csv(RAW_PATH, index=False, encoding="utf-8")
    print(f"💾 已保存: {RAW_PATH}")
    return RAW_PATH


def load_raw(pool: str = "hs300") -> pd.DataFrame:
    """加载原始长表（不存在则自动下载）。

    pool: "hs300" 沪深300（不存在自动下载）/ "ashare" 全 A（不存在报错提示先跑脚本）。
    注意：全 A 文件由 scripts/build_data_ashare.py 生成，只写数据行不写表头，
    且列顺序为 code,date,...（脚本 FIELDS 定义）——需按实际顺序指定列名。
    """
    if pool == "ashare":
        if not os.path.exists(A_SHARE_RAW_PATH):
            raise FileNotFoundError(
                "全 A 数据尚未生成，请先运行: PYTHONIOENCODING=utf-8 "
                "python scripts/build_data_ashare.py（约 4 小时，支持断点续传）"
            )
        ashare_cols = ["code", "date", "close", "high", "low",
                       "volume", "amount", "turn", "peTTM", "pbMRQ"]
        return pd.read_csv(A_SHARE_RAW_PATH, header=None, names=ashare_cols)
    if not os.path.exists(RAW_PATH):
        df = download_hs300_data()
        save_raw(df)
    return pd.read_csv(RAW_PATH)


def load_industry() -> pd.Series:
    """行业映射：code → 证监会行业分类（pd.Series，全 A 池中性化用）。"""
    if not os.path.exists(INDUSTRY_PATH):
        raise FileNotFoundError(
            "行业映射表不存在，请运行: PYTHONIOENCODING=utf-8 "
            "python scripts/build_data_ashare.py 完成下载"
        )
    df = pd.read_csv(INDUSTRY_PATH)
    return pd.Series(df["industry"].values, index=df["code"])


def download_index(start_date: str = "2023-06-01", end_date: str = "2026-08-15") -> pd.DataFrame:
    """下载沪深300指数（sh.000300）日线——策略模块的业绩基准。"""
    bs = _ensure_baostock()
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    try:
        rs = bs.query_history_k_data_plus(
            "sh.000300", "date,close",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3",
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=["date", "close"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        print(f"✅ 指数下载完成: {len(df)} 行")
        return df
    finally:
        bs.logout()


def load_index() -> pd.Series:
    """加载指数收盘价序列（不存在则下载）。"""
    if not os.path.exists(INDEX_PATH):
        df = download_index()
        df.to_csv(INDEX_PATH, index=False, encoding="utf-8")
    idx = pd.read_csv(INDEX_PATH)
    return pd.Series(idx["close"].values, index=idx["date"])


def _attach_open_cache(df: pd.DataFrame, pool: str) -> pd.DataFrame:
    """面板缺 open 列时，从 ashare_open.csv（backfill_open.py 产物）合并补上。

    背景：全 A 数据文件生成时未含 open 列（build_data_ashare.py 的 FIELDS
    历史版本无 open），而 alpha101 因子库大量公式需要 open。不重下 434MB
    主文件，只按 (code, date) 左连接补一列——多出来的缓存行无害，
    主表没有的日期不会进面板。

    返回：补上 open 列后的长表；缓存不存在时给出明确指引而非静默崩溃。
    """
    if not os.path.exists(OPEN_CACHE_PATH):
        raise RuntimeError(
            f"面板缺 open 列且找不到补数缓存 {OPEN_CACHE_PATH}\n"
            "请先运行: PYTHONIOENCODING=utf-8 python scripts/backfill_open.py"
            "（只补 open 一列，断点续传，不用重下全 A 主数据）"
        )
    op = pd.read_csv(OPEN_CACHE_PATH, usecols=["code", "date", "open"])
    op["open"] = pd.to_numeric(op["open"], errors="coerce")
    merged = df.merge(op, on=["code", "date"], how="left")
    missing = merged["open"].isna().mean()
    if missing > 0.5:
        print(f"⚠️  open 补数覆盖率偏低（缺失 {missing:.0%}），可能是补数尚未跑完，"
              f"open 相关因子将出现缺失值（如实保留 NaN，不填 0）")
    return merged


def load_panel(pool: str = "hs300") -> dict[str, pd.DataFrame]:
    """
    把长表转成 (date × code) 的 wide 面板，供因子 DSL 向量化求值。

    参数：
        pool — "hs300"（默认，300 只，快）/ "ashare"（全 A 5000+ 只，首次加载慢）

    返回：
        dict：key 为数据名（open/close/high/low/volume/amount/vwap/turn/pe/pb），
              value 为 DataFrame（index=date, columns=code）

    原理：长表 → 以 (date, code) 为多级索引 → unstack 成宽表。
    unstack 比 pivot_table 快一个量级（全 A 400 万行时差异显著）。
    """
    df = load_raw(pool)
    if "open" not in df.columns:
        df = _attach_open_cache(df, pool)
    df = df.set_index(["date", "code"]).sort_index()
    # vwap（成交量加权均价）= 全天成交额 / 成交量。
    # WorldQuant 101 公式大量使用 vwap，作为派生字段而非独立下载列
    df["vwap"] = df["amount"] / df["volume"].replace(0, np.nan)
    panel = {}
    for name, col in [
        ("open", "open"),
        ("close", "close"),
        ("high", "high"),
        ("low", "low"),
        ("volume", "volume"),
        ("amount", "amount"),
        ("vwap", "vwap"),
        ("turn", "turn"),
        ("pe", "peTTM"),
        ("pb", "pbMRQ"),
    ]:
        panel[name] = df[col].unstack().sort_index()
    return panel


if __name__ == "__main__":
    # 小样测试：先拉 5 只股票验证接口，再决定是否全量下载
    import sys

    test = "--test" in sys.argv
    if test:
        print("🧪 小样测试模式：只下载 5 只股票")
        df = download_hs300_data(limit=5)
        print(df.head(10))
        print(df.info())
    else:
        df = download_hs300_data()
        save_raw(df)

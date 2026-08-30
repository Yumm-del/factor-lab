"""
alpha101 因子库 ———— WorldQuant 101 公式的 A 股可复现子集（81/101）
====================================================================
目的：把 WorldQuant 官方 101 Alpha 公式翻译进本项目的 DSL 表达式树，
      与经典因子 / LLM 因子走完全相同的【表达式树 → 求值 → 体检】管线。

翻译原则（如实披露，全部为真实约束）：
    1. 公式来源：WorldQuant 官方 101 Alphas 文档（STHSF/alpha101 仓库转录版）
    2. 可移植 81 个：19 个依赖美股特有数据（indneutralize 行业中性化、
       cap 市值、adv81 等）在 A 股不可复现，不移植；#57 官方文档缺失
    3. 小数窗口取整：官方优化版参数（如 16.1219）取整为整数——
       pandas 滚动窗口不支持小数；取整损失已在对比中披露
    4. 不做 fillna(0)：参考实现为写库习惯填 0，本库保持 NaN 由体检管线处理
    5. 全部为受信内建表达式，不走 LLM 的 MAX_DEPTH/MAX_NODES 校验
"""

from . import dsdl
from .factor_engine import _leaf, _ts, _cross, _neg, _combine

# ============================================================
# 表达式树助手（alpha101 专用构造）
# ============================================================


def _ts2(op: str, a: dict, b: dict, p: int) -> dict:
    """二元时序算子节点：ts_corr(rank(open), rank(volume), 10) 这类。"""
    return {"op": op, "args": [a, b], "param": p}


def _cond(c: dict, a: dict, b: dict) -> dict:
    """三元条件节点：c 为真取 a，否则取 b（WorldQuant '? :' 语法）。"""
    return {"op": "cond", "args": [c, a, b]}


def _b(op: str, a: dict, b: dict) -> dict:
    """二元节点（min/max/pow/gt/lt/eq/and/or/add/sub/mul/div）。"""
    return {"op": op, "args": [a, b]}


def _u(op: str, a: dict) -> dict:
    """一元节点（sign/ln/scale/abs/neg）。"""
    return {"op": op, "args": [a]}


def _c(v: float) -> dict:
    """常量节点。"""
    return {"op": "const", "value": v}


def _neg1(x: dict) -> dict:
    """布尔条件 × -1：官方 'cond ? -1 : 0' 语义。

    注意不能写成 _neg(cond)——pandas 对布尔 DataFrame 取负是**按位取反**
    （True→False），得到的是布尔结果而不是数值 -1/0。必须先用 mul 乘
    数值 -1，让 numpy 把 bool 广播成 int64。"""
    return _combine("mul", x, _c(-1))


# —— 数据叶子缩写（公式可读性）——
close = _leaf("close")
open_ = _leaf("open")   # open 是 Python 内建名，用下划线区分
high = _leaf("high")
low = _leaf("low")
volume = _leaf("volume")
vwap = _leaf("vwap")

# —— 派生量缩写 ——
r1 = _ts("ts_returns", close, 1)          # 日收益率 returns
ADV_WINDOWS = {5: 5, 10: 10, 15: 15, 20: 20, 30: 30, 40: 40,
               50: 50, 60: 60, 120: 120, 180: 180, 150: 150}


def adv(n: int) -> dict:
    """n 日平均成交量（WorldQuant adv_n = SMA(volume, n)，滚动均量）。"""
    return _ts("ts_mean", volume, n)


# ============================================================
# 公式翻译（81 条，按官方编号）
# ============================================================

ALPHA101_FACTORS: dict[str, dict] = {
    # ——— #1-26：整数参数版（教科书公式） ———
    "alpha101_001": {
        "name": "Alpha#1",
        "category": "时序结构",
        "description": "条件替换后的价格位置：收益为负时用 20 日波动率替换，再取 5 日窗口内最高点位置。",
        "expr": _combine(
            "sub",
            _cross("rank", _ts("ts_argmax",
                                _ts("signed_power",
                                   _cond(_b("lt", r1, _c(0)),
                                         _ts("ts_std", r1, 20), close),
                                   2),
                                5)),
            _c(0.5),
        ),
    },
    "alpha101_002": {
        "name": "Alpha#2",
        "category": "量价相关",
        "description": "成交量对数差与日内涨跌幅（收盘-开盘）/开盘 的排名相关性取负。",
        "expr": _neg(_ts2("ts_corr",
                          _cross("rank", _ts("delta", _u("ln", volume), 2)),
                          _cross("rank", _combine("div", _combine("sub", close, open_),
                                                  open_)),
                          6)),
    },
    "alpha101_003": {
        "name": "Alpha#3",
        "category": "量价相关",
        "description": "开盘价排名与成交量排名的 10 日滚动相关取负：放量高开背离信号。",
        "expr": _neg(_ts2("ts_corr",
                          _cross("rank", open_), _cross("rank", volume), 10)),
    },
    "alpha101_004": {
        "name": "Alpha#4",
        "category": "时序结构",
        "description": "最低价截面排名的 9 日时间序列排名取负。",
        "expr": _neg(_ts("ts_rank", _cross("rank", low), 9)),
    },
    "alpha101_005": {
        "name": "Alpha#5",
        "category": "波动形态",
        "description": "开盘价相对 10 日均价的偏离排名 × 收盘价与 VWAP 偏离排名的绝对值取负。",
        "expr": _combine(
            "mul",
            _cross("rank", _combine("sub", open_,
                                    _combine("div", _ts("ts_sum", vwap, 10), _c(10)))),
            _neg(_u("abs", _cross("rank", _combine("sub", close, vwap)))),
        ),
    },
    "alpha101_006": {
        "name": "Alpha#6",
        "category": "量价相关",
        "description": "开盘价与成交量的 10 日滚动相关取负。",
        "expr": _neg(_ts2("ts_corr", open_, volume, 10)),
    },
    "alpha101_007": {
        "name": "Alpha#7",
        "category": "条件结构",
        "description": "量能条件式：放量日取波动衰减信号，缩量日记 -1。A 股放量/缩量分域的最经典公式。",
        "expr": _cond(_b("lt", adv(20), volume),
                      _combine("mul",
                               _neg(_ts("ts_rank", _u("abs", _ts("delta", close, 7)), 60)),
                               _u("sign", _ts("delta", close, 7))),
                      _neg(_c(1))),
    },
    "alpha101_008": {
        "name": "Alpha#8",
        "category": "动量反转",
        "description": "（5 日开盘和 × 5 日收益和）与其 10 日前取值的差取负排名。",
        "expr": _neg(_cross("rank", _combine(
            "sub",
            _combine("mul", _ts("ts_sum", open_, 5), _ts("ts_sum", r1, 5)),
            _ts("delay", _combine("mul", _ts("ts_sum", open_, 5), _ts("ts_sum", r1, 5)), 10),
        ))),
    },
    "alpha101_009": {
        "name": "Alpha#9",
        "category": "条件结构",
        "description": "5 日连涨（或连跌）时保留当日变动，否则取负——趋势确认反转过滤。",
        "expr": _cond(
            _b("or",
               _b("lt", _c(0), _ts("ts_min", _ts("delta", close, 1), 5)),
               _b("lt", _ts("ts_max", _ts("delta", close, 1), 5), _c(0))),
            _ts("delta", close, 1),
            _neg(_ts("delta", close, 1))),
    },
    "alpha101_010": {
        "name": "Alpha#10",
        "category": "条件结构",
        "description": "同 Alpha#9，4 日窗口：4 日连涨（或连跌）时保留当日变动，否则取负。",
        "expr": _cond(
            _b("or",
               _b("lt", _c(0), _ts("ts_min", _ts("delta", close, 1), 4)),
               _b("lt", _ts("ts_max", _ts("delta", close, 1), 4), _c(0))),
            _ts("delta", close, 1),
            _neg(_ts("delta", close, 1))),
    },
    "alpha101_011": {
        "name": "Alpha#11",
        "category": "波动形态",
        "description": "VWAP 与收盘偏离的 3 日高低点排名之和 × 成交量 3 日变化排名。",
        "expr": _combine(
            "mul",
            _combine("add",
                     _cross("rank", _ts("ts_max", _combine("sub", vwap, close), 3)),
                     _cross("rank", _ts("ts_min", _combine("sub", vwap, close), 3))),
            _cross("rank", _ts("delta", volume, 3))),
    },
    "alpha101_012": {
        "name": "Alpha#12",
        "category": "动量反转",
        "description": "成交量方向符号 × 收盘价 1 日变化取负——量价配合的短期反转。",
        "expr": _combine("mul", _u("sign", _ts("delta", volume, 1)),
                         _neg(_ts("delta", close, 1))),
    },
    "alpha101_013": {
        "name": "Alpha#13",
        "category": "量价相关",
        "description": "收盘价排名与成交量排名的 5 日滚动协方差取负排名。",
        "expr": _neg(_cross("rank", _ts2("ts_cov",
                                         _cross("rank", close), _cross("rank", volume), 5))),
    },
    "alpha101_014": {
        "name": "Alpha#14",
        "category": "量价相关",
        "description": "收益 3 日变化取负排名 × 开盘价与成交量 10 日相关。",
        "expr": _combine("mul",
                         _neg(_cross("rank", _ts("delta", r1, 3))),
                         _ts2("ts_corr", open_, volume, 10)),
    },
    "alpha101_015": {
        "name": "Alpha#15",
        "category": "量价相关",
        "description": "高价排名与成交量排名的 3 日相关，排名后 3 日求和取负。",
        "expr": _neg(_ts("ts_sum",
                         _cross("rank", _ts2("ts_corr",
                                             _cross("rank", high), _cross("rank", volume), 3)),
                         3)),
    },
    "alpha101_016": {
        "name": "Alpha#16",
        "category": "量价相关",
        "description": "高价排名与成交量排名的 5 日滚动协方差取负排名。",
        "expr": _neg(_cross("rank", _ts2("ts_cov",
                                         _cross("rank", high), _cross("rank", volume), 5))),
    },
    "alpha101_017": {
        "name": "Alpha#17",
        "category": "时序结构",
        "description": "三重结构：价格时序排名、收盘价二阶差分、量比时序排名连乘。",
        "expr": _combine(
            "mul",
            _combine("mul",
                     _neg(_cross("rank", _ts("ts_rank", close, 10))),
                     _cross("rank", _ts("delta", _ts("delta", close, 1), 1))),
            _cross("rank", _ts("ts_rank", _combine("div", volume, adv(20)), 5))),
    },
    "alpha101_018": {
        "name": "Alpha#18",
        "category": "波动形态",
        "description": "收盘-开盘绝对差的 5 日波动 + 当日实体 + 收盘开盘 10 日相关，取负排名。",
        "expr": _neg(_cross("rank", _combine(
            "add",
            _combine("add",
                     _ts("ts_std", _u("abs", _combine("sub", close, open_)), 5),
                     _combine("sub", close, open_)),
            _ts2("ts_corr", close, open_, 10)))),
    },
    "alpha101_019": {
        "name": "Alpha#19",
        "category": "动量反转",
        "description": "收盘 7 日位移符号取负 ×（1 + 250 日收益和的排名）。",
        "expr": _combine(
            "mul",
            _neg(_u("sign", _combine("add",
                                     _combine("sub", close, _ts("delay", close, 7)),
                                     _ts("delta", close, 7)))),
            _combine("add", _c(1), _cross("rank", _combine("add", _c(1), _ts("ts_sum", r1, 250))))),
    },
    "alpha101_020": {
        "name": "Alpha#20",
        "category": "波动形态",
        "description": "开盘价对昨日高/收/低三个参照的偏离排名连乘取负。",
        "expr": _combine(
            "mul",
            _combine("mul",
                     _neg(_cross("rank", _combine("sub", open_, _ts("delay", high, 1)))),
                     _cross("rank", _combine("sub", open_, _ts("delay", close, 1)))),
            _cross("rank", _combine("sub", open_, _ts("delay", low, 1)))),
    },
    "alpha101_021": {
        "name": "Alpha#21",
        "category": "条件结构",
        "description": "三重条件：均线带穿越判涨跌，量能确认。经典均线+量能组合。",
        "expr": _cond(
            _b("or",
               _b("lt", _combine("add", _combine("div", _ts("ts_sum", close, 8), _c(8)),
                                 _ts("ts_std", close, 8)),
                   _combine("div", _ts("ts_sum", close, 2), _c(2))),
               _b("lt", _combine("div", _ts("ts_sum", close, 2), _c(2)),
                   _combine("sub", _combine("div", _ts("ts_sum", close, 8), _c(8)),
                            _ts("ts_std", close, 8)))),
            _neg(_c(1)),
            _cond(_b("or", _b("lt", _c(1), _combine("div", volume, adv(20))),
                     _b("eq", _combine("div", volume, adv(20)), _c(1))),
                  _c(1), _neg(_c(1)))),
    },
    "alpha101_022": {
        "name": "Alpha#22",
        "category": "量价相关",
        "description": "高量 5 日相关的变化取负 × 收盘 20 日波动排名。",
        "expr": _neg(_combine(
            "mul",
            _ts("delta", _ts2("ts_corr", high, volume, 5), 5),
            _cross("rank", _ts("ts_std", close, 20)))),
    },
    "alpha101_023": {
        "name": "Alpha#23",
        "category": "条件结构",
        "description": "20 日均高价被突破时取高价 2 日变化取负，否则 0。",
        "expr": _cond(_b("lt", _combine("div", _ts("ts_sum", high, 20), _c(20)), high),
                      _neg(_ts("delta", high, 2)), _c(0)),
    },
    "alpha101_024": {
        "name": "Alpha#24",
        "category": "条件结构",
        "description": "100 日均线斜率小于等于 5% 时取 100 日低位回撤，否则取 3 日变化取负。",
        "expr": _cond(
            _b("or",
               _b("lt", _combine("div",
                                 _ts("delta", _combine("div", _ts("ts_sum", close, 100), _c(100)), 100),
                                 _ts("delay", close, 100)), _c(0.05)),
               _b("eq", _combine("div",
                                 _ts("delta", _combine("div", _ts("ts_sum", close, 100), _c(100)), 100),
                                 _ts("delay", close, 100)), _c(0.05))),
            _neg(_combine("sub", close, _ts("ts_min", close, 100))),
            _neg(_ts("delta", close, 3))),
    },
    "alpha101_025": {
        "name": "Alpha#25",
        "category": "动量反转",
        "description": "收益取负 × 量能 × VWAP × 振幅的排名——量价形态综合。",
        "expr": _cross("rank", _combine(
            "mul",
            _combine("mul", _combine("mul", _neg(r1), adv(20)), vwap),
            _combine("sub", high, close))),
    },
    "alpha101_026": {
        "name": "Alpha#26",
        "category": "量价相关",
        "description": "量排名与高价排名的 5 日相关，再取 3 日最高取负。",
        "expr": _neg(_ts("ts_max", _ts2("ts_corr",
                                        _ts("ts_rank", volume, 5), _ts("ts_rank", high, 5), 5), 3)),
    },
    # ——— #27-40：条件/复合结构 ———
    "alpha101_027": {
        "name": "Alpha#27",
        "category": "条件结构",
        "description": "量价排名相关的 2 日均值排名高于 0.5 时记 -1，否则 1。",
        "expr": _cond(_b("lt", _c(0.5),
                         _cross("rank", _combine("div",
                                                 _ts("ts_sum", _ts2("ts_corr", _cross("rank", volume),
                                                                     _cross("rank", vwap), 6), 2),
                                                 _c(2)))),
                      _neg(_c(1)), _c(1)),
    },
    "alpha101_028": {
        "name": "Alpha#28",
        "category": "量价相关",
        "description": "量价相关的截面缩放：相关 + 价格中枢 - 收盘价。",
        "expr": _u("scale", _combine(
            "sub", _combine("add", _ts2("ts_corr", adv(20), low, 5),
                            _combine("div", _combine("add", high, low), _c(2))),
            close)),
    },
    "alpha101_029": {
        "name": "Alpha#29",
        "category": "时序结构",
        "description": "深度复合：对数量能的时间结构最小化 + 收益反向滞后 6 日时序排名。",
        "expr": _combine(
            "add",
            _b("min", _ts("ts_product",
                          _cross("rank", _cross("rank", _u("scale", _u("ln", _ts("ts_sum",
                              _ts("ts_min", _cross("rank", _cross("rank", _neg(_cross("rank",
                                  _ts("delta", _combine("sub", close, _c(1)), 5))))), 2),
                              1))))),
                          1),
               _c(5)),
            _ts("ts_rank", _ts("delay", _neg(r1), 6), 5)),
    },
    "alpha101_030": {
        "name": "Alpha#30",
        "category": "动量反转",
        "description": "三日连涨方向信号的排名取反 × 短期量 / 长期量。",
        "expr": _combine(
            "div",
            _combine("mul",
                     _combine("sub", _c(1), _cross("rank", _combine(
                         "add",
                         _combine("add",
                                  _u("sign", _combine("sub", close, _ts("delay", close, 1))),
                                  _u("sign", _combine("sub", _ts("delay", close, 1),
                                                      _ts("delay", close, 2)))),
                         _u("sign", _combine("sub", _ts("delay", close, 2),
                                             _ts("delay", close, 3)))))),
                     _ts("ts_sum", volume, 5)),
            _ts("ts_sum", volume, 20)),
    },
    "alpha101_031": {
        "name": "Alpha#31",
        "category": "时序结构",
        "description": "三重排名衰减 + 3 日变化取负排名 + 量价相关符号的复合。",
        "expr": _combine(
            "add",
            _combine("add",
                     _cross("rank", _cross("rank", _cross("rank", _ts(
                         "decay_linear", _neg(_cross("rank", _cross("rank", _ts("delta", close, 10)))),
                         10)))),
                     _cross("rank", _neg(_ts("delta", close, 3)))),
            _u("sign", _u("scale", _ts2("ts_corr", adv(20), low, 12)))),
    },
    "alpha101_032": {
        "name": "Alpha#32",
        "category": "量价相关",
        "description": "7 日均线偏离的截面缩放 + 20 倍量价相关（VWAP 滞后 5 日）的截面缩放。",
        "expr": _combine(
            "add",
            _u("scale", _combine("sub", _combine("div", _ts("ts_sum", close, 7), _c(7)), close)),
            _combine("mul", _c(20),
                     _u("scale", _ts2("ts_corr", vwap, _ts("delay", close, 5), 230)))),
    },
    "alpha101_033": {
        "name": "Alpha#33",
        "category": "动量反转",
        "description": "（1 - 开盘/收盘）的 1 次幂取负排名——日内方向的反转信号。",
        "expr": _cross("rank", _neg(_ts("signed_power",
                                       _combine("sub", _c(1), _combine("div", open_, close)), 1))),
    },
    "alpha101_034": {
        "name": "Alpha#34",
        "category": "时序结构",
        "description": "短长波动比排名与 1 日变化排名反向的组合。",
        "expr": _cross("rank", _combine(
            "add",
            _combine("sub", _c(1), _cross("rank", _combine("div",
                                                           _ts("ts_std", r1, 2), _ts("ts_std", r1, 5)))),
            _combine("sub", _c(1), _cross("rank", _ts("delta", close, 1))))),
    },
    "alpha101_035": {
        "name": "Alpha#35",
        "category": "时序结构",
        "description": "量 32 日时序排名 ×（1 - 价高差 16 日时序排名）×（1 - 收益 32 日时序排名）。",
        "expr": _combine(
            "mul",
            _combine("mul", _ts("ts_rank", volume, 32),
                     _combine("sub", _c(1), _ts("ts_rank",
                                                _combine("sub", _combine("add", close, high), low), 16))),
            _combine("sub", _c(1), _ts("ts_rank", r1, 32))),
    },
    "alpha101_036": {
        "name": "Alpha#36",
        "category": "复合",
        "description": "五段式复合：量价相关、开盘反转、收益衰减、量价相关、200 日均线偏离（官方系数 2.21/0.7/0.73/0.6）。",
        "expr": _combine(
            "add",
            _combine("add",
                     _combine("add",
                              _combine("mul", _c(2.21),
                                       _cross("rank", _ts2("ts_corr",
                                                           _combine("sub", close, open_),
                                                           _ts("delay", volume, 1), 15))),
                              _combine("mul", _c(0.7),
                                       _cross("rank", _combine("sub", open_, close)))),
                     _combine("mul", _c(0.73),
                              _cross("rank", _ts("ts_rank", _ts("delay", _neg(r1), 6), 5)))),
            _combine("add",
                     _cross("rank", _u("abs", _ts2("ts_corr", vwap, adv(20), 6))),
                     _combine("mul", _c(0.6), _cross("rank", _combine(
                         "mul",
                         _combine("sub", _combine("div", _ts("ts_sum", close, 200), _c(200)), open_),
                         _combine("sub", close, open_)))))),
    },
    "alpha101_037": {
        "name": "Alpha#37",
        "category": "量价相关",
        "description": "昨日实体与收盘 200 日相关排名 + 当日实体排名。",
        "expr": _combine("add",
                         _cross("rank", _ts2("ts_corr",
                                             _ts("delay", _combine("sub", open_, close), 1),
                                             close, 200)),
                         _cross("rank", _combine("sub", open_, close))),
    },
    "alpha101_038": {
        "name": "Alpha#38",
        "category": "动量反转",
        "description": "收盘 10 日时序排名取负 × 收盘/开盘排名。",
        "expr": _combine("mul", _neg(_cross("rank", _ts("ts_rank", close, 10))),
                         _cross("rank", _combine("div", close, open_))),
    },
    "alpha101_039": {
        "name": "Alpha#39",
        "category": "动量反转",
        "description": "7 日变化 ×（1 - 量比衰减排名）取负排名 ×（1 + 250 日收益和排名）。",
        "expr": _combine(
            "mul",
            _neg(_cross("rank", _combine(
                "mul", _ts("delta", close, 7),
                _combine("sub", _c(1), _cross("rank", _ts(
                    "decay_linear", _combine("div", volume, adv(20)), 9)))))),
            _combine("add", _c(1), _cross("rank", _ts("ts_sum", r1, 250)))),
    },
    "alpha101_040": {
        "name": "Alpha#40",
        "category": "量价相关",
        "description": "高价 10 日波动取负排名 × 高量 10 日相关。",
        "expr": _combine("mul",
                         _neg(_cross("rank", _ts("ts_std", high, 10))),
                         _ts2("ts_corr", high, volume, 10)),
    },
}

# ——— 共用缩写（后半段公式的公共子表达式） ———

p_hl2 = _combine("div", _combine("add", high, low), _c(2))     # (high + low) / 2
# 20 日斜率 − 10 日斜率：((delay(c,20)-delay(c,10))/10) - ((delay(c,10)-c)/10)，#46/#49/#51 共用
slope10 = _combine(
    "sub",
    _combine("div", _combine("sub", _ts("delay", close, 20), _ts("delay", close, 10)), _c(10)),
    _combine("div", _combine("sub", _ts("delay", close, 10), close), _c(10)))
hl_ma5 = _combine("div", _combine("sub", high, low),
                  _combine("div", _ts("ts_sum", close, 5), _c(5)))  # (high-low)/(5日均价)，#83 用
# #73 的混合价（open 14.72% + low 85.28%），公式内出现两次
mixed73 = _combine("add", _combine("mul", open_, _c(0.147155)),
                   _combine("mul", low, _c(0.852845)))

ALPHA101_FACTORS.update({
    # ——— #41-47：条件/复合结构（续） ———
    "alpha101_041": {
        "name": "Alpha#41",
        "category": "波动形态",
        "description": "高低价几何均值（平方根）与 VWAP 的差——价格中枢偏离。",
        "expr": _combine("sub", _b("pow", _combine("mul", high, low), _c(0.5)), vwap),
    },
    "alpha101_042": {
        "name": "Alpha#42",
        "category": "动量反转",
        "description": "VWAP 偏离排名 / VWAP 中枢排名——收盘位置在成交量加权均值中的相对位。",
        "expr": _combine("div",
                         _cross("rank", _combine("sub", vwap, close)),
                         _cross("rank", _combine("add", vwap, close))),
    },
    "alpha101_043": {
        "name": "Alpha#43",
        "category": "时序结构",
        "description": "量比 20 日时序排名 × 负收益 8 日时序排名——放量与回调叠加。",
        "expr": _combine("mul",
                         _ts("ts_rank", _combine("div", volume, adv(20)), 20),
                         _ts("ts_rank", _neg(_ts("delta", close, 7)), 8)),
    },
    "alpha101_044": {
        "name": "Alpha#44",
        "category": "量价相关",
        "description": "高价与成交量排名的 5 日相关取负——价涨量缩背离。",
        "expr": _neg(_ts2("ts_corr", high, _cross("rank", volume), 5)),
    },
    "alpha101_045": {
        "name": "Alpha#45",
        "category": "量价相关",
        "description": "三重排名结构：5 日滞后均价排名 × 量价相关 × 5 日/20 日均价相关，整体取负。",
        "expr": _neg(_combine(
            "mul",
            _combine("mul",
                     _cross("rank", _combine("div",
                                             _ts("ts_sum", _ts("delay", close, 5), 20), _c(20))),
                     _ts2("ts_corr", close, volume, 2)),
            _cross("rank", _ts2("ts_corr", _ts("ts_sum", close, 5),
                                _ts("ts_sum", close, 20), 2)))),
    },
    "alpha101_046": {
        "name": "Alpha#46",
        "category": "条件结构",
        "description": "斜率加速（20 日斜率 > 10 日斜率 0.25）记 -1；斜率转负记 1；否则当日变动取负。",
        "expr": _cond(_b("lt", _c(0.25), slope10),
                      _neg(_c(1)),
                      _cond(_b("lt", slope10, _c(0)),
                            _c(1),
                            _neg(_ts("delta", close, 1)))),
    },
    "alpha101_047": {
        "name": "Alpha#47",
        "category": "复合",
        "description": "倒挂排名 × 量能 × 高位反转排名 - VWAP 滞后偏离排名。",
        "expr": _combine(
            "sub",
            _combine("mul",
                     _combine("div",
                              _combine("mul", _cross("rank", _combine("div", _c(1), close)), volume),
                              adv(20)),
                     _combine("div",
                              _combine("mul", high, _cross("rank", _combine("sub", high, close))),
                              _combine("div", _ts("ts_sum", high, 5), _c(5)))),
            _cross("rank", _combine("sub", vwap, _ts("delay", vwap, 5)))),
    },
    # ——— #49-55：回调结构族 ———
    "alpha101_049": {
        "name": "Alpha#49",
        "category": "条件结构",
        "description": "斜率下穿 -0.1 时记 1，否则当日变动取负——20 日斜率拐头信号。",
        "expr": _cond(_b("lt", slope10, _neg(_c(0.1))), _c(1),
                      _neg(_ts("delta", close, 1))),
    },
    "alpha101_050": {
        "name": "Alpha#50",
        "category": "量价相关",
        "description": "量价排名相关的 5 日时序最高取负——量价联动极值反转。",
        "expr": _neg(_ts("ts_max",
                         _cross("rank", _ts2("ts_corr",
                                             _cross("rank", volume), _cross("rank", vwap), 5)),
                         5)),
    },
    "alpha101_051": {
        "name": "Alpha#51",
        "category": "条件结构",
        "description": "同 Alpha#49，阈值放宽到 -0.05。",
        "expr": _cond(_b("lt", slope10, _neg(_c(0.05))), _c(1),
                      _neg(_ts("delta", close, 1))),
    },
    "alpha101_052": {
        "name": "Alpha#52",
        "category": "动量反转",
        "description": "5 日低点缺口（负）与 240 日/20 日收益差排名 × 量 5 日时序排名。",
        "expr": _combine(
            "mul",
            _combine("mul",
                     _combine("add", _neg(_ts("ts_min", low, 5)),
                              _ts("delay", _ts("ts_min", low, 5), 5)),
                     _cross("rank", _combine("div",
                                             _combine("sub", _ts("ts_sum", r1, 240),
                                                      _ts("ts_sum", r1, 20)),
                                             _c(220)))),
            _ts("ts_rank", volume, 5)),
    },
    "alpha101_053": {
        "name": "Alpha#53",
        "category": "动量反转",
        "description": "（收盘位置 - 1）型形态的 9 日变化取负——实体位置漂移反转。",
        "expr": _neg(_ts("delta",
                         _combine("div",
                                  _combine("sub", _combine("sub", close, low),
                                           _combine("sub", high, close)),
                                  _combine("sub", close, low)),
                         9)),
    },
    "alpha101_054": {
        "name": "Alpha#54",
        "category": "波动形态",
        "description": "低开缺口 × 开盘 5 次方 /（低高差 × 收盘 5 次方）取负——非线性放大的低开信号。",
        "expr": _combine("div",
                         _neg(_combine("mul", _combine("sub", low, close),
                                       _b("pow", open_, _c(5)))),
                         _combine("mul", _combine("sub", low, high),
                                  _b("pow", close, _c(5)))),
    },
    "alpha101_055": {
        "name": "Alpha#55",
        "category": "波动形态",
        "description": "12 日布林位置排名与量排名的 6 日相关取负。",
        "expr": _neg(_ts2("ts_corr",
                          _cross("rank", _combine("div",
                                                  _combine("sub", close, _ts("ts_min", low, 12)),
                                                  _combine("sub", _ts("ts_max", high, 12),
                                                           _ts("ts_min", low, 12)))),
                          _cross("rank", volume), 6)),
    },
    # ——— #60-62、#64-66、#68：小数窗口优化版（round 取整） ———
    "alpha101_060": {
        "name": "Alpha#60",
        "category": "复合",
        "description": "收盘位置 × 量能的缩放排名 × 2 - 10 日最高点位置缩放排名，取负。",
        "expr": _neg(_combine(
            "sub",
            _combine("mul", _c(2),
                     _u("scale", _cross("rank", _combine(
                         "mul",
                         _combine("div", _combine("sub", _combine("sub", close, low),
                                                  _combine("sub", high, close)),
                                  _combine("sub", high, low)),
                         volume)))),
            _u("scale", _cross("rank", _ts("ts_argmax", close, 10))))),
    },
    "alpha101_061": {
        "name": "Alpha#61",
        "category": "量价相关",
        "description": "VWAP 与 16 日低点之差排名 < VWAP 与 180 日均量相关排名（窗口 16.12/17.93 取整）。",
        "expr": _b("lt",
                   _cross("rank", _combine("sub", vwap, _ts("ts_min", vwap, 16))),
                   _cross("rank", _ts2("ts_corr", vwap, adv(180), 18))),
    },
    "alpha101_062": {
        "name": "Alpha#62",
        "category": "条件结构",
        "description": "VWAP-均量相关排名 < 开盘双排名布尔（官方原式 rank(open)+rank(open)，疑似笔误照搬）。",
        "expr": _neg1(_b("lt",
                        _cross("rank", _ts2("ts_corr", vwap, _ts("ts_sum", adv(20), 22), 10)),
                        _cross("rank", _b("lt",
                                          _combine("add", _cross("rank", open_),
                                                   _cross("rank", open_)),
                                          _combine("add", _cross("rank", p_hl2),
                                                   _cross("rank", high)))))),
    },
    "alpha101_064": {
        "name": "Alpha#64",
        "category": "量价相关",
        "description": "13 日均量加权价与 13 日均量的相关排名 < 混合价 4 日变化排名，取负。",
        "expr": _neg1(_b("lt",
                        _cross("rank", _ts2("ts_corr",
                                            _ts("ts_sum", _combine(
                                                "add", _combine("mul", open_, _c(0.178404)),
                                                _combine("mul", low, _c(0.821596))), 13),
                                            _ts("ts_sum", adv(120), 13), 17)),
                        _cross("rank", _ts("delta", _combine(
                            "add", _combine("mul", p_hl2, _c(0.178404)),
                            _combine("mul", vwap, _c(0.821596))), 4)))),
    },
    "alpha101_065": {
        "name": "Alpha#65",
        "category": "量价相关",
        "description": "开/VWAP 混合价与 9 日均量的相关排名 < 开盘 14 日低点回撤排名，取负。",
        "expr": _neg1(_b("lt",
                        _cross("rank", _ts2("ts_corr",
                                            _combine("add", _combine("mul", open_, _c(0.00817205)),
                                                     _combine("mul", vwap, _c(0.99182795))),
                                            _ts("ts_sum", adv(60), 9), 6)),
                        _cross("rank", _combine("sub", open_, _ts("ts_min", open_, 14))))),
    },
    "alpha101_066": {
        "name": "Alpha#66",
        "category": "复合",
        "description": "VWAP 4 日变化衰减排名 + 低点形态 11 日衰减的 7 日时序排名，取负。",
        "expr": _neg(_combine(
            "add",
            _cross("rank", _ts("decay_linear", _ts("delta", vwap, 4), 7)),
            _ts("ts_rank", _ts("decay_linear",
                               _combine("div",
                                        _combine("sub", _combine(
                                            "add", _combine("mul", low, _c(0.96633)),
                                            _combine("mul", low, _c(0.03367))), vwap),
                                        _combine("sub", open_, p_hl2)),
                               11), 7))),
    },
    "alpha101_068": {
        "name": "Alpha#68",
        "category": "条件结构",
        "description": "高价/均量排名的 9 日相关时序排名 < 混合价 1 日变化排名，取负。",
        "expr": _neg1(_b("lt",
                        _ts("ts_rank", _ts2("ts_corr", _cross("rank", high),
                                             _cross("rank", adv(15)), 9), 14),
                        _cross("rank", _ts("delta", _combine(
                            "add", _combine("mul", close, _c(0.518371)),
                            _combine("mul", low, _c(0.481629))), 1)))),
    },
    # ——— #71-75、#77-78：深度复合优化版 ———
    "alpha101_071": {
        "name": "Alpha#71",
        "category": "复合",
        "description": "价格/均量时序排名的相关衰减时序排名 vs 低开双 VWAP 偏离平方衰减排名，取大。",
        "expr": _b("max",
                   _ts("ts_rank", _ts("decay_linear",
                                      _ts2("ts_corr", _ts("ts_rank", close, 3),
                                           _ts("ts_rank", adv(180), 12), 18), 4), 16),
                   _ts("ts_rank", _ts("decay_linear",
                                      _b("pow", _cross("rank", _combine(
                                          "sub", _combine("add", low, open_),
                                          _combine("add", vwap, vwap))), _c(2)),
                                      16), 4)),
    },
    "alpha101_072": {
        "name": "Alpha#72",
        "category": "量价相关",
        "description": "中价/均量的 9 日相关衰减排名 ÷ VWAP/量时序排名的 7 日相关衰减排名。",
        "expr": _combine("div",
                         _cross("rank", _ts("decay_linear",
                                            _ts2("ts_corr", p_hl2, adv(40), 9), 10)),
                         _cross("rank", _ts("decay_linear",
                                            _ts2("ts_corr", _ts("ts_rank", vwap, 4),
                                                 _ts("ts_rank", volume, 19), 7), 3))),
    },
    "alpha101_073": {
        "name": "Alpha#73",
        "category": "复合",
        "description": "VWAP 5 日变化衰减排名 vs 混合价日收益负向衰减的时序排名，取大取负。",
        "expr": _neg(_b("max",
                        _cross("rank", _ts("decay_linear", _ts("delta", vwap, 5), 3)),
                        _ts("ts_rank", _ts("decay_linear",
                                           _neg(_combine("div", _ts("delta", mixed73, 2), mixed73)),
                                           3), 17))),
    },
    "alpha101_074": {
        "name": "Alpha#74",
        "category": "量价相关",
        "description": "收盘与 37 日均量相关的排名 < 混合价/量排名相关的排名，取负。",
        "expr": _neg1(_b("lt",
                        _cross("rank", _ts2("ts_corr", close, _ts("ts_sum", adv(30), 37), 15)),
                        _cross("rank", _ts2("ts_corr",
                                            _cross("rank", _combine(
                                                "add", _combine("mul", high, _c(0.0261661)),
                                                _combine("mul", vwap, _c(0.9738339)))),
                                            _cross("rank", volume), 11)))),
    },
    "alpha101_075": {
        "name": "Alpha#75",
        "category": "量价相关",
        "description": "VWAP-量相关排名 < 低价/均量排名相关排名（README 转录笔误已按官方语义还原）。",
        "expr": _b("lt",
                   _cross("rank", _ts2("ts_corr", vwap, volume, 4)),
                   _cross("rank", _ts2("ts_corr", _cross("rank", low),
                                       _cross("rank", adv(50)), 12))),
    },
    "alpha101_077": {
        "name": "Alpha#77",
        "category": "复合",
        "description": "中价-缺口差的 20 日衰减排名 vs 中价/均量的 3 日相关衰减排名，取小。",
        "expr": _b("min",
                   _cross("rank", _ts("decay_linear",
                                      _combine("sub", _combine("add", p_hl2, high),
                                               _combine("add", vwap, high)), 20)),
                   _cross("rank", _ts("decay_linear",
                                      _ts2("ts_corr", p_hl2, adv(40), 3), 6))),
    },
    "alpha101_078": {
        "name": "Alpha#78",
        "category": "量价相关",
        "description": "低/VWAP 混合价与 40 日均量相关的排名，以量价相关排名为幂指数。",
        "expr": _b("pow",
                   _cross("rank", _ts2("ts_corr",
                                       _ts("ts_sum", _combine(
                                           "add", _combine("mul", low, _c(0.352233)),
                                           _combine("mul", vwap, _c(0.647767))), 20),
                                       _ts("ts_sum", adv(40), 20), 7)),
                   _cross("rank", _ts2("ts_corr", _cross("rank", vwap),
                                       _cross("rank", volume), 6))),
    },
    # ——— #81、#83-86、#88、#92、#94-96、#98-99、#101 ———
    "alpha101_081": {
        "name": "Alpha#81",
        "category": "复合",
        "description": "量价相关 4 次方排名的 15 日乘积对数排名 < 量价相关排名，取负。",
        "expr": _neg1(_b("lt",
                        _cross("rank", _u("ln", _ts("ts_product",
                                                    _cross("rank", _b("pow",
                                                                      _cross("rank", _ts2(
                                                                          "ts_corr", vwap,
                                                                          _ts("ts_sum", adv(10), 50), 8)),
                                                                      _c(4))),
                                                    15))),
                        _cross("rank", _ts2("ts_corr", _cross("rank", vwap),
                                            _cross("rank", volume), 5)))),
    },
    "alpha101_083": {
        "name": "Alpha#83",
        "category": "量价相关",
        "description": "振幅/5 日均价比的 2 日滞后排名 × 量排名平方，除以振幅比/(VWAP-收盘)。",
        "expr": _combine("div",
                         _combine("mul",
                                  _cross("rank", _ts("delay", hl_ma5, 2)),
                                  _cross("rank", _cross("rank", volume))),
                         _combine("div", hl_ma5, _combine("sub", vwap, close))),
    },
    "alpha101_084": {
        "name": "Alpha#84",
        "category": "时序结构",
        "description": "VWAP 与 15 日高点回撤的 21 日时序排名，以收盘 5 日变化为幂指数（面板指数）。",
        "expr": _b("pow",
                   _ts("ts_rank", _combine("sub", vwap, _ts("ts_max", vwap, 15)), 21),
                   _ts("delta", close, 5)),
    },
    "alpha101_085": {
        "name": "Alpha#85",
        "category": "量价相关",
        "description": "高/收混合价与 30 日均量相关的排名，以中价/量时序排名相关的排名为幂指数。",
        "expr": _b("pow",
                   _cross("rank", _ts2("ts_corr",
                                       _combine("add", _combine("mul", high, _c(0.876703)),
                                                _combine("mul", close, _c(0.123297))),
                                       adv(30), 10)),
                   _cross("rank", _ts2("ts_corr",
                                       _ts("ts_rank", p_hl2, 4),
                                       _ts("ts_rank", volume, 10), 7))),
    },
    "alpha101_086": {
        "name": "Alpha#86",
        "category": "条件结构",
        "description": "收盘/均量相关的 20 日时序排名 < 收盘-中价差排名，取负。",
        "expr": _neg1(_b("lt",
                        _ts("ts_rank", _ts2("ts_corr", close,
                                            _ts("ts_sum", adv(20), 15), 6), 20),
                        _cross("rank", _combine("sub", _combine("add", open_, close),
                                                _combine("add", vwap, open_))))),
    },
    "alpha101_088": {
        "name": "Alpha#88",
        "category": "复合",
        "description": "开盘低价排名和 - 高价收盘排名和的 8 日衰减排名 vs 收盘/均量时序排名相关的衰减时序排名，取小。",
        "expr": _b("min",
                   _cross("rank", _ts("decay_linear",
                                      _combine("sub",
                                               _combine("add", _cross("rank", open_),
                                                        _cross("rank", low)),
                                               _combine("add", _cross("rank", high),
                                                        _cross("rank", close))), 8)),
                   _ts("ts_rank", _ts("decay_linear",
                                      _ts2("ts_corr", _ts("ts_rank", close, 8),
                                           _ts("ts_rank", adv(60), 21), 8), 7), 3)),
    },
    "alpha101_092": {
        "name": "Alpha#92",
        "category": "条件结构",
        "description": "中价收盘高于低开点的布尔面板直接衰减（15 日）后 19 日时序排名 vs 低价/均量相关衰减，取小。",
        "expr": _b("min",
                   _ts("ts_rank", _ts("decay_linear",
                                      _b("lt", _combine("add", p_hl2, close),
                                         _combine("add", low, open_)), 15), 19),
                   _ts("ts_rank", _ts("decay_linear",
                                      _ts2("ts_corr", _cross("rank", low),
                                           _cross("rank", adv(30)), 8), 7), 7)),
    },
    "alpha101_094": {
        "name": "Alpha#94",
        "category": "时序结构",
        "description": "VWAP 与 12 日低点回撤的排名，以 VWAP/均量时序相关时序排名为幂指数，取负。",
        "expr": _neg(_b("pow",
                        _cross("rank", _combine("sub", vwap, _ts("ts_min", vwap, 12))),
                        _ts("ts_rank", _ts2("ts_corr", _ts("ts_rank", vwap, 20),
                                            _ts("ts_rank", adv(60), 4), 18), 3))),
    },
    "alpha101_095": {
        "name": "Alpha#95",
        "category": "条件结构",
        "description": "开盘 12 日低点回撤排名 < 中价/均量相关排名 5 次方的 12 日时序排名。",
        "expr": _b("lt",
                   _cross("rank", _combine("sub", open_, _ts("ts_min", open_, 12))),
                   _ts("ts_rank", _b("pow",
                                     _cross("rank", _ts2("ts_corr",
                                                         _ts("ts_sum", p_hl2, 19),
                                                         _ts("ts_sum", adv(40), 19), 13)),
                                     _c(5)), 12)),
    },
    "alpha101_096": {
        "name": "Alpha#96",
        "category": "复合",
        "description": "量价相关排名衰减的时序排名 vs 相关位置 13 日最高点衰减时序排名，取大取负。",
        "expr": _neg(_b("max",
                        _ts("ts_rank", _ts("decay_linear",
                                           _ts2("ts_corr", _cross("rank", vwap),
                                                _cross("rank", volume), 4), 4), 8),
                        _ts("ts_rank", _ts("decay_linear",
                                           _ts("ts_argmax", _ts2("ts_corr",
                                                                 _ts("ts_rank", close, 7),
                                                                 _ts("ts_rank", adv(60), 4), 4), 13),
                                           14), 13))),
    },
    "alpha101_098": {
        "name": "Alpha#98",
        "category": "复合",
        "description": "VWAP/5 日均量相关衰减排名 - 开盘/均量相关最低点位置衰减时序排名。",
        "expr": _combine(
            "sub",
            _cross("rank", _ts("decay_linear",
                               _ts2("ts_corr", vwap, _ts("ts_sum", adv(5), 26), 5), 7)),
            _cross("rank", _ts("decay_linear",
                               _ts("ts_rank", _ts("ts_argmin",
                                                  _ts2("ts_corr", _cross("rank", open_),
                                                       _cross("rank", adv(15)), 21), 9), 7),
                               8))),
    },
    "alpha101_099": {
        "name": "Alpha#99",
        "category": "量价相关",
        "description": "中价/60 日均量相关的排名 < 低价量相关的排名，取负。",
        "expr": _neg1(_b("lt",
                        _cross("rank", _ts2("ts_corr",
                                            _ts("ts_sum", p_hl2, 20),
                                            _ts("ts_sum", adv(60), 20), 9)),
                        _cross("rank", _ts2("ts_corr", low, volume, 6)))),
    },
    "alpha101_101": {
        "name": "Alpha#101",
        "category": "动量反转",
        "description": "（收盘-开盘）/（高低差+0.001）——归一化实体信号，101 个公式中的收盘收官。",
        "expr": _combine("div",
                         _combine("sub", close, open_),
                         _combine("add", _combine("sub", high, low), _c(0.001))),
    },
})

# ============================================================
# 接口
# ============================================================


def list_alpha101() -> list[dict]:
    """列出全部 81 个 alpha101 因子（含可读公式字符串，供报告/体检管线调用）。"""
    from .dsdl import to_formula

    out = []
    for key, meta in ALPHA101_FACTORS.items():
        out.append({
            "key": key,
            "name": meta["name"],
            "category": meta["category"],
            "description": meta["description"],
            "formula": to_formula(meta["expr"]),
        })
    return out


def get_alpha101(key: str) -> dict | None:
    """按 key（如 'alpha101_007'）取单个因子元数据（含表达式树）。"""
    return ALPHA101_FACTORS.get(key)


# ============================================================
# 自测：合成面板上全部 81 个公式求值 + 公式字符串回读
# ============================================================

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from .factor_engine import compute_factor
    from .dsdl import to_formula, parse_formula

    rng = np.random.default_rng(42)
    n_days, n_stocks = 260, 20
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    codes = [f"c{i:03d}" for i in range(n_stocks)]

    # 合成 OHLCV：随机游走 + 日内扰动，amount=volume*close 保证 vwap≈close
    ret = rng.normal(0.0005, 0.02, (n_days, n_stocks))
    close_ = 10 * np.exp(np.cumsum(ret, axis=0))
    open_ = close_ * (1 + rng.normal(0, 0.005, (n_days, n_stocks)))
    high_ = np.maximum(open_, close_) * (1 + np.abs(rng.normal(0, 0.005, (n_days, n_stocks))))
    low_ = np.minimum(open_, close_) * (1 - np.abs(rng.normal(0, 0.005, (n_days, n_stocks))))
    volume_ = rng.lognormal(15, 0.4, (n_days, n_stocks))

    panel = {
        "open": pd.DataFrame(open_, index=dates, columns=codes),
        "close": pd.DataFrame(close_, index=dates, columns=codes),
        "high": pd.DataFrame(high_, index=dates, columns=codes),
        "low": pd.DataFrame(low_, index=dates, columns=codes),
        "volume": pd.DataFrame(volume_, index=dates, columns=codes),
        "amount": pd.DataFrame(volume_ * close_, index=dates, columns=codes),
        "vwap": pd.DataFrame(close_ * (1 + rng.normal(0, 0.003, (n_days, n_stocks))),
                             index=dates, columns=codes),
    }

    failed, ok = [], 0
    print(f"🧪 alpha101 因子库自测：{len(ALPHA101_FACTORS)} 个公式")
    for key in sorted(ALPHA101_FACTORS, key=lambda k: int(k.split("_")[1])):
        meta = ALPHA101_FACTORS[key]
        try:
            out = compute_factor(meta["expr"], panel)
            assert out.shape == (n_days, n_stocks), f"shape={out.shape}"
            assert out.iloc[60:, :].notna().any().any(), "全 NaN"
            formula = to_formula(meta["expr"])          # 公式字符串可生成
            reparsed = parse_formula(formula, trusted=True)  # 字符串可回读成树（受信路径）
            assert reparsed == meta["expr"], "round-trip 不一致"
            ok += 1
        except Exception as e:  # noqa: BLE001 —— 自测需要收集所有失败
            failed.append((key, str(e)))

    print(f"✅ 通过 {ok}/{len(ALPHA101_FACTORS)}")
    if failed:
        for key, err in failed:
            print(f"❌ {key}: {err}")
        raise SystemExit(1)
    print("alpha101 因子库自测通过 ✅")


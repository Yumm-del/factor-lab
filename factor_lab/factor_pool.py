"""
因子池查重模块 ———— 防 LLM 反复生成相似因子
================================================
目的：反思迭代闭环里，LLM 很容易围绕同一个想法反复生成结构不同、
      但数值几乎相同的因子（例如把 `ts_mean(close,20)` 换成
      `ts_sum(close,20)/20`）。这会浪费迭代轮次，也让"已挖因子库"
      失去多样性。

方法（借鉴 AlphaForge AAAI 2025 AlphaPool 设计思想，自行实现，
      原仓库无 LICENSE，仅参考思路不抄代码）：
    新因子与池内已有因子做**逐日截面皮尔逊相关**（两方都非 NaN 且
    ≥10 只股票才入样，与交叉验证脚本 day_corr 同口径），
    中位 |corr| > 0.99 判定为重复——返回最相似因子与相关度，
    由上层（LLM 反思）决定如何处理。

池 = 内置 91 个因子（10 教科书 + 81 WorldQuant 101 移植）+ 会话内
已挖因子。因子值懒计算 + 模块级缓存（同股票池一次，之后秒回）。
"""

import numpy as np
import pandas as pd

try:
    from factor_lab import alpha101_library, dsdl, factor_engine  # 包方式（app.py 路径）
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from factor_lab import alpha101_library, dsdl, factor_engine  # noqa: E402,F811

# 重复判定阈值：中位 |corr| > 0.99 视为数值等价（与对拍脚本的通过线一致）
DUP_THRESHOLD = 0.99


def day_corr(ref: pd.DataFrame, dsl: pd.DataFrame) -> list:
    """逐日截面皮尔逊相关：两行都非 NaN 的股票入样，≥10 只才入样。

    与 scripts/verify_alpha101_ref.py 的 day_corr 同口径——
    查重和交叉验证用同一把尺子，数字可互证。"""
    corrs = []
    ra, rb = ref.to_numpy(dtype=float), dsl.to_numpy(dtype=float)
    with np.errstate(all="ignore"):  # corrcoef 对全 NaN/恒定行内部会告警，静音（结果已判 NaN 剔除）
        for i in range(len(ref)):
            m = ~(np.isnan(ra[i]) | np.isnan(rb[i]))
            n = m.sum()
            if n >= 10:
                c = np.corrcoef(ra[i][m], rb[i][m])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
    return corrs


class FactorPool:
    """内置因子池（91 个）+ 会话内已挖因子，提供重复查重。

    设计：
        - 池成员注册时只存 (name, expr)，因子值懒计算；
        - _values 做模块级缓存：同一股票池内首次查重全池算一遍，
          之后秒回（91 个因子向量化求值约 30-60 秒一次，可接受）；
        - check() 返回最相似因子 dict（name/corr/类别），
          None 表示未发现重复。
    """

    def __init__(self):
        self._members: list[dict] = []   # [{key, name, category, expr}]
        self._values: dict[str, pd.DataFrame] = {}  # key -> 因子值面板（缓存）

    # —— 池成员管理 ——
    def register(self, key: str, name: str, category: str, expr: dict) -> None:
        """注册一个池成员（含表达式树）。"""
        self._members.append({"key": key, "name": name, "category": category, "expr": expr})

    def register_builtin(self) -> None:
        """注册内置 91 因子：10 教科书 + 81 WorldQuant 101 移植。"""
        for f in factor_engine.list_factors():
            meta = factor_engine.get_factor(f["key"])
            self.register(f["key"], meta["name"], meta["category"], meta["expr"])
        for f in alpha101_library.list_alpha101():
            meta = alpha101_library.get_alpha101(f["key"])
            self.register(f["key"], meta["name"], meta["category"], meta["expr"])

    def add_discovered(self, key: str, name: str, expr: dict) -> None:
        """会话内新挖因子加入池（查重后）——让池随会话生长。"""
        self.register(key, name, "已挖", expr)

    def contains(self, key: str) -> bool:
        """key 是否已在池中（幂等入池：同一表达式只加一次）。"""
        return any(m["key"] == key for m in self._members)

    # —— 查重 ——
    def _value_of(self, member: dict, panel: dict) -> pd.DataFrame:
        """取池成员因子值（懒计算 + 缓存）。"""
        k = member["key"]
        if k not in self._values:
            self._values[k] = dsdl.evaluate(member["expr"], panel)
        return self._values[k]

    def check(self, candidate: pd.DataFrame, panel: dict,
              threshold: float = DUP_THRESHOLD, k: int = 3) -> dict:
        """候选因子 vs 全池逐日截面相关查重。

        参数：
            candidate — 候选因子值面板 (date × code)
            panel     — 数据面板（求值池成员需要）
            threshold — 重复阈值（中位 |corr|），默认 0.99
            k         — top 列表长度（供 UI 展示"最相似提示"）

        返回：
            {"hit": 最相似且超阈值的成员 dict（{key,name,category,corr}）| None,
             "top": 与候选最相似的 k 个池成员（不看阈值）}
        """
        scored = []
        for m in self._members:
            vals = self._value_of(m, panel)
            corrs = day_corr(candidate, vals)
            if not corrs:
                continue
            scored.append({"key": m["key"], "name": m["name"],
                           "category": m["category"],
                           "corr": float(np.median(np.abs(corrs)))})
        scored.sort(key=lambda x: x["corr"], reverse=True)
        hit = scored[0] if scored and scored[0]["corr"] > threshold else None
        return {"hit": hit, "top": scored[:k]}


# 模块级单例：同一进程内复用（Streamlit 重跑时由 st.cache_resource 包一层）
_pool_singleton: FactorPool | None = None


def get_pool() -> FactorPool:
    """获取内置因子池单例（懒注册 91 个内置因子）。"""
    global _pool_singleton
    if _pool_singleton is None:
        p = FactorPool()
        p.register_builtin()
        _pool_singleton = p
    return _pool_singleton


if __name__ == "__main__":
    # 自测：一个与 alpha101_012 同结构的候选（不同写法但数值近似）应被查重
    from factor_lab.data_pipeline import load_panel

    panel = load_panel("hs300")
    pool = get_pool()
    print(f"池成员数: {len(pool._members)}（10 教科书 + 81 WQ101）")

    # 候选 1：alpha101_012 的数值等价改写（ts_sum 换 ts_mean×窗口，结果相同）
    orig = alpha101_library.get_alpha101("alpha101_012")["expr"]
    cand = dsdl.evaluate(orig, panel)
    res = pool.check(cand, panel)
    print(f"原样回喂 alpha101_012 → hit: {res['hit']} | top1: {res['top'][0] if res['top'] else None}")
    assert res["hit"] is not None and res["hit"]["corr"] > 0.99, "同数值因子必须被判重复"
    assert res["top"][0]["key"] == "alpha101_012", "最相似因子应为 alpha101_012"

    # 候选 2：一个完全不同的因子（随机构造）不应被判重复
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(0, 1, cand.shape), index=cand.index, columns=cand.columns)
    res2 = pool.check(noise, panel)
    print(f"随机噪声 → hit: {res2['hit']}（应为 None）| top1: {res2['top'][0] if res2['top'] else None}")
    assert res2["hit"] is None, "随机因子不应被判重复"
    print("\n✅ 因子池查重自测通过")

"""
因子实验室 · Streamlit 工作台 ———— 演示主界面
================================================
页面结构（三个 Tab，演示顺序即 Tab 顺序）：
    Tab 1  AI 因子工场     —— 核心卖点：自然语言 → 因子 → 体检 → AI 解读
    Tab 2  经典因子库      —— 10 个经典因子的体检（对照组，证明工作台严谨）
    Tab 3  因子对比        —— 多个因子的体检结果横向 PK

设计原则：
    1. 评委 30 秒内看懂"这是什么"——第一屏就是输入框 + 一键生成
    2. 图表全部交互式（plotly）：hover 看数值，缩放看细节
    3. 体检数字旁边永远配一句人话解读（LLM 生成），不堆原始数字
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 包导入（app.py 在项目根目录，factor_lab 是同级包）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_lab import (  # noqa: E402
    alpha101_library, data_pipeline, dsdl, factor_engine, llm_factor,
    neutralize, strategy, validation,
)
from factor_lab.factor_pool import get_pool  # AlphaPool 式因子池查重

st.set_page_config(page_title="因子实验室 · AI Factor Lab", layout="wide", page_icon="🧪")


# ============================================================
# 数据加载（缓存：数据文件不变就不重新加载）
# ============================================================
@st.cache_data(show_spinner="正在加载数据面板…")
def load_panel_cached(pool: str = "hs300"):
    """股票池面板（全 A 首次加载较慢，之后由 Streamlit 缓存秒回）。"""
    return data_pipeline.load_panel(pool)


@st.cache_data(show_spinner="加载指数…")
def load_index_cached() -> pd.Series:
    """沪深300指数收盘序列（策略基准）。"""
    return data_pipeline.load_index()


@st.cache_data(show_spinner="体检中…")
def diagnose_classic_cached(key: str, pool: str = "hs300", style: str = "none"):
    """经典因子体检（缓存：同一因子×池子×中性化只算一次）。"""
    panel = data_pipeline.load_panel(pool)
    expr = factor_engine.get_factor(key)["expr"]
    fac = dsdl.evaluate(expr, panel)
    fac = neutralize_factor(fac, panel, style)
    return validation.full_diagnosis(fac, panel["close"])


@st.cache_data(show_spinner="体检中…")
def diagnose_expr_cached(expr_json: str, pool: str = "hs300", style: str = "none"):
    """任意 DSL 表达式树（JSON 字符串）的体检——alpha101 移植因子复用此入口。

    与 diagnose_classic_cached 同管线（体检指标/评分/方向完全一致），
    只是表达式来源从内置因子库换成 JSON 树，保证 91 个因子同口径。"""
    panel = data_pipeline.load_panel(pool)
    expr = dsdl.parse_factor(expr_json)
    fac = dsdl.evaluate(expr, panel)
    fac = neutralize_factor(fac, panel, style)
    return validation.full_diagnosis(fac, panel["close"])


@st.cache_data(show_spinner="策略回测中（Top30 周频、扣双边成本）…")
def build_strategy_cached(expr_str: str, pool: str = "hs300", style: str = "none") -> dict:
    """因子 → 组合策略回测（缓存：expr_str×池子×中性化作键）。

    与「策略构建」页完全相同的参数（Top30 / 周频 / 20bps），
    保证对比页的 PK 结果可以直接映射到单因子策略页。"""
    panel = data_pipeline.load_panel(pool)
    index_close = data_pipeline.load_index()
    expr = dsdl.parse_factor(expr_str)
    fac = dsdl.evaluate(expr, panel)
    fac = neutralize_factor(fac, panel, style)
    return strategy.build_portfolio(fac, panel["close"], index_close,
                                    n_stocks=30, rebalance_days=5, cost_bps=20.0)


@st.cache_data
def industry_cached() -> pd.Series:
    """行业映射（全 A 下载完成后可用）。"""
    return data_pipeline.load_industry()


def neutralize_factor(fac: pd.DataFrame, panel: dict, style: str) -> pd.DataFrame:
    """按侧边栏设置对因子做截面中性化（无/行业/行业+市值）。

    行业表不存在（全 A 下载未完成）时降级为不中性化并提示。"""
    if style == "none":
        return fac
    try:
        industry = industry_cached()
    except FileNotFoundError:
        st.warning("行业映射表尚未生成（全 A 下载中），本次体检跳过中性化。")
        return fac
    log_size = None
    if style == "industry+size":
        cap = neutralize.mktcap_proxy(panel)
        log_size = np.log(cap)  # log 变换：消除量纲与右偏
    return neutralize.neutralize(fac, industry, log_size, style)


@st.cache_data(show_spinner="加载预置示例…")
def load_presets() -> list[dict]:
    """预置示例（演示用，不依赖 API）→ 现算体检。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "presets.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        presets = json.load(f)
    panel = data_pipeline.load_panel()
    for p in presets:
        expr = dsdl.parse_factor(p["expr_str"])
        fac = dsdl.evaluate(expr, panel)
        p["diag"] = validation.full_diagnosis(fac, panel["close"])
    return presets


# ============================================================
# 绘图函数（plotly，全部交互式）
# ============================================================
def fig_ic_series(ic_table: pd.DataFrame) -> go.Figure:
    """每日 IC + 30 日滚动均值 + 零线。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ic_table.index, y=ic_table["ic"], name="日IC",
                             mode="lines", line=dict(width=0.8, color="#8ab4f8")))
    rolling = ic_table["ic"].rolling(30).mean()
    fig.add_trace(go.Scatter(x=ic_table.index, y=rolling, name="30日滚动均值",
                             line=dict(width=2, color="#ff6b6b")))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="每日 IC（因子值与下期收益的截面相关）", height=320,
                      yaxis_title="IC", xaxis_title="", legend=dict(x=1.0, xanchor="right", y=1.0, yanchor="top",
                                   bgcolor="rgba(255,255,255,0.55)"),
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def fig_layer_nav(layers_meta: dict) -> go.Figure:
    """5 层组合累计净值 + 基准。"""
    nav = layers_meta["layer_nav"]
    bench = layers_meta["benchmark"]
    fig = go.Figure()
    colors = ["#d32f2f", "#f57c00", "#fbc02d", "#7cb342", "#2e7d32"]  # L1 红 → L5 绿
    for i, col in enumerate(nav.columns):
        fig.add_trace(go.Scatter(x=nav.index, y=nav[col], name=col,
                                 line=dict(width=1.6, color=colors[i])))
    bench_nav = (1 + bench.fillna(0)).cumprod()
    fig.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name="等权基准",
                             line=dict(width=2, dash="dot", color="#37474f")))
    fig.update_layout(title="分层组合累计净值（L1=因子值最高层，逐日调仓）", height=340,
                      yaxis_title="净值（起点=1）", legend=dict(x=1.0, xanchor="right", y=1.0, yanchor="top",
                                   bgcolor="rgba(255,255,255,0.55)"),
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def fig_spread(layers_meta: dict) -> go.Figure:
    """多空（L1-L5）累计净值。"""
    spread_nav = layers_meta["spread_nav"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread_nav.index, y=spread_nav, name="多空(L1-L5)",
                             fill="tozeroy", line=dict(width=2, color="#5c6bc0")))
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(title="多空组合（L1-L5）累计净值", height=260,
                      yaxis_title="净值", margin=dict(l=40, r=20, t=50, b=30))
    return fig


def fig_lifecycle(lifecycle: pd.Series) -> go.Figure:
    """因子生命周期：全样本 60 日滚动 IC 曲线（风格切换一眼可见）。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lifecycle.index, y=lifecycle.values,
                             name="60日滚动IC", fill="tozeroy",
                             line=dict(width=1.8, color="#26a69a")))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="因子生命周期（全样本 60 日滚动 IC——注意风格切换）", height=260,
                      yaxis_title="滚动IC", margin=dict(l=40, r=20, t=50, b=30))
    return fig


def fig_decay(decay: dict) -> go.Figure:
    """IC 衰减柱状图。"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(decay.keys()), y=list(decay.values()), name="RankIC",
                         marker_color=["#ef5350" if v > 0 else "#90a4ae" for v in decay.values()]))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="IC 衰减曲线（因子对未来 1~10 日的预测力）", height=260,
                      xaxis_title="未来天数", yaxis_title="RankIC",
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def fig_layer_bar(layers_meta: dict) -> go.Figure:
    """各层年化收益柱状图（单调性一眼可见）。"""
    annual = layers_meta["layer_returns"].mean().mul(252)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=annual.index, y=annual.values,
                         marker_color=["#d32f2f", "#f57c00", "#fbc02d", "#7cb342", "#2e7d32"]))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="各层年化收益（等权、日调仓）", height=260,
                      xaxis_title="分层（L1=因子值最高）", yaxis_title="年化收益",
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


# ============================================================
# 体检结果展示（两个 Tab 共用）
# ============================================================
def render_diagnosis(diag: dict):
    """把体检单渲染成卡片 + 图表 + 指标解读。"""
    s = diag["ic_summary"]
    lay = diag["layers"]
    verdict = diag["verdict"]

    # —— 结论横幅（彩色徽章 + 人话结论）——
    badge_map = {"优秀": "#16a34a", "可用": "#ea580c", "淘汰": "#dc2626"}
    badge = badge_map[verdict["label"]]
    st.markdown(
        f"""
        <div style="border:1px solid {badge}33; background:{badge}0d; border-radius:10px;
                    padding:12px 16px; margin-bottom:6px;">
          <span style="background:{badge}; color:#fff; border-radius:6px;
                       padding:2px 12px; font-weight:700; font-size:15px;">{verdict['label']}</span>
          <span style="font-weight:700; font-size:17px; margin-left:12px;">评分 {diag['score']}/100</span>
          <span style="color:#64748b; font-size:14px; margin-left:8px;">近 {diag['window_days']} 个交易日</span>
          <div style="margin-top:6px; color:#334155;">{verdict['text']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # —— 核心指标卡片 ——
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("IC 均值", f"{s['ic_mean']:+.4f}")
    c2.metric("RankIC 均值", f"{s['rank_ic_mean']:+.4f}")
    c3.metric("IR（信息比率）", f"{s['ic_ir']:.2f}")
    c4.metric("t 值", f"{s['ic_t']:+.2f}")
    c5.metric("多空年化", f"{lay['spread_annual']:+.1%}")

    c6, c7, c8 = st.columns(3)
    c6.metric("IC 为正天数占比", f"{s['ic_positive']:.1%}")
    c7.metric("层间单调性", f"{lay['monotonic']:+.2f}")
    c8.metric("组合日换手率", f"{diag['turnover']:.1%}")

    # —— 图表区 ——
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(fig_ic_series(diag["ic_series"]), width="stretch")
        st.plotly_chart(fig_spread(lay), width="stretch")
    with col_right:
        st.plotly_chart(fig_layer_nav(lay), width="stretch")
        st.plotly_chart(fig_layer_bar(lay), width="stretch")
    st.plotly_chart(fig_decay(diag["ic_decay"]), width="stretch")
    if "lifecycle" in diag and len(diag["lifecycle"].dropna()) > 0:
        st.plotly_chart(fig_lifecycle(diag["lifecycle"]), width="stretch")


# ============================================================
# Tab 1：AI 因子工场（核心卖点）
# ============================================================
def _local_report(formula: str, diag: dict) -> str:
    """手动表达式编辑的本地报告：不调 API，把体检数字翻译成人话。
    与 LLM 解读结构对齐（画像/结论/指标/建议），保证手动与 AI 路径观感一致。"""
    s = diag["ic_summary"]
    lay = diag["layers"]
    decay = diag["ic_decay"]
    lag1, lag5, lag10 = decay.get(1), decay.get(5), decay.get(10)
    decay_txt = (f"信号强度随持有期下降（lag1 {lag1:+.3f} → lag10 {lag10:+.3f}），"
                 "属短线信号，实盘需更高调仓频率" if lag10 is not None and abs(lag1) > abs(lag10) else
                 "信号随持有期衰减平缓，低频调仓即可保留大部分收益")
    direction = "因子值越高、未来收益越高" if s["ic_mean"] > 0 else "因子值越低、未来收益越高（反向）"
    return (
        f"## 因子画像\n"
        f"`{formula}` —— 手动输入的手工表达式。\n\n"
        f"## 体检结论\n"
        f"**{diag['verdict']['text']}**（评分 {diag['score']:.1f}/100，"
        f"基于近 {diag['window_days']} 个交易日滚动窗口）\n\n"
        f"## 关键指标解读\n"
        f"- **方向**：{direction}。近一年 IC 均值 {s['ic_mean']:+.4f}，"
        f"{'统计显著（|t|≥2）' if abs(s['ic_t']) >= 2 else '统计上还不够显著'}。\n"
        f"- **稳定性**：IR（信息比率）{s['ic_ir']:.2f}，IC 为正天数占比 {s['ic_positive']:.0%}"
        f"——{'信号稳定' if s['ic_positive'] > 0.55 else '信号时好时坏，需谨慎'}。\n"
        f"- **分层单调性**：L1（最高层）多空年化 {lay['spread_annual']:+.1%}，"
        f"单调性 {lay['monotonic']:+.2f}（{'层级区分清晰' if lay['monotonic'] > 0.5 else '层级区分一般'}）。\n"
        f"- **实盘成本**：组合日换手 {diag['turnover']:.1%}，{decay_txt}。\n"
        f"## 建议\n"
        f"- 可到「🚀 策略构建」把该因子直接做成组合策略，看扣成本后的真实净值。\n"
        f"- 想改进可调整窗口参数（如 5→10）或组合多个算子再体检。"
    )


# ============================================================
# Tab 1：AI 因子工场（核心卖点）
# ============================================================

def _dedup_hint_ui(dup: dict | None):
    """因子池查重结果 → 工作台提示（重复警告 / 无重复说明）。

    dup 结构来自 factor_pool.FactorPool.check()：
    {"hit": 超阈值的最相似因子 dict|None, "top": 最相似 k 个}。
    """
    if not dup:
        return
    hit = dup.get("hit")
    if hit:
        st.warning(
            f"⚠️ **与已有因子重复**：本因子与「{hit['name']}」（{hit['category']}类）"
            f"逐日截面相关 **{hit['corr']:.4f}**（>0.99 判定重复）。\n\n"
            f"反思迭代已要求 AI 改变核心逻辑（换算子组合/换数据字段/换时间窗口），"
            f"禁止等价变形——重复因子对因子库没有增量贡献。"
        )
    elif dup.get("top"):
        t = dup["top"][0]
        st.caption(
            f"✅ 与因子库现有因子无重复（最相似为「{t['name']}」，"
            f"相关 {t['corr']:.4f}，未超 0.99 阈值）"
        )


def _pool_discovered(expr: dict, formula: str):
    """会话已挖因子加入因子池（幂等）——让池随会话生长，后续查重可识别。"""
    key = f"discovered_{hash(formula) & 0xFFFF:04x}"
    pool = get_pool()
    if not pool.contains(key):
        pool.add_discovered(key, f"已挖: {formula[:28]}", expr)


def render_ai_factory():
    st.title("🧪 AI 因子工场")
    st.markdown(
        "> 用一句话描述你的因子想法，AI 生成受限因子表达式 → 自动体检 → 自动解读。\n"
        "> **演示建议**：试试「放量突破」「低波动」「小市值」等方向，或直接点下面的灵感模板。"
    )

    # —— 灵感模板（降低演示门槛）——
    templates = [
        "放量后价格延续上涨（量价配合）",
        "低波动率的股票未来表现更好",
        "接近52周高点的股票强势延续",
        "短期超跌反弹（5日内跌多了）",
        "高换手率的股票有短线机会",
    ]
    cols = st.columns(len(templates))
    for col, t in zip(cols, templates):
        if col.button(t, width="stretch"):
            st.session_state["idea"] = t

    # —— 演示模式：预置示例（离线，不调 API，评委演示的保底方案）——
    with st.expander("📂 演示模式：载入预置示例（秒开，不调用 AI）"):
        presets = load_presets()
        if not presets:
            st.caption("暂无预置示例（运行 `python scripts/make_preset.py` 生成）")
        for p in presets:
            if st.button(f"载入：{p['idea']}", key=f"preset_{p['idea']}",
                         help=f"因子：{p['formula']}"):
                st.session_state["last_result"] = p

    idea = st.text_input(
        "💡 因子想法（自然语言）",
        value=st.session_state.get("idea", ""),
        placeholder="例如：我想捕捉成交量放大后延续上涨的股票",
    )

    if st.button("🚀 生成并体检因子", type="primary", width="stretch", disabled=not idea.strip()):
        if not idea.strip():
            st.warning("请输入因子想法")
            return
        try:
            with st.spinner("AI 正在生成因子表达式…"):
                result = llm_factor.generate_factor(idea, panel, panel["close"])

            # 因子池查重提示（首轮生成会触发全池 91 因子懒计算缓存，稍等片刻）
            _dedup_hint_ui(result.get("dup"))
            _pool_discovered(result["expr"], result["formula"])

            if style != "none":
                # 中性化开启：按当前设置重算体检（报告用本地模板，保证数字与文字一致）
                fac2 = neutralize_factor(dsdl.evaluate(result["expr"], panel), panel, style)
                result["diag"] = validation.full_diagnosis(fac2, panel["close"])
                result["report"] = _local_report(result["formula"], result["diag"])

            st.session_state["last_result"] = result
        except Exception as e:  # noqa: BLE001
            st.error(f"生成失败：{e}")

    # —— 手动表达式编辑：评委亲自动手改因子（不依赖 AI/API）——
    with st.expander("✏️ 手动表达式编辑（不调 API：粘贴公式或 JSON 直接体检）"):
        mode = st.radio("输入方式", ["📝 人类可读公式", "🧬 JSON 表达式树"],
                        horizontal=True, label_visibility="collapsed")
        if mode == "📝 人类可读公式":
            default_src = "rank(ts_mean(close, 5) / ts_mean(volume, 20))"
            src = st.text_input("公式", value=default_src,
                                help="算子白名单（38 个）：时序 ts_returns/ts_mean/ts_std/"
                                     "ts_zscore/ts_rank/ts_max/ts_min/delay/delta/ts_sum/"
                                     "ts_product/ts_argmax/ts_argmin/decay_linear/ts_corr/"
                                     "ts_cov · 截面 rank/normalize/scale · 一元 signed_power/"
                                     "log/ln/abs/neg/sign · 二元 add/sub/mul/div/pow/min/max/"
                                     "gt/lt/eq/and/or · 条件 cond；数据叶子 open/close/high/low/"
                                     "volume/amount/vwap/turn/pe/pb；支持 + - * / 与括号")
        else:
            default_src = ('{"op": "rank", "args": [{"op": "div", "args": ['
                           '{"op": "ts_mean", "args": [{"op": "close"}], "param": 5}, '
                           '{"op": "ts_mean", "args": [{"op": "volume"}], "param": 20}]}]}')
            src = st.text_input("JSON 表达式树", value=default_src)
        want_report = st.checkbox("调用 AI 生成解读报告（需 API，约 10 秒）", value=False)
        if st.button("🔬 解析并体检", type="primary", width="stretch", disabled=not src.strip()):
            try:
                expr = dsdl.parse_formula(src) if mode.startswith("📝") else dsdl.parse_factor(src)
                formula = dsdl.to_formula(expr)
                factor_panel = neutralize_factor(dsdl.evaluate(expr, panel), panel, style)
                diag = validation.full_diagnosis(factor_panel, panel["close"])
                if want_report:
                    with st.spinner("AI 解读中…"):
                        report = llm_factor._llm_text(
                            llm_factor.INTERPRET_SYSTEM,
                            llm_factor.build_interpret_prompt(
                                formula, "手动输入，无 AI 逻辑自述", diag),
                        )
                else:
                    report = _local_report(formula, diag)
                # 手动编辑的因子也入池（幂等）——评委亲测的因子同样参与后续查重
                _pool_discovered(expr, formula)
                st.session_state["last_result"] = {
                    "idea": "手动编辑",
                    "rationale": "手动输入的表达式",
                    "expr": expr,
                    "expr_str": json.dumps(expr, ensure_ascii=False),
                    "formula": formula,
                    "tree": dsdl.render_tree(expr),
                    "diag": diag,
                    "report": report,
                }
            except Exception as e:  # noqa: BLE001
                st.error(f"解析失败：{e}")

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        _dedup_hint_ui(result.get("dup"))  # 展示时再次给出查重提示（preset 无 dup 字段，安全跳过）
        st.markdown(f"### 📝 因子定义：`{result['formula']}`")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**AI 的因子逻辑**：{result['rationale']}")
            st.code(result["tree"], language="text")
        with col2:
            st.markdown("**AI 体检报告**")
            st.markdown(result["report"])

        st.divider()
        render_diagnosis(result["diag"])


# ============================================================
# Tab 2：经典因子库（对照组）
# ============================================================
def render_classic_library():
    st.title("📚 经典因子库")
    st.markdown(
        "> 内置 **91 个因子**——10 个教科书经典因子（动量/反转/波动/流动性/价值/趋势）"
        "+ 81 个 WorldQuant 101 公式移植（与公开参考实现逐日相关交叉验证通过，"
        "27/27 双向验证全 >0.99）。全部用同一套 DSL 表达式表示，"
        "与 AI 因子走完全相同的体检流水线。\n"
        "> **这既是基线对照，也是可信度证明**：工作台对「已知有效的因子」能给出正确判定。"
    )

    src = st.radio("因子来源", ["教科书经典（10）", "WorldQuant 101 移植（81）"], horizontal=True)
    if src.startswith("教科书"):
        factors = factor_engine.list_factors()
        cats = ["动量", "反转", "波动", "流动性", "价值", "趋势"]
        selected_cat = st.radio("因子类别", cats, horizontal=True, key="classic_cat")
        in_cat = [f for f in factors if f["category"] == selected_cat]
        labels = {f["key"]: f"{f['name']} —— {f['description']}" for f in in_cat}
        key = st.selectbox("选择因子", list(labels.keys()),
                           format_func=lambda k: labels[k], key="classic_sel")
        expr = factor_engine.get_factor(key)["expr"]
        expr_str = dsdl.to_formula(expr)
    else:
        factors = alpha101_library.list_alpha101()
        cats = sorted({f["category"] for f in factors})
        selected_cat = st.radio("因子类别", cats, horizontal=True, key="wq101_cat")
        kw = st.text_input("搜索（编号/名称/描述）", placeholder="如：007 或 量价",
                           key="wq101_kw").strip()
        in_cat = [f for f in factors if f["category"] == selected_cat]
        if kw:
            in_cat = [f for f in in_cat
                      if kw in f["key"] or kw in f["name"] or kw in f["description"]]
        labels = {f["key"]: f"#{f['key'][-3:]} {f['name']} —— {f['description']}"
                  for f in in_cat}
        key = st.selectbox("选择因子", list(labels.keys()),
                           format_func=lambda k: labels[k], key="wq101_sel")
        expr = alpha101_library.get_alpha101(key)["expr"]
        expr_str = dsdl.to_formula(expr)

    if st.button("🔬 开始体检", type="primary"):
        with st.spinner("体检中…"):
            if src.startswith("教科书"):
                st.session_state[f"diag_{key}"] = diagnose_classic_cached(key, pool, style)
            else:
                st.session_state[f"diag_{key}"] = diagnose_expr_cached(
                    json.dumps(expr, ensure_ascii=False), pool, style)

    diag = st.session_state.get(f"diag_{key}")
    if diag:
        st.markdown(f"**DSL 表达式**：`{expr_str}`")
        st.code(dsdl.render_tree(expr), language="text")
        st.divider()
        render_diagnosis(diag)


# ============================================================
# Tab 3：因子对比
# ============================================================
def render_compare():
    st.title("⚖️ 因子对比")
    st.markdown(
        "> 把多个因子（经典 + AI 生成）的体检指标放在同一张表里 PK。\n"
        "> 演示时：先生成一个 AI 因子，再来这里和经典因子对照——"
        "如果 AI 因子跑赢了经典因子，就是全场最有力的演示瞬间。"
    )

    # 收集所有可对比因子：经典因子（缓存体检）+ WQ101 移植（勾选）+ AI 因子（会话内生成）
    rows = []
    with st.spinner("体检全部经典因子（首次约 10 秒，之后秒开）…"):
        for f in factor_engine.list_factors():
            d = diagnose_classic_cached(f["key"], pool, style)
            rows.append({
                "因子": f["name"], "类型": "经典",
                "expr_str": json.dumps(factor_engine.get_factor(f["key"])["expr"],
                                       ensure_ascii=False),
                "IC均值": d["ic_summary"]["ic_mean"],
                "IR": d["ic_summary"]["ic_ir"],
                "多空年化": d["layers"]["spread_annual"],
                "单调性": d["layers"]["monotonic"],
                "换手率": d["turnover"],
                "评分": d["score"],
            })

    # —— WorldQuant 101 移植因子可选加入 PK（默认 3 个代表性因子）——
    wq_all = alpha101_library.list_alpha101()
    wq_default = ["alpha101_001", "alpha101_012", "alpha101_101"]
    wq_keys = st.multiselect(
        "额外加入 WorldQuant 101 移植因子（81 个可选，默认 3 个代表性）",
        [f["key"] for f in wq_all],
        default=wq_default,
        format_func=lambda k: f"#{k[-3:]} {alpha101_library.get_alpha101(k)['name']}"
                              f" —— {alpha101_library.get_alpha101(k)['description'][:26]}…",
    )
    with st.spinner("体检勾选的 WQ101 因子…"):
        for k in wq_keys:
            d = diagnose_expr_cached(
                json.dumps(alpha101_library.get_alpha101(k)["expr"], ensure_ascii=False),
                pool, style)
            rows.append({
                "因子": f"WQ#{k[-3:]}", "类型": "经典",
                "expr_str": json.dumps(alpha101_library.get_alpha101(k)["expr"],
                                       ensure_ascii=False),
                "IC均值": d["ic_summary"]["ic_mean"],
                "IR": d["ic_summary"]["ic_ir"],
                "多空年化": d["layers"]["spread_annual"],
                "单调性": d["layers"]["monotonic"],
                "换手率": d["turnover"],
                "评分": d["score"],
            })
    result = st.session_state.get("last_result")
    if result:
        d = result["diag"]
        rows.append({
            "因子": result["formula"][:30] + "…", "类型": "AI 生成",
            "expr_str": result["expr_str"],
            "IC均值": d["ic_summary"]["ic_mean"],
            "IR": d["ic_summary"]["ic_ir"],
            "多空年化": d["layers"]["spread_annual"],
            "单调性": d["layers"]["monotonic"],
            "换手率": d["turnover"],
            "评分": d["score"],
        })

    df = pd.DataFrame(rows).sort_values("评分", ascending=False).reset_index(drop=True)
    st.dataframe(df.style
                 .bar(subset=["评分"], color="#5c6bc0")
                 .format({"IC均值": "{:+.4f}", "IR": "{:.2f}",
                          "多空年化": "{:+.1%}", "单调性": "{:+.2f}", "换手率": "{:.1%}"}),
                 width="stretch", hide_index=True)

    # 评分条形图
    fig = go.Figure()
    colors = ["#ff7043" if t == "AI 生成" else "#5c6bc0" for t in df["类型"]]
    fig.add_trace(go.Bar(x=df["因子"], y=df["评分"], marker_color=colors,
                         text=df["评分"], textposition="outside"))
    fig.update_layout(title="因子综合评分 PK（橙色=AI 生成）", height=380,
                      yaxis_title="评分", yaxis=dict(range=[0, 105]),
                      margin=dict(l=40, r=20, t=50, b=80))
    st.plotly_chart(fig, width="stretch")

    # ============ 策略层 PK：体检只是门槛，真金白银才是结果 ============
    st.divider()
    st.subheader("🚀 策略层 PK：谁的组合真的赚得多？")
    st.markdown(
        "> 每个因子直接做成 **Top-30 周频组合**（每周调仓、双边成本 20bps、与沪深300指数对比）"
        "——体检评分是「因子该不该信」，这里是「实际能赚多少」。"
        "首次计算约 15 秒（11 个因子全量回测），之后秒开。"
    )
    with st.spinner("正在回测全部因子的组合策略…"):
        strat_rows = []
        for r in rows:
            s = build_strategy_cached(r["expr_str"], pool, style)
            m = s["metrics"]
            strat_rows.append({
                "因子": r["因子"], "类型": r["类型"],
                "年化": m["annual_return"],
                "超额(年化)": m.get("excess_annual"),
                "Sharpe": m.get("sharpe"),
                "最大回撤": m.get("max_drawdown"),
                "平均换手": m.get("avg_turnover"),
                "评分": r["评分"],
                "nav": s["nav"], "benchmark_nav": s["benchmark_nav"],
            })
    strat_df = pd.DataFrame(strat_rows).sort_values("年化", ascending=False).reset_index(drop=True)

    # 净值对比图：每因子一条线，AI 因子橙色加粗
    fig_pk = go.Figure()
    for row in strat_df.itertuples():
        if row.nav is None:
            continue
        is_ai = row.类型 == "AI 生成"
        fig_pk.add_trace(go.Scatter(
            x=row.nav.index, y=row.nav.values, name=row.因子,
            line=dict(width=3 if is_ai else 1.4, color="#ff7043" if is_ai else "#90a4ae"),
            opacity=1.0 if is_ai else 0.8))
    bench_ref = next((r.benchmark_nav for r in strat_df.itertuples()
                      if r.benchmark_nav is not None), None)
    if bench_ref is not None:
        fig_pk.add_trace(go.Scatter(x=bench_ref.index, y=bench_ref.values,
                                    name="沪深300指数", line=dict(width=2, dash="dot",
                                                                  color="#37474f")))
    fig_pk.update_layout(title="组合累计净值对比（起点=1，已扣成本；橙色=AI 因子，点击图例可开关）",
                         height=420, yaxis_title="净值",
                         legend=dict(x=1.0, xanchor="right", y=1.0, yanchor="top",
                                     bgcolor="rgba(255,255,255,0.55)", font=dict(size=10)),
                         margin=dict(l=40, r=20, t=50, b=30))
    st.plotly_chart(fig_pk, width="stretch")

    # 指标表：按年化排序
    show_cols = ["因子", "类型", "年化", "超额(年化)", "Sharpe", "最大回撤", "平均换手", "评分"]
    st.dataframe(strat_df[show_cols].style
                 .bar(subset=["年化"], color="#5c6bc0")
                 .highlight_max(subset=["年化", "超额(年化)", "Sharpe"], color="#e8f5e9")
                 .format({"年化": "{:+.1%}", "超额(年化)": "{:+.1%}",
                          "Sharpe": "{:.2f}", "最大回撤": "{:.1%}",
                          "平均换手": "{:.0%}", "评分": "{:.1f}"}),
                 width="stretch", hide_index=True)


# ============================================================
# Tab 4：策略构建（因子的最后一公里）
# ============================================================
def fig_strategy_nav(result: dict) -> go.Figure:
    """组合净值 vs 沪深300指数。"""
    nav = result["nav"]
    bench = result["benchmark_nav"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name="因子组合（Top30 周调仓）",
                             line=dict(width=2.5, color="#d32f2f")))
    if bench is not None:
        fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="沪深300指数",
                                 line=dict(width=2, dash="dot", color="#37474f")))
    fig.update_layout(title="组合净值 vs 沪深300（起点归一化=1，已扣交易成本）", height=380,
                      yaxis_title="净值", legend=dict(x=1.0, xanchor="right", y=1.0, yanchor="top",
                                   bgcolor="rgba(255,255,255,0.55)"),
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def fig_strategy_excess(result: dict) -> go.Figure:
    """超额净值（组合/基准）。"""
    nav, bench = result["nav"], result["benchmark_nav"]
    excess = nav / bench
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=excess.index, y=excess.values, name="超额净值",
                             fill="tozeroy", line=dict(width=2, color="#5c6bc0")))
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(title="超额净值（组合/基准，>1 表示跑赢指数）", height=260,
                      yaxis_title="超额", margin=dict(l=40, r=20, t=50, b=30))
    return fig


def render_strategy():
    st.title("🚀 策略构建：从因子到组合")
    st.markdown(
        "> 把体检通过的因子做成**可实盘的 Top-N 等权组合**：每周按因子值选股，"
        "扣交易成本，与沪深300指数对比。\n"
        "> **这是回答「因子能赚多少钱」的最后一公里**——也是赛道二评委最关心的部分。"
    )

    # —— 因子来源 ——
    source = st.radio("因子来源", ["经典因子", "AI 因子（会话内生成）"], horizontal=True)

    factor_panel = None
    if source == "经典因子":
        # 91 个内置因子：10 教科书 + 81 WorldQuant 101 移植（同一体检/回测管线）
        factors = factor_engine.list_factors() + alpha101_library.list_alpha101()
        labels = {f["key"]: f"{f['key']} {f['name']}（{f['formula'][:40]}…）"
                  for f in factors}
        key = st.selectbox("选择因子", list(labels.keys()), format_func=lambda k: labels[k])
        meta = (alpha101_library.get_alpha101(key) if key.startswith("alpha101_")
                else factor_engine.get_factor(key))
        factor_panel = neutralize_factor(
            dsdl.evaluate(meta["expr"], panel), panel, style)
        factor_desc = f"{key} {meta['name']}"
    else:
        last = st.session_state.get("last_result")
        if not last:
            st.info("先在「AI 因子工场」生成一个因子，再来这里构建策略")
            return
        factor_panel = neutralize_factor(
            dsdl.evaluate(dsdl.parse_factor(last["expr_str"]), panel), panel, style)
        factor_desc = last["formula"]

    # —— 参数 ——
    c1, c2, c3 = st.columns(3)
    n_stocks = c1.slider("持仓数量", 10, 100, 30, step=5)
    reb_label = c2.selectbox("调仓频率", ["周频（5日）", "双周（10日）", "月频（20日）"], index=0)
    rebalance = {"周频（5日）": 5, "双周（10日）": 10, "月频（20日）": 20}[reb_label]
    cost = c3.selectbox("双边交易成本", [10, 20, 40], index=1, format_func=lambda x: f"{x} bps")

    if st.button("📈 构建策略", type="primary"):
        with st.spinner("构建组合并回测中…"):
            index_close = load_index_cached()
            result = strategy.build_portfolio(
                factor_panel, panel["close"], index_close,
                n_stocks=n_stocks, rebalance_days=rebalance, cost_bps=cost,
            )
            st.session_state["strategy_result"] = result
            st.session_state["strategy_desc"] = factor_desc

    result = st.session_state.get("strategy_result")
    if result:
        m = result["metrics"]
        st.markdown(f"**当前因子**：`{st.session_state.get('strategy_desc', '')}`")
        # —— 指标卡片 ——
        bench = result["benchmark_nav"]
        index_annual = float(bench.pct_change().dropna().mean() * 252) if bench is not None else 0.0
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric("组合年化收益", f"{m['annual_return']:+.1%}")
        col_b.metric("沪深300同期年化", f"{index_annual:+.1%}")
        col_c.metric("Sharpe", f"{m['sharpe']:.2f}")
        col_d.metric("最大回撤", f"{m['max_drawdown']:.1%}")
        col_e.metric("超额年化", f"{m.get('excess_annual', 0):+.1%}")

        fig_nav = fig_strategy_nav(result)
        st.plotly_chart(fig_nav, width="stretch")
        col_f, col_g = st.columns(2)
        with col_f:
            if result["benchmark_nav"] is not None:
                st.plotly_chart(fig_strategy_excess(result), width="stretch")
        with col_g:
            # 指标明细
            st.markdown("**绩效明细**")
            rows = [
                ("信息比率", m.get("information_ratio", 0)),
                ("日胜率（vs 指数）", m.get("win_rate", 0)),
                ("回测天数", m["n_days"]),
                ("调仓频率", f"每 {result['rebalance_days']} 日"),
            ]
            for name, v in rows:
                if name == "回测天数":
                    st.markdown(f"- {name}：**{v}** 天")
                elif name == "调仓频率":
                    st.markdown(f"- {name}：**{v}**")
                else:
                    st.markdown(f"- {name}：**{v:.2f}**")


# ============================================================
# 主入口
# ============================================================
st.sidebar.title("🧪 因子实验室")
st.sidebar.markdown(
    "**AI 因子分析工作台**\n\n"
    "自然语言 → 受限因子表达式 → 机构级体检 → AI 解读\n\n"
    "— 北大金融AI智能体创新大赛 · 赛道二 —"
)

# —— 股票池 + 中性化设置（全局生效）——
pool_label = st.sidebar.radio(
    "股票池", ["沪深300（300 只，快）", "全 A（5000+ 只，首次加载慢）"], key="pool_sel")
pool = "hs300" if pool_label.startswith("沪深300") else "ashare"
style = st.sidebar.selectbox(
    "因子中性化",
    ["无", "行业", "行业+市值"],
    help="把因子值对行业/市值回归取残差，剥离『选股其实在选行业/大小盘』的成分。"
         "行业映射表需全 A 数据下载完成后才可用。",
)
style_map = {"无": "none", "行业": "industry", "行业+市值": "industry+size"}

panel = load_panel_cached(pool)

# —— 首屏 hero：30 秒讲清"这是什么"——
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1e3a8a 0%,#2563eb 55%,#3b82f6 100%);
                border-radius:16px; padding:28px 32px; margin-bottom:8px;">
      <div style="font-size:30px; font-weight:700; color:#ffffff; letter-spacing:1px;">
        🧪 因子实验室 <span style="font-size:18px; font-weight:400; opacity:.85;">AI Factor Lab</span>
      </div>
      <div style="font-size:16px; color:#dbeafe; margin-top:6px;">
        用一句话描述因子想法 → AI 生成<b>受限表达式</b> → 机构级体检 → 可实盘策略
      </div>
      <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;">
        <span style="background:rgba(255,255,255,.18); color:#fff; border-radius:20px;
                     padding:4px 14px; font-size:13px;">🔒 38 个白名单算子 · 不执行任意代码</span>
        <span style="background:rgba(255,255,255,.18); color:#fff; border-radius:20px;
                     padding:4px 14px; font-size:13px;">📊 IC / 分层 / 换手 / 衰减 · 全套体检</span>
        <span style="background:rgba(255,255,255,.18); color:#fff; border-radius:20px;
                     padding:4px 14px; font-size:13px;">💰 Top30 周频 · 扣双边成本 · 对沪深300</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.sidebar:
    st.markdown("#### 数据状态")
    st.markdown(
        f"- 股票池：{'沪深300' if pool == 'hs300' else '全 A'}（{panel['close'].shape[1]} 只）\n"
        f"- 区间：{panel['close'].index[0]} ~ {panel['close'].index[-1]}\n"
        f"- 交易日：{panel['close'].shape[0]} 天\n"
        f"- 中性化：{'无' if style == 'none' else style}"
    )
    st.caption("数据源：baostock 日线（前复权）")

tab_ai, tab_classic, tab_compare, tab_strategy = st.tabs(
    ["🧪 AI 因子工场", "📚 经典因子库", "⚖️ 因子对比", "🚀 策略构建"]
)
with tab_ai:
    render_ai_factory()
with tab_classic:
    render_classic_library()
with tab_compare:
    render_compare()
with tab_strategy:
    render_strategy()

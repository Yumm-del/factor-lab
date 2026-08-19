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

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 包导入（app.py 在项目根目录，factor_lab 是同级包）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_lab import data_pipeline, dsdl, factor_engine, llm_factor, validation  # noqa: E402

st.set_page_config(page_title="因子实验室 · AI Factor Lab", layout="wide", page_icon="🧪")


# ============================================================
# 数据加载（缓存：数据文件不变就不重新加载）
# ============================================================
@st.cache_data(show_spinner="正在加载沪深300数据面板…")
def load_panel_cached():
    return data_pipeline.load_panel()


@st.cache_data(show_spinner="体检中…")
def diagnose_classic_cached(key: str):
    """经典因子体检（缓存：同一因子只算一次，对比页重进秒开）。"""
    panel = data_pipeline.load_panel()
    expr = factor_engine.get_factor(key)["expr"]
    fac = dsdl.evaluate(expr, panel)
    return validation.full_diagnosis(fac, panel["close"])


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
                      yaxis_title="IC", xaxis_title="", legend=dict(orientation="h", y=1.1),
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
                      yaxis_title="净值（起点=1）", legend=dict(orientation="h", y=1.1),
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

    # —— 结论横幅 ——
    color_map = {"优秀": "🟢", "可用": "🟠", "淘汰": "🔴"}
    st.markdown(
        f"### {color_map[verdict['label']]} 体检结论：**{verdict['label']}**　评分 **{diag['score']}/100**\n\n"
        f"> {verdict['text']}"
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


# ============================================================
# Tab 1：AI 因子工场（核心卖点）
# ============================================================
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

            st.session_state["last_result"] = result
        except Exception as e:  # noqa: BLE001
            st.error(f"生成失败：{e}")

    result = st.session_state.get("last_result")
    if result:
        st.divider()
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
        "> 内置 10 个 A 股实证研究中最稳健的经典因子（动量/反转/波动/流动性/价值/趋势），"
        "全部用同一套 DSL 表达式表示——与 AI 因子走完全相同的体检流水线。\n"
        "> **这既是基线对照，也是可信度证明**：工作台对「已知有效的因子」能给出正确判定。"
    )

    factors = factor_engine.list_factors()
    cat_map = {f["key"]: f for f in factors}
    cats = ["动量", "反转", "波动", "流动性", "价值", "趋势"]
    selected_cat = st.radio("因子类别", cats, horizontal=True)
    in_cat = [f for f in factors if f["category"] == selected_cat]
    labels = {f["key"]: f"{f['name']} —— {f['description']}" for f in in_cat}
    key = st.selectbox("选择因子", list(labels.keys()),
                       format_func=lambda k: labels[k])

    meta = cat_map[key]
    expr = factor_engine.get_factor(key)["expr"]
    expr_str = dsdl.to_formula(expr)

    if st.button("🔬 开始体检", type="primary"):
        with st.spinner("体检中…"):
            st.session_state[f"diag_{key}"] = diagnose_classic_cached(key)

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

    # 收集所有可对比因子：经典因子（缓存体检）+ AI 因子（会话内生成过的）
    rows = []
    with st.spinner("体检全部经典因子（首次约 10 秒，之后秒开）…"):
        for f in factor_engine.list_factors():
            d = diagnose_classic_cached(f["key"])
            rows.append({
                "因子": f["name"], "类型": "经典",
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


# ============================================================
# 主入口
# ============================================================
st.sidebar.title("🧪 因子实验室")
st.sidebar.markdown(
    "**AI 因子分析工作台**\n\n"
    "自然语言 → 受限因子表达式 → 机构级体检 → AI 解读\n\n"
    "— 北大金融AI智能体创新大赛 · 赛道二 —"
)

panel = load_panel_cached()
with st.sidebar:
    st.markdown("#### 数据状态")
    st.markdown(
        f"- 股票池：沪深300（{panel['close'].shape[1]} 只）\n"
        f"- 区间：{panel['close'].index[0]} ~ {panel['close'].index[-1]}\n"
        f"- 交易日：{panel['close'].shape[0]} 天"
    )
    st.caption("数据源：baostock 日线（前复权）")

tab_ai, tab_classic, tab_compare = st.tabs(["🧪 AI 因子工场", "📚 经典因子库", "⚖️ 因子对比"])
with tab_ai:
    render_ai_factory()
with tab_classic:
    render_classic_library()
with tab_compare:
    render_compare()

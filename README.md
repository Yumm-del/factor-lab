# 🧪 因子实验室 · AI Factor Lab

**让 AI 帮你挖 A 股因子：一句话描述想法 → 受限因子表达式 → 机构级体检 → AI 解读报告**

> 北京大学金融 AI 智能体创新大赛 · 赛道二（量化投资策略与金融科技工具研发）
> 用真实沪深300数据（baostock）驱动的 AI 因子分析工作台。

---

## 为什么做这个（核心叙事）

多因子选股是量化投资的基本功，但传统因子挖掘有两个致命痛点：

1. **门槛高**：写因子需要编程 + 金融知识，研究员把大量时间花在调试代码而不是思考逻辑
2. **不可信**：新手（和 AI）挖出的因子经常是"回测好看、实盘失效"——因为缺少机构级的统计检验（IC 是否显著？分层是否单调？换手是否吃掉收益？信号衰减多快？）

**因子实验室**把这两件事同时解决：

```
自然语言想法              受限 DSL 表达式               机构级体检                AI 解读
"放量后延续上涨" ──LLM──▶ rank(ts_mean(volume,5)/     ──自动──▶ IC / RankIC      ──LLM──▶ 大白话报告
                           ts_mean(volume,20))                   分层回测            + 改进建议
                           ↓ 白名单算子 + 深度限制                 换手率
                           （不能执行任意代码）                    IC衰减曲线
                                                                  综合评分
```

三个差异化设计：

- **安全受限的因子 DSL**：LLM 只能从白名单算子（时序/截面/组合共 18 个）组合出 JSON 表达式树，**不执行任意代码**——这是"AI 挖因子"能落地的安全地基，也是区别于"让 AI 写代码"类玩具的核心架构
- **一切因子同一套体检**：经典因子与 AI 因子走完全相同的验证流水线（IC/RankIC、5 层分层回测、换手率、IC 衰减、综合评分）——经典因子是"已知答案的对照组"，工作台能正确判定它们，AI 因子就值得信
- **可视化表达式树**：每个因子都可以渲染成树状结构图，评委一眼看懂"AI 在想什么"，全流程可解释

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 DeepSeek API key（因子生成/解读用）
cp .env.example .env   # 填入 DEEPSEEK_API_KEY

# 3. 下载沪深300数据（约 300 只 × 1 年日线，几分钟）
python scripts/build_data.py

# 4. 启动工作台
streamlit run app.py
```

打开 http://localhost:8501，在「AI 因子工场」输入想法，例如：

> 「放量后价格延续上涨（量价配合）」

等待约 20 秒，你会看到：AI 生成的因子公式、表达式树、全套体检图表、以及 AI 撰写的体检报告。

## 目录结构

```
factor-lab/
├── app.py                    # Streamlit 工作台（演示主界面）
├── factor_lab/
│   ├── data_pipeline.py      # baostock 沪深300 数据管道（长表→面板）
│   ├── dsdl.py               # 因子 DSL：算子白名单 + 解析校验 + 向量化求值
│   ├── factor_engine.py      # 10 个经典因子族（全部用 DSL 表达式表示）
│   ├── validation.py         # 因子体检：IC/分层/换手/衰减/综合评分
│   └── llm_factor.py         # LLM 因子生成 + 体检报告解读（DeepSeek）
├── scripts/build_data.py     # 数据构建脚本
├── requirements.txt
└── .env.example              # API key 模板
```

## 技术细节

- **数据**：baostock 沪深300 成分股日线（前复权），含 close/high/low/volume/amount/turn/pe/pb，全部真实数据
- **DSL 算子**（白名单）：`ts_returns/ts_mean/ts_std/ts_zscore/ts_rank/ts_max/ts_min/ts_corr/delay`（时序）+ `rank/normalize`（截面）+ `add/sub/mul/div/signed_power/log/abs/neg`（组合）
- **体检指标**：IC 均值/IR/t 值、RankIC、5 层分层回测（含单调性）、多空年化、组合换手率、IC 衰减曲线（未来 1~10 日）
- **综合评分**：IC 强度（30 分）＋ 信息比率（25 分）＋ 多空年化（30 分）＋ 层间单调性（15 分），用 tanh 压缩防止过拟合因子刷分
- **LLM**：DeepSeek API（`deepseek-v4-pro`），输出受 JSON Schema 校验 + 深度/节点数限制，非法输出自动重试

## 演示脚本（5 分钟版）

1. **开场 30s**：展示数据面板（300 只 × 250 天真实数据）与经典因子库
2. **1 分钟**：经典因子体检——选「低波动」，展示 IC 0.03、分层单调、多空曲线（证明工作台可信）
3. **2 分钟**：AI 因子工场——输入「放量延续上涨」，AI 生成表达式 + 树状图 + 体检 + 报告
4. **1 分钟**：因子对比页——AI 因子 vs 10 个经典因子评分 PK
5. **30s**：讲安全设计（白名单 DSL，不执行任意代码）与可解释性（表达式树）

## 参赛信息

- 赛道：量化投资策略与金融科技工具研发
- 创新点：LLM 受限因子生成 × 机构级自动验证 × 全链路可解释
- 技术栈：Python 3.14 / pandas / baostock / Streamlit / plotly / DeepSeek API

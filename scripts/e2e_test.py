"""
端到端测试 ———— 合成面板 + 真实 DeepSeek API
验证：自然语言 → LLM 生成因子 → DSL 校验 → 求值 → 体检 → AI 解读 全链路
用法：python scripts/e2e_test.py "你的因子想法"
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_lab import llm_factor, validation  # noqa: E402


def synthetic_panel(n_days=260, n_stocks=100):
    """合成面板：带一个隐藏动量结构的假数据（真实管线验证用）。"""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2025-06-02", periods=n_days, freq="B")
    codes = [f"sh.60{i:04d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, codes])
    drift = rng.normal(0.0003, 0.02, (len(dates), len(codes)))
    close = pd.DataFrame(10 * np.exp(np.cumsum(drift, axis=0)), index=dates, columns=codes)
    return {
        "close": close,
        "volume": pd.DataFrame(rng.lognormal(15, 0.4, (n_days, n_stocks)), index=dates, columns=codes),
        "amount": pd.DataFrame(rng.lognormal(8, 0.4, (n_days, n_stocks)), index=dates, columns=codes),
        "turn": pd.DataFrame(rng.lognormal(0.4, 0.5, (n_days, n_stocks)), index=dates, columns=codes),
        "pe": pd.DataFrame(rng.lognormal(2.6, 0.35, (n_days, n_stocks)), index=dates, columns=codes),
        "pb": pd.DataFrame(rng.lognormal(1.1, 0.3, (n_days, n_stocks)), index=dates, columns=codes),
    }


if __name__ == "__main__":
    idea = sys.argv[1] if len(sys.argv) > 1 else "放量后价格延续上涨（量价配合）"
    print(f"🧪 端到端测试 | 想法：{idea}\n" + "=" * 50)
    panel = synthetic_panel()
    result = llm_factor.generate_factor(idea, panel, panel["close"])

    print("✅ 因子生成成功")
    print(f"📝 公式: {result['formula']}")
    print(f"💡 逻辑: {result['rationale']}")
    print("🌲 表达式树:")
    print(result["tree"])
    s = result["diag"]["ic_summary"]
    print(f"\n📊 体检: IC {s['ic_mean']:+.4f} | IR {s['ic_ir']:.2f} | 评分 {result['diag']['score']}/100")
    print(f"📋 AI 解读:\n{result['report']}")

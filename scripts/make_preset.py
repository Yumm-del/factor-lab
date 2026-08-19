"""
生成预置示例 ———— 把 LLM 生成结果固化为离线示例（演示不依赖 API）
用法：python scripts/make_preset.py
输出：data/presets.json（formula/rationale/tree/report/expr_str，diag 加载时现算）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_lab import data_pipeline, llm_factor  # noqa: E402

PRESET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "presets.json")

IDEAS = [
    "低波动率的股票未来表现更好",
    "放量后价格延续上涨（量价配合）",
]

if __name__ == "__main__":
    panel = data_pipeline.load_panel()
    presets = []
    for idea in IDEAS:
        print(f"🔄 生成预置: {idea}")
        r = llm_factor.generate_factor(idea, panel, panel["close"])
        presets.append({
            "idea": idea,
            "formula": r["formula"],
            "rationale": r["rationale"],
            "tree": r["tree"],
            "report": r["report"],
            "expr_str": r["expr_str"],
        })
        print(f"  ✅ {r['formula']}")
    with open(PRESET_PATH, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存 {len(presets)} 个预置: {PRESET_PATH}")

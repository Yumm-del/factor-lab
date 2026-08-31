"""Agent 反思闭环消融实验：单轮生成 vs 体检反馈后的完整 Agent。

目的：
    用同一次首轮生成作为共同起点，比较“到此停止”的单轮 LLM 与
    “读取真实体检结果后继续反思”的 Agent。这样不会把两次随机采样
    的差异误写成反思模块的收益。

输出：
    docs/evidence/agent_ablation.json  原始逐轮记录，可审计、可续跑
    docs/evidence/agent_ablation.md    评委可直接阅读的汇总证据

用法：
    python scripts/agent_ablation.py
    python scripts/agent_ablation.py --limit 1  # 先做一条链路冒烟测试
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_lab import data_pipeline, dsdl, llm_factor, validation  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "docs" / "evidence" / "agent_ablation.json"
OUTPUT_MD = ROOT / "docs" / "evidence" / "agent_ablation.md"

# 预先固定的五类常见研究意图，覆盖趋势、反转、量价、风险和基本面。
# 不根据实验结果替换题目，避免挑选“刚好有效”的成功案例。
IDEAS = [
    "寻找中期趋势延续但短期不过热的股票",
    "捕捉短期超跌后的均值回归，同时避免接住持续下跌的股票",
    "成交量温和放大且价格开始突破的股票可能延续上涨",
    "偏好低波动、走势稳定的股票，并控制信号换手",
    "寻找估值较低但基本面不过度异常的股票",
]


def file_sha256(path: Path) -> str:
    """返回数据文件 SHA-256，证明复现实验使用的是同一份输入。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_result(result: dict, panel: dict) -> tuple[dict, str, str]:
    """校验 LLM JSON、计算因子并返回体检单、公式和表达式树 JSON。"""
    expr_dict = result.get("expr")
    if not isinstance(expr_dict, dict):
        raise RuntimeError("LLM 输出缺少 expr 对象")
    expr = dsdl.parse_factor(json.dumps(expr_dict, ensure_ascii=False))
    formula = dsdl.to_formula(expr)
    factor_panel = dsdl.evaluate(expr, panel)
    diag = validation.full_diagnosis(factor_panel, panel["close"])
    return diag, formula, json.dumps(expr_dict, ensure_ascii=False)


def summarize(trials: list[dict]) -> dict:
    """汇总成功试验；失败样本保留在分母信息中，不静默丢弃。"""
    successful = [trial for trial in trials if trial.get("status") == "ok"]
    if not successful:
        return {
            "planned_trials": len(IDEAS),
            "completed_trials": len(trials),
            "successful_trials": 0,
            "failed_trials": len(trials),
        }

    initial_scores = [trial["rounds"][0]["score"] for trial in successful]
    final_scores = [trial["rounds"][-1]["score"] for trial in successful]
    threshold = llm_factor.REFLECT_THRESHOLD
    return {
        "planned_trials": len(IDEAS),
        "completed_trials": len(trials),
        "successful_trials": len(successful),
        "failed_trials": len(trials) - len(successful),
        "single_shot_pass_rate": sum(v >= threshold for v in initial_scores) / len(successful),
        "full_agent_pass_rate": sum(v >= threshold for v in final_scores) / len(successful),
        "mean_initial_score": sum(initial_scores) / len(successful),
        "mean_final_score": sum(final_scores) / len(successful),
        "mean_score_delta": sum(b - a for a, b in zip(initial_scores, final_scores)) / len(successful),
        "improved_trials": sum(b > a for a, b in zip(initial_scores, final_scores)),
        "degraded_trials": sum(b < a for a, b in zip(initial_scores, final_scores)),
        "unchanged_trials": sum(b == a for a, b in zip(initial_scores, final_scores)),
        "total_llm_generations": sum(len(trial["rounds"]) for trial in successful),
    }


def write_outputs(payload: dict) -> None:
    """原子性要求不高，但每轮都落盘，API 中断后可以从下一题续跑。"""
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = payload["summary"]
    lines = [
        "# Agent 反思闭环消融实验",
        "",
        "> 口径：同一次首轮生成同时作为 single-shot 基线；仅当评分低于 50 时，",
        "> 完整 Agent 才读取体检反馈并最多反思 2 轮。固定 5 个意图，不按结果换题。",
        "",
        "## 汇总",
        "",
        f"- 计划 / 已完成 / 成功：{summary['planned_trials']} / "
        f"{summary['completed_trials']} / {summary['successful_trials']}",
    ]
    if summary["successful_trials"]:
        lines.extend([
            f"- single-shot 达标率：{summary['single_shot_pass_rate']:.1%}",
            f"- 完整 Agent 达标率：{summary['full_agent_pass_rate']:.1%}",
            f"- 平均评分：{summary['mean_initial_score']:.2f} → "
            f"{summary['mean_final_score']:.2f}（Δ {summary['mean_score_delta']:+.2f}）",
            f"- 改善 / 下降 / 不变：{summary['improved_trials']} / "
            f"{summary['degraded_trials']} / {summary['unchanged_trials']}",
            f"- LLM 生成总次数：{summary['total_llm_generations']}",
            "",
            "## 逐题记录",
            "",
            "| 意图 | 首轮评分 | 最终评分 | 轮数 | 结果 |",
            "|---|---:|---:|---:|---|",
        ])
        for trial in payload["trials"]:
            if trial.get("status") != "ok":
                continue
            rounds = trial["rounds"]
            lines.append(
                f"| {trial['idea']} | {rounds[0]['score']:.2f} | "
                f"{rounds[-1]['score']:.2f} | {len(rounds)} | "
                f"{rounds[-1]['verdict']} |"
            )
    lines.extend([
        "",
        "## 边界",
        "",
        "这是固定小样本的工程消融，不等于统计显著的投资结论；它用于回答反思模块",
        "是否在相同起点上带来可观测变化。所有逐轮公式、评分与错误均保存在 JSON 中。",
        "",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 Agent 反思闭环消融实验")
    parser.add_argument("--limit", type=int, default=len(IDEAS), help="本次最多运行几道固定题")
    args = parser.parse_args()

    existing = {trial["idea"]: trial for trial in []}
    if OUTPUT_JSON.exists():
        prior = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        existing = {trial["idea"]: trial for trial in prior.get("trials", [])}

    panel = data_pipeline.load_panel("hs300")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": llm_factor.MODEL,
        "threshold": llm_factor.REFLECT_THRESHOLD,
        "max_reflect_rounds": llm_factor.MAX_REFLECT_ROUNDS,
        "dedup_enabled": False,
        "interpretation_call_enabled": False,
        "dataset": str(Path(data_pipeline.RAW_PATH).relative_to(ROOT)),
        "dataset_sha256": file_sha256(Path(data_pipeline.RAW_PATH)),
        "n_dates": int(panel["close"].shape[0]),
        "n_stocks": int(panel["close"].shape[1]),
    }

    selected = IDEAS[: max(0, min(args.limit, len(IDEAS)))]
    for index, idea in enumerate(selected, start=1):
        if existing.get(idea, {}).get("status") == "ok":
            print(f"[{index}/{len(selected)}] 已有成功记录，跳过：{idea}")
            continue

        print(f"[{index}/{len(selected)}] {idea}")
        trial = {"idea": idea, "status": "running", "rounds": []}
        existing[idea] = trial
        try:
            system, user = llm_factor.build_generation_prompt(idea)
            result = llm_factor._llm_json(system, user)
            for round_index in range(1, llm_factor.MAX_REFLECT_ROUNDS + 2):
                diag, formula, expr_json = evaluate_result(result, panel)
                rationale = result.get("rationale", "")
                trial["rounds"].append({
                    "round": round_index,
                    "formula": formula,
                    "expr": json.loads(expr_json),
                    "rationale": rationale,
                    "score": float(diag["score"]),
                    "verdict": diag["verdict"]["label"],
                    "ic_mean": float(diag["ic_summary"]["ic_mean"]),
                    "ic_ir": float(diag["ic_summary"]["ic_ir"]),
                    "turnover": float(diag["turnover"]),
                })
                print(f"  round {round_index}: {diag['score']:.2f} / {diag['verdict']['label']}")
                if diag["score"] >= llm_factor.REFLECT_THRESHOLD or round_index > llm_factor.MAX_REFLECT_ROUNDS:
                    break
                result = llm_factor._llm_json(
                    llm_factor.REFLECT_SYSTEM,
                    llm_factor.build_reflect_prompt(idea, formula, rationale, diag),
                )
            trial["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - 失败必须入证据，不能静默删样本
            trial["status"] = "error"
            trial["error_type"] = type(exc).__name__
            trial["error"] = str(exc)
            print(f"  ERROR: {type(exc).__name__}: {exc}")

        ordered = [existing[item] for item in IDEAS if item in existing]
        payload = {"metadata": metadata, "trials": ordered, "summary": summarize(ordered)}
        write_outputs(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

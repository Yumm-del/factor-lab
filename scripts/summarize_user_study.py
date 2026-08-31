"""汇总匿名用户测试 CSV，拒绝在样本不足时生成夸大的落地结论。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "evidence" / "user_study.csv"
DEFAULT_JSON = ROOT / "docs" / "evidence" / "user_study_summary.json"
DEFAULT_MD = ROOT / "docs" / "evidence" / "user_study_summary.md"
MIN_SAMPLE = 5

REQUIRED_COLUMNS = {
    "participant_id",
    "experience_level",
    "duration_minutes",
    "completed",
    "decision_correct",
    "false_positive_avoided",
    "help_count",
    "ease_score",
    "trust_score",
    "comment",
}


def _parse_binary(row: dict[str, str], key: str) -> int:
    value = row[key].strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{key} 必须是 0 或 1，实际为 {value!r}")
    return int(value)


def _parse_float(
    row: dict[str, str], key: str, minimum: float, maximum: float
) -> float:
    value = float(row[key])
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} 必须在 {minimum}~{maximum}，实际为 {value}")
    return value


def load_rows(path: Path) -> list[dict]:
    """读取并严格校验用户测试记录。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"CSV 缺少字段: {', '.join(sorted(missing))}")

        rows = []
        seen_ids = set()
        for line_number, raw in enumerate(reader, start=2):
            participant_id = raw["participant_id"].strip()
            if not participant_id:
                raise ValueError(f"第 {line_number} 行 participant_id 为空")
            if participant_id in seen_ids:
                raise ValueError(f"participant_id 重复: {participant_id}")
            seen_ids.add(participant_id)

            rows.append(
                {
                    "participant_id": participant_id,
                    "experience_level": raw["experience_level"].strip(),
                    "duration_minutes": _parse_float(
                        raw, "duration_minutes", 0.1, 120
                    ),
                    "completed": _parse_binary(raw, "completed"),
                    "decision_correct": _parse_binary(raw, "decision_correct"),
                    "false_positive_avoided": _parse_binary(
                        raw, "false_positive_avoided"
                    ),
                    "help_count": int(
                        _parse_float(raw, "help_count", 0, 100)
                    ),
                    "ease_score": _parse_float(raw, "ease_score", 1, 5),
                    "trust_score": _parse_float(raw, "trust_score", 1, 5),
                    "comment": raw["comment"].strip(),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    """计算项目书可直接引用的聚合指标。"""

    count = len(rows)
    enough_evidence = count >= MIN_SAMPLE
    if not rows:
        metrics = {}
    else:
        metrics = {
            "completion_rate": statistics.mean(row["completed"] for row in rows),
            "correct_decision_rate": statistics.mean(
                row["decision_correct"] for row in rows
            ),
            "false_positive_avoidance_rate": statistics.mean(
                row["false_positive_avoided"] for row in rows
            ),
            "median_duration_minutes": statistics.median(
                row["duration_minutes"] for row in rows
            ),
            "mean_help_count": statistics.mean(row["help_count"] for row in rows),
            "mean_ease_score": statistics.mean(row["ease_score"] for row in rows),
            "mean_trust_score": statistics.mean(row["trust_score"] for row in rows),
        }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": count,
        "minimum_sample": MIN_SAMPLE,
        "evidence_status": "sufficient" if enough_evidence else "insufficient",
        "metrics": metrics,
        "anonymous_comments": [row["comment"] for row in rows if row["comment"]],
    }


def write_outputs(summary: dict, json_path: Path, md_path: Path) -> None:
    """保存机器可读结果与项目书证据页。"""

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    status = "证据充分" if summary["evidence_status"] == "sufficient" else "证据不足"
    rows = [
        "# 真实用户测试汇总",
        "",
        f"- 有效样本：{summary['sample_size']} 人",
        f"- 最低样本要求：{summary['minimum_sample']} 人",
        f"- 当前状态：**{status}**",
        "",
    ]
    metrics = summary["metrics"]
    if metrics:
        rows.extend(
            [
                "| 指标 | 结果 |",
                "|---|---:|",
                f"| 任务完成率 | {metrics['completion_rate']:.1%} |",
                f"| 正确判断率 | {metrics['correct_decision_rate']:.1%} |",
                f"| 避免假阳性比例 | {metrics['false_positive_avoidance_rate']:.1%} |",
                f"| 完成时长中位数 | {metrics['median_duration_minutes']:.1f} 分钟 |",
                f"| 平均求助次数 | {metrics['mean_help_count']:.2f} |",
                f"| 易用性均分 | {metrics['mean_ease_score']:.2f}/5 |",
                f"| 可解释可信度均分 | {metrics['mean_trust_score']:.2f}/5 |",
                "",
            ]
        )
    if summary["evidence_status"] != "sufficient":
        rows.append(
            "当前样本不足，不得在项目书中表述为已经完成真实用户验证。"
        )
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"尚无真实用户记录: {args.input}")
        print("先复制 user_study_template.csv，再按方案逐人记录。")
        return 2

    summary = summarize(load_rows(args.input))
    write_outputs(summary, args.json, args.markdown)
    print(
        f"User study: n={summary['sample_size']}, "
        f"status={summary['evidence_status']}"
    )
    return 0 if summary["evidence_status"] == "sufficient" else 1


if __name__ == "__main__":
    raise SystemExit(main())

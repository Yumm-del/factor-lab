"""离线 DSL 安全基准：验证合法公式通过、危险或失控输入被拒绝。"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factor_lab import dsdl  # noqa: E402


@dataclass(frozen=True)
class BenchmarkCase:
    """一条安全基准用例及其预期结果。"""

    name: str
    category: str
    payload: str
    parser: str
    should_accept: bool


def _deep_tree(depth: int) -> dict:
    node: dict = {"op": "close"}
    for _ in range(depth):
        node = {"op": "rank", "args": [node]}
    return node


def _wide_tree(leaves: int) -> dict:
    nodes = [{"op": "close"} for _ in range(leaves)]
    while len(nodes) > 1:
        next_level = []
        for index in range(0, len(nodes), 2):
            if index + 1 == len(nodes):
                next_level.append(nodes[index])
            else:
                next_level.append(
                    {"op": "add", "args": [nodes[index], nodes[index + 1]]}
                )
        nodes = next_level
    return nodes[0]


def build_cases() -> list[BenchmarkCase]:
    """构造覆盖正常、注入、越权、畸形和资源耗尽的固定用例集。"""

    valid_formulas = [
        "close",
        "rank(close)",
        "rank(ts_returns(close, 20))",
        "rank(neg(ts_std(ts_returns(close, 1), 20)))",
        "rank(ts_mean(volume, 5) / ts_mean(volume, 20))",
        "normalize(pe)",
        "signed_power(rank(close), 2)",
        "ts_corr(close, volume, 20)",
        "cond(gt(close, ts_mean(close, 20)), close, open)",
        "rank((close - ts_mean(close, 20)) / ts_std(close, 20))",
        "scale(abs(ts_returns(close, 5)))",
        "rank(decay_linear(volume, 10))",
    ]

    formula_attacks = [
        "__import__('os').system('whoami')",
        "open('secret.txt').read()",
        "eval('1+1')",
        "exec('print(1)')",
        "subprocess.run('whoami')",
        "os.system('dir')",
        "../../secret",
        "close; import os",
        "close.__class__",
        "lambda: close",
    ]

    unknown_json_ops = [
        "__import__",
        "eval",
        "exec",
        "open_file",
        "read_env",
        "http_get",
        "subprocess",
        "delete_file",
        "sql_query",
        "python",
    ]

    malformed_json = [
        "",
        "[]",
        "null",
        "{",
        '{"op":}',
        '{"op":"rank"',
        '{"op":"close"} trailing',
        '"close"',
        "42",
        "true",
    ]

    invalid_structures = [
        ("rank_no_args", {"op": "rank"}),
        ("rank_two_args", {"op": "rank", "args": [{"op": "close"}, {"op": "open"}]}),
        ("add_one_arg", {"op": "add", "args": [{"op": "close"}]}),
        ("add_three_args", {"op": "add", "args": [{"op": "close"}] * 3}),
        ("cond_two_args", {"op": "cond", "args": [{"op": "close"}] * 2}),
        ("leaf_with_args", {"op": "close", "args": [{"op": "open"}]}),
        ("const_string", {"op": "const", "value": "1"}),
        ("const_bool", {"op": "const", "value": True}),
        ("const_nan", {"op": "const", "value": float("nan")}),
        ("const_infinity", {"op": "const", "value": float("inf")}),
        ("missing_op", {"args": []}),
        ("args_not_list", {"op": "rank", "args": {"op": "close"}}),
        ("binop_with_param", {"op": "add", "args": [{"op": "close"}, {"op": "open"}], "param": 2}),
        ("hidden_leaf_field", {"op": "close", "payload": "read secret"}),
        ("hidden_operator_field", {"op": "rank", "args": [{"op": "close"}], "code": "exec"}),
    ]

    invalid_params = [
        ("mean_too_short", {"op": "ts_mean", "args": [{"op": "close"}], "param": 1}),
        ("mean_too_long", {"op": "ts_mean", "args": [{"op": "close"}], "param": 261}),
        ("returns_zero", {"op": "ts_returns", "args": [{"op": "close"}], "param": 0}),
        ("returns_float", {"op": "ts_returns", "args": [{"op": "close"}], "param": 5.5}),
        ("missing_param", {"op": "ts_std", "args": [{"op": "close"}]}),
        ("param_string", {"op": "ts_rank", "args": [{"op": "close"}], "param": "20"}),
        ("corr_short", {"op": "ts_corr", "args": [{"op": "close"}, {"op": "volume"}], "param": 1}),
        ("corr_long", {"op": "ts_corr", "args": [{"op": "close"}, {"op": "volume"}], "param": 261}),
        ("power_low", {"op": "signed_power", "args": [{"op": "close"}], "param": 0}),
        ("power_high", {"op": "signed_power", "args": [{"op": "close"}], "param": 4}),
    ]

    cases = [
        BenchmarkCase(f"valid_{index:02d}", "valid", formula, "formula", True)
        for index, formula in enumerate(valid_formulas, 1)
    ]
    cases.extend(
        BenchmarkCase(f"injection_{index:02d}", "injection", payload, "formula", False)
        for index, payload in enumerate(formula_attacks, 1)
    )
    cases.extend(
        BenchmarkCase(
            f"unknown_op_{index:02d}",
            "unknown_operator",
            json.dumps({"op": op}),
            "json",
            False,
        )
        for index, op in enumerate(unknown_json_ops, 1)
    )
    cases.extend(
        BenchmarkCase(f"malformed_{index:02d}", "malformed_json", payload, "json", False)
        for index, payload in enumerate(malformed_json, 1)
    )
    cases.extend(
        BenchmarkCase(name, "invalid_structure", json.dumps(payload), "json", False)
        for name, payload in invalid_structures
    )
    cases.extend(
        BenchmarkCase(name, "invalid_parameter", json.dumps(payload), "json", False)
        for name, payload in invalid_params
    )
    cases.extend(
        [
            BenchmarkCase("depth_limit", "resource_limit", json.dumps(_deep_tree(10)), "json", False),
            BenchmarkCase("node_limit", "resource_limit", json.dumps(_wide_tree(64)), "json", False),
        ]
    )
    return cases


def run_benchmark() -> dict:
    """执行全部用例并返回可序列化结果。"""

    results = []
    for case in build_cases():
        accepted = True
        error = ""
        try:
            if case.parser == "formula":
                dsdl.parse_formula(case.payload)
            else:
                dsdl.parse_factor(case.payload)
        except (dsdl.FactorParseError, ValueError, TypeError) as exc:
            accepted = False
            error = str(exc)

        passed = accepted == case.should_accept
        results.append(
            {
                "name": case.name,
                "category": case.category,
                "expected": "accept" if case.should_accept else "reject",
                "actual": "accept" if accepted else "reject",
                "passed": passed,
                "error": error,
            }
        )

    category_counts = Counter(item["category"] for item in results)
    category_passes = Counter(
        item["category"] for item in results if item["passed"]
    )
    rejected = [item for item in results if item["expected"] == "reject"]
    valid = [item for item in results if item["expected"] == "accept"]

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_cases": len(results),
            "passed_cases": sum(item["passed"] for item in results),
            "valid_acceptance_rate": sum(item["passed"] for item in valid) / len(valid),
            "unsafe_rejection_rate": sum(item["passed"] for item in rejected) / len(rejected),
        },
        "categories": {
            category: {
                "cases": category_counts[category],
                "passed": category_passes[category],
            }
            for category in sorted(category_counts)
        },
        "results": results,
    }


def write_report(result: dict) -> tuple[Path, Path]:
    """保存机器可读 JSON 与项目书可引用的 Markdown 证据。"""

    evidence_dir = ROOT / "docs" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    json_path = evidence_dir / "security_benchmark.json"
    md_path = evidence_dir / "security_benchmark.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = result["summary"]
    rows = [
        "# DSL 安全基准",
        "",
        "该基准完全离线运行，不调用 LLM API，不读取行情或用户文件。",
        "",
        f"- 总用例：{summary['total_cases']}",
        f"- 通过：{summary['passed_cases']}/{summary['total_cases']}",
        f"- 合法公式通过率：{summary['valid_acceptance_rate']:.1%}",
        f"- 非法或危险输入拦截率：{summary['unsafe_rejection_rate']:.1%}",
        "",
        "| 类别 | 用例数 | 通过数 |",
        "|---|---:|---:|",
    ]
    for category, values in result["categories"].items():
        rows.append(f"| {category} | {values['cases']} | {values['passed']} |")
    rows.extend(
        [
            "",
            "## 复现",
            "",
            "```powershell",
            r".\.venv\Scripts\python.exe scripts\security_benchmark.py",
            "```",
            "",
            "完整逐例结果见 `security_benchmark.json`。",
        ]
    )
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    result = run_benchmark()
    json_path, md_path = write_report(result)
    summary = result["summary"]
    print(
        "DSL security benchmark: "
        f"{summary['passed_cases']}/{summary['total_cases']} passed; "
        f"unsafe rejection {summary['unsafe_rejection_rate']:.1%}"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0 if summary["passed_cases"] == summary["total_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

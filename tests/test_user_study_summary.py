"""用户测试汇总器的边界测试。"""

import unittest

from scripts.summarize_user_study import summarize


def _row(index: int, completed: int = 1) -> dict:
    return {
        "participant_id": f"P{index:02d}",
        "experience_level": "beginner",
        "duration_minutes": 8.0 + index,
        "completed": completed,
        "decision_correct": completed,
        "false_positive_avoided": completed,
        "help_count": 1,
        "ease_score": 4,
        "trust_score": 4,
        "comment": "",
    }


class UserStudySummaryTests(unittest.TestCase):
    def test_small_sample_is_marked_insufficient(self) -> None:
        result = summarize([_row(index) for index in range(1, 5)])
        self.assertEqual(result["evidence_status"], "insufficient")

    def test_minimum_sample_is_sufficient(self) -> None:
        result = summarize([_row(index) for index in range(1, 6)])
        self.assertEqual(result["evidence_status"], "sufficient")
        self.assertEqual(result["metrics"]["completion_rate"], 1.0)

    def test_failures_are_not_dropped(self) -> None:
        rows = [_row(index) for index in range(1, 5)] + [_row(5, completed=0)]
        result = summarize(rows)
        self.assertAlmostEqual(result["metrics"]["completion_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()

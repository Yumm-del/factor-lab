"""Agent 消融汇总口径测试。"""

import unittest

from scripts.agent_ablation import IDEAS, summarize


class AgentAblationSummaryTests(unittest.TestCase):
    def test_summary_compares_same_trial_first_and_last_round(self):
        trials = [{
            "idea": IDEAS[0],
            "status": "ok",
            "rounds": [{"score": 30.0}, {"score": 55.0}],
        }]
        result = summarize(trials)
        self.assertEqual(result["mean_score_delta"], 25.0)
        self.assertEqual(result["single_shot_pass_rate"], 0.0)
        self.assertEqual(result["full_agent_pass_rate"], 1.0)

    def test_failed_trials_are_retained(self):
        trials = [{"idea": IDEAS[0], "status": "error", "rounds": []}]
        result = summarize(trials)
        self.assertEqual(result["completed_trials"], 1)
        self.assertEqual(result["failed_trials"], 1)
        self.assertEqual(result["successful_trials"], 0)


if __name__ == "__main__":
    unittest.main()

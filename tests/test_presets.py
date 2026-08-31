"""离线预置案例必须与跨环境最终结论一致。"""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PresetEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presets = json.loads(
            (ROOT / "data" / "presets.json").read_text(encoding="utf-8")
        )

    def test_every_preset_exposes_independent_decision(self):
        self.assertGreaterEqual(len(self.presets), 2)
        for preset in self.presets:
            with self.subTest(idea=preset["idea"]):
                result = preset.get("independent_validation")
                self.assertIsInstance(result, dict)
                self.assertEqual(result.get("status"), "rejected")
                self.assertTrue(result.get("summary"))

    def test_reports_do_not_claim_final_usability(self):
        for preset in self.presets:
            with self.subTest(idea=preset["idea"]):
                self.assertNotIn("判定为**可用**", preset["report"])
                self.assertIn("跨环境接纳结论", preset["report"])


if __name__ == "__main__":
    unittest.main()

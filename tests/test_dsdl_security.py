"""DSL 安全边界的最小回归测试。"""

import json
import unittest

from factor_lab import dsdl


class DSDLSecurityTests(unittest.TestCase):
    """确保合法研究公式可用，越权和资源耗尽输入被拒绝。"""

    def test_valid_formula_is_accepted(self) -> None:
        expression = dsdl.parse_formula("rank(ts_returns(close, 20))")
        self.assertEqual(expression["op"], "rank")

    def test_unknown_operator_is_rejected(self) -> None:
        with self.assertRaises(dsdl.FactorParseError):
            dsdl.parse_factor(json.dumps({"op": "read_env"}))

    def test_python_injection_is_rejected(self) -> None:
        with self.assertRaises(dsdl.FactorParseError):
            dsdl.parse_formula("__import__('os').system('whoami')")

    def test_depth_limit_is_enforced(self) -> None:
        node: dict = {"op": "close"}
        for _ in range(dsdl.MAX_DEPTH + 2):
            node = {"op": "rank", "args": [node]}
        with self.assertRaises(dsdl.FactorParseError):
            dsdl.parse_factor(json.dumps(node))

    def test_leaf_cannot_take_arguments(self) -> None:
        payload = {"op": "close", "args": [{"op": "open"}]}
        with self.assertRaises(dsdl.FactorParseError):
            dsdl.parse_factor(json.dumps(payload))

    def test_window_range_is_enforced(self) -> None:
        payload = {
            "op": "ts_returns",
            "args": [{"op": "close"}],
            "param": 0,
        }
        with self.assertRaises(dsdl.FactorParseError):
            dsdl.parse_factor(json.dumps(payload))

    def test_signed_power_range_is_enforced(self) -> None:
        payload = {
            "op": "signed_power",
            "args": [{"op": "close"}],
            "param": 4,
        }
        with self.assertRaises(dsdl.FactorParseError):
            dsdl.parse_factor(json.dumps(payload))

    def test_non_finite_constant_is_rejected(self) -> None:
        with self.assertRaises(dsdl.FactorParseError):
            dsdl.parse_factor('{"op":"const","value":NaN}')

    def test_hidden_fields_are_rejected(self) -> None:
        payload = {"op": "close", "payload": "read secret"}
        with self.assertRaises(dsdl.FactorParseError):
            dsdl.parse_factor(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()

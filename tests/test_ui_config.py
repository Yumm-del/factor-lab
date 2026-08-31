"""公共Demo只能展示当前环境真正可执行的选项。"""

import unittest

from factor_lab import ui_config


class UIConfigTests(unittest.TestCase):
    def test_cloud_without_large_files_hides_unavailable_features(self):
        self.assertEqual(ui_config.pool_options(False), ["沪深300（300 只，快）"])
        self.assertEqual(ui_config.neutralization_options(False), ["无"])

    def test_local_data_enables_full_capabilities(self):
        self.assertEqual(len(ui_config.pool_options(True)), 2)
        self.assertEqual(len(ui_config.neutralization_options(True)), 3)

    def test_none_label_maps_to_computation_enum(self):
        self.assertEqual(ui_config.normalize_style("无"), "none")

    def test_unknown_style_fails_closed(self):
        with self.assertRaises(KeyError):
            ui_config.normalize_style("未知")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import importlib
import unittest
from unittest import mock


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())


class GoldMainTests(unittest.TestCase):
    def test_gold_module_main_delegates_to_local_gold_main(self) -> None:
        module = importlib.import_module("apps.warehouse.run_gold.main")

        with mock.patch(
            "praedixa.platform.warehouse.local_gold.main"
        ) as gold_main:
            module.main()

        gold_main.assert_called_once_with()

    def test_gold_module_main_uses_no_cli_parser(self) -> None:
        module_path = PROJECT_ROOT / "apps" / "warehouse" / "run_gold" / "main.py"
        source = module_path.read_text(encoding="utf-8")

        self.assertNotIn("argparse", source)
        self.assertNotIn("parse_args", source)


if __name__ == "__main__":
    unittest.main()

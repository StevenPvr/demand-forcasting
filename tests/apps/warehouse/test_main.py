from __future__ import annotations

from pathlib import Path
import importlib
import unittest
from unittest import mock


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())


class WarehouseMainTests(unittest.TestCase):
    def test_warehouse_module_main_delegates_to_local_medallion_main(self) -> None:
        module = importlib.import_module("apps.warehouse.main")

        with mock.patch(
            "praedixa.platform.warehouse.local_medallion.main"
        ) as medallion_main:
            module.main()

        medallion_main.assert_called_once_with()

    def test_warehouse_module_main_uses_no_cli_parser(self) -> None:
        module_path = PROJECT_ROOT / "apps" / "warehouse" / "main.py"
        source = module_path.read_text(encoding="utf-8")

        self.assertNotIn("argparse", source)
        self.assertNotIn("parse_args", source)


if __name__ == "__main__":
    unittest.main()

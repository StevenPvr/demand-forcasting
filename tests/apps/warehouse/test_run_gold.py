from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())


class GoldMainTests(unittest.TestCase):
    def test_gold_module_main_can_show_help(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        result = subprocess.run(
            [sys.executable, "-m", "apps.warehouse.run_gold.main", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Run the local Praedixa silver -> gold workflow.", result.stdout)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.global_dataset.main import parse_args  # noqa: E402


class GlobalDatasetMainTests(unittest.TestCase):
    def test_parse_args_uses_default_output_dir(self) -> None:
        with mock.patch.object(sys, "argv", ["global_dataset_main"]):
            args = parse_args()

        self.assertEqual(args.output_dir, "data/global_dataset")


if __name__ == "__main__":
    unittest.main()

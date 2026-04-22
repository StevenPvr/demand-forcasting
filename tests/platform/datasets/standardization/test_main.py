from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.main import (  # noqa: E402
    build_default_global_dataset_main_config,
)
from praedixa.platform.runtime.paths import GLOBAL_DATASET_DIR  # noqa: E402


class GlobalDatasetMainTests(unittest.TestCase):
    def test_default_global_dataset_main_config_uses_default_output_dir(self) -> None:
        config = build_default_global_dataset_main_config()
        self.assertEqual(config.output_dir, str(GLOBAL_DATASET_DIR))


if __name__ == "__main__":
    unittest.main()

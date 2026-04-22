from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.runtime.paths import CACHE_DIR, GLOBAL_DATASET_DIR, SOURCES_DIR, VAR_DIR  # noqa: E402


class RuntimePathsTests(unittest.TestCase):
    def test_runtime_paths_are_derived_from_var_dir(self) -> None:
        self.assertEqual(SOURCES_DIR.parent, VAR_DIR)
        self.assertEqual(CACHE_DIR.parent, VAR_DIR)
        self.assertEqual(GLOBAL_DATASET_DIR.parent.parent, VAR_DIR)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.runtime.constants import RANDOM_SEED  # noqa: E402


class RuntimeConstantsTests(unittest.TestCase):
    def test_random_seed_matches_repo_default(self) -> None:
        self.assertEqual(RANDOM_SEED, 7)


if __name__ == "__main__":
    unittest.main()

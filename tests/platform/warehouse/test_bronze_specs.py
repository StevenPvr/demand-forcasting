from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.platform.warehouse.bronze_specs import supplemental_corpus_daily_ddl  # noqa: E402


class BronzeSpecsTests(unittest.TestCase):
    def test_supplemental_corpus_bronze_ddl_keeps_label_quality_columns(self) -> None:
        ddl = supplemental_corpus_daily_ddl("bronze")

        self.assertIn("target_semantics VARCHAR", ddl)
        self.assertIn("censor_flag BOOLEAN", ddl)
        self.assertIn("target_source VARCHAR", ddl)
        self.assertIn("label_quality_score DOUBLE", ddl)
        self.assertIn("usable_for_training_flag BOOLEAN", ddl)


if __name__ == "__main__":
    unittest.main()

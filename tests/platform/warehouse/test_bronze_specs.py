from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.platform.warehouse.bronze_specs import (  # noqa: E402
    supplemental_corpus_daily_spec,
    supplemental_corpus_daily_ddl,
    synthetic_foodservice_daily_ddl,
)


class BronzeSpecsTests(unittest.TestCase):
    def test_supplemental_corpus_bronze_ddl_keeps_label_quality_columns(self) -> None:
        ddl = supplemental_corpus_daily_ddl("bronze")

        self.assertIn("target_semantics VARCHAR", ddl)
        self.assertIn("censor_flag BOOLEAN", ddl)
        self.assertIn("target_source VARCHAR", ddl)
        self.assertIn("label_quality_score DOUBLE", ddl)
        self.assertIn("usable_for_training_flag BOOLEAN", ddl)

    def test_supplemental_corpus_bronze_spec_points_to_stable_table(self) -> None:
        spec = supplemental_corpus_daily_spec(
            source_path=Path("supplemental_corpus_daily.parquet"),
            schema_name="bronze",
        )

        self.assertEqual(spec.table_name, "bronze_supplemental_corpus_daily")
        self.assertEqual(spec.source_name, "supplemental_corpus_daily")
        self.assertFalse(spec.required)
        self.assertIn("dataset_source", spec.expected_columns)

    def test_synthetic_foodservice_bronze_ddl_uses_stable_source_table(self) -> None:
        ddl = synthetic_foodservice_daily_ddl("bronze")

        self.assertIn("bronze_synthetic_foodservice_daily", ddl)
        self.assertIn("target_semantics VARCHAR", ddl)
        self.assertIn("source_policy_id VARCHAR", ddl)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import pandas as pd
import polars as pl


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.pipeline import build_global_daily_standardization  # noqa: E402


def _canonical_fixture(dataset_source: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dataset_source": [dataset_source],
            "source_partition": ["historical"],
            "source_run_id": ["manual"],
            "series_id": [f"{dataset_source}_site__sku_1"],
            "dt": [pd.Timestamp("2024-01-01")],
            "location_id": [f"{dataset_source}_site"],
            "product_id": ["sku_1"],
            "region_id": [None],
            "org_group_id": [None],
            "category_level_1": [None],
            "category_level_2": [None],
            "category_level_3": [None],
            "observed_demand_qty": [12.0],
            "target_semantics": ["observed_sales"],
            "censor_flag": [False],
            "target_source": ["fixture"],
            "label_quality_score": [1.0],
            "usable_for_training_flag": [True],
            "observed_revenue_net": [None],
            "observed_discount_amount": [None],
            "promo_flag": [None],
            "holiday_flag": [None],
            "activity_flag": [None],
            "observed_stockout_flag": [None],
            "observed_stockout_available": [False],
            "observed_stockout_intensity": [None],
            "day_complete_flag": [True],
            "calendar_weekday_name": ["Monday"],
            "calendar_day_of_week": [0],
            "calendar_month": [1],
            "calendar_year": [2024],
            "calendar_week_key": [1],
            "event_name_1": [None],
            "event_type_1": [None],
            "event_name_2": [None],
            "event_type_2": [None],
            "weather_precipitation": [None],
            "weather_temperature": [None],
            "weather_humidity": [None],
            "weather_wind_level": [None],
            "silver_run_id": ["manual"],
        }
    )


class GlobalDatasetPipelineTests(unittest.TestCase):
    def test_build_global_daily_standardization_returns_in_memory_summary_only(self) -> None:
        with (
            mock.patch(
                "praedixa.platform.datasets.standardization.pipeline.build_freshretail_standardized_frame",
                return_value=_canonical_fixture("freshretail"),
            ),
            mock.patch(
                "praedixa.platform.datasets.standardization.pipeline.build_supplemental_corpus_frame",
                return_value=_canonical_fixture("first_party_daily"),
            ),
        ):
            artifacts = build_global_daily_standardization(bakery_input_path=None)

        self.assertEqual(artifacts.source_summaries["freshretail"]["rows"], 1)
        self.assertEqual(artifacts.source_summaries["supplemental_corpus"]["rows"], 1)
        self.assertEqual(artifacts.source_summaries["combined"]["rows"], 2)
        self.assertEqual(artifacts.data_quality["sources"]["freshretail"]["error_count"], 0)
        self.assertEqual(artifacts.data_quality["combined"]["error_count"], 0)

    def test_build_global_daily_standardization_fails_on_duplicate_canonical_grain(self) -> None:
        duplicated_fresh = pl.concat(
            [_canonical_fixture("freshretail"), _canonical_fixture("freshretail")],
            how="vertical_relaxed",
        )
        with (
            mock.patch(
                "praedixa.platform.datasets.standardization.pipeline.build_freshretail_standardized_frame",
                return_value=duplicated_fresh,
            ),
            mock.patch(
                "praedixa.platform.datasets.standardization.pipeline.build_supplemental_corpus_frame",
                return_value=_canonical_fixture("first_party_daily"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate_grain"):
                build_global_daily_standardization(bakery_input_path=None)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd
import polars as pl


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.quality import raise_on_error_issues, validate_canonical_frame  # noqa: E402


def _canonical_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dataset_source": ["freshretail"],
            "source_partition": ["train"],
            "source_run_id": ["manual"],
            "series_id": ["store_1__sku_1"],
            "dt": [pd.Timestamp("2024-01-01")],
            "location_id": ["store_1"],
            "product_id": ["sku_1"],
            "region_id": ["city_a"],
            "org_group_id": ["mg_1"],
            "category_level_1": ["cat_1"],
            "category_level_2": ["dept_1"],
            "category_level_3": ["family_1"],
            "observed_demand_qty": [12.0],
            "observed_revenue_net": [18.0],
            "observed_discount_amount": [0.1],
            "avg_selling_price": [1.5],
            "promo_flag": [True],
            "holiday_flag": [False],
            "activity_flag": [False],
            "observed_stockout_flag": [False],
            "observed_stockout_available": [True],
            "observed_stockout_intensity": [0.0],
            "location_open_flag": [True],
            "day_complete_flag": [True],
            "missing_sales_flag": [False],
            "calendar_weekday_name": ["Monday"],
            "calendar_day_of_week": [0],
            "calendar_month": [1],
            "calendar_year": [2024],
            "calendar_week_key": [1],
            "event_name_1": [None],
            "event_type_1": [None],
            "event_name_2": [None],
            "event_type_2": [None],
            "weather_precipitation": [1.0],
            "weather_temperature": [18.0],
            "weather_humidity": [0.6],
            "weather_wind_level": [4.0],
            "anomaly_flag": [False],
            "silver_run_id": ["manual"],
        }
    )


class GlobalDatasetQualityTests(unittest.TestCase):
    def test_validate_canonical_frame_passes_clean_frame(self) -> None:
        report = validate_canonical_frame(_canonical_frame(), dataset_name="freshretail")

        self.assertEqual(report.error_count, 0)
        self.assertEqual(report.warning_count, 0)

    def test_validate_canonical_frame_detects_duplicate_grain(self) -> None:
        duplicated = pl.concat([_canonical_frame(), _canonical_frame()], how="vertical_relaxed")

        report = validate_canonical_frame(duplicated, dataset_name="freshretail")

        self.assertTrue(any(issue.code == "duplicate_grain" for issue in report.issues))
        with self.assertRaisesRegex(ValueError, "duplicate_grain"):
            raise_on_error_issues(report)

    def test_validate_canonical_frame_detects_negative_demand(self) -> None:
        negative = _canonical_frame().with_columns(pl.lit(-3.0).alias("observed_demand_qty"))

        report = validate_canonical_frame(negative, dataset_name="freshretail")

        self.assertTrue(any(issue.code == "negative_observed_demand_qty" for issue in report.issues))
        with self.assertRaisesRegex(ValueError, "negative_observed_demand_qty"):
            raise_on_error_issues(report)

    def test_validate_canonical_frame_detects_invalid_series_id(self) -> None:
        invalid_series = _canonical_frame().with_columns(pl.lit("broken_series").alias("series_id"))

        report = validate_canonical_frame(invalid_series, dataset_name="freshretail")

        self.assertTrue(any(issue.code == "invalid_series_id" for issue in report.issues))


if __name__ == "__main__":
    unittest.main()

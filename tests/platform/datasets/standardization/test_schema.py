from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd
import polars as pl


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.schema import align_lazy_frame_to_canonical_schema  # noqa: E402


def _minimal_canonical_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dataset_source": ["freshretail"],
            "source_partition": ["train"],
            "series_id": ["store_1__sku_1"],
            "location_id": ["store_1"],
            "product_id": ["sku_1"],
            "region_id": ["city_a"],
            "org_group_id": ["mg_1"],
            "category_level_1": ["cat_1"],
            "category_level_2": ["dept_1"],
            "category_level_3": ["family_1"],
            "dt": [pd.Timestamp("2024-01-01")],
            "observed_demand_qty": [12.0],
            "observed_revenue_net": [None],
            "observed_discount_amount": [0.1],
            "avg_selling_price": [None],
            "promo_flag": [True],
            "holiday_flag": [True],
            "activity_flag": [False],
            "observed_stockout_flag": [True],
            "observed_stockout_available": [True],
            "observed_stockout_intensity": [2.0],
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
            "source_run_id": ["manual"],
            "silver_run_id": ["manual"],
        }
    )


class GlobalDatasetSchemaTests(unittest.TestCase):
    def test_align_lazy_frame_to_canonical_schema_casts_date_and_numeric_columns(self) -> None:
        aligned = align_lazy_frame_to_canonical_schema(_minimal_canonical_frame().lazy()).collect()

        self.assertEqual(aligned.schema["dt"], pl.Date)
        self.assertEqual(aligned.schema["observed_demand_qty"], pl.Float32)
        self.assertEqual(aligned.schema["calendar_day_of_week"], pl.Int8)


if __name__ == "__main__":
    unittest.main()

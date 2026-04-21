from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd
import polars as pl


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.freshretail import standardize_freshretail_lazy_frame  # noqa: E402


class GlobalDatasetFreshRetailTests(unittest.TestCase):
    def test_standardize_freshretail_lazy_frame_preserves_native_observed_demand(self) -> None:
        frame = pl.DataFrame(
            {
                "city_id": ["city_a"],
                "store_id": ["store_1"],
                "management_group_id": ["mg_1"],
                "first_category_id": ["cat_1"],
                "second_category_id": ["dept_1"],
                "third_category_id": ["family_1"],
                "product_id": ["sku_1"],
                "dt": [pd.Timestamp("2024-01-01")],
                "sale_amount": [12.0],
                "stock_hour6_22_cnt": [2],
                "discount": [0.1],
                "holiday_flag": [True],
                "activity_flag": [False],
                "precpt": [1.5],
                "avg_temperature": [18.0],
                "avg_humidity": [0.6],
                "avg_wind_level": [4.0],
                "is_censored": [True],
            }
        )

        record = standardize_freshretail_lazy_frame(frame.lazy(), source_partition="train").collect().to_dicts()[0]

        self.assertEqual(record["dataset_source"], "freshretail")
        self.assertEqual(record["series_id"], "store_1__sku_1")
        self.assertEqual(record["location_id"], "store_1")
        self.assertEqual(record["product_id"], "sku_1")
        self.assertEqual(record["observed_demand_qty"], 12.0)
        self.assertTrue(record["observed_stockout_flag"])
        self.assertTrue(record["observed_stockout_available"])
        self.assertAlmostEqual(record["observed_discount_amount"], 0.1, places=6)
        self.assertTrue(record["promo_flag"])


if __name__ == "__main__":
    unittest.main()

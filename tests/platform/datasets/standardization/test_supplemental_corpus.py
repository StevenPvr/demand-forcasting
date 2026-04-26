from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import polars as pl


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.supplemental_corpus import (  # noqa: E402
    build_supplemental_corpus_standardized_dataset,
    supplemental_corpus_compatibility_matrix,
    standardize_freshretail_lt_lazy_frame,
)
from praedixa.platform.datasets.standardization.supplemental_corpus_standardizers import (  # noqa: E402
    standardize_m5_sales_lazy_frame,
)


class SupplementalCorpusTests(unittest.TestCase):
    def test_freshretail_lt_promo_flag_treats_discount_one_as_non_promo(self) -> None:
        frame = pl.DataFrame(
            {
                "city_id": ["city_a", "city_a"],
                "store_id": ["store_1", "store_1"],
                "management_group_id": ["mg_1", "mg_1"],
                "first_category_id": ["cat_1", "cat_1"],
                "second_category_id": ["dept_1", "dept_1"],
                "third_category_id": ["family_1", "family_1"],
                "product_id": ["sku_1", "sku_2"],
                "dt": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "sale_amount": [12.0, 9.0],
                "stock_hour6_22_cnt": [1, 0],
                "discount": [1.0, 0.8],
                "holiday_flag": [False, False],
                "activity_flag": [False, False],
                "precpt": [0.0, 0.0],
                "avg_temperature": [18.0, 18.0],
                "avg_humidity": [0.6, 0.6],
                "avg_wind_level": [3.0, 3.0],
                "is_censored": [False, False],
            }
        )

        records = standardize_freshretail_lt_lazy_frame(
            frame.lazy(), source_partition="historical_train"
        ).collect()
        self.assertFalse(
            records.filter(pl.col("product_id") == "sku_1").select("promo_flag").item()
        )
        self.assertTrue(
            records.filter(pl.col("product_id") == "sku_2").select("promo_flag").item()
        )
        self.assertEqual(
            records.filter(pl.col("product_id") == "sku_1")
            .select("target_semantics")
            .item(),
            "observed_sales",
        )
        self.assertNotIn("location_open_flag", records.columns)

    def test_m5_maps_to_canonical_daily_contract(self) -> None:
        m5_sales = pl.DataFrame(
            {
                "id": ["FOODS_1_001_CA_1_evaluation"],
                "item_id": ["FOODS_1_001"],
                "dept_id": ["FOODS_1"],
                "cat_id": ["FOODS"],
                "store_id": ["CA_1"],
                "state_id": ["CA"],
                "d_1": [3],
                "d_2": [4],
            }
        )
        m5_calendar = pl.DataFrame(
            {
                "d": ["d_1", "d_2"],
                "date": ["2011-01-29", "2011-01-30"],
                "event_name_1": [None, "Sporting"],
                "event_type_1": [None, "Sporting"],
                "event_name_2": [None, None],
                "event_type_2": [None, None],
            }
        )

        frames = [
            standardize_m5_sales_lazy_frame(
                m5_sales.lazy(),
                m5_calendar.lazy(),
                source_partition="test",
            ),
        ]
        records = pl.concat(frames, how="vertical_relaxed").collect()

        self.assertEqual(
            set(records["dataset_source"].to_list()),
            {"m5_forecasting_accuracy"},
        )
        self.assertFalse(records["series_id"].is_null().any())
        self.assertFalse(records["observed_demand_qty"].is_null().any())
        self.assertEqual(set(records["target_semantics"].to_list()), {"observed_sales"})

    def test_build_supplemental_corpus_dataset_loads_m5_raw_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            self._write_small_m5_fixture(tmp_dir / "m5_forecasting_accuracy_zenodo")

            output_path = tmp_dir / "supplemental_corpus_daily.parquet"
            result_path = build_supplemental_corpus_standardized_dataset(
                output_path=output_path,
                raw_dir=tmp_dir,
                source_run_id="test_run",
                silver_run_id="test_silver",
            )
            records = pl.read_parquet(result_path).select("dataset_source").unique()

        self.assertEqual(result_path, output_path)
        self.assertEqual(
            set(records["dataset_source"].to_list()),
            {"m5_forecasting_accuracy"},
        )

    def _write_small_m5_fixture(self, root: Path) -> None:
        root.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "id": "FOODS_1_001_CA_1_evaluation",
                    "item_id": "FOODS_1_001",
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "store_id": "CA_1",
                    "state_id": "CA",
                    "d_1": 3,
                    "d_2": 4,
                }
            ]
        ).to_csv(root / "sales_train_evaluation.csv", index=False)
        pd.DataFrame(
            [
                {"d": "d_1", "date": "2011-01-29"},
                {"d": "d_2", "date": "2011-01-30"},
            ]
        ).to_csv(root / "calendar.csv", index=False)


class SupplementalCorpusSmokeTests(unittest.TestCase):
    def test_compatibility_matrix_matches_the_curated_training_corpus(self) -> None:
        compatibility_by_source = {
            item.dataset_source: item
            for item in supplemental_corpus_compatibility_matrix()
        }

        self.assertTrue(
            compatibility_by_source["freshretail_lt"].compatible_with_pipeline
        )
        self.assertTrue(
            compatibility_by_source["m5_forecasting_accuracy"].compatible_with_pipeline
        )
        self.assertEqual(
            set(compatibility_by_source),
            {
                "freshretail_lt",
                "m5_forecasting_accuracy",
            },
        )


if __name__ == "__main__":
    unittest.main()

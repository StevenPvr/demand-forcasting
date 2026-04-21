from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
import polars as pl


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.pipeline import build_global_daily_standardization  # noqa: E402


def _freshretail_fixture() -> dict[str, list[object]]:
    return {
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


def _commercial_fixture() -> dict[str, list[object]]:
    return {
        "dataset_source": ["uci_online_retail"],
        "source_partition": ["historical"],
        "source_run_id": ["manual"],
        "series_id": ["uk_online_retail_1__sku_1"],
        "location_id": ["uk_online_retail_1"],
        "product_id": ["sku_1"],
        "region_id": [None],
        "org_group_id": [None],
        "category_level_1": [None],
        "category_level_2": [None],
        "category_level_3": [None],
        "dt": [pd.Timestamp("2011-01-29")],
        "observed_demand_qty": [2.0],
        "observed_revenue_net": [7.0],
        "observed_discount_amount": [None],
        "avg_selling_price": [3.5],
        "promo_flag": [None],
        "holiday_flag": [None],
        "activity_flag": [None],
        "observed_stockout_flag": [None],
        "observed_stockout_available": [False],
        "observed_stockout_intensity": [None],
        "location_open_flag": [True],
        "day_complete_flag": [True],
        "missing_sales_flag": [False],
        "calendar_weekday_name": ["Saturday"],
        "calendar_day_of_week": [0],
        "calendar_month": [1],
        "calendar_year": [2011],
        "calendar_week_key": [4],
        "event_name_1": [None],
        "event_type_1": [None],
        "event_name_2": [None],
        "event_type_2": [None],
        "weather_precipitation": [None],
        "weather_temperature": [None],
        "weather_humidity": [None],
        "weather_wind_level": [None],
        "anomaly_flag": [False],
        "silver_run_id": ["manual"],
    }


def _bakery_fixture() -> dict[str, list[object]]:
    return {
        "dataset_source": ["bakery"],
        "source_partition": ["historical"],
        "source_run_id": ["manual"],
        "series_id": ["bakery_store_1__sku_1"],
        "location_id": ["bakery_store_1"],
        "product_id": ["sku_1"],
        "region_id": [None],
        "org_group_id": [None],
        "category_level_1": [None],
        "category_level_2": [None],
        "category_level_3": [None],
        "dt": [pd.Timestamp("2024-01-02")],
        "observed_demand_qty": [2.0],
        "observed_revenue_net": [5.0],
        "observed_discount_amount": [None],
        "avg_selling_price": [2.5],
        "promo_flag": [None],
        "holiday_flag": [None],
        "activity_flag": [None],
        "observed_stockout_flag": [None],
        "observed_stockout_available": [False],
        "observed_stockout_intensity": [None],
        "location_open_flag": [True],
        "day_complete_flag": [True],
        "missing_sales_flag": [False],
        "calendar_weekday_name": ["Tuesday"],
        "calendar_day_of_week": [1],
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
        "anomaly_flag": [False],
        "silver_run_id": ["manual"],
    }


def _fixture_frame(dataset_source: str) -> dict[str, list[object]]:
    if dataset_source == "freshretail":
        return _freshretail_fixture()
    if dataset_source == "commercial_external":
        return _commercial_fixture()
    return _bakery_fixture()


def _write_fixture_parquet(path: Path, dataset_source: str) -> None:
    pl.DataFrame(_fixture_frame(dataset_source)).write_parquet(path)


def _copy_fixture(source_path: Path, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.write_bytes(source_path.read_bytes())
    return destination


class GlobalDatasetPipelineTests(unittest.TestCase):
    def test_build_global_daily_standardization_writes_manifest_from_source_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fresh_path = root / "freshretail_daily.parquet"
            commercial_path = root / "commercial_external_daily.parquet"
            _write_fixture_parquet(fresh_path, "freshretail")
            _write_fixture_parquet(commercial_path, "commercial_external")

            def copy_fresh_fixture(output_path: str | Path) -> Path:
                return _copy_fixture(fresh_path, output_path)

            def copy_commercial_fixture(output_path: str | Path, raw_dir: str | Path) -> Path:
                _ = raw_dir
                return _copy_fixture(commercial_path, output_path)

            with mock.patch(
                "praedixa.platform.datasets.standardization.pipeline.build_freshretail_standardized_dataset",
                side_effect=copy_fresh_fixture,
            ):
                with mock.patch(
                    "praedixa.platform.datasets.standardization.pipeline.build_commercial_external_standardized_dataset",
                    side_effect=copy_commercial_fixture,
                ):
                    artifacts = build_global_daily_standardization(
                        output_dir=root / "global_dataset",
                        bakery_input_path=None,
                    )

            manifest = json.loads(Path(artifacts.manifest_path).read_text(encoding="utf-8"))
            combined = pl.read_parquet(artifacts.global_gold)

        self.assertEqual(combined.height, 2)
        self.assertEqual(manifest["sources"]["combined"]["rows"], 2)
        self.assertEqual(manifest["sources"]["freshretail"]["rows"], 1)
        self.assertEqual(manifest["sources"]["commercial_external"]["rows"], 1)
        self.assertIn("data_quality", manifest)
        self.assertEqual(manifest["data_quality"]["sources"]["freshretail"]["error_count"], 0)
        self.assertEqual(manifest["data_quality"]["combined"]["error_count"], 0)

    def test_build_global_daily_standardization_fails_on_duplicate_canonical_grain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fresh_path = root / "freshretail_daily.parquet"
            commercial_path = root / "commercial_external_daily.parquet"
            _write_fixture_parquet(fresh_path, "freshretail")
            _write_fixture_parquet(commercial_path, "commercial_external")
            duplicated_fresh = pl.concat([pl.read_parquet(fresh_path), pl.read_parquet(fresh_path)], how="vertical_relaxed")
            duplicated_fresh.write_parquet(fresh_path)

            def copy_fresh_fixture(output_path: str | Path) -> Path:
                return _copy_fixture(fresh_path, output_path)

            def copy_commercial_fixture(output_path: str | Path, raw_dir: str | Path) -> Path:
                _ = raw_dir
                return _copy_fixture(commercial_path, output_path)

            with mock.patch(
                "praedixa.platform.datasets.standardization.pipeline.build_freshretail_standardized_dataset",
                side_effect=copy_fresh_fixture,
            ):
                with mock.patch(
                    "praedixa.platform.datasets.standardization.pipeline.build_commercial_external_standardized_dataset",
                    side_effect=copy_commercial_fixture,
                ):
                    with self.assertRaisesRegex(ValueError, "duplicate_grain"):
                        build_global_daily_standardization(
                            output_dir=root / "global_dataset",
                            bakery_input_path=None,
                        )


if __name__ == "__main__":
    unittest.main()

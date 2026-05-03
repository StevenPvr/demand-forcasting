from __future__ import annotations

# ruff: noqa: E402

from collections.abc import Mapping
from pathlib import Path
import sys
import unittest
from typing import cast
from unittest import mock

import pandas as pd
import polars as pl


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.pipeline import (
    build_global_daily_standardization,
)  # noqa: E402


def _mapping_value(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise AssertionError(f"Expected mapping payload for `{key}`.")
    return cast(Mapping[str, object], value)


def _int_value(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int):
        raise AssertionError(f"Expected integer value for `{key}`.")
    return value


def _source_rows(source_summaries: Mapping[str, object], source_name: str) -> int:
    return _int_value(_mapping_value(source_summaries, source_name), "rows")


def _quality_error_count(data_quality: Mapping[str, object], source_name: str) -> int:
    sources = _mapping_value(data_quality, "sources")
    return _int_value(_mapping_value(sources, source_name), "error_count")


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
    def test_build_global_daily_standardization_returns_in_memory_summary_only(
        self,
    ) -> None:
        with (
            mock.patch(
                "praedixa.platform.datasets.standardization.pipeline._build_source_frames",
                return_value={
                    "synthetic_foodservice_qsr": _canonical_fixture(
                        "synthetic_foodservice_qsr"
                    )
                },
            ),
        ):
            artifacts = build_global_daily_standardization()

        self.assertEqual(
            _source_rows(artifacts.source_summaries, "synthetic_foodservice_qsr"), 1
        )
        self.assertNotIn("supplemental_corpus", artifacts.source_summaries)
        self.assertEqual(
            _source_rows(artifacts.source_summaries, "combined"), 1
        )
        self.assertEqual(
            _quality_error_count(
                artifacts.data_quality, "synthetic_foodservice_qsr"
            ),
            0,
        )
        self.assertEqual(
            _int_value(
                _mapping_value(artifacts.data_quality, "combined"), "error_count"
            ),
            0,
        )

    def test_build_global_daily_standardization_fails_on_duplicate_canonical_grain(
        self,
    ) -> None:
        duplicated_fresh = pl.concat(
            [_canonical_fixture("freshretail"), _canonical_fixture("freshretail")],
            how="vertical_relaxed",
        )
        with (
            mock.patch(
                "praedixa.platform.datasets.standardization.pipeline._build_source_frames",
                return_value={"freshretail": duplicated_fresh},
            ),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate_grain"):
                build_global_daily_standardization()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, patch

import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.evaluation.reference_mode import (  # noqa: E402
    load_gold_reference_mode_frames,
)


class EvaluationReferenceModeTests(unittest.TestCase):
    def test_gold_reference_mode_samples_only_train_and_validation(self) -> None:
        train_frame = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01"]), "series_id": ["a"]})
        valid_frame = pd.DataFrame({"dt": pd.to_datetime(["2024-01-02"]), "series_id": ["a"]})
        overlap_test_frame = pd.DataFrame({"dt": pd.to_datetime(["2024-04-01"]), "series_id": ["bakery_a"]})
        reference_train = pd.DataFrame({"date": pd.to_datetime(["2024-03-30"]), "product": ["croissant"], "quantity": [9.0], "is_missing_day": [0]})
        reference_val = pd.DataFrame({"date": pd.to_datetime(["2024-03-31"]), "product": ["croissant"], "quantity": [10.0], "is_missing_day": [0]})
        scored_reference_test = pd.DataFrame({"date": pd.to_datetime(["2024-04-01"]), "product": ["croissant"], "quantity": [11.0], "is_missing_day": [0]})

        with (
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode.load_gold_train_tuning_frames",
                return_value=(train_frame, valid_frame, "location_id", {}, {}),
            ) as mocked_load_gold,
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode.build_bakery_reference_splits_from_gold",
                return_value=SimpleNamespace(
                    train_df=reference_train,
                    val_df=reference_val,
                    test_df=scored_reference_test,
                    metadata={
                        "reference_protocol": "bakery_product_arima_equivalent_from_gold",
                        "reference_test_rows": 1,
                    },
                ),
            ) as mocked_reference_bundle,
            patch(
                "praedixa.demand_forecast.evaluation.reference_mode.load_gold_bakery_overlap_test_frame",
                return_value=(
                    overlap_test_frame,
                    scored_reference_test,
                    {"reference_test_rows": 1, "overlap_test_rows": 1},
                ),
            ) as mocked_overlap,
        ):
            resolved_train_frame, resolved_valid_frame, resolved_test_frame, _, _, overlap_metadata = load_gold_reference_mode_frames(
                duckdb_path="warehouse.duckdb",
                gold_table="gold.gold_daily_product_forecast_panel_d1",
                train_sample_fraction=0.05,
                tuning_sample_fraction=0.07,
            )

        mocked_load_gold.assert_called_once_with(
            duckdb_path="warehouse.duckdb",
            gold_table="gold.gold_daily_product_forecast_panel_d1",
            logger=ANY,
            date_col="dt",
            dataset_source_col="dataset_source",
            train_sample_fraction=0.05,
            tuning_sample_fraction=0.07,
        )
        mocked_reference_bundle.assert_called_once_with(
            duckdb_path="warehouse.duckdb",
            logger=ANY,
        )
        mocked_overlap.assert_called_once_with(
            duckdb_path="warehouse.duckdb",
            reference_full_df=ANY,
            reference_test_df=scored_reference_test,
        )
        self.assertIs(resolved_train_frame, train_frame)
        self.assertIs(resolved_valid_frame, valid_frame)
        self.assertIs(resolved_test_frame, overlap_test_frame)
        self.assertEqual(
            overlap_metadata,
            {
                "reference_protocol": "bakery_product_arima_equivalent_from_gold",
                "reference_test_rows": 1,
                "overlap_test_rows": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()

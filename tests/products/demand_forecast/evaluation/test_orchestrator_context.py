from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.contracts.targets import DEFAULT_VARIATION_TARGET_COL  # noqa: E402
from praedixa.demand_forecast.evaluation.bakery_metrics import (  # noqa: E402
    REFERENCE_DATE_COL,
    REFERENCE_PRODUCT_COL,
    REFERENCE_TARGET_COL,
)
from praedixa.demand_forecast.evaluation.orchestrator import EvaluationBuildRequest  # noqa: E402
from praedixa.demand_forecast.evaluation.orchestrator_context import (  # noqa: E402
    LoadedEvaluationFrames,
    load_evaluation_frames,
    prepare_evaluation_context,
)


def _delta_log_wow(target_values: list[float], lag_values: list[float]) -> np.ndarray:
    target = np.asarray(target_values, dtype=float)
    lag = np.asarray(lag_values, dtype=float)
    return np.log1p(target) - np.log1p(lag)


def _build_bakery_loaded_frames() -> LoadedEvaluationFrames:
    train_targets = [10.0, 11.0, 12.0, 13.0]
    train_lags = [8.0, 9.0, 10.0, 11.0]
    valid_targets = [14.0, 15.0]
    valid_lags = [12.0, 13.0]
    test_targets = [16.0, 17.0]
    test_lags = [14.0, 15.0]

    train_frame = pd.DataFrame(
        {
            "dt": pd.date_range("2024-01-01", periods=4, freq="D"),
            "dataset_source": ["bakery"] * 4,
            "location_id": ["bakery_store_1"] * 4,
            "product_id": ["croissant"] * 4,
            "series_id": ["bakery_store_1__croissant"] * 4,
            "client_id": ["public_bakery_sales"] * 4,
            "avg_selling_price": [2.0, 2.0, 2.1, 2.1],
            "weather_humidity": [40.0, 42.0, 41.0, 43.0],
            "event_count_local": [1.0, 2.0, 3.0, 4.0],
            "target_demand_qty_d_plus_1": train_targets,
            "target_lag_7": train_lags,
            DEFAULT_VARIATION_TARGET_COL: _delta_log_wow(train_targets, train_lags),
        }
    )
    valid_frame = pd.DataFrame(
        {
            "dt": pd.date_range("2024-01-05", periods=2, freq="D"),
            "dataset_source": ["bakery"] * 2,
            "location_id": ["bakery_store_1"] * 2,
            "product_id": ["croissant"] * 2,
            "series_id": ["bakery_store_1__croissant"] * 2,
            "client_id": ["public_bakery_sales"] * 2,
            "avg_selling_price": [2.2, 2.2],
            "weather_humidity": [44.0, 45.0],
            "event_count_local": [5.0, 6.0],
            "target_demand_qty_d_plus_1": valid_targets,
            "target_lag_7": valid_lags,
            DEFAULT_VARIATION_TARGET_COL: _delta_log_wow(valid_targets, valid_lags),
        }
    )
    test_frame = pd.DataFrame(
        {
            "dt": pd.date_range("2024-01-07", periods=2, freq="D"),
            "dataset_source": ["bakery"] * 2,
            "location_id": ["bakery_store_1"] * 2,
            "product_id": ["croissant"] * 2,
            "series_id": ["bakery_store_1__croissant"] * 2,
            "client_id": ["public_bakery_sales"] * 2,
            "avg_selling_price": [2.3, 2.4],
            "weather_humidity": [np.nan, np.nan],
            "event_count_local": [np.nan, 7.0],
            "target_demand_qty_d_plus_1": test_targets,
            "target_lag_7": test_lags,
            DEFAULT_VARIATION_TARGET_COL: _delta_log_wow(test_targets, test_lags),
            REFERENCE_DATE_COL: pd.date_range("2024-01-08", periods=2, freq="D"),
            REFERENCE_PRODUCT_COL: ["croissant"] * 2,
        }
    )
    history_reference = pd.DataFrame(
        {
            REFERENCE_DATE_COL: pd.date_range("2024-01-02", periods=6, freq="D"),
            REFERENCE_PRODUCT_COL: ["croissant"] * 6,
            REFERENCE_TARGET_COL: train_targets + valid_targets,
        }
    )
    scored_reference_test = pd.DataFrame(
        {
            REFERENCE_DATE_COL: pd.date_range("2024-01-08", periods=2, freq="D"),
            REFERENCE_PRODUCT_COL: ["croissant"] * 2,
            REFERENCE_TARGET_COL: test_targets,
        }
    )
    return LoadedEvaluationFrames(
        evaluation_mode="bakery_reference_overlap",
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        history_reference=history_reference,
        scored_reference_test=scored_reference_test,
        overlap_metadata={
            "reference_test_rows": 2,
            "overlap_test_rows": 2,
            "missing_reference_test_rows": 0,
            "overlap_start_date": "2024-01-08",
            "overlap_end_date": "2024-01-09",
            "product_count": 1,
        },
    )


class EvaluationOrchestratorContextTests(unittest.TestCase):
    def test_evaluation_build_request_uses_current_bakery_sample_defaults(self) -> None:
        request = EvaluationBuildRequest()

        self.assertEqual(request.train_sample_fraction, 0.05)
        self.assertEqual(request.tuning_sample_fraction, 0.05)

    def test_load_evaluation_frames_routes_bakery_reference_mode_with_current_defaults(self) -> None:
        request = EvaluationBuildRequest()
        expected_loaded = _build_bakery_loaded_frames()
        logger = logging.getLogger(__name__)

        with patch(
            "praedixa.demand_forecast.evaluation.orchestrator_context._load_gold_evaluation_frames",
            return_value=expected_loaded,
        ) as mocked_load_gold:
            loaded = load_evaluation_frames(
                train_selection_input_path=None,
                train_tuning_input_path=None,
                val_input_path=None,
                requested_target_col=DEFAULT_VARIATION_TARGET_COL,
                duckdb_path=request.duckdb_path,
                gold_table=request.gold_table,
                train_sample_fraction=request.train_sample_fraction,
                tuning_sample_fraction=request.tuning_sample_fraction,
                logger=logger,
            )

        self.assertIs(loaded, expected_loaded)
        mocked_load_gold.assert_called_once_with(
            duckdb_path=request.duckdb_path,
            gold_table=request.gold_table,
            train_sample_fraction=0.05,
            tuning_sample_fraction=0.05,
        )

    def test_prepare_evaluation_context_accepts_large_gpu_best_params_for_bakery_overlap(self) -> None:
        loaded = _build_bakery_loaded_frames()
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            best_params_path = Path(temp_dir) / "best_optuna_params.json"
            best_params_path.write_text(
                json.dumps(
                    {
                        "runtime_profile": "scaleway_l40s",
                        "batch_size": 8192,
                        "learning_rate": 0.04,
                        "max_encoder_length": 28,
                    }
                ),
                encoding="utf-8",
            )
            context = prepare_evaluation_context(
                loaded=loaded,
                requested_target_col=DEFAULT_VARIATION_TARGET_COL,
                best_params_path=best_params_path,
                logger=logger,
            )

        self.assertEqual(context.evaluation_mode, "bakery_reference_overlap")
        self.assertEqual(context.best_params["runtime_profile"], "scaleway_l40s")
        self.assertEqual(context.best_params["batch_size"], 8192)
        self.assertEqual(context.target_contract.learning_target_col, DEFAULT_VARIATION_TARGET_COL)
        self.assertEqual(context.target_contract.absolute_target_col, "target_demand_qty_d_plus_1")
        self.assertEqual(context.dropped_test_rows, 0)
        self.assertIn("avg_selling_price", context.feature_cols)
        self.assertIn("weather_humidity", context.missing_in_test_feature_cols)
        self.assertNotIn("weather_humidity", context.feature_cols)
        self.assertIn("event_count_local", context.feature_cols)
        self.assertEqual(context.test_frame["event_count_local"].tolist(), [2.5, 7.0])
        self.assertIsNotNone(context.overlap_metadata)
        assert context.overlap_metadata is not None
        self.assertEqual(context.overlap_metadata["scorable_test_rows"], 2)


if __name__ == "__main__":
    unittest.main()

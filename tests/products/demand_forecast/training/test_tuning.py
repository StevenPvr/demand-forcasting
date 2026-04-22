from pathlib import Path
import sys
from typing import Any, cast
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.contracts.targets import resolve_target_contract  # noqa: E402
from praedixa.demand_forecast.training.tuning import (  # noqa: E402
    fit_and_score_tft_model_on_tuning,
    optimize_tft_model_params,
    resolve_hpo_execution_policy,
)
from praedixa.demand_forecast.training.tuning_scoring import (  # noqa: E402
    filter_predictable_validation_rows,
    _resolve_fold_preparation_workers,
)


class TuningPolicyTests(unittest.TestCase):
    def _build_target_contract(self) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
        train_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a"],
                "dataset_source": ["source_a", "source_a", "source_a"],
                "dt": pd.date_range("2024-01-01", periods=3, freq="D"),
                "target_demand_qty_d_plus_1": [10.0, 11.0, 12.0],
            }
        )
        tuning_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a"],
                "dataset_source": ["source_a", "source_a", "source_a"],
                "dt": pd.date_range("2024-01-04", periods=3, freq="D"),
                "target_demand_qty_d_plus_1": [13.0, 14.0, 15.0],
            }
        )
        target_contract = resolve_target_contract(
            train_frame,
            tuning_frame,
            requested_target_col="target_demand_qty_d_plus_1",
        )
        return train_frame, tuning_frame, target_contract

    def test_gpu_runtime_profile_forces_single_fold_worker(self) -> None:
        plan = resolve_hpo_execution_policy(
            folds=[{"fold": 0}, {"fold": 1}, {"fold": 2}],
            model_params={"runtime_profile": "scaleway_l40s", "n_jobs": 8},
            total_threads=8,
            logger=__import__("logging").getLogger(__name__),
        )

        self.assertEqual(plan["accelerator"], "gpu")
        self.assertTrue(plan["gpu_safe_mode"])
        self.assertEqual(plan["fold_workers"], 1)
        self.assertEqual(plan["threads_per_fold"], 8)

    def test_mac_metal_runtime_profile_forces_single_fold_worker(self) -> None:
        plan = resolve_hpo_execution_policy(
            folds=[{"fold": 0}, {"fold": 1}],
            model_params={"runtime_profile": "mac_metal", "n_jobs": 6},
            total_threads=6,
            logger=__import__("logging").getLogger(__name__),
        )

        self.assertEqual(plan["runtime_profile"], "mac_metal")
        self.assertIn(plan["accelerator"], {"mps", "cpu"})
        self.assertEqual(plan["fold_workers"], 1)
        self.assertEqual(plan["threads_per_fold"], 6)

    def test_gpu_runtime_profile_forces_single_fold_preparation_worker(self) -> None:
        prep_workers = _resolve_fold_preparation_workers(
            folds=[{"fold": 0}, {"fold": 1}],
            execution_plan={"total_threads": 8, "gpu_safe_mode": True},
        )

        self.assertEqual(prep_workers, 1)

    def test_optuna_hpo_returns_stage_and_runtime_metadata(self) -> None:
        train_frame, tuning_frame, target_contract = self._build_target_contract()

        def _fake_fit_and_score(**kwargs: object) -> dict[str, object]:
            model_params = cast(dict[str, Any], kwargs["model_params"])
            max_epochs = int(model_params["max_epochs"])
            return {
                "macro_mean_wape": max(0.05, 1.0 / float(max_epochs)),
                "dataset_mean_wape": {"source_a": max(0.05, 1.0 / float(max_epochs))},
                "mean_abs_bias": 0.05,
                "mean_coverage_80": 0.85,
                "mean_coverage_95": 0.95,
                "fold_results": [],
                "folds_completed": len(cast(list[dict[str, object]], kwargs["folds"])),
            }

        with patch(
            "praedixa.demand_forecast.training.tuning.fit_and_score_tft_model_on_tuning",
            side_effect=_fake_fit_and_score,
        ):
            best_params, tuning_report, hpo_metadata = optimize_tft_model_params(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[{"fold": 0, "train_idx": [0], "valid_idx": [1]}],
                feature_cols=["series_id"],
                baseline_wape=0.5,
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                tuning_trials=4,
                model_params={"runtime_profile": "scaleway_l40s", "n_jobs": 8},
            )
        typed_hpo_metadata = cast(dict[str, Any], hpo_metadata)
        execution_policy = cast(dict[str, Any], typed_hpo_metadata["execution_policy"])
        pruner = cast(dict[str, Any], typed_hpo_metadata["pruner"])
        stage_policy = cast(dict[str, Any], typed_hpo_metadata["stage_policy"])

        self.assertEqual(best_params["runtime_profile"], "scaleway_l40s")
        self.assertIn("stage_name", tuning_report.columns)
        self.assertIn("trial_status", tuning_report.columns)
        self.assertIn("trial_duration_seconds", tuning_report.columns)
        self.assertIn("mean_abs_bias", tuning_report.columns)
        self.assertIn("coverage_80", tuning_report.columns)
        self.assertIn("coverage_95", tuning_report.columns)
        self.assertEqual(execution_policy["accelerator"], "gpu")
        self.assertEqual(execution_policy["fold_workers"], 1)
        self.assertEqual(pruner["type"], "MedianPruner")
        self.assertIn("stage_a", stage_policy)
        self.assertIn("stage_b", stage_policy)

    def test_fit_and_score_keeps_dt_and_series_id_for_tft_folds(self) -> None:
        train_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a", "a"],
                "dataset_source": ["source_a"] * 4,
                "dt": pd.date_range("2024-01-01", periods=4, freq="D"),
                "rolling_mean_7": [1.0, 2.0, 3.0, 4.0],
                "target_demand_qty_d_plus_1": [10.0, 11.0, 12.0, 13.0],
            }
        )
        tuning_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a", "a"],
                "dataset_source": ["source_a"] * 4,
                "dt": pd.date_range("2024-01-05", periods=4, freq="D"),
                "rolling_mean_7": [5.0, 6.0, 7.0, 8.0],
                "target_demand_qty_d_plus_1": [14.0, 15.0, 16.0, 17.0],
            }
        )
        target_contract = resolve_target_contract(
            train_frame,
            tuning_frame,
            requested_target_col="target_demand_qty_d_plus_1",
        )

        def _fake_fit_tft_model(
            fold_train_frame: pd.DataFrame,
            feature_cols: list[str],
            *,
            target_col: str,
            valid_weights: np.ndarray | None = None,
            **_: object,
        ) -> object:
            self.assertIn("dt", fold_train_frame.columns)
            self.assertIn("series_id", fold_train_frame.columns)
            self.assertEqual(feature_cols, ["rolling_mean_7"])
            self.assertEqual(target_col, "target_demand_qty_d_plus_1")
            self.assertIsNotNone(valid_weights)
            self.assertEqual(len(cast(np.ndarray, valid_weights)), 2)
            return object()

        with (
            patch(
                "praedixa.demand_forecast.training.tuning_scoring.fit_tft_model",
                side_effect=_fake_fit_tft_model,
            ),
            patch(
                "praedixa.demand_forecast.training.tuning_scoring.predict_with_tft_model",
                return_value=np.array([15.0, 16.0], dtype=float),
            ),
            patch(
                "praedixa.demand_forecast.training.tuning_scoring.predict_quantiles_with_tft_model",
                return_value=pd.DataFrame(
                    {
                        "prediction_p2_5": [14.0, 15.0],
                        "prediction_p10": [14.5, 15.5],
                        "prediction_p50": [15.0, 16.0],
                        "prediction_p90": [15.5, 16.5],
                        "prediction_p97_5": [16.0, 17.0],
                    }
                ),
            ),
        ):
            result = fit_and_score_tft_model_on_tuning(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[{"fold": 0, "train_idx": [0, 1], "valid_idx": [2, 3]}],
                feature_cols=["rolling_mean_7"],
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                model_params={"runtime_profile": "local_cpu", "n_jobs": 2, "max_encoder_length": 1},
            )

        self.assertEqual(result["folds_completed"], 1)
        self.assertIn("macro_mean_wape", result)
        self.assertIn("mean_abs_bias", result)
        self.assertIn("mean_coverage_80", result)
        self.assertIn("mean_coverage_95", result)

    def test_fit_and_score_caches_fold_dataset_artifacts_by_encoder_length(self) -> None:
        train_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a", "a"],
                "location_id": ["loc_1"] * 4,
                "product_id": ["sku_1"] * 4,
                "dataset_source": ["source_a"] * 4,
                "dt": pd.date_range("2024-01-01", periods=4, freq="D"),
                "rolling_mean_7": [1.0, 2.0, 3.0, 4.0],
                "target_demand_qty_d_plus_1": [10.0, 11.0, 12.0, 13.0],
            }
        )
        tuning_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a", "a"],
                "location_id": ["loc_1"] * 4,
                "product_id": ["sku_1"] * 4,
                "dataset_source": ["source_a"] * 4,
                "dt": pd.date_range("2024-01-05", periods=4, freq="D"),
                "rolling_mean_7": [5.0, 6.0, 7.0, 8.0],
                "target_demand_qty_d_plus_1": [14.0, 15.0, 16.0, 17.0],
            }
        )
        target_contract = resolve_target_contract(
            train_frame,
            tuning_frame,
            requested_target_col="target_demand_qty_d_plus_1",
        )

        with (
            patch(
                "praedixa.demand_forecast.training.tuning_scoring.build_training_dataset_artifacts",
                return_value=cast(Any, object()),
            ) as mocked_build_dataset_artifacts,
            patch(
                "praedixa.demand_forecast.training.tuning_scoring.fit_tft_model",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.training.tuning_scoring.predict_with_tft_model",
                return_value=np.array([15.0, 16.0], dtype=float),
            ),
            patch(
                "praedixa.demand_forecast.training.tuning_scoring.predict_quantiles_with_tft_model",
                return_value=pd.DataFrame(
                    {
                        "prediction_p2_5": [14.0, 15.0],
                        "prediction_p10": [14.5, 15.5],
                        "prediction_p50": [15.0, 16.0],
                        "prediction_p90": [15.5, 16.5],
                        "prediction_p97_5": [16.0, 17.0],
                    }
                ),
            ),
        ):
            fit_and_score_tft_model_on_tuning(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[{"fold": 0, "train_idx": [0, 1], "valid_idx": [2, 3]}],
                feature_cols=["rolling_mean_7"],
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                model_params={"runtime_profile": "local_cpu", "n_jobs": 2, "max_encoder_length": 1},
            )
            fit_and_score_tft_model_on_tuning(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[{"fold": 0, "train_idx": [0, 1], "valid_idx": [2, 3]}],
                feature_cols=["rolling_mean_7"],
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                model_params={"runtime_profile": "local_cpu", "n_jobs": 2, "max_encoder_length": 1},
            )

        self.assertEqual(mocked_build_dataset_artifacts.call_count, 1)

    def test_filter_predictable_validation_rows_drops_insufficient_history(self) -> None:
        fold_train_frame = pd.DataFrame(
            {
                "series_id": ["series_1"] * 7,
                "location_id": ["store_1"] * 7,
                "product_id": ["sku_1"] * 7,
                "dt": pd.date_range("2024-01-01", periods=7, freq="D"),
            }
        )
        fold_valid_frame = pd.DataFrame(
            {
                "series_id": ["series_1", "series_2"],
                "location_id": ["store_1", "store_2"],
                "product_id": ["sku_1", "sku_2"],
                "dt": pd.to_datetime(["2024-01-08", "2024-01-08"]),
            }
        )

        filtered_frame, filtered_indices = filter_predictable_validation_rows(
            fold_train_frame=fold_train_frame,
            fold_valid_frame=fold_valid_frame,
            valid_indices=np.asarray([10, 11], dtype=np.int32),
            max_encoder_length=7,
        )

        self.assertEqual(len(filtered_frame), 1)
        self.assertEqual(filtered_frame.iloc[0]["location_id"], "store_1")
        np.testing.assert_array_equal(filtered_indices, np.asarray([10], dtype=np.int32))

    def test_filter_predictable_validation_rows_handles_multiple_groups_without_full_history_scan(self) -> None:
        fold_train_frame = pd.DataFrame(
            {
                "series_id": ["series_1"] * 7 + ["series_2"] * 3,
                "location_id": ["store_1"] * 7 + ["store_2"] * 3,
                "product_id": ["sku_1"] * 7 + ["sku_2"] * 3,
                "dt": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                        "2024-01-06",
                        "2024-01-07",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                    ]
                ),
            }
        )
        fold_valid_frame = pd.DataFrame(
            {
                "series_id": ["series_1", "series_2", "series_1"],
                "location_id": ["store_1", "store_2", "store_1"],
                "product_id": ["sku_1", "sku_2", "sku_1"],
                "dt": pd.to_datetime(["2024-01-08", "2024-01-04", "2024-01-09"]),
            }
        )

        filtered_frame, filtered_indices = filter_predictable_validation_rows(
            fold_train_frame=fold_train_frame,
            fold_valid_frame=fold_valid_frame,
            valid_indices=np.asarray([20, 21, 22], dtype=np.int32),
            max_encoder_length=7,
        )

        self.assertEqual(filtered_frame["series_id"].tolist(), ["series_1", "series_1"])
        np.testing.assert_array_equal(filtered_indices, np.asarray([20, 22], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()

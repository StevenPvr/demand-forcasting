from pathlib import Path
import sys
from typing import Any, cast
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
if str(PLATFORM_SRC) not in sys.path:
    sys.path.insert(0, str(PLATFORM_SRC))
if str(PRODUCT_SRC) not in sys.path:
    sys.path.insert(0, str(PRODUCT_SRC))

from praedixa.demand_forecast.contracts.targets import resolve_target_contract  # noqa: E402
from praedixa.demand_forecast.training.tft.tuning import (  # noqa: E402
    fit_and_score_tft_model_on_tuning,
    optimize_tft_model_params,
    resolve_hpo_execution_policy,
)
from praedixa.demand_forecast.training.tft.policy import (  # noqa: E402
    resolved_trial_status_name,
)
import praedixa.demand_forecast.training.tft.scoring as tuning_scoring_module  # noqa: E402
from praedixa.demand_forecast.training.tft.scoring import (  # noqa: E402
    clear_tuning_fold_caches,
    filter_predictable_validation_rows,
    prewarm_tft_fold_cores_for_optuna,
)


class TuningPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_tuning_fold_caches()

    def tearDown(self) -> None:
        clear_tuning_fold_caches()

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

    def test_local_cpu_runtime_profile_forces_single_fold_worker_on_macos(self) -> None:
        with patch(
            "praedixa.demand_forecast.training.tft.policy.should_serialize_local_cpu_folds",
            return_value=True,
        ):
            plan = resolve_hpo_execution_policy(
                folds=[{"fold": 0}, {"fold": 1}],
                model_params={"runtime_profile": "local_cpu", "n_jobs": 6},
                total_threads=6,
                logger=__import__("logging").getLogger(__name__),
            )

        self.assertEqual(plan["runtime_profile"], "local_cpu")
        self.assertEqual(plan["accelerator"], "cpu")
        self.assertTrue(plan["gpu_safe_mode"])
        self.assertEqual(plan["fold_workers"], 1)
        self.assertEqual(plan["threads_per_fold"], 6)

    def test_resolved_trial_status_name_supports_live_trials(self) -> None:
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        self.assertEqual(
            resolved_trial_status_name(trial, default_status="PRUNED"), "PRUNED"
        )

        trial.set_user_attr("terminal_status", "REJECTED")
        self.assertEqual(
            resolved_trial_status_name(trial, default_status="PRUNED"), "REJECTED"
        )

    def test_fold_preparation_workers_are_runtime_agnostic_before_fit(self) -> None:
        prep_workers = cast(
            Any, tuning_scoring_module
        )._resolve_fold_preparation_workers(
            folds=[{"fold": 0}, {"fold": 1}],
            execution_plan={"total_threads": 8, "gpu_safe_mode": True},
        )

        self.assertEqual(prep_workers, 1)

    def test_hpo_resolved_inputs_apply_gpu_runtime_profile_defaults(self) -> None:
        train_frame, tuning_frame, target_contract = self._build_target_contract()

        resolved_params, execution_plan, _ = cast(
            Any, tuning_scoring_module
        )._resolved_tft_tuning_inputs(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            folds=[{"fold": 0, "train_idx": [0], "valid_idx": [1]}],
            feature_cols=["series_id"],
            target_contract=target_contract,
            logger=__import__("logging").getLogger(__name__),
            model_params={
                "runtime_profile": "scaleway_l40s",
                "n_jobs": 8,
                "max_encoder_length": 14,
            },
            total_threads=8,
            target_transform="identity",
            hpo_mode=True,
        )

        self.assertEqual(resolved_params["runtime_profile"], "scaleway_l40s")
        self.assertEqual(resolved_params["accelerator"], "gpu")
        self.assertEqual(resolved_params["determinism_mode"], "off")
        self.assertFalse(bool(resolved_params["use_learning_rate_finder"]))
        self.assertTrue(bool(resolved_params["enable_progress_bar"]))
        self.assertFalse(bool(resolved_params["enable_csv_logger"]))
        self.assertFalse(bool(resolved_params["enable_lr_monitor"]))
        self.assertFalse(bool(resolved_params["enable_validation_metric_logging"]))
        self.assertFalse(bool(resolved_params["enable_device_stats_monitor"]))
        self.assertEqual(int(resolved_params["log_every_n_steps"]), 50)
        self.assertEqual(resolved_params["dataset_core_max_encoder_length"], 14)
        self.assertEqual(execution_plan["fold_workers"], 1)

    def test_prewarm_skips_when_encoder_length_is_variable_across_trials(self) -> None:
        train_frame, tuning_frame, target_contract = self._build_target_contract()

        with patch(
            "praedixa.demand_forecast.training.tft.scoring.build_training_dataset_core",
        ) as mocked_build_training_core:
            result = prewarm_tft_fold_cores_for_optuna(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[{"fold": 0, "train_idx": [0], "valid_idx": [1]}],
                feature_cols=["series_id"],
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                model_params={"runtime_profile": "scaleway_l40s", "n_jobs": 8},
            )

        self.assertEqual(result["cached_cores"], 0)
        self.assertEqual(result["core_max_encoder_length"], None)
        self.assertEqual(
            cast(dict[str, Any], result["execution_policy"])["reason"],
            "variable_encoder_length",
        )
        mocked_build_training_core.assert_not_called()

    def test_prewarm_skips_multi_fold_sequential_execution_in_memory_safe_mode(
        self,
    ) -> None:
        train_frame, tuning_frame, target_contract = self._build_target_contract()

        with patch(
            "praedixa.demand_forecast.training.tft.scoring.build_training_dataset_core",
        ) as mocked_build_training_core:
            result = prewarm_tft_fold_cores_for_optuna(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[
                    {"fold": 0, "train_idx": [0], "valid_idx": [1]},
                    {"fold": 1, "train_idx": [0, 1], "valid_idx": [2]},
                ],
                feature_cols=["series_id"],
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                model_params={
                    "runtime_profile": "scaleway_l40s",
                    "n_jobs": 8,
                    "dataset_core_max_encoder_length": 14,
                    "max_encoder_length": 14,
                },
            )

        self.assertEqual(result["cached_cores"], 0)
        self.assertEqual(
            cast(dict[str, Any], result["execution_policy"])["reason"],
            "memory_safe_sequential_folds",
        )
        mocked_build_training_core.assert_not_called()

    def test_fit_and_score_uses_memory_safe_path_for_multi_fold_serial_execution(
        self,
    ) -> None:
        train_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a"],
                "location_id": ["loc_1"] * 3,
                "product_id": ["sku_1"] * 3,
                "dataset_source": ["source_a"] * 3,
                "dt": pd.date_range("2024-01-01", periods=3, freq="D"),
                "rolling_mean_7": [1.0, 2.0, 3.0],
                "target_demand_qty_d_plus_1": [10.0, 11.0, 12.0],
            }
        )
        tuning_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a"],
                "location_id": ["loc_1"] * 3,
                "product_id": ["sku_1"] * 3,
                "dataset_source": ["source_a"] * 3,
                "dt": pd.date_range("2024-01-04", periods=3, freq="D"),
                "rolling_mean_7": [4.0, 5.0, 6.0],
                "target_demand_qty_d_plus_1": [13.0, 14.0, 15.0],
            }
        )
        target_contract = resolve_target_contract(
            train_frame,
            tuning_frame,
            requested_target_col="target_demand_qty_d_plus_1",
        )

        with (
            patch(
                "praedixa.demand_forecast.training.tft.scoring._cached_fold_artifacts"
            ) as mocked_cached_fold_artifacts,
            patch(
                "praedixa.demand_forecast.training.tft.scoring._iterative_trial_scoring_memory_safe",
                return_value=([], 2, 0.001),
            ) as mocked_memory_safe_scoring,
        ):
            result = fit_and_score_tft_model_on_tuning(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[
                    {"fold": 0, "train_idx": [0], "valid_idx": [1]},
                    {"fold": 1, "train_idx": [0, 1], "valid_idx": [2]},
                ],
                feature_cols=["rolling_mean_7"],
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                model_params={
                    "runtime_profile": "scaleway_l40s",
                    "n_jobs": 8,
                    "max_encoder_length": 1,
                },
            )

        mocked_cached_fold_artifacts.assert_not_called()
        mocked_memory_safe_scoring.assert_called_once()
        self.assertEqual(result["folds_completed"], 2)
        self.assertEqual(float(cast(Any, result["selected_learning_rate"])), 0.001)

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
                "selected_learning_rate": 0.003,
                "fold_results": [],
                "folds_completed": len(cast(list[dict[str, object]], kwargs["folds"])),
            }

        with (
            patch(
                "praedixa.demand_forecast.training.tft.tuning.fit_and_score_tft_model_on_tuning",
                side_effect=_fake_fit_and_score,
            ),
            patch(
                "praedixa.demand_forecast.training.tft.tuning.prewarm_tft_fold_cores_for_optuna",
                return_value={},
            ) as mocked_prewarm,
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
        self.assertIn("selected_learning_rate", tuning_report.columns)
        self.assertEqual(float(cast(Any, best_params["learning_rate"])), 0.003)
        self.assertFalse(bool(best_params["use_learning_rate_finder"]))
        self.assertEqual(execution_policy["accelerator"], "gpu")
        self.assertEqual(execution_policy["fold_workers"], 1)
        self.assertEqual(pruner["type"], "MedianPruner")
        self.assertIn("stage_a", stage_policy)
        self.assertIn("stage_b", stage_policy)
        mocked_prewarm.assert_called_once()

    def test_optuna_hpo_refuses_to_promote_rejected_trials(
        self,
    ) -> None:
        train_frame, tuning_frame, target_contract = self._build_target_contract()

        def _always_rejected(**_: object) -> dict[str, object]:
            return {
                "macro_mean_wape": 1.2,
                "dataset_mean_wape": {"source_a": 1.2},
                "mean_abs_bias": 0.1,
                "mean_coverage_80": 0.8,
                "mean_coverage_95": 0.95,
                "selected_learning_rate": 0.0025,
                "fold_results": [],
                "folds_completed": 1,
            }

        with (
            patch(
                "praedixa.demand_forecast.training.tft.tuning.fit_and_score_tft_model_on_tuning",
                side_effect=_always_rejected,
            ),
            patch(
                "praedixa.demand_forecast.training.tft.tuning.prewarm_tft_fold_cores_for_optuna",
                return_value={},
            ) as mocked_prewarm,
        ):
            with self.assertRaises(RuntimeError):
                optimize_tft_model_params(
                    train_frame=train_frame,
                    tuning_frame=tuning_frame,
                    folds=[{"fold": 0, "train_idx": [0], "valid_idx": [1]}],
                    feature_cols=["series_id"],
                    baseline_wape=0.5,
                    target_contract=target_contract,
                    logger=__import__("logging").getLogger(__name__),
                    tuning_trials=2,
                    random_seed=7,
                    model_params={"runtime_profile": "local_cpu", "n_jobs": 2},
                )

        mocked_prewarm.assert_called_once()

    def test_fold_completion_log_reports_business_metrics(self) -> None:
        logger = __import__("logging").getLogger("praedixa.test.fold_metrics")

        with self.assertLogs(logger, level="INFO") as captured:
            cast(Any, tuning_scoring_module)._log_fold_completion(
                logger=logger,
                fold_number=1,
                total_folds=2,
                dataset_mean_wape={"source_a": 0.4, "source_b": 0.6},
                mean_abs_bias=0.2,
                mean_coverage_80=0.75,
                mean_coverage_95=0.9,
            )

        joined_output = "\n".join(captured.output)
        self.assertIn("business_macro_wape=0.500000", joined_output)
        self.assertIn("business_mean_abs_bias=0.200000", joined_output)
        self.assertIn("business_mean_coverage_80=0.750000", joined_output)
        self.assertIn("business_mean_coverage_95=0.900000", joined_output)
        self.assertIn("dataset_business_mean_wape=", joined_output)

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
                "praedixa.demand_forecast.training.tft.scoring.fit_tft_model",
                side_effect=_fake_fit_tft_model,
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.predict_with_tft_model",
                return_value=np.array([15.0, 16.0], dtype=float),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.predict_quantiles_with_tft_model",
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
                model_params={
                    "runtime_profile": "local_cpu",
                    "n_jobs": 2,
                    "max_encoder_length": 1,
                },
            )

        self.assertEqual(result["folds_completed"], 1)
        self.assertIn("macro_mean_wape", result)
        self.assertIn("mean_abs_bias", result)
        self.assertIn("mean_coverage_80", result)
        self.assertIn("mean_coverage_95", result)

    def test_fit_and_score_filters_unusable_tuning_rows_from_fold_train(self) -> None:
        train_frame = pd.DataFrame(
            {
                "series_id": ["a", "a"],
                "location_id": ["loc_1"] * 2,
                "product_id": ["sku_1"] * 2,
                "dataset_source": ["source_a"] * 2,
                "dt": pd.date_range("2024-01-01", periods=2, freq="D"),
                "rolling_mean_7": [1.0, 2.0],
                "target_demand_qty_d_plus_1": [10.0, 11.0],
                "usable_for_training_flag": [True, True],
                "censor_flag": [False, False],
                "label_quality_score": [1.0, 1.0],
            }
        )
        tuning_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a"],
                "location_id": ["loc_1"] * 3,
                "product_id": ["sku_1"] * 3,
                "dataset_source": ["source_a"] * 3,
                "dt": pd.date_range("2024-01-03", periods=3, freq="D"),
                "rolling_mean_7": [3.0, 4.0, 5.0],
                "target_demand_qty_d_plus_1": [12.0, 13.0, 14.0],
                "usable_for_training_flag": [False, True, True],
                "censor_flag": [False, False, False],
                "label_quality_score": [1.0, 1.0, 1.0],
            }
        )
        target_contract = resolve_target_contract(
            train_frame,
            tuning_frame,
            requested_target_col="target_demand_qty_d_plus_1",
        )
        training_targets_seen: list[float] = []

        def _fake_build_training_dataset_core(
            *_args: object,
            **kwargs: object,
        ) -> object:
            training_frame = cast(pd.DataFrame, kwargs["training_frame"])
            training_targets_seen.extend(
                training_frame["target_demand_qty_d_plus_1"].astype(float).tolist()
            )
            return object()

        with (
            patch(
                "praedixa.demand_forecast.training.tft.scoring.build_training_dataset_core",
                side_effect=_fake_build_training_dataset_core,
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.build_validation_dataset_core",
                return_value=cast(Any, object()),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.clone_training_dataset_from_core",
                return_value=cast(Any, object()),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.clone_validation_dataset_from_core",
                return_value=cast(Any, object()),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.fit_tft_model",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.predict_with_tft_model",
                return_value=np.array([14.0], dtype=float),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.predict_quantiles_with_tft_model",
                return_value=pd.DataFrame({"prediction_p50": [14.0]}),
            ),
        ):
            fit_and_score_tft_model_on_tuning(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[{"fold": 0, "train_idx": [0, 1], "valid_idx": [2]}],
                feature_cols=["rolling_mean_7"],
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                model_params={
                    "runtime_profile": "local_cpu",
                    "n_jobs": 2,
                    "max_encoder_length": 1,
                    "dataset_core_max_encoder_length": 1,
                },
            )

        self.assertIn(13.0, training_targets_seen)
        self.assertNotIn(12.0, training_targets_seen)

    def test_fit_and_score_reuses_fold_cores_across_encoder_lengths(self) -> None:
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
                "praedixa.demand_forecast.training.tft.scoring.build_training_dataset_core",
                return_value=cast(Any, object()),
            ) as mocked_build_training_core,
            patch(
                "praedixa.demand_forecast.training.tft.scoring.build_validation_dataset_core",
                return_value=cast(Any, object()),
            ) as mocked_build_validation_core,
            patch(
                "praedixa.demand_forecast.training.tft.scoring.clone_training_dataset_from_core",
                return_value=cast(Any, object()),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.clone_validation_dataset_from_core",
                return_value=cast(Any, object()),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.fit_tft_model",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.predict_with_tft_model",
                return_value=np.array([15.0, 16.0], dtype=float),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.predict_quantiles_with_tft_model",
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
                model_params={
                    "runtime_profile": "local_cpu",
                    "n_jobs": 2,
                    "max_encoder_length": 1,
                    "dataset_core_max_encoder_length": 3,
                },
            )
            fit_and_score_tft_model_on_tuning(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[{"fold": 0, "train_idx": [0, 1], "valid_idx": [2, 3]}],
                feature_cols=["rolling_mean_7"],
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                model_params={
                    "runtime_profile": "local_cpu",
                    "n_jobs": 2,
                    "max_encoder_length": 2,
                    "dataset_core_max_encoder_length": 3,
                },
            )

        self.assertEqual(mocked_build_training_core.call_count, 1)
        self.assertEqual(mocked_build_validation_core.call_count, 2)

    def test_prewarm_tft_fold_cores_moves_core_build_before_first_trial(self) -> None:
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
                "praedixa.demand_forecast.training.tft.scoring.build_training_dataset_core",
                return_value=cast(Any, object()),
            ) as mocked_build_training_core,
            patch(
                "praedixa.demand_forecast.training.tft.scoring.build_validation_dataset_core",
                return_value=cast(Any, object()),
            ) as mocked_build_validation_core,
            patch(
                "praedixa.demand_forecast.training.tft.scoring.clone_training_dataset_from_core",
                return_value=cast(Any, object()),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.clone_validation_dataset_from_core",
                return_value=cast(Any, object()),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.fit_tft_model",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.predict_with_tft_model",
                return_value=np.array([15.0, 16.0], dtype=float),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.predict_quantiles_with_tft_model",
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
            prewarm_tft_fold_cores_for_optuna(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[{"fold": 0, "train_idx": [0, 1], "valid_idx": [2, 3]}],
                feature_cols=["rolling_mean_7"],
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                model_params={
                    "runtime_profile": "local_cpu",
                    "n_jobs": 2,
                    "dataset_core_max_encoder_length": 3,
                },
            )
            fit_and_score_tft_model_on_tuning(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                folds=[{"fold": 0, "train_idx": [0, 1], "valid_idx": [2, 3]}],
                feature_cols=["rolling_mean_7"],
                target_contract=target_contract,
                logger=__import__("logging").getLogger(__name__),
                model_params={
                    "runtime_profile": "local_cpu",
                    "n_jobs": 2,
                    "max_encoder_length": 1,
                    "dataset_core_max_encoder_length": 3,
                },
            )

        self.assertEqual(mocked_build_training_core.call_count, 1)
        self.assertEqual(mocked_build_validation_core.call_count, 1)

    def test_prepare_fold_frame_recomputes_group_and_time_when_support_columns_are_mixed(
        self,
    ) -> None:
        train_frame = pd.DataFrame(
            {
                "series_id": ["a", "a"],
                "dataset_source": ["source_a", "source_a"],
                "dt": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "target_demand_qty_d_plus_1": [10.0, 11.0],
            }
        )
        valid_frame = pd.DataFrame(
            {
                "series_id": ["a", "a"],
                "dataset_source": ["source_a", "source_a"],
                "dt": pd.to_datetime(["2024-01-03", "2024-01-04"]),
                "target_demand_qty_d_plus_1": [12.0, 13.0],
                tuning_scoring_module.GROUP_COL: ["a", "a"],
                tuning_scoring_module.TIME_IDX_COL: [17, 18],
            }
        )

        prepared = cast(Any, tuning_scoring_module)._prepare_fold_frame(
            fold_number=1,
            stage_name="validation_core",
            train_frame=train_frame,
            valid_frame=valid_frame,
            train_weights=np.ones(len(train_frame), dtype=float),
            valid_weights=np.ones(len(valid_frame), dtype=float),
            feature_cols=[],
            logger=__import__("logging").getLogger(__name__),
        )

        self.assertTrue(prepared[tuning_scoring_module.TIME_IDX_COL].notna().all())
        self.assertEqual(
            str(prepared[tuning_scoring_module.TIME_IDX_COL].dtype), "int32"
        )
        np.testing.assert_array_equal(
            prepared[tuning_scoring_module.TIME_IDX_COL].to_numpy(),
            np.array([0, 1, 2, 3], dtype=np.int32),
        )
        self.assertEqual(
            prepared[tuning_scoring_module.GROUP_COL].astype("string").tolist(),
            ["a", "a", "a", "a"],
        )
        self.assertEqual(
            prepared[tuning_scoring_module.SPLIT_COL].tolist(),
            ["train", "train", "valid", "valid"],
        )

    def test_prepare_fold_frame_raises_when_polars_contract_path_fails(self) -> None:
        train_frame = pd.DataFrame(
            {
                "series_id": ["a", "a"],
                "dataset_source": ["source_a", "source_a"],
                "dt": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "target_demand_qty_d_plus_1": [10.0, 11.0],
                tuning_scoring_module.GROUP_COL: ["a", "a"],
                tuning_scoring_module.TIME_IDX_COL: [0, 1],
            }
        )

        with patch(
            "praedixa.demand_forecast.training.tft.scoring._build_combined_frame_with_polars",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Polars fold frame preparation failed"
            ):
                cast(Any, tuning_scoring_module)._prepare_fold_frame(
                    fold_number=1,
                    stage_name="training_core",
                    train_frame=train_frame,
                    valid_frame=None,
                    train_weights=np.ones(len(train_frame), dtype=float),
                    valid_weights=None,
                    feature_cols=[],
                    logger=__import__("logging").getLogger(__name__),
                )

    def test_fit_and_score_uses_absolute_target_quantiles_without_reconstruction(
        self,
    ) -> None:
        train_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a"],
                "location_id": ["loc_1"] * 3,
                "product_id": ["sku_1"] * 3,
                "dataset_source": ["source_a"] * 3,
                "dt": pd.date_range("2024-01-01", periods=3, freq="D"),
                "rolling_mean_7": [1.0, 2.0, 3.0],
                "target_demand_qty_d_plus_1": [10.0, 11.0, 12.0],
            }
        )
        tuning_frame = pd.DataFrame(
            {
                "series_id": ["a", "a", "a", "a"],
                "location_id": ["loc_1"] * 4,
                "product_id": ["sku_1"] * 4,
                "dataset_source": ["source_a"] * 4,
                "dt": pd.date_range("2024-01-04", periods=4, freq="D"),
                "rolling_mean_7": [4.0, 5.0, 6.0, 7.0],
                "target_demand_qty_d_plus_1": [13.0, 14.0, 15.0, 16.0],
            }
        )
        target_contract = resolve_target_contract(
            train_frame,
            tuning_frame,
            requested_target_col="target_demand_qty_d_plus_1",
        )
        valid_raw_predictions = tuning_frame.loc[
            [2, 3], "target_demand_qty_d_plus_1"
        ].to_numpy(dtype=float)

        with (
            patch(
                "praedixa.demand_forecast.training.tft.scoring.fit_tft_model",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.predict_with_tft_model",
                return_value=valid_raw_predictions,
            ),
            patch(
                "praedixa.demand_forecast.training.tft.scoring.predict_quantiles_with_tft_model",
                return_value=pd.DataFrame(
                    {
                        "prediction_p2_5": valid_raw_predictions - 0.03,
                        "prediction_p10": valid_raw_predictions - 0.02,
                        "prediction_p50": valid_raw_predictions,
                        "prediction_p90": valid_raw_predictions + 0.02,
                        "prediction_p97_5": valid_raw_predictions + 0.03,
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
                model_params={
                    "runtime_profile": "local_cpu",
                    "n_jobs": 2,
                    "max_encoder_length": 1,
                },
            )

        self.assertEqual(result["folds_completed"], 1)
        self.assertAlmostEqual(
            float(cast(Any, result["macro_mean_wape"])), 0.0, places=8
        )
        self.assertAlmostEqual(
            float(cast(Any, result["mean_coverage_80"])), 1.0, places=8
        )
        self.assertAlmostEqual(
            float(cast(Any, result["mean_coverage_95"])), 1.0, places=8
        )

    def test_filter_predictable_validation_rows_drops_insufficient_history(
        self,
    ) -> None:
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
        np.testing.assert_array_equal(
            filtered_indices, np.asarray([10], dtype=np.int32)
        )

    def test_filter_predictable_validation_rows_handles_multiple_groups_without_full_history_scan(
        self,
    ) -> None:
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
        np.testing.assert_array_equal(
            filtered_indices, np.asarray([20, 22], dtype=np.int32)
        )


if __name__ == "__main__":
    unittest.main()

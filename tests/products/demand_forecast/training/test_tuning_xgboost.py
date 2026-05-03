from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable, cast
import unittest

import numpy as np
import optuna
import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.contracts.targets import resolve_target_contract  # noqa: E402
from praedixa.demand_forecast.training.validation.folds import (  # noqa: E402
    build_grouped_tuning_walk_forward_folds_by_dataset,
)
from praedixa.demand_forecast.training.config.constants import (  # noqa: E402
    DEFAULT_XGBOOST_MAX_CAT_THRESHOLD,
    DEFAULT_XGBOOST_MAX_CAT_TO_ONEHOT,
    DEFAULT_XGBOOST_TUNING_COLSAMPLE_BYTREE_RANGE,
    DEFAULT_XGBOOST_TUNING_LEARNING_RATE_RANGE,
    DEFAULT_XGBOOST_TUNING_MAX_BIN_CHOICES,
    DEFAULT_XGBOOST_TUNING_MAX_DEPTH_CHOICES,
    DEFAULT_XGBOOST_TUNING_MIN_CHILD_WEIGHT_RANGE,
    DEFAULT_XGBOOST_TUNING_REG_ALPHA_RANGE,
    DEFAULT_XGBOOST_TUNING_REG_LAMBDA_RANGE,
    DEFAULT_XGBOOST_TUNING_SUBSAMPLE_RANGE,
)
import praedixa.demand_forecast.training.xgboost.tuning as xgboost_tuning  # noqa: E402
from praedixa.demand_forecast.training.xgboost.tuning import (  # noqa: E402
    fit_and_score_xgboost_model_on_tuning,
    guardrail_failure_reason,
    sample_xgboost_optuna_params,
)
from praedixa.demand_forecast.training.xgboost.scoring import (  # noqa: E402
    build_xgboost_fold_matrix_cache,
    xgboost_execution_policy,
)
from praedixa.demand_forecast.training.shared.hpo import (  # noqa: E402
    build_hpo_runtime_metadata,
)


ObjectiveScoreFn = Callable[[dict[str, object]], float]
FinalBestParamsFn = Callable[..., dict[str, object]]

OBJECTIVE_SCORE = cast(
    ObjectiveScoreFn,
    getattr(xgboost_tuning, "_objective_score"),
)
FINAL_BEST_PARAMS = cast(
    FinalBestParamsFn,
    getattr(xgboost_tuning, "_final_best_params"),
)


class XGBoostTuningTests(unittest.TestCase):
    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_dates = pd.date_range("2024-01-01", periods=24, freq="D")
        tuning_dates = pd.date_range("2024-01-25", periods=18, freq="D")
        train_signal = np.arange(24, dtype=float)
        tuning_signal = np.arange(24, 42, dtype=float)
        train_frame = pd.DataFrame(
            {
                "dataset_source": ["fixture"] * 24,
                "client_id": ["loc_1__sku_1"] * 24,
                "location_id": ["loc_1"] * 24,
                "product_id": ["sku_1"] * 24,
                "dt": train_dates,
                "target_day_of_week": train_dates.dayofweek.astype(int),
                "target_demand_qty_d_plus_1": 5.0 + train_signal,
                "usable_for_training_flag": [True] * 24,
                "censor_flag": [False] * 24,
                "label_quality_score": [1.0] * 24,
            }
        )
        tuning_frame = pd.DataFrame(
            {
                "dataset_source": ["fixture"] * 18,
                "client_id": ["loc_1__sku_1"] * 18,
                "location_id": ["loc_1"] * 18,
                "product_id": ["sku_1"] * 18,
                "dt": tuning_dates,
                "target_day_of_week": tuning_dates.dayofweek.astype(int),
                "target_demand_qty_d_plus_1": 5.0 + tuning_signal,
                "usable_for_training_flag": [True] * 18,
                "censor_flag": [False] * 18,
                "label_quality_score": [1.0] * 18,
            }
        )
        return train_frame, tuning_frame

    def test_sample_xgboost_optuna_params_returns_xgboost_space(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "learning_rate": 0.04,
                "max_depth": 6,
                "min_child_weight": 64.0,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "reg_alpha": 1e-2,
                "reg_lambda": 20.0,
                "max_bin": 256,
                "max_cat_to_onehot": 16,
                "max_cat_threshold": 128,
            }
        )

        params = sample_xgboost_optuna_params(cast(Any, trial), random_seed=11)

        self.assertNotIn("n_estimators", params)
        self.assertEqual(params["max_depth"], 6)
        self.assertTrue(params["enable_early_stopping"])
        self.assertTrue(params["enable_categorical"])
        self.assertEqual(params["early_stopping_rounds"], 100)
        self.assertEqual(params["max_bin"], 256)
        self.assertEqual(params["max_cat_to_onehot"], 16)
        self.assertEqual(params["max_cat_threshold"], 128)
        self.assertEqual(params["random_state"], 12)

    def test_xgboost_tuning_space_is_expressive_for_fine_metadata_panels(
        self,
    ) -> None:
        self.assertIn(14, DEFAULT_XGBOOST_TUNING_MAX_DEPTH_CHOICES)
        self.assertEqual(DEFAULT_XGBOOST_TUNING_LEARNING_RATE_RANGE, (0.005, 0.12))
        self.assertEqual(DEFAULT_XGBOOST_TUNING_MAX_BIN_CHOICES, (256,))
        self.assertEqual(DEFAULT_XGBOOST_TUNING_MIN_CHILD_WEIGHT_RANGE, (0.5, 64.0))
        self.assertEqual(DEFAULT_XGBOOST_TUNING_REG_ALPHA_RANGE, (1e-8, 10.0))
        self.assertEqual(DEFAULT_XGBOOST_TUNING_REG_LAMBDA_RANGE, (0.1, 300.0))
        self.assertLessEqual(DEFAULT_XGBOOST_TUNING_SUBSAMPLE_RANGE[0], 0.65)
        self.assertGreaterEqual(DEFAULT_XGBOOST_TUNING_SUBSAMPLE_RANGE[1], 1.0)
        self.assertGreaterEqual(DEFAULT_XGBOOST_TUNING_COLSAMPLE_BYTREE_RANGE[1], 1.0)
        self.assertEqual(DEFAULT_XGBOOST_MAX_CAT_TO_ONEHOT, 8)
        self.assertEqual(DEFAULT_XGBOOST_MAX_CAT_THRESHOLD, 64)

    def test_fit_and_score_xgboost_uses_walk_forward_folds(self) -> None:
        train_frame, tuning_frame = self._frames()
        target_contract = resolve_target_contract(
            train_frame,
            tuning_frame,
            requested_target_col="target_demand_qty_d_plus_1",
        )
        folds = build_grouped_tuning_walk_forward_folds_by_dataset(
            tuning_frame,
            n_folds=5,
        )

        result = fit_and_score_xgboost_model_on_tuning(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            folds=folds,
            feature_cols=[
                "client_id",
                "location_id",
                "product_id",
                "target_day_of_week",
            ],
            target_contract=target_contract,
            logger=__import__("logging").getLogger(__name__),
            model_params={
                "n_estimators": 8,
                "early_stopping_rounds": 2,
                "max_parallel_fold_workers": 5,
                "max_depth": 2,
                "n_jobs": 2,
                "random_state": 7,
            },
        )

        self.assertEqual(result["folds_completed"], 5)
        self.assertIsInstance(result["selected_n_estimators"], int)
        self.assertLessEqual(int(cast(Any, result["selected_n_estimators"])), 8)
        self.assertTrue(bool(np.isfinite(float(cast(Any, result["macro_mean_wape"])))))
        self.assertTrue(
            bool(np.isfinite(float(cast(Any, result["mean_abs_normalized_bias"]))))
        )
        self.assertTrue(
            bool(np.isfinite(float(cast(Any, result["mean_decision_loss"]))))
        )
        self.assertEqual(
            result["validation_monitor_metric"],
            "segmented_asymmetric_decision_loss",
        )
        self.assertIn("economic_objective_config", result)
        self.assertIn("segmented_economic_objective_config", result)
        self.assertIn("segment_decision_loss", result)
        self.assertIn("product_decision_loss", result)
        execution_policy = cast(dict[str, object], result["execution_policy"])
        self.assertEqual(execution_policy["backend"], "xgboost")
        self.assertEqual(execution_policy["fold_workers"], 5)
        self.assertEqual(execution_policy["threads_per_fold"], 2)
        self.assertEqual(execution_policy["total_threads"], 10)

    def test_cpu_scoring_can_reuse_prebuilt_fold_matrix_cache(self) -> None:
        train_frame, tuning_frame = self._frames()
        target_contract = resolve_target_contract(
            train_frame,
            tuning_frame,
            requested_target_col="target_demand_qty_d_plus_1",
        )
        folds = build_grouped_tuning_walk_forward_folds_by_dataset(
            tuning_frame,
            n_folds=5,
        )
        feature_cols = [
            "client_id",
            "location_id",
            "product_id",
            "target_day_of_week",
        ]
        model_params: dict[str, object] = {
            "n_estimators": 8,
            "early_stopping_rounds": 2,
            "max_parallel_fold_workers": 1,
            "max_depth": 2,
            "max_bin": 128,
            "n_jobs": 1,
            "random_state": 7,
        }
        logger = __import__("logging").getLogger(__name__)

        cache = build_xgboost_fold_matrix_cache(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            folds=folds,
            feature_cols=feature_cols,
            target_contract=target_contract,
            logger=logger,
            model_params=model_params,
            total_threads=1,
            train_frame_is_eligible=True,
        )
        self.assertIsNotNone(cache)
        assert cache is not None

        result = fit_and_score_xgboost_model_on_tuning(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            folds=folds,
            feature_cols=feature_cols,
            target_contract=target_contract,
            logger=logger,
            model_params=model_params,
            total_threads=1,
            train_frame_is_eligible=True,
            fold_matrix_cache=cache,
        )

        self.assertEqual(result["folds_completed"], 5)
        execution_policy = cast(dict[str, object], result["execution_policy"])
        self.assertEqual(execution_policy["accelerator"], "cpu")
        self.assertEqual(execution_policy["fold_matrix_cache"], "prebuilt")
        self.assertEqual(execution_policy["fold_matrix_cache_folds"], 5)

    def test_cuda_execution_policy_serializes_folds_on_single_gpu(self) -> None:
        execution_policy = xgboost_execution_policy(
            {
                "runtime_profile": "scaleway_l40s",
                "device": "cuda",
                "n_jobs": 8,
                "max_parallel_fold_workers": 5,
            },
            fold_count=5,
        )

        self.assertEqual(execution_policy["backend"], "xgboost")
        self.assertEqual(execution_policy["accelerator"], "gpu")
        self.assertEqual(execution_policy["gpu_safe_mode"], True)
        self.assertEqual(execution_policy["fold_workers"], 1)
        self.assertEqual(execution_policy["threads_per_fold"], 1)
        self.assertEqual(execution_policy["total_threads"], 1)

    def test_guardrail_uses_normalized_bias_instead_of_raw_demand_units(
        self,
    ) -> None:
        accepted_result: dict[str, object] = {
            "dataset_mean_wape": {"fixture": 0.40},
            "mean_abs_bias": 50.0,
            "mean_abs_normalized_bias": 0.20,
        }
        rejected_result: dict[str, object] = {
            "dataset_mean_wape": {"fixture": 0.40},
            "mean_abs_bias": 50.0,
            "mean_abs_normalized_bias": 0.90,
        }

        self.assertIsNone(
            guardrail_failure_reason(
                tuning_result=accepted_result,
                baseline_wape=0.50,
                baseline_dataset_wape={"fixture": 0.50},
            )
        )
        self.assertEqual(
            guardrail_failure_reason(
                tuning_result=rejected_result,
                baseline_wape=0.50,
                baseline_dataset_wape={"fixture": 0.50},
            ),
            "absolute_bias_too_high",
        )

    def test_xgboost_objective_prefers_decision_loss_over_wape(self) -> None:
        lower_wape_but_overproducing: dict[str, object] = {
            "macro_mean_wape": 0.10,
            "mean_decision_loss": 0.40,
        }
        higher_wape_but_better_decision: dict[str, object] = {
            "macro_mean_wape": 0.12,
            "mean_decision_loss": 0.20,
        }

        self.assertGreater(
            OBJECTIVE_SCORE(higher_wape_but_better_decision),
            OBJECTIVE_SCORE(lower_wape_but_overproducing),
        )

    def test_final_best_params_keep_economic_objective_artifacts(self) -> None:
        economic_config = {"version": "segmented_asymmetric_economic_objective_final"}
        best_trial = optuna.trial.create_trial(
            params={},
            distributions={},
            value=-0.25,
            user_attrs={
                "economic_objective_config": economic_config,
                "mean_decision_loss": 0.25,
                "validation_economic_loss": 0.12,
                "validation_bias": -0.02,
                "validation_positive_bias_penalty": 0.0,
                "validation_severe_negative_bias_penalty": 0.0,
                "validation_wape": 0.18,
                "segmented_economic_objective_config": economic_config,
                "segment_decision_loss": {"bread": 0.25},
                "product_decision_loss": {"BAGUETTE": 0.25},
            },
        )

        params = FINAL_BEST_PARAMS(
            best_trial=best_trial,
            model_params={"n_jobs": 1, "runtime_profile": "local_cpu"},
            random_seed=7,
            total_threads=1,
        )

        self.assertEqual(
            params["validation_monitor_metric"],
            "segmented_asymmetric_decision_loss",
        )
        self.assertEqual(params["economic_objective_config"], economic_config)
        self.assertEqual(params["segmented_economic_objective_config"], economic_config)
        self.assertEqual(params["mean_decision_loss"], 0.25)
        self.assertEqual(params["validation_bias"], -0.02)
        self.assertEqual(params["segment_decision_loss"], {"bread": 0.25})

    def test_hpo_metadata_serializes_economic_objective(self) -> None:
        economic_config = {"version": "segmented_asymmetric_economic_objective_final"}
        study = optuna.create_study(direction="maximize")
        study.add_trial(
            optuna.trial.create_trial(
                params={},
                distributions={},
                value=-0.25,
                user_attrs={
                    "mean_wape": 0.18,
                    "mean_abs_bias": 0.8,
                    "mean_decision_loss": 0.25,
                    "mean_normalized_economic_loss": 0.12,
                    "mean_normalized_bias": -0.02,
                    "mean_positive_bias_penalty": 0.0,
                    "mean_severe_negative_bias_penalty": 0.0,
                    "validation_monitor_metric": "segmented_asymmetric_decision_loss",
                    "economic_objective_config": economic_config,
                    "segmented_economic_objective_config": economic_config,
                    "segment_decision_loss": {"bread": 0.25},
                    "product_decision_loss": {"BAGUETTE": 0.25},
                },
            )
        )

        metadata = build_hpo_runtime_metadata(
            study=study,
            execution_policy={
                "runtime_profile": "local_cpu",
                "fold_workers": 1,
                "threads_per_fold": 1,
            },
            tuning_trials=1,
            stage_budget="quick",
        )

        self.assertEqual(metadata["best_decision_loss"], 0.25)
        self.assertEqual(metadata["best_normalized_bias"], -0.02)
        self.assertEqual(metadata["economic_objective_config"], economic_config)
        self.assertEqual(
            metadata["segmented_economic_objective_config"],
            economic_config,
        )
        self.assertEqual(metadata["segment_decision_loss"], {"bread": 0.25})


if __name__ == "__main__":
    unittest.main()

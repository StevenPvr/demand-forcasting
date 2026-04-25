from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast
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
from praedixa.demand_forecast.training.xgboost.tuning import (  # noqa: E402
    fit_and_score_xgboost_model_on_tuning,
    sample_xgboost_optuna_params,
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
                "learning_rate": 0.05,
                "max_depth": 4,
                "min_child_weight": 2.0,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_alpha": 1e-4,
                "reg_lambda": 1.0,
                "max_bin": 256,
            }
        )

        params = sample_xgboost_optuna_params(cast(Any, trial), random_seed=11)

        self.assertNotIn("n_estimators", params)
        self.assertEqual(params["max_depth"], 4)
        self.assertTrue(params["enable_early_stopping"])
        self.assertTrue(params["enable_categorical"])
        self.assertEqual(params["early_stopping_rounds"], 100)
        self.assertEqual(params["random_state"], 12)

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
            feature_cols=["client_id", "location_id", "product_id", "target_day_of_week"],
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
        execution_policy = cast(dict[str, object], result["execution_policy"])
        self.assertEqual(execution_policy["backend"], "xgboost")
        self.assertEqual(execution_policy["fold_workers"], 5)
        self.assertEqual(execution_policy["threads_per_fold"], 2)
        self.assertEqual(execution_policy["total_threads"], 10)


if __name__ == "__main__":
    unittest.main()

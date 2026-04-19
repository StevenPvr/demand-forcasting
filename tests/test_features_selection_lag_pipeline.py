import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.features_selection_lag.pipeline import (  # noqa: E402
    build_lag_selection_outputs,
    build_walk_forward_folds,
    filter_correlated_lag_features,
    get_lag_candidate_columns,
    optimize_non_lag_model_params,
    resolve_parallelism,
    split_chronological_train_tuning,
)


class FeatureSelectionLagPipelineTests(unittest.TestCase):
    def test_get_lag_candidate_columns_keeps_only_temporal_candidates(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-04-01"]),
                "target": [0.2],
                "city_id": [1],
                "sale_amount_lag_1": [0.1],
                "sale_amount_rolling_mean_7": [0.1],
                "sale_amount_ewm_mean_7": [0.1],
            }
        )

        candidates = get_lag_candidate_columns(frame)

        self.assertEqual(
            candidates,
            [
                "sale_amount_lag_1",
                "sale_amount_rolling_mean_7",
                "sale_amount_ewm_mean_7",
            ],
        )

    def test_split_chronological_train_tuning_uses_unique_dates(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(
                    [
                        "2024-04-01",
                        "2024-04-01",
                        "2024-04-02",
                        "2024-04-03",
                        "2024-04-04",
                        "2024-04-05",
                        "2024-04-06",
                        "2024-04-07",
                        "2024-04-08",
                        "2024-04-09",
                    ]
                ),
                "target": np.arange(10),
            }
        )

        selection_train, tuning_holdout, metadata = split_chronological_train_tuning(frame, train_fraction=0.7)

        self.assertEqual(selection_train["dt"].nunique(), 6)
        self.assertEqual(tuning_holdout["dt"].nunique(), 3)
        self.assertLess(selection_train["dt"].max(), tuning_holdout["dt"].min())
        self.assertEqual(metadata["train_unique_dates"], 6)
        self.assertEqual(metadata["holdout_unique_dates"], 3)

    def test_filter_correlated_lag_features_applies_pearson_then_spearman(self) -> None:
        target = pd.Series(np.arange(1, 51), dtype=float)
        frame = pd.DataFrame(
            {
                "target": target,
                "lag_anchor": target,
                "lag_dup": target + 0.001,
                "lag_cube": target**3,
                "lag_noise": np.where((target % 2) == 0, 1.0, 0.0),
            }
        )

        selected, report = filter_correlated_lag_features(
            frame=frame,
            candidate_cols=["lag_anchor", "lag_dup", "lag_cube", "lag_noise"],
            target_col="target",
            pearson_threshold=0.95,
            spearman_threshold=0.95,
        )

        self.assertIn("lag_anchor", selected)
        self.assertIn("lag_noise", selected)
        self.assertNotIn("lag_dup", selected)
        self.assertNotIn("lag_cube", selected)

        dropped_by = dict(zip(report["feature"], report["dropped_by"], strict=False))
        self.assertEqual(dropped_by["lag_dup"], "pearson")
        self.assertEqual(dropped_by["lag_cube"], "spearman")

    def test_build_walk_forward_folds_returns_five_expanding_folds(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-04-01", periods=42, freq="D"),
                "target": np.arange(42, dtype=float),
            }
        )

        folds = build_walk_forward_folds(frame, n_folds=5)

        self.assertEqual(len(folds), 5)
        self.assertEqual(folds[0]["train_dates"], 7)
        self.assertEqual(folds[0]["valid_dates"], 7)
        self.assertEqual(folds[-1]["train_dates"], 35)
        self.assertEqual(folds[-1]["valid_dates"], 7)

    def test_resolve_parallelism_balances_workers_and_threads(self) -> None:
        with mock.patch("research_praedixa.features_selection_lag.pipeline._is_macos", return_value=False):
            workers, threads = resolve_parallelism(requested_workers=8, candidate_count=3, total_threads=12)

        self.assertEqual(workers, 3)
        self.assertEqual(threads, 4)

    def test_resolve_parallelism_forces_single_outer_worker_on_macos(self) -> None:
        with mock.patch("research_praedixa.features_selection_lag.pipeline._is_macos", return_value=True):
            workers, threads = resolve_parallelism(requested_workers=8, candidate_count=10, total_threads=12)

        self.assertEqual(workers, 1)
        self.assertEqual(threads, 12)

    def test_optimize_non_lag_model_params_returns_best_params_and_report(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=42, freq="D"),
                "target": np.sin(np.arange(42) / 4.0) + 0.1,
                "city_id": np.tile([1, 2], 21),
                "holiday_flag": np.tile([0, 1, 0], 14),
                "store_id": 1,
                "product_id": 10,
                "sale_amount_lag_1": np.roll(np.sin(np.arange(42) / 4.0), 1),
            }
        )
        folds = build_walk_forward_folds(frame, n_folds=5)

        best_params, tuning_report = optimize_non_lag_model_params(
            frame=frame,
            folds=folds,
            feature_cols=["city_id", "holiday_flag", "store_id", "product_id"],
            n_trials=4,
            baseline_wape=0.8,
            num_workers=2,
            base_model_params={"n_jobs": 1, "verbosity": 0, "n_estimators": 40},
        )

        self.assertEqual(len(tuning_report), 4)
        self.assertIn("mean_wape", tuning_report.columns)
        self.assertIn("baseline_wape_improvement_pct", tuning_report.columns)
        self.assertIn("learning_rate", tuning_report.columns)
        self.assertIn("max_depth", best_params)
        self.assertEqual(best_params["n_jobs"], 1)
        self.assertEqual(best_params["n_estimators"], 40)

    def test_build_lag_selection_outputs_writes_selected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "data_train_cleaned.parquet"
            output_dir = root / "outputs"

            dates = pd.date_range("2024-01-01", periods=60, freq="D")
            base_signal = np.sin(np.arange(60) / 5.0)
            lag_signal = np.roll(base_signal, 1)
            lag_signal[0] = lag_signal[1]
            target = (2.5 * lag_signal) + (0.2 * np.cos(np.arange(60) / 3.0))

            frame = pd.DataFrame(
                {
                    "dt": dates,
                    "target": target,
                    "store_id": 1,
                    "product_id": 10,
                    "city_id": 100,
                    "holiday_flag": 0,
                    "sale_amount_lag_1": lag_signal,
                    "sale_amount_lag_2": np.roll(lag_signal, 1),
                    "sale_amount_lag_7": np.roll(lag_signal, 7),
                    "sale_amount_lag_14": np.roll(lag_signal, 14),
                    "sale_amount_lag_21": np.roll(lag_signal, 21),
                    "sale_amount_lag_28": np.roll(lag_signal, 28),
                    "sale_amount_rolling_mean_7": pd.Series(lag_signal).rolling(7, min_periods=1).mean(),
                    "sale_amount_rolling_mean_28": pd.Series(lag_signal).rolling(28, min_periods=1).mean(),
                    "sale_amount_ewm_mean_7": pd.Series(lag_signal).ewm(span=7, adjust=False).mean(),
                    "noise_lag_99": np.linspace(0.1, 0.9, 60),
                }
            )
            frame.to_parquet(input_path, index=False)

            outputs = build_lag_selection_outputs(
                input_path=input_path,
                output_dir=output_dir,
                train_fraction=0.7,
                n_folds=5,
                pearson_threshold=0.95,
                spearman_threshold=0.95,
                tuning_trials=4,
                model_params={
                    "n_estimators": 40,
                    "learning_rate": 0.1,
                    "max_depth": 4,
                    "min_child_weight": 5.0,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "random_state": 42,
                    "n_jobs": 1,
                    "verbosity": 0,
                    "tree_method": "hist",
                },
                num_workers=2,
            )

            self.assertEqual(
                set(outputs.keys()),
                {
                    "train_selected",
                    "tuning_selected",
                    "selected_lag_features",
                    "lag_correlation_report",
                    "non_lag_model_tuning_report",
                    "best_non_lag_model_params",
                    "lag_addon_importance_report",
                    "baseline_report",
                    "split_metadata",
                },
            )

            train_selected = pd.read_parquet(outputs["train_selected"])
            tuning_selected = pd.read_parquet(outputs["tuning_selected"])
            tuning_report = pd.read_csv(outputs["non_lag_model_tuning_report"])
            addon_report = pd.read_csv(outputs["lag_addon_importance_report"])
            selected_features_payload = json.loads(Path(outputs["selected_lag_features"]).read_text())
            kept_features = addon_report.loc[addon_report["keep"] == True, "feature"].tolist()  # noqa: E712

            self.assertIn("target", train_selected.columns)
            self.assertIn("dt", train_selected.columns)
            self.assertIn("store_id", train_selected.columns)
            self.assertIn("sale_amount_lag_1", train_selected.columns)
            self.assertEqual(list(train_selected.columns), list(tuning_selected.columns))
            self.assertEqual(len(tuning_report), 4)
            self.assertIn("mean_wape", tuning_report.columns)
            self.assertIn("baseline_wape_improvement_pct", tuning_report.columns)
            self.assertIn("wape_improvement_pct", addon_report.columns)
            temporal_output_cols = [
                column
                for column in train_selected.columns
                if any(token in column for token in ("_lag_", "_rolling_", "_ewm_"))
            ]
            self.assertTrue(set(kept_features).issubset(set(temporal_output_cols)))
            self.assertIn("sale_amount_lag_7", temporal_output_cols)
            self.assertEqual(selected_features_payload["evaluated_feature_count"], len(addon_report))


if __name__ == "__main__":
    unittest.main()

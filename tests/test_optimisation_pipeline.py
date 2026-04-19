import json
from pathlib import Path
import sys
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.optimisation.pipeline import (  # noqa: E402
    build_optimisation_outputs,
    build_grouped_tuning_walk_forward_folds_by_dataset,
    build_tuning_walk_forward_folds,
    build_tuning_walk_forward_folds_by_dataset,
    evaluate_statistical_baselines_macro,
)


class OptimisationPipelineTests(unittest.TestCase):
    def test_build_tuning_walk_forward_folds_uses_expanding_train_on_tuning_block(self) -> None:
        tuning_frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-06-01", periods=18, freq="D"),
                "target": np.arange(18, dtype=float),
            }
        )

        folds = build_tuning_walk_forward_folds(tuning_frame, n_folds=5)

        self.assertEqual(len(folds), 5)
        self.assertEqual(folds[0]["train_dates"], 3)
        self.assertEqual(folds[0]["valid_dates"], 3)
        self.assertEqual(folds[-1]["train_dates"], 15)
        self.assertEqual(folds[-1]["valid_dates"], 3)

    def test_build_tuning_walk_forward_folds_by_dataset_returns_positional_indices(self) -> None:
        base_frame = pd.DataFrame(
            {
                "dataset_source": ["a"] * 8 + ["b"] * 8,
                "dt": list(pd.date_range("2024-06-01", periods=8, freq="D")) * 2,
                "target": np.arange(16, dtype=float),
            }
        )
        tuning_frame = base_frame.drop(index=[1, 3, 10]).copy()

        folds = build_tuning_walk_forward_folds_by_dataset(tuning_frame, n_folds=2)

        for fold in folds:
            valid_frame = (
                tuning_frame.loc[tuning_frame["dataset_source"] == fold["dataset_source"]]
                .sort_values("dt")
                .reset_index(drop=True)
                .iloc[fold["valid_idx"]]
            )
            self.assertFalse(valid_frame.empty)
            self.assertTrue((valid_frame["dataset_source"] == fold["dataset_source"]).all())

    def test_build_grouped_tuning_walk_forward_folds_by_dataset_returns_fold_unions(self) -> None:
        tuning_frame = pd.DataFrame(
            {
                "dataset_source": ["a"] * 8 + ["b"] * 8,
                "dt": list(pd.date_range("2024-06-01", periods=8, freq="D")) * 2,
                "target": np.arange(16, dtype=float),
            }
        ).sort_values("dt").reset_index(drop=True)

        folds = build_grouped_tuning_walk_forward_folds_by_dataset(tuning_frame, n_folds=2)

        self.assertEqual(len(folds), 2)
        for fold in folds:
            valid_frame = tuning_frame.iloc[fold["valid_idx"]]
            self.assertEqual(set(valid_frame["dataset_source"]), {"a", "b"})
            self.assertEqual(len(fold["per_dataset"]), 2)

    def test_evaluate_statistical_baselines_macro_ignores_partial_nan_predictions(self) -> None:
        tuning_frame = pd.DataFrame(
            {
                "dataset_source": ["a"] * 6 + ["b"] * 6,
                "dt": list(pd.date_range("2024-06-01", periods=6, freq="D")) * 2,
                "target": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0] * 2,
                "lag_1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0] * 2,
                "lag_7": [1.0] * 12,
                "rolling_mean_7": [1.0] * 12,
                "rolling_mean_28": [1.0] * 12,
                "same_dow_mean_4w": [1.0] * 12,
            }
        )

        folds = build_tuning_walk_forward_folds_by_dataset(tuning_frame, n_folds=2)
        baseline_rows, best_baseline = evaluate_statistical_baselines_macro(
            tuning_frame,
            folds,
            absolute_target_col="target",
        )

        self.assertTrue(baseline_rows)
        self.assertEqual(best_baseline["baseline_name"], "naive_lag_1")
        self.assertTrue(np.isfinite(best_baseline["mean_wape"]))
        naive_row = next(row for row in baseline_rows if row["baseline_name"] == "naive_lag_1")
        self.assertTrue(all(result["rows_scored"] > 0 for result in naive_row["fold_wape_scores"]))

    def test_build_optimisation_outputs_writes_reports_and_selected_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_input_path = root / "train_selection_70_selected.parquet"
            tuning_input_path = root / "train_tuning_30_selected.parquet"
            output_dir = root / "outputs"

            train_dates = pd.date_range("2024-04-01", periods=30, freq="D")
            tuning_dates = pd.date_range("2024-05-01", periods=18, freq="D")
            signal = np.sin(np.arange(48) / 4.0)
            lag1 = np.roll(signal, 1)
            lag1[0] = lag1[1]

            shifted_signal = signal + 3.0
            shifted_lag1 = np.roll(shifted_signal, 1)
            shifted_lag1[0] = shifted_lag1[1]

            train_frame = pd.DataFrame(
                {
                    "dt": train_dates,
                    "target": shifted_signal[:30] * 2.0 + 0.1,
                    "store_id": 1,
                    "product_id": 10,
                    "city_id": 100,
                    "holiday_flag": np.tile([0, 1, 0], 10),
                    "sale_amount_lag_1": shifted_lag1[:30],
                    "sale_amount_lag_7": np.roll(shifted_lag1[:30], 7),
                }
            )
            tuning_frame = pd.DataFrame(
                {
                    "dt": tuning_dates,
                    "target": shifted_signal[30:] * 2.0 + 0.1,
                    "store_id": 1,
                    "product_id": 10,
                    "city_id": 100,
                    "holiday_flag": np.tile([0, 1, 0], 6),
                    "sale_amount_lag_1": shifted_lag1[30:],
                    "sale_amount_lag_7": np.roll(shifted_lag1[30:], 7),
                }
            )

            train_frame.to_parquet(train_input_path, index=False)
            tuning_frame.to_parquet(tuning_input_path, index=False)

            outputs = build_optimisation_outputs(
                train_input_path=train_input_path,
                tuning_input_path=tuning_input_path,
                output_dir=output_dir,
                n_folds=5,
                tuning_trials=4,
                train_sample_fraction=1.0,
                tuning_sample_fraction=1.0,
                model_params={
                    "n_estimators": 40,
                    "learning_rate": 0.05,
                    "max_depth": 4,
                    "min_child_weight": 2.0,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_lambda": 1.0,
                    "metric": "l2",
                    "random_state": 42,
                    "n_jobs": 1,
                    "verbosity": 0,
                },
            )

            self.assertEqual(
                set(outputs.keys()),
                {
                    "best_params",
                    "optuna_trials_report",
                    "baseline_report",
                    "optimisation_metadata",
                    "optimisation_feature_audit",
                },
            )

            trials_report = pd.read_csv(outputs["optuna_trials_report"])
            baseline_payload = json.loads(Path(outputs["baseline_report"]).read_text())
            metadata_payload = json.loads(Path(outputs["optimisation_metadata"]).read_text())
            feature_audit_payload = json.loads(Path(outputs["optimisation_feature_audit"]).read_text())
            best_params_payload = json.loads(Path(outputs["best_params"]).read_text())

            self.assertEqual(len(trials_report), 4)
            self.assertIn("baseline_wape_improvement_pct", trials_report.columns)
            self.assertIn("mean_wape", trials_report.columns)
            self.assertIn("best_baseline", baseline_payload)
            self.assertIn("folds", metadata_payload)
            self.assertEqual(metadata_payload["tuning_trials"], 4)
            self.assertEqual(metadata_payload["target_transform"], "delta_log_wow")
            self.assertEqual(metadata_payload["absolute_target_col"], "target")
            self.assertEqual(metadata_payload["reconstruction_anchor_col"], "sale_amount_lag_7")
            self.assertIn("learning_rate", best_params_payload)
            self.assertEqual(best_params_payload["n_estimators"], 40)
            self.assertIn("max_depth", best_params_payload)
            self.assertIn("constant_feature_cols", feature_audit_payload)

    def test_build_optimisation_outputs_loads_gold_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            duckdb_path = root / "praedixa.duckdb"
            output_dir = root / "outputs"

            frames: list[pd.DataFrame] = []
            dataset_specs = (
                ("m5", "store_1", "sku_1"),
                ("m5", "store_2", "sku_2"),
                ("freshretail", "store_3", "sku_3"),
                ("freshretail", "store_4", "sku_4"),
            )
            for dataset_offset, (dataset_source, location_id, product_id) in enumerate(dataset_specs):
                dates = pd.date_range("2024-01-01", periods=120, freq="D") + pd.Timedelta(days=dataset_offset)
                base_signal = np.sin((np.arange(120) + dataset_offset) / 5.0) + 2.0
                frames.append(
                    pd.DataFrame(
                        {
                            "dataset_source": [dataset_source] * 120,
                            "series_id": [f"{location_id}__{product_id}"] * 120,
                            "dt": dates,
                            "location_id": [location_id] * 120,
                            "product_id": [product_id] * 120,
                            "target_demand_qty_d_plus_1": base_signal,
                            "lag_1": np.roll(base_signal, 1),
                            "lag_7": np.roll(base_signal, 7),
                            "rolling_mean_7": pd.Series(base_signal).rolling(7, min_periods=1).mean(),
                            "policy_rate_latest": [1.5 + dataset_offset] * 120,
                            "food_cpi_latest": np.linspace(100.0, 101.0, 120),
                            "split_bucket": ["train"] * 80 + ["val"] * 40,
                        }
                    )
                )
            frame = pd.concat(frames, ignore_index=True)

            connection = duckdb.connect(str(duckdb_path))
            try:
                connection.execute("create schema gold")
                connection.register("gold_frame", frame)
                connection.execute("create table gold.gold_daily_product_forecast_panel_d1 as select * from gold_frame")
                connection.unregister("gold_frame")
            finally:
                connection.close()

            outputs = build_optimisation_outputs(
                train_input_path=None,
                tuning_input_path=None,
                output_dir=output_dir,
                duckdb_path=duckdb_path,
                gold_table="gold.gold_daily_product_forecast_panel_d1",
                n_folds=1,
                tuning_trials=2,
                model_params={
                    "n_estimators": 20,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "min_child_weight": 1.0,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "n_jobs": 1,
                    "verbosity": 0,
                },
            )

            metadata_payload = json.loads(Path(outputs["optimisation_metadata"]).read_text())
            self.assertEqual(metadata_payload["duckdb_path"], str(duckdb_path))
            self.assertEqual(metadata_payload["gold_table"], "gold.gold_daily_product_forecast_panel_d1")
            self.assertIsNone(metadata_payload["train_input_path"])
            self.assertIsNone(metadata_payload["tuning_input_path"])
            self.assertEqual(metadata_payload["target_transform"], "delta_log_wow")
            self.assertEqual(metadata_payload["absolute_target_col"], "target_demand_qty_d_plus_1")
            self.assertEqual(metadata_payload["reconstruction_anchor_col"], "lag_7")
            self.assertEqual(metadata_payload["sample_store_col"], "location_id")
            self.assertEqual(metadata_payload["train_sampling"]["sample_strategy"], "date_store_stratified")
            self.assertEqual(metadata_payload["tuning_sampling"]["sample_strategy"], "date_store_stratified")
            self.assertEqual(metadata_payload["train_sampling"]["sample_fraction"], 0.2)
            self.assertEqual(metadata_payload["tuning_sampling"]["sample_fraction"], 0.2)
            self.assertEqual(metadata_payload["train_sampling"]["datasets"]["m5"]["sampled_rows"], 160)
            self.assertEqual(metadata_payload["train_sampling"]["datasets"]["freshretail"]["sampled_rows"], 160)
            self.assertEqual(metadata_payload["tuning_sampling"]["datasets"]["m5"]["sampled_rows"], 80)
            self.assertEqual(metadata_payload["tuning_sampling"]["datasets"]["freshretail"]["sampled_rows"], 80)
            self.assertEqual(metadata_payload["train_dataset_weights"]["m5"]["total_weight"], 0.5)
            self.assertEqual(metadata_payload["train_dataset_weights"]["freshretail"]["total_weight"], 0.5)
            self.assertEqual(metadata_payload["tuning_dataset_weights"]["m5"]["total_weight"], 0.5)
            self.assertEqual(metadata_payload["tuning_dataset_weights"]["freshretail"]["total_weight"], 0.5)


if __name__ == "__main__":
    unittest.main()

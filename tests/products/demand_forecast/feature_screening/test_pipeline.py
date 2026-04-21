import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.feature_screening.pipeline import (  # noqa: E402
    build_lag_selection_outputs,
    build_walk_forward_folds,
    filter_correlated_lag_features,
    get_lag_candidate_columns,
    optimize_non_lag_model_params,
    resolve_parallelism,
    split_chronological_train_tuning,
)
from praedixa.demand_forecast.feature_screening.constants import DEFAULT_GOLD_TABLE  # noqa: E402


def _build_feature_selection_input_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    base_signal = np.sin(np.arange(60) / 5.0)
    lag_signal = np.roll(base_signal, 1)
    lag_signal[0] = lag_signal[1]
    return pd.DataFrame(
        {
            "dt": dates,
            "target": (2.5 * lag_signal) + (0.2 * np.cos(np.arange(60) / 3.0)),
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


def _assert_lag_selection_outputs_exist(output_dir: Path) -> None:
    expected = [
        "train_selection_70_selected.parquet",
        "train_tuning_30_selected.parquet",
        "selected_lag_features.json",
        "best_non_lag_model_params.json",
    ]
    for name in expected:
        if not (output_dir / name).exists():
            raise AssertionError(f"Missing lag selection artifact: {output_dir / name}")


def _addon_report_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": "sale_amount_lag_1",
                "keep": True,
                "selection_stage": "addon",
                "wape_improvement_pct": 0.05,
            }
        ]
    )


class FeatureSelectionLagPipelineTests(unittest.TestCase):
    def test_feature_screening_defaults_to_model_facing_gold_panel(self) -> None:
        self.assertEqual(DEFAULT_GOLD_TABLE, "gold.gold_daily_product_forecast_panel_d1")

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
        with mock.patch("praedixa.demand_forecast.feature_screening.pipeline._is_macos", return_value=False):
            workers, threads = resolve_parallelism(requested_workers=8, candidate_count=3, total_threads=12)

        self.assertEqual(workers, 3)
        self.assertEqual(threads, 4)

    def test_resolve_parallelism_forces_single_outer_worker_on_macos(self) -> None:
        with mock.patch("praedixa.demand_forecast.feature_screening.pipeline._is_macos", return_value=True):
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

        with mock.patch(
            "praedixa.demand_forecast.feature_screening.tuning.fit_and_score_tft_model",
            return_value=[0.6, 0.65, 0.62, 0.61, 0.63],
        ):
            best_params, tuning_report = optimize_non_lag_model_params(
                frame=frame,
                folds=folds,
                feature_cols=["city_id", "holiday_flag", "store_id", "product_id"],
                n_trials=4,
                baseline_wape=0.8,
                num_workers=2,
                base_model_params={"n_jobs": 1},
            )

        self.assertEqual(len(tuning_report), 4)
        self.assertEqual(best_params["n_jobs"], 1)
        self.assertIn("hidden_size", best_params)

    def test_build_lag_selection_outputs_writes_selected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "data_train_cleaned.parquet"
            output_dir = root / "outputs"
            _build_feature_selection_input_frame().to_parquet(input_path, index=False)
            with (
                mock.patch(
                    "praedixa.demand_forecast.feature_screening.orchestrator.optimize_non_lag_model_params",
                    return_value=(
                        {"hidden_size": 8, "max_epochs": 1, "n_jobs": 1},
                        pd.DataFrame(
                            [
                                {
                                    "trial": 0,
                                    "mean_wape": 0.2,
                                    "baseline_wape_improvement_pct": 0.1,
                                }
                            ]
                        ),
                    ),
                ),
                mock.patch(
                    "praedixa.demand_forecast.feature_screening.orchestrator.run_single_addon_lag_selection_memmap",
                    return_value=(
                        ["sale_amount_lag_1"],
                        _addon_report_fixture(),
                        [{"feature_count": 4, "mean_wape": 0.2}],
                    ),
                ),
            ):
                build_lag_selection_outputs(
                    input_path=input_path,
                    output_dir=output_dir,
                    train_fraction=0.7,
                    n_folds=5,
                    pearson_threshold=0.95,
                    spearman_threshold=0.95,
                    tuning_trials=4,
                    model_params={"n_jobs": 1},
                    num_workers=2,
                )

            _assert_lag_selection_outputs_exist(output_dir)

    def test_build_lag_selection_outputs_correlation_only_caps_selected_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "data_train_cleaned.parquet"
            output_dir = root / "outputs"

            dates = pd.date_range("2024-01-01", periods=80, freq="D")
            target = np.linspace(1.0, 4.0, len(dates))
            frame = pd.DataFrame(
                {
                    "dt": dates,
                    "target": target,
                    "store_id": 1,
                    "product_id": 10,
                }
            )
            frame["sale_amount_lag_1"] = target
            frame["sale_amount_lag_2"] = target**2
            frame["sale_amount_lag_3"] = target**3
            frame["sale_amount_lag_4"] = np.sin(target)
            frame["sale_amount_lag_5"] = np.cos(target)
            frame.to_parquet(input_path, index=False)

            outputs = build_lag_selection_outputs(
                input_path=input_path,
                output_dir=output_dir,
                correlation_only=True,
                max_selected_lag_features=2,
            )

            selected_features_payload = json.loads(Path(outputs["selected_lag_features"]).read_text())
            split_metadata = json.loads(Path(outputs["split_metadata"]).read_text())
            addon_report = pd.read_csv(outputs["lag_addon_importance_report"])
            tuning_report = pd.read_csv(outputs["non_lag_model_tuning_report"])
            train_selected = pd.read_parquet(outputs["train_selected"])

            self.assertTrue(split_metadata["correlation_only"])
            self.assertEqual(selected_features_payload["selected_lag_feature_count"], 2)
            self.assertEqual(len(addon_report), 2)
            self.assertEqual(tuning_report.loc[0, "mode"], "correlation_only")
            kept_temporal_cols = [column for column in train_selected.columns if "_lag_" in column]
            self.assertEqual(len(kept_temporal_cols), 2)

    def test_build_lag_selection_outputs_can_materialize_from_gold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            duckdb_path = root / "praedixa.duckdb"
            output_dir = root / "outputs"
            frame = pd.DataFrame(
                {
                    "dt": pd.date_range("2024-01-01", periods=20, freq="D"),
                    "split_bucket": ["train"] * 20,
                    "target_delta_log_wow_d_plus_1": np.linspace(1.0, 3.0, 20),
                    "target_demand_qty_d_plus_1": np.linspace(10.0, 30.0, 20),
                    "store_id": 1,
                    "product_id": 10,
                    "lag_1": np.linspace(0.5, 2.5, 20),
                    "lag_7": np.linspace(0.2, 1.2, 20),
                    "rolling_mean_7": np.linspace(0.3, 1.3, 20),
                }
            )
            import duckdb

            connection = duckdb.connect(str(duckdb_path))
            try:
                connection.execute("create schema gold")
                connection.register("gold_frame", frame)
                connection.execute("create table gold.gold_feature_panel_d1 as select * from gold_frame")
                connection.unregister("gold_frame")
            finally:
                connection.close()

            outputs = build_lag_selection_outputs(
                input_path=None,
                output_dir=output_dir,
                duckdb_path=duckdb_path,
                gold_table="gold.gold_feature_panel_d1",
                target_col="target_delta_log_wow_d_plus_1",
                correlation_only=True,
                max_selected_lag_features=2,
            )

            train_selected = pd.read_parquet(outputs["train_selected"])
            self.assertIn("target_delta_log_wow_d_plus_1", train_selected.columns)
            self.assertTrue((output_dir / "_cache" / "feature_selection_source_train.parquet").exists())


if __name__ == "__main__":
    unittest.main()

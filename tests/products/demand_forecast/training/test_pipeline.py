from pathlib import Path
import json
import sys
import tempfile
import unittest
from typing import Any, cast
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.training.pipeline import (  # noqa: E402
    OptimisationBuildRequest,
    build_optimisation_outputs,
    build_grouped_tuning_walk_forward_folds_by_dataset,
    build_tuning_walk_forward_folds,
    build_tuning_walk_forward_folds_by_dataset,
    evaluate_statistical_baselines_macro,
)
from praedixa.demand_forecast.training.constants import DEFAULT_GOLD_TABLE  # noqa: E402
from praedixa.demand_forecast.training.feature_audit import drop_constant_feature_columns  # noqa: E402


def _mock_hpo_result(*, mean_wape: float, stage_a_trials: int, stage_b_trials: int) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    return (
        {"hidden_size": 8, "max_epochs": 1, "n_jobs": 1},
        pd.DataFrame(
            [
                {
                    "trial": 0,
                    "mean_wape": mean_wape,
                    "baseline_wape_improvement_pct": 0.10,
                    "hidden_size": 8,
                }
            ]
        ),
        {
            "execution_policy": {"accelerator": "cpu", "fold_workers": 2, "gpu_safe_mode": False},
            "pruner": {"type": "MedianPruner"},
            "stage_policy": {
                "stage_a": {"trial_count": stage_a_trials},
                "stage_b": {"trial_count": stage_b_trials},
            },
        },
    )


def _sinusoidal_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dates = pd.date_range("2024-04-01", periods=30, freq="D")
    tuning_dates = pd.date_range("2024-05-01", periods=18, freq="D")
    shifted_signal = np.sin(np.arange(48) / 4.0) + 3.0
    return (
        pd.DataFrame(
            {
                "dt": train_dates,
                "target": shifted_signal[:30] * 2.0 + 0.1,
                "location_id": "store_1",
                "product_id": "sku_1",
                "client_id": "public_client",
                "target_day_of_week": [int(value) for value in train_dates.dayofweek],
                "target_holiday_flag": np.tile([0, 1, 0], 10),
                "current_day_demand_qty": shifted_signal[:30] * 2.0 + 0.1,
                "avg_selling_price": [2.4] * 30,
            }
        ),
        pd.DataFrame(
            {
                "dt": tuning_dates,
                "target": shifted_signal[30:] * 2.0 + 0.1,
                "location_id": "store_1",
                "product_id": "sku_1",
                "client_id": "public_client",
                "target_day_of_week": [int(value) for value in tuning_dates.dayofweek],
                "target_holiday_flag": np.tile([0, 1, 0], 6),
                "current_day_demand_qty": shifted_signal[30:] * 2.0 + 0.1,
                "avg_selling_price": [2.4] * 18,
            }
        ),
    )


def _write_selected_inputs(root: Path) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    train_input_path = root / "train_selection_70_selected.parquet"
    tuning_input_path = root / "train_tuning_30_selected.parquet"
    train_frame, tuning_frame = _sinusoidal_frames()
    train_frame.to_parquet(train_input_path, index=False)
    tuning_frame.to_parquet(tuning_input_path, index=False)
    return train_input_path, tuning_input_path, train_frame, tuning_frame


def _assert_output_artifacts(output_dir: Path) -> dict[str, Any]:
    assert (output_dir / "best_optuna_params.json").exists()
    assert (output_dir / "optuna_trials_report.csv").exists()
    assert (output_dir / "baseline_report.json").exists()
    return json.loads((output_dir / "optimisation_metadata.json").read_text(encoding="utf-8"))


def _gold_frame() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    dataset_specs = (
        ("freshretail_lt", "store_1", "sku_1"),
        ("first_party_daily", "store_2", "sku_2"),
        ("freshretail", "store_3", "sku_3"),
        ("bakery", "store_4", "sku_4"),
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
                    "current_day_demand_qty": base_signal,
                    "avg_selling_price": np.full(120, 2.5 + dataset_offset * 0.1),
                    "target_day_of_week": dates.dayofweek.astype(int),
                    "target_holiday_flag": np.zeros(120, dtype=bool),
                    "split_bucket": ["train"] * 80 + ["val"] * 40,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _write_gold_table(duckdb_path: Path) -> None:
    connection = duckdb.connect(str(duckdb_path))
    try:
        connection.execute("create schema gold")
        connection.register("gold_frame", _gold_frame())
        connection.execute("create table gold.gold_daily_product_forecast_panel_d1 as select * from gold_frame")
        connection.unregister("gold_frame")
    finally:
        connection.close()


class OptimisationPipelineTests(unittest.TestCase):
    def test_training_defaults_to_model_facing_gold_panel(self) -> None:
        self.assertEqual(DEFAULT_GOLD_TABLE, "gold.gold_daily_product_forecast_panel_d1")

    def test_optimisation_request_defaults_to_full_sampling(self) -> None:
        request = OptimisationBuildRequest()

        self.assertEqual(request.train_sample_fraction, 1.0)
        self.assertEqual(request.tuning_sample_fraction, 1.0)

    def test_constant_metadata_features_are_kept_for_tft_training(self) -> None:
        frame = pd.DataFrame(
            {
                "country_code": ["FR", "FR", "FR"],
                "city_name": ["Paris", "Paris", "Paris"],
                "current_day_demand_qty": [1.0, 2.0, 3.0],
            }
        )

        feature_cols, constant_feature_cols = drop_constant_feature_columns(
            frame,
            ["country_code", "city_name", "current_day_demand_qty"],
        )

        self.assertEqual(
            feature_cols,
            ["country_code", "city_name", "current_day_demand_qty"],
        )
        self.assertEqual(constant_feature_cols, [])

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
            valid_frame = cast(
                pd.DataFrame,
                tuning_frame.loc[tuning_frame["dataset_source"] == fold["dataset_source"]]
                .sort_values("dt")
                .reset_index(drop=True)
                .iloc[cast(Any, fold["valid_idx"])]
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
            valid_frame = cast(pd.DataFrame, tuning_frame.iloc[cast(Any, fold["valid_idx"])])
            self.assertEqual(set(valid_frame["dataset_source"]), {"a", "b"})
            self.assertEqual(len(cast(list[dict[str, object]], fold["per_dataset"])), 2)

    def test_build_grouped_tuning_walk_forward_folds_by_dataset_skips_sparse_dataset_from_validation(self) -> None:
        tuning_frame = pd.DataFrame(
            {
                "dataset_source": ["a"] * 8 + ["b"] * 4,
                "dt": list(pd.date_range("2024-06-01", periods=8, freq="D")) + list(pd.date_range("2024-06-01", periods=4, freq="D")),
                "target": np.arange(12, dtype=float),
            }
        ).sort_values(["dataset_source", "dt"]).reset_index(drop=True)

        flat_folds = build_tuning_walk_forward_folds_by_dataset(tuning_frame, n_folds=5)
        grouped_folds = build_grouped_tuning_walk_forward_folds_by_dataset(tuning_frame, n_folds=5)

        dataset_b_flat_folds = [fold for fold in flat_folds if fold["dataset_source"] == "b"]
        self.assertEqual(len(dataset_b_flat_folds), 0)
        self.assertEqual(len(grouped_folds), 5)
        self.assertEqual(set(cast(list[str], grouped_folds[0]["datasets"])), {"a"})
        self.assertEqual(set(cast(list[str], grouped_folds[-1]["datasets"])), {"a"})

    def test_evaluate_statistical_baselines_macro_ignores_partial_nan_predictions(self) -> None:
        tuning_frame = pd.DataFrame(
            {
                "dataset_source": ["a"] * 6 + ["b"] * 6,
                "dt": list(pd.date_range("2024-06-01", periods=6, freq="D")) * 2,
                "target": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0] * 2,
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
        self.assertTrue(bool(np.isfinite(float(cast(Any, best_baseline["mean_wape"])))))
        naive_row = next(row for row in baseline_rows if row["baseline_name"] == "naive_lag_1")
        self.assertTrue(
            all(
                cast(int, result["rows_scored"]) > 0
                for result in cast(list[dict[str, object]], naive_row["fold_wape_scores"])
            )
        )

    def test_build_optimisation_outputs_writes_reports_and_selected_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "outputs"
            train_input_path, tuning_input_path, train_frame, tuning_frame = _write_selected_inputs(root)

            with patch(
                "praedixa.demand_forecast.training.orchestrator.optimize_tft_model_params",
                return_value=_mock_hpo_result(mean_wape=0.15, stage_a_trials=3, stage_b_trials=1),
            ):
                build_optimisation_outputs(
                    OptimisationBuildRequest(
                        train_input_path=train_input_path,
                        tuning_input_path=tuning_input_path,
                        output_dir=output_dir,
                        n_folds=5,
                        tuning_trials=4,
                        train_sample_fraction=1.0,
                        tuning_sample_fraction=1.0,
                        model_params={"n_jobs": 1},
                    )
                )

            metadata = _assert_output_artifacts(output_dir)
            self.assertGreaterEqual(metadata["feature_count"], 1)
            self.assertEqual(metadata["train_rows"], len(train_frame))
            self.assertEqual(metadata["tuning_rows"], len(tuning_frame))
            self.assertEqual(metadata["hpo_runtime"]["pruner"]["type"], "MedianPruner")
            self.assertEqual(metadata["hpo_runtime"]["execution_policy"]["accelerator"], "cpu")

    def test_build_optimisation_outputs_loads_gold_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            duckdb_path = root / "praedixa.duckdb"
            output_dir = root / "outputs"
            _write_gold_table(duckdb_path)

            with patch(
                "praedixa.demand_forecast.training.orchestrator.optimize_tft_model_params",
                return_value=_mock_hpo_result(mean_wape=0.25, stage_a_trials=1, stage_b_trials=1),
            ):
                build_optimisation_outputs(
                    OptimisationBuildRequest(
                        train_input_path=None,
                        tuning_input_path=None,
                        output_dir=output_dir,
                        duckdb_path=duckdb_path,
                        gold_table="gold.gold_daily_product_forecast_panel_d1",
                        n_folds=1,
                        tuning_trials=2,
                        train_sample_fraction=1.0,
                        tuning_sample_fraction=1.0,
                        model_params={"n_jobs": 1},
                    )
                )

            metadata = _assert_output_artifacts(output_dir)
            self.assertIsNone(metadata["train_input_path"])
            self.assertEqual(metadata["gold_table"], "gold.gold_daily_product_forecast_panel_d1")


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from typing import Callable, cast
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.evaluation import pipeline as evaluation_pipeline  # noqa: E402
import praedixa.demand_forecast.evaluation.orchestrator_runtime as evaluation_orchestrator_runtime  # noqa: E402
from praedixa.demand_forecast.evaluation.modeling import select_feature_columns  # noqa: E402
from praedixa.demand_forecast.evaluation.bakery_metrics import (  # noqa: E402
    build_predictions_frame,
    compute_probabilistic_metrics_payload,
    rmse_score,
)
from praedixa.demand_forecast.contracts.targets import TargetContract  # noqa: E402


evaluate_daily_refit_predictions = evaluation_pipeline.evaluate_daily_refit_predictions
build_daily_walk_forward_folds = evaluation_pipeline.build_daily_walk_forward_folds
build_evaluation_outputs = evaluation_pipeline.build_evaluation_outputs
EvaluationBuildRequest = evaluation_pipeline.EvaluationBuildRequest
_FIT_FINAL_MODEL = cast(
    Callable[..., object],
    getattr(evaluation_orchestrator_runtime, "_fit_final_model"),
)


def _build_eval_output_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_selection = pd.DataFrame(
        {
            "dt": pd.date_range("2024-01-01", periods=8, freq="D"),
            "location_id": ["store_1"] * 8,
            "product_id": ["sku_1"] * 8,
            "client_id": ["public_client"] * 8,
            "target_day_of_week": list(range(7)) + [0],
            "lag_1": [1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45],
            "lag_7": [1.0] * 8,
            "rolling_mean_7": [1.0, 1.05, 1.1, 1.12, 1.15, 1.18, 1.2, 1.22],
            "target": [1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5],
        }
    )
    train_tuning = pd.DataFrame(
        {
            "dt": pd.date_range("2024-01-09", periods=4, freq="D"),
            "location_id": ["store_1"] * 4,
            "product_id": ["sku_1"] * 4,
            "client_id": ["public_client"] * 4,
            "target_day_of_week": [1, 2, 3, 4],
            "lag_1": [1.5, 1.55, 1.6, 1.65],
            "lag_7": [1.1, 1.2, 1.25, 1.3],
            "rolling_mean_7": [1.28, 1.32, 1.36, 1.4],
            "target": [1.55, 1.6, 1.65, 1.7],
        }
    )
    val_frame = pd.DataFrame(
        {
            "dt": pd.date_range("2024-01-13", periods=3, freq="D"),
            "location_id": ["store_1"] * 3,
            "product_id": ["sku_1"] * 3,
            "client_id": ["public_client"] * 3,
            "target_day_of_week": [5, 6, 0],
            "lag_1": [1.7, 1.75, 1.8],
            "lag_7": [1.35, 1.4, 1.45],
            "rolling_mean_7": [1.45, 1.5, 1.55],
            "target": [1.75, 1.8, 1.85],
        }
    )
    return train_selection, train_tuning, val_frame


def _write_eval_fixture_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path, pd.DataFrame]:
    selection_path = root / "train_selection_70_selected.parquet"
    tuning_path = root / "train_tuning_30_selected.parquet"
    val_path = root / "data_val_cleaned.parquet"
    params_path = root / "best_optuna_params.json"
    output_dir = root / "evaluation"
    train_selection, train_tuning, val_frame = _build_eval_output_frames()
    train_selection.to_parquet(selection_path, index=False)
    train_tuning.to_parquet(tuning_path, index=False)
    val_frame.to_parquet(val_path, index=False)
    params_path.write_text(json.dumps({"n_jobs": 1}), encoding="utf-8")
    return selection_path, tuning_path, val_path, params_path, output_dir, val_frame


def _build_stub_predictions(val_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        build_predictions_frame(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-13", "2024-01-14", "2024-01-15"]),
                    "product": ["sku_1", "sku_1", "sku_1"],
                }
            ),
            actual=val_frame["target"],
            prediction_raw=np.array([1.72, 1.79, 1.88], dtype=float),
            train_rows_used=12,
            quantile_predictions=pd.DataFrame(
                {
                    "prediction_p2_5": [1.54, 1.61, 1.70],
                    "prediction_p10": [1.60, 1.68, 1.76],
                    "prediction_p50": [1.72, 1.79, 1.88],
                    "prediction_p90": [1.84, 1.92, 2.00],
                    "prediction_p97_5": [1.90, 1.98, 2.08],
                }
            ),
        ),
        pd.DataFrame(
            [
                {
                    "fold": 1,
                    "eval_date": "2024-01-13",
                    "history_rows": 12,
                    "bakery_history_rows": 0,
                    "valid_rows": 3,
                    "best_iteration": 1,
                    "mae": 0.05,
                    "rmse": 0.06,
                }
            ]
        ),
    )


def _write_dummy_artifact(_model: object, path: Path) -> Path:
    path.write_bytes(b"tft-model")
    return path


def _dummy_final_model() -> SimpleNamespace:
    return SimpleNamespace(
        runtime_profile="local_cpu",
        system_info={"runtime_profile": "local_cpu"},
        git_sha="test-sha",
        normalization_strategy={"kind": "group_normalizer"},
        artifact_bundle_version=2,
        interpretability_payload={"variable_selection": {"encoder": {"feat": 1.0}}},
        bundle_manifest=None,
        data_hashes={},
    )


def _write_dummy_plot(_frame: pd.DataFrame, path: Path) -> None:
    path.write_bytes(b"plot")


def _assert_written_artifacts(output_dir: Path) -> None:
    expected_names = [
        "foundation_tft_test_metrics.json",
        "foundation_tft_probabilistic_metrics.json",
        "foundation_tft_test_predictions.csv",
        "foundation_tft_probabilistic_predictions.csv",
        "foundation_tft_final_model.pt",
        "foundation_tft_feature_manifest.json",
        "foundation_tft_feature_roles.json",
        "foundation_tft_split_manifest.json",
        "foundation_tft_target_contract.json",
        "foundation_tft_interpretability.json",
        "foundation_tft_promotable_bundle_manifest.json",
        "foundation_tft_actual_vs_predicted.png",
        "foundation_tft_residuals.png",
        "foundation_tft_test_diagnostics.json",
        "foundation_tft_model_card.json",
    ]
    for name in expected_names:
        assert_path = output_dir / name
        if not assert_path.exists():
            raise AssertionError(f"Expected artifact missing: {assert_path}")


def _assert_prediction_exports(output_dir: Path) -> None:
    canonical_predictions = pd.read_csv(output_dir / "foundation_tft_test_predictions.csv")
    probabilistic_predictions = pd.read_csv(output_dir / "foundation_tft_probabilistic_predictions.csv")
    assert "prediction_p50" not in canonical_predictions.columns
    assert "prediction_p50" in probabilistic_predictions.columns
    for column in ["lower_80", "upper_80", "lower_95", "upper_95"]:
        assert canonical_predictions[column].isna().all()


def _assert_metric_exports(output_dir: Path) -> None:
    canonical_metrics_payload = json.loads((output_dir / "foundation_tft_test_metrics.json").read_text("utf-8"))
    canonical_overall = cast(dict[str, object], canonical_metrics_payload["overall_metrics"])
    canonical_product = cast(dict[str, dict[str, object]], canonical_metrics_payload["per_product_metrics"])
    assert "pinball_loss_p50" not in canonical_overall
    assert "wape" in canonical_overall
    assert "bias" in canonical_overall
    assert "segment_metrics" in canonical_metrics_payload
    assert canonical_overall["coverage_80"] is None
    assert canonical_overall["coverage_95"] is None
    assert "pinball_loss_p50" not in canonical_product["sku_1"]
    probabilistic_payload = json.loads((output_dir / "foundation_tft_probabilistic_metrics.json").read_text("utf-8"))
    probabilistic_overall = cast(dict[str, object], probabilistic_payload["overall_metrics"])
    assert "pinball_loss_p50" in probabilistic_overall
    assert probabilistic_overall["coverage_80"] == 1.0
    assert probabilistic_overall["coverage_95"] == 1.0


def _assert_diagnostics_exports(output_dir: Path) -> None:
    diagnostics_payload = json.loads((output_dir / "foundation_tft_test_diagnostics.json").read_text("utf-8"))
    probabilistic_summary = diagnostics_payload["probabilistic_summary"]
    assert probabilistic_summary["quantile_columns"] == [
        "prediction_p2_5",
        "prediction_p10",
        "prediction_p50",
        "prediction_p90",
        "prediction_p97_5",
    ]
    assert probabilistic_summary["intervals"]["80"]["lower"] == "lower_80"
    assert "quantile_calibration_error" in diagnostics_payload
    model_card_payload = json.loads((output_dir / "foundation_tft_model_card.json").read_text("utf-8"))
    assert model_card_payload["forecast_output_contract"]["median_forecast_column"] == "prediction_p50"
    assert model_card_payload["probabilistic_forecast"]["intervals"]["95"]["coverage"] == 1.0
    assert model_card_payload["interpretability_path"].endswith("foundation_tft_interpretability.json")
    assert model_card_payload["feature_roles_path"].endswith("foundation_tft_feature_roles.json")
    promotable_manifest = json.loads((output_dir / "foundation_tft_promotable_bundle_manifest.json").read_text("utf-8"))
    assert promotable_manifest["bundle_version"] == 2
    assert promotable_manifest["artifact_paths"]["feature_manifest_json"].endswith("foundation_tft_feature_manifest.json")


def _build_refit_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TargetContract]:
    train_frame = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "product": ["A", "A"], "target": [1.0, 2.0], "target_abs": [1.0, 2.0], "feat": [0.1, 0.2]}
    )
    valid_frame = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-03"]), "product": ["A"], "target": [3.0], "target_abs": [3.0], "feat": [0.3]}
    )
    test_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-01", "2024-02-01", "2024-02-02", "2024-02-02"]),
            "product": ["B", "A", "A", "B"],
            "target": [10.0, 20.0, 30.0, 40.0],
            "target_abs": [10.0, 20.0, 30.0, 40.0],
            "feat": [1.0, 2.0, 3.0, 4.0],
            "reference_only_col": ["x", "y", "z", "w"],
        }
    )
    contract = TargetContract("target", "target_abs", "identity", None)
    return train_frame, valid_frame, test_frame, contract


class EvaluationPipelineTests(unittest.TestCase):
    def test_constant_metadata_features_are_kept_for_tft_evaluation(self) -> None:
        train_frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=3, freq="D"),
                "target": [1.0, 2.0, 3.0],
                "target_demand_qty_d_plus_1": [1.0, 2.0, 3.0],
                "country_code": ["FR", "FR", "FR"],
                "city_name": ["Paris", "Paris", "Paris"],
                "rolling_mean_7": [1.0, 2.0, 3.0],
            }
        )

        feature_cols, constant_feature_cols, identifier_feature_cols = select_feature_columns(
            train_frame,
            learning_target_col="target",
            absolute_target_col="target_demand_qty_d_plus_1",
        )

        self.assertIn("country_code", feature_cols)
        self.assertIn("city_name", feature_cols)
        self.assertIn("rolling_mean_7", feature_cols)
        self.assertEqual(constant_feature_cols, [])
        self.assertEqual(identifier_feature_cols, [])

    def test_xgboost_evaluation_keeps_store_product_and_client_identifiers(self) -> None:
        train_frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=3, freq="D"),
                "client_id": ["store_1_sku_1"] * 3,
                "location_id": ["store_1"] * 3,
                "product_id": ["sku_1"] * 3,
                "target": [1.0, 2.0, 3.0],
                "target_demand_qty_d_plus_1": [1.0, 2.0, 3.0],
                "country_code": ["FR", "FR", "FR"],
                "rolling_mean_7": [1.0, 2.0, 3.0],
            }
        )

        feature_cols, _, identifier_feature_cols = select_feature_columns(
            train_frame,
            learning_target_col="target",
            absolute_target_col="target_demand_qty_d_plus_1",
            model_backend="xgboost",
        )

        self.assertIn("client_id", feature_cols)
        self.assertIn("location_id", feature_cols)
        self.assertIn("product_id", feature_cols)
        self.assertNotIn("gold_run_id", feature_cols)
        self.assertEqual(identifier_feature_cols, [])

    def test_build_predictions_frame_populates_quantile_columns_and_intervals(self) -> None:
        reference_test_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-13", "2024-01-14"]),
                "product": ["10", "10"],
            }
        )

        predictions_df = build_predictions_frame(
            reference_test_df,
            actual=np.array([1.75, 1.80], dtype=float),
            prediction_raw=np.array([1.72, 1.79], dtype=float),
            train_rows_used=12,
            quantile_predictions=pd.DataFrame(
                {
                    "prediction_p2_5": [1.54, 1.62],
                    "prediction_p10": [1.60, 1.68],
                    "prediction_p50": [1.72, 1.79],
                    "prediction_p90": [1.84, 1.92],
                    "prediction_p97_5": [1.90, 1.98],
                }
            ),
        )

        self.assertEqual(
            list(predictions_df.columns),
            [
                "origin_date",
                "target_date",
                "actual",
                "prediction_raw",
                "prediction_rounded",
                "lower_80",
                "upper_80",
                "lower_95",
                "upper_95",
                "train_rows_used",
                "product",
                "prediction_p2_5",
                "prediction_p10",
                "prediction_p50",
                "prediction_p90",
                "prediction_p97_5",
            ],
        )
        self.assertEqual(predictions_df["lower_80"].tolist(), [1.60, 1.68])
        self.assertEqual(predictions_df["upper_80"].tolist(), [1.84, 1.92])
        self.assertEqual(predictions_df["lower_95"].tolist(), [1.54, 1.62])
        self.assertEqual(predictions_df["upper_95"].tolist(), [1.90, 1.98])
        self.assertEqual(predictions_df["prediction_p50"].tolist(), [1.72, 1.79])

    def test_compute_probabilistic_metrics_payload_exposes_quantile_metrics(self) -> None:
        predictions_df = build_predictions_frame(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
                    "product": ["10", "10"],
                }
            ),
            actual=np.array([1.75, 1.80], dtype=float),
            prediction_raw=np.array([1.72, 1.79], dtype=float),
            train_rows_used=12,
            quantile_predictions=pd.DataFrame(
                {
                    "prediction_p2_5": [1.54, 1.62],
                    "prediction_p10": [1.60, 1.68],
                    "prediction_p50": [1.72, 1.79],
                    "prediction_p90": [1.84, 1.92],
                    "prediction_p97_5": [1.90, 1.98],
                }
            ),
        )

        metrics_payload = compute_probabilistic_metrics_payload(predictions_df)
        overall_metrics = cast(dict[str, object], metrics_payload["overall_metrics"])
        per_product_metrics = cast(dict[str, dict[str, object]], metrics_payload["per_product_metrics"])

        self.assertIn("pinball_loss_p10", overall_metrics)
        self.assertIn("pinball_loss_p50", overall_metrics)
        self.assertIn("pinball_loss_p90", overall_metrics)
        self.assertIn("pinball_loss_p2_5", overall_metrics)
        self.assertIn("pinball_loss_p97_5", overall_metrics)
        self.assertIn("interval_width_80", overall_metrics)
        self.assertIn("interval_width_95", overall_metrics)
        self.assertEqual(overall_metrics["coverage_95"], 1.0)
        self.assertIn("pinball_loss_p50", per_product_metrics["10"])

    def test_rmse_score_handles_large_residuals_without_overflow(self) -> None:
        rmse = rmse_score(
            np.array([1.0e200, 1.0e200], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        )

        self.assertTrue(np.isfinite(rmse))
        self.assertEqual(rmse, 1.0e200)

    def test_build_daily_walk_forward_folds_creates_one_fold_per_day(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-04-01", "2024-04-01", "2024-04-02", "2024-04-03"]),
                "target": [1.0, 2.0, 3.0, 4.0],
            }
        )

        folds = build_daily_walk_forward_folds(frame)

        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[0]["eval_date"], "2024-04-01")
        self.assertEqual(folds[0]["history_rows"], 0)
        self.assertEqual(folds[1]["history_rows"], 2)
        self.assertEqual(folds[2]["valid_rows"], 1)

    def test_build_evaluation_outputs_writes_metrics_predictions_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selection_path, tuning_path, val_path, params_path, output_dir, val_frame = _write_eval_fixture_inputs(root)
            stub_predictions, stub_daily_report = _build_stub_predictions(val_frame)

            with (
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.evaluate_daily_refit_predictions",
                    return_value=(stub_predictions, stub_daily_report, 1),
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.fit_final_model",
                    return_value=_dummy_final_model(),
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.save_tft_model",
                    side_effect=_write_dummy_artifact,
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.plot_actual_vs_predicted",
                    side_effect=_write_dummy_plot,
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.plot_residuals",
                    side_effect=_write_dummy_plot,
                ),
            ):
                build_evaluation_outputs(
                    EvaluationBuildRequest(
                        train_selection_input_path=selection_path,
                        train_tuning_input_path=tuning_path,
                        val_input_path=val_path,
                        best_params_path=params_path,
                        output_dir=output_dir,
                    )
                )
            _assert_written_artifacts(output_dir)
            _assert_prediction_exports(output_dir)
            _assert_metric_exports(output_dir)
            _assert_diagnostics_exports(output_dir)

    def test_build_xgboost_evaluation_outputs_writes_xgboost_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selection_path, tuning_path, val_path, params_path, output_dir, val_frame = _write_eval_fixture_inputs(root)
            params_path.write_text(
                json.dumps({"model_backend": "xgboost", "n_jobs": 1}),
                encoding="utf-8",
            )
            stub_predictions, stub_daily_report = _build_stub_predictions(val_frame)

            with (
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.raise_if_xgboost_backend_required",
                    return_value=None,
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.evaluate_xgboost_daily_refit_predictions",
                    return_value=(stub_predictions, stub_daily_report, 1),
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.fit_final_xgboost_model",
                    return_value=_dummy_final_model(),
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.save_xgboost_model",
                    side_effect=_write_dummy_artifact,
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.plot_actual_vs_predicted",
                    side_effect=_write_dummy_plot,
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.plot_residuals",
                    side_effect=_write_dummy_plot,
                ),
            ):
                build_evaluation_outputs(
                    EvaluationBuildRequest(
                        train_selection_input_path=selection_path,
                        train_tuning_input_path=tuning_path,
                        val_input_path=val_path,
                        best_params_path=params_path,
                        output_dir=output_dir,
                        model_backend="xgboost",
                    )
                )

            self.assertTrue((output_dir / "xgboost_test_metrics.json").exists())
            self.assertTrue((output_dir / "xgboost_test_predictions.csv").exists())
            self.assertTrue((output_dir / "xgboost_final_model.json").exists())
            metadata = json.loads((output_dir / "xgboost_evaluation_metadata.json").read_text("utf-8"))
            self.assertEqual(metadata["model_backend"], "xgboost")
            self.assertEqual(metadata["model_family"], "xgboost")
            self.assertTrue(metadata["daily_refit"])

    def test_daily_refit_predictions_keep_actuals_aligned_with_product_and_target_date(self) -> None:
        train_frame, valid_frame, test_frame, target_contract = _build_refit_frames()
        train_row_counts: list[int] = []
        train_fit_columns: list[list[str]] = []

        def _record_fit_call(**kwargs: object) -> tuple[str, int]:
            train_row_counts.append(len(cast(pd.DataFrame, kwargs["train_frame"])))
            train_fit_columns.append(list(cast(pd.DataFrame, kwargs["train_frame"]).columns))
            return "model", 7

        def _predict_absolute_stub(_model: object, frame: pd.DataFrame, **_: object) -> np.ndarray:
            return cast(np.ndarray, frame["target_abs"].to_numpy())

        with (
            patch("praedixa.demand_forecast.evaluation.pipeline._fit_evaluation_model", side_effect=_record_fit_call),
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._predict_absolute",
                side_effect=_predict_absolute_stub,
            ),
        ):
            predictions_df, daily_report_df, best_iteration = evaluate_daily_refit_predictions(
                train_frame=train_frame,
                valid_frame=valid_frame,
                test_frame=test_frame,
                feature_cols=["feat"],
                target_contract=target_contract,
                model_params={},
            )

        self.assertEqual(best_iteration, 7)
        self.assertEqual(len(daily_report_df), 2)
        self.assertEqual(train_row_counts, [2, 4])
        self.assertEqual(train_fit_columns, [list(train_frame.columns), list(train_frame.columns)])
        self.assertEqual(daily_report_df["bakery_history_rows"].tolist(), [0, 2])
        pd.testing.assert_frame_equal(
            predictions_df.loc[:, ["product", "target_date", "actual"]].reset_index(drop=True),
            pd.DataFrame(
                {
                    "product": ["A", "A", "B", "B"],
                    "target_date": ["2024-02-01", "2024-02-02", "2024-02-01", "2024-02-02"],
                    "actual": [20.0, 30.0, 10.0, 40.0],
                }
            ),
        )

    def test_final_model_fit_aligns_test_frame_to_train_valid_schema(self) -> None:
        train_frame, valid_frame, test_frame, target_contract = _build_refit_frames()
        recorded: dict[str, object] = {}
        context = SimpleNamespace(
            train_frame=train_frame,
            valid_frame=valid_frame,
            test_frame=test_frame,
            feature_cols=["feat"],
            target_contract=target_contract,
            best_params={"runtime_profile": "local_cpu"},
        )

        def _record_final_fit(**kwargs: object) -> str:
            fit_frame = cast(pd.DataFrame, kwargs["fit_frame"])
            recorded["columns"] = list(fit_frame.columns)
            recorded["rows"] = len(fit_frame)
            return "final-model"

        result = _FIT_FINAL_MODEL(
            context=context,
            best_iteration=2,
            fit_final_model_fn=_record_final_fit,
        )

        self.assertEqual(result, "final-model")
        self.assertEqual(recorded["columns"], list(train_frame.columns))
        self.assertEqual(recorded["rows"], len(train_frame) + len(valid_frame))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from typing import Callable, cast
import unittest
import warnings
from unittest.mock import patch

import pandas as pd
import numpy as np


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.evaluation import pipeline as evaluation_pipeline  # noqa: E402
import praedixa.demand_forecast.evaluation.orchestrator_runtime as evaluation_orchestrator_runtime  # noqa: E402
import praedixa.demand_forecast.evaluation.modeling as evaluation_modeling  # noqa: E402
import praedixa.demand_forecast.evaluation.xgboost_runtime as xgboost_runtime  # noqa: E402
import praedixa.demand_forecast.evaluation.chronos2_runtime as chronos2_runtime  # noqa: E402
import praedixa.demand_forecast.evaluation.moirai_runtime as moirai_runtime  # noqa: E402
import praedixa.demand_forecast.evaluation.timesfm_runtime as timesfm_runtime  # noqa: E402
from praedixa.demand_forecast.backends.chronos2.model import (  # noqa: E402
    Chronos2ZeroShotModel,
    fit_chronos2_zero_shot_model,
    predict_chronos2_quantiles,
)
from praedixa.demand_forecast.backends.moirai.model import (  # noqa: E402
    MoiraiZeroShotModel,
    predict_moirai_quantiles,
)
from praedixa.demand_forecast.backends.timesfm.model import (  # noqa: E402
    TimesFMZeroShotModel,
    predict_timesfm_quantiles,
)
from praedixa.demand_forecast.backends.foundation_covariates import (  # noqa: E402
    build_covariate_schema,
)
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
ModelParamsBuilder = Callable[[dict[str, object]], dict[str, object]]
CHRONOS2_MODEL_PARAMS = cast(
    ModelParamsBuilder,
    getattr(chronos2_runtime, "_chronos2_model_params"),
)
MOIRAI_MODEL_PARAMS = cast(
    ModelParamsBuilder,
    getattr(moirai_runtime, "_moirai_model_params"),
)
TIMESFM_MODEL_PARAMS = cast(
    ModelParamsBuilder,
    getattr(timesfm_runtime, "_timesfm_model_params"),
)


def _build_eval_output_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_selection = pd.DataFrame(
        {
            "dt": pd.date_range("2024-01-01", periods=8, freq="D"),
            "dataset_source": ["unit_test"] * 8,
            "location_id": ["store_1"] * 8,
            "product_id": ["sku_1"] * 8,
            "client_id": ["public_client"] * 8,
            "target_day_of_week": list(range(7)) + [0],
            "lag_1": [1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45],
            "lag_7": [1.0] * 8,
            "rolling_mean_7": [1.0, 1.05, 1.1, 1.12, 1.15, 1.18, 1.2, 1.22],
            "target": [1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5],
            "target_demand_qty_d_plus_1": [
                1.1,
                1.2,
                1.25,
                1.3,
                1.35,
                1.4,
                1.45,
                1.5,
            ],
        }
    )
    train_tuning = pd.DataFrame(
        {
            "dt": pd.date_range("2024-01-09", periods=4, freq="D"),
            "dataset_source": ["unit_test"] * 4,
            "location_id": ["store_1"] * 4,
            "product_id": ["sku_1"] * 4,
            "client_id": ["public_client"] * 4,
            "target_day_of_week": [1, 2, 3, 4],
            "lag_1": [1.5, 1.55, 1.6, 1.65],
            "lag_7": [1.1, 1.2, 1.25, 1.3],
            "rolling_mean_7": [1.28, 1.32, 1.36, 1.4],
            "target": [1.55, 1.6, 1.65, 1.7],
            "target_demand_qty_d_plus_1": [1.55, 1.6, 1.65, 1.7],
        }
    )
    val_frame = pd.DataFrame(
        {
            "dt": pd.date_range("2024-01-13", periods=3, freq="D"),
            "dataset_source": ["unit_test"] * 3,
            "location_id": ["store_1"] * 3,
            "product_id": ["sku_1"] * 3,
            "client_id": ["public_client"] * 3,
            "target_day_of_week": [5, 6, 0],
            "lag_1": [1.7, 1.75, 1.8],
            "lag_7": [1.35, 1.4, 1.45],
            "rolling_mean_7": [1.45, 1.5, 1.55],
            "target": [1.75, 1.8, 1.85],
            "target_demand_qty_d_plus_1": [1.75, 1.8, 1.85],
        }
    )
    return train_selection, train_tuning, val_frame


def _write_eval_fixture_inputs(
    root: Path,
) -> tuple[Path, Path, Path, Path, Path, pd.DataFrame]:
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


def _build_stub_predictions(
    val_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    canonical_predictions = pd.read_csv(
        output_dir / "foundation_tft_test_predictions.csv"
    )
    probabilistic_predictions = pd.read_csv(
        output_dir / "foundation_tft_probabilistic_predictions.csv"
    )
    assert "prediction_p50" not in canonical_predictions.columns
    assert "prediction_p50" in probabilistic_predictions.columns
    for column in ["lower_80", "upper_80", "lower_95", "upper_95"]:
        assert canonical_predictions[column].isna().all()


def _assert_metric_exports(output_dir: Path) -> None:
    canonical_metrics_payload = json.loads(
        (output_dir / "foundation_tft_test_metrics.json").read_text("utf-8")
    )
    canonical_overall = cast(
        dict[str, object], canonical_metrics_payload["overall_metrics"]
    )
    canonical_product = cast(
        dict[str, dict[str, object]], canonical_metrics_payload["per_product_metrics"]
    )
    assert "pinball_loss_p50" not in canonical_overall
    assert "wape" in canonical_overall
    assert "bias" in canonical_overall
    assert "segment_metrics" in canonical_metrics_payload
    assert canonical_overall["coverage_80"] is None
    assert canonical_overall["coverage_95"] is None
    assert "pinball_loss_p50" not in canonical_product["sku_1"]
    probabilistic_payload = json.loads(
        (output_dir / "foundation_tft_probabilistic_metrics.json").read_text("utf-8")
    )
    probabilistic_overall = cast(
        dict[str, object], probabilistic_payload["overall_metrics"]
    )
    assert "pinball_loss_p50" in probabilistic_overall
    assert probabilistic_overall["coverage_80"] == 1.0
    assert probabilistic_overall["coverage_95"] == 1.0


def _assert_diagnostics_exports(output_dir: Path) -> None:
    diagnostics_payload = json.loads(
        (output_dir / "foundation_tft_test_diagnostics.json").read_text("utf-8")
    )
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
    model_card_payload = json.loads(
        (output_dir / "foundation_tft_model_card.json").read_text("utf-8")
    )
    assert (
        model_card_payload["forecast_output_contract"]["median_forecast_column"]
        == "prediction_p50"
    )
    assert (
        model_card_payload["probabilistic_forecast"]["intervals"]["95"]["coverage"]
        == 1.0
    )
    assert model_card_payload["interpretability_path"].endswith(
        "foundation_tft_interpretability.json"
    )
    assert model_card_payload["feature_roles_path"].endswith(
        "foundation_tft_feature_roles.json"
    )
    promotable_manifest = json.loads(
        (output_dir / "foundation_tft_promotable_bundle_manifest.json").read_text(
            "utf-8"
        )
    )
    assert promotable_manifest["bundle_version"] == 2
    assert promotable_manifest["artifact_paths"]["feature_manifest_json"].endswith(
        "foundation_tft_feature_manifest.json"
    )


def _build_refit_frames() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, TargetContract
]:
    train_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "dataset_source": ["synthetic_foodservice_bakery"] * 2,
            "product": ["A", "A"],
            "target": [1.0, 2.0],
            "target_abs": [1.0, 2.0],
            "feat": [0.1, 0.2],
        }
    )
    valid_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03"]),
            "dataset_source": ["synthetic_foodservice_bakery"],
            "product": ["A"],
            "target": [3.0],
            "target_abs": [3.0],
            "feat": [0.3],
        }
    )
    test_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-02-01", "2024-02-01", "2024-02-02", "2024-02-02"]
            ),
            "dataset_source": ["bakery"] * 4,
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
    def test_foundation_daily_calibration_defaults_match_xgboost_raw_refit(
        self,
    ) -> None:
        for params_builder in (
            CHRONOS2_MODEL_PARAMS,
            MOIRAI_MODEL_PARAMS,
            TIMESFM_MODEL_PARAMS,
        ):
            with self.subTest(params_builder=getattr(params_builder, "__name__", "")):
                params = params_builder({})
                self.assertFalse(params["enable_daily_prediction_calibration"])
                explicit_params = params_builder(
                    {"enable_daily_prediction_calibration": True}
                )
                self.assertTrue(
                    explicit_params["enable_daily_prediction_calibration"]
                )

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

        feature_cols, constant_feature_cols, identifier_feature_cols = (
            select_feature_columns(
                train_frame,
                learning_target_col="target",
                absolute_target_col="target_demand_qty_d_plus_1",
            )
        )

        self.assertIn("country_code", feature_cols)
        self.assertIn("city_name", feature_cols)
        self.assertIn("rolling_mean_7", feature_cols)
        self.assertEqual(constant_feature_cols, [])
        self.assertEqual(identifier_feature_cols, [])

    def test_xgboost_evaluation_keeps_store_product_and_client_identifiers(
        self,
    ) -> None:
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

    def test_build_predictions_frame_populates_quantile_columns_and_intervals(
        self,
    ) -> None:
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

    def test_compute_probabilistic_metrics_payload_exposes_quantile_metrics(
        self,
    ) -> None:
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
        per_product_metrics = cast(
            dict[str, dict[str, object]], metrics_payload["per_product_metrics"]
        )

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
                "dt": pd.to_datetime(
                    ["2024-04-01", "2024-04-01", "2024-04-02", "2024-04-03"]
                ),
                "target": [1.0, 2.0, 3.0, 4.0],
            }
        )

        folds = build_daily_walk_forward_folds(frame)

        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[0]["eval_date"], "2024-04-01")
        self.assertEqual(folds[0]["history_rows"], 0)
        self.assertEqual(folds[1]["history_rows"], 2)
        self.assertEqual(folds[2]["valid_rows"], 1)

    def test_build_evaluation_outputs_writes_metrics_predictions_and_plots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (
                selection_path,
                tuning_path,
                val_path,
                params_path,
                output_dir,
                val_frame,
            ) = _write_eval_fixture_inputs(root)
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
            (
                selection_path,
                tuning_path,
                val_path,
                params_path,
                output_dir,
                val_frame,
            ) = _write_eval_fixture_inputs(root)
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
            metadata = json.loads(
                (output_dir / "xgboost_evaluation_metadata.json").read_text("utf-8")
            )
            self.assertEqual(metadata["model_backend"], "xgboost")
            self.assertEqual(metadata["model_family"], "xgboost")
            self.assertTrue(metadata["daily_refit"])

    def test_build_chronos2_evaluation_outputs_writes_chronos2_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (
                selection_path,
                tuning_path,
                val_path,
                params_path,
                output_dir,
                val_frame,
            ) = _write_eval_fixture_inputs(root)
            params_path.write_text(
                json.dumps(
                    {
                        "model_backend": "chronos2",
                        "model_path": "amazon/chronos-2",
                        "device_map": "cpu",
                    }
                ),
                encoding="utf-8",
            )
            stub_predictions, stub_daily_report = _build_stub_predictions(val_frame)

            with (
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.raise_if_chronos2_backend_required",
                    return_value=None,
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.evaluate_chronos2_daily_refit_predictions",
                    return_value=(stub_predictions, stub_daily_report, 0),
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.fit_final_chronos2_model",
                    return_value=_dummy_final_model(),
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.save_chronos2_model",
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
                        model_backend="chronos2",
                    )
                )

            self.assertTrue((output_dir / "chronos2_test_metrics.json").exists())
            self.assertTrue(
                (output_dir / "chronos2_probabilistic_predictions.csv").exists()
            )
            self.assertTrue((output_dir / "chronos2_final_model.json").exists())
            metadata = json.loads(
                (output_dir / "chronos2_evaluation_metadata.json").read_text("utf-8")
            )
            self.assertEqual(metadata["model_backend"], "chronos2")
            self.assertEqual(metadata["model_family"], "chronos2")
            self.assertTrue(metadata["daily_refit"])

    def test_build_moirai_evaluation_outputs_writes_moirai_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (
                selection_path,
                tuning_path,
                val_path,
                params_path,
                output_dir,
                val_frame,
            ) = _write_eval_fixture_inputs(root)
            params_path.write_text(
                json.dumps(
                    {
                        "model_backend": "moirai",
                        "model_path": "Salesforce/moirai-2.0-R-small",
                    }
                ),
                encoding="utf-8",
            )
            stub_predictions, stub_daily_report = _build_stub_predictions(val_frame)

            with (
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.raise_if_moirai_backend_required",
                    return_value=None,
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.evaluate_moirai_daily_refit_predictions",
                    return_value=(stub_predictions, stub_daily_report, 0),
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.fit_final_moirai_model",
                    return_value=_dummy_final_model(),
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.save_moirai_model",
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
                        model_backend="moirai",
                    )
                )

            self.assertTrue((output_dir / "moirai_test_metrics.json").exists())
            self.assertTrue(
                (output_dir / "moirai_probabilistic_predictions.csv").exists()
            )
            self.assertTrue((output_dir / "moirai_final_model.json").exists())
            metadata = json.loads(
                (output_dir / "moirai_evaluation_metadata.json").read_text("utf-8")
            )
            self.assertEqual(metadata["model_backend"], "moirai")
            self.assertEqual(metadata["model_family"], "moirai")
            self.assertTrue(metadata["daily_refit"])

    def test_build_timesfm_evaluation_outputs_writes_timesfm_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (
                selection_path,
                tuning_path,
                val_path,
                params_path,
                output_dir,
                val_frame,
            ) = _write_eval_fixture_inputs(root)
            params_path.write_text(
                json.dumps(
                    {
                        "model_backend": "timesfm",
                        "model_path": "google/timesfm-2.0-500m-pytorch",
                    }
                ),
                encoding="utf-8",
            )
            stub_predictions, stub_daily_report = _build_stub_predictions(val_frame)

            with (
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.raise_if_timesfm_backend_required",
                    return_value=None,
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.evaluate_timesfm_daily_refit_predictions",
                    return_value=(stub_predictions, stub_daily_report, 0),
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.fit_final_timesfm_model",
                    return_value=_dummy_final_model(),
                ),
                patch(
                    "praedixa.demand_forecast.evaluation.orchestrator.save_timesfm_model",
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
                        model_backend="timesfm",
                    )
                )

            self.assertTrue((output_dir / "timesfm_test_metrics.json").exists())
            self.assertTrue(
                (output_dir / "timesfm_probabilistic_predictions.csv").exists()
            )
            self.assertTrue((output_dir / "timesfm_final_model.json").exists())
            metadata = json.loads(
                (output_dir / "timesfm_evaluation_metadata.json").read_text("utf-8")
            )
            self.assertEqual(metadata["model_backend"], "timesfm")
            self.assertEqual(metadata["model_family"], "timesfm")
            self.assertTrue(metadata["daily_refit"])

    def test_chronos2_quantile_predictions_align_by_series_and_timestamp(self) -> None:
        class FakeChronosPipeline:
            def predict_df(self, *args: object, **kwargs: object) -> pd.DataFrame:
                future_df = cast(pd.DataFrame, kwargs["future_df"])
                return pd.DataFrame(
                    {
                        "series_id": future_df["series_id"].tolist(),
                        "timestamp": future_df["timestamp"].tolist(),
                        "0.1": [9.0, 19.0],
                        "0.5": [10.0, 20.0],
                        "0.9": [11.0, 21.0],
                    }
                )

        context_frame = pd.DataFrame(
            {
                "series_id": ["store__A", "store__B"],
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "target": [8.0, 18.0],
                "calendar_feature": [1, 1],
            }
        )
        future_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "product_id": ["A", "B"],
                "client_id": ["store__A", "store__B"],
                "target_abs": [10.0, 20.0],
                "calendar_feature": [2, 2],
            }
        )
        model = Chronos2ZeroShotModel(
            pipeline=FakeChronosPipeline(),
            context_frame=context_frame,
            feature_cols=["calendar_feature"],
            series_order=["store__A", "store__B"],
            covariate_kinds={"calendar_feature": "numeric"},
        )

        quantiles = predict_chronos2_quantiles(model, future_frame)

        pd.testing.assert_frame_equal(
            quantiles,
            pd.DataFrame(
                {
                    "prediction_p10": [9.0, 19.0],
                    "prediction_p50": [10.0, 20.0],
                    "prediction_p90": [11.0, 21.0],
                }
            ),
        )

    def test_chronos2_prediction_frame_build_avoids_fragmentation_warning(self) -> None:
        feature_cols = [f"feature_{idx}" for idx in range(180)]

        class FakeChronosPipeline:
            def predict_df(self, *args: object, **kwargs: object) -> pd.DataFrame:
                future_df = cast(pd.DataFrame, kwargs["future_df"])
                return pd.DataFrame(
                    {
                        "series_id": future_df["series_id"].tolist(),
                        "timestamp": future_df["timestamp"].tolist(),
                        "0.1": [1.0, 2.0],
                        "0.5": [1.5, 2.5],
                        "0.9": [2.0, 3.0],
                    }
                )

        context_frame = pd.DataFrame(
            {
                "series_id": ["store__A", "store__B"],
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "target": [8.0, 18.0],
                **{column: [1.0, 2.0] for column in feature_cols},
            }
        )
        future_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "product_id": ["A", "B"],
                "client_id": ["store__A", "store__B"],
                "target_abs": [10.0, 20.0],
                **{column: [3.0, 4.0] for column in feature_cols},
            }
        )
        model = Chronos2ZeroShotModel(
            pipeline=FakeChronosPipeline(),
            context_frame=context_frame,
            feature_cols=feature_cols,
            series_order=["store__A", "store__B"],
            covariate_kinds={column: "numeric" for column in feature_cols},
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            quantiles = predict_chronos2_quantiles(model, future_frame)

        self.assertEqual(len(quantiles), 2)

    def test_chronos2_cold_start_context_avoids_fragmentation_warning(self) -> None:
        feature_cols = [f"feature_{idx}" for idx in range(180)]
        rows = 64
        train_frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=rows, freq="D"),
                "product_id": ["sku_a"] * rows,
                "client_id": ["store_a"] * rows,
                "target_abs": np.linspace(1.0, 2.0, rows),
                **{
                    column: np.linspace(0.0, 1.0, rows)
                    for column in feature_cols
                },
            }
        )

        with (
            warnings.catch_warnings(),
            patch(
                "praedixa.demand_forecast.backends.chronos2.model.load_chronos2_pipeline",
                return_value=object(),
            ),
        ):
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            model, _ = fit_chronos2_zero_shot_model(
                train_frame=train_frame,
                valid_frame=train_frame.head(0),
                feature_cols=feature_cols,
                target_col="target_abs",
                model_params={},
            )

        self.assertIn("__praedixa_foundation_cold_start__", model.series_order)

    def test_chronos2_quantile_predictions_use_cold_start_context_for_unknown_series(
        self,
    ) -> None:
        class FakeChronosPipeline:
            def predict_df(self, *args: object, **kwargs: object) -> pd.DataFrame:
                _ = args
                future_df = cast(pd.DataFrame, kwargs["future_df"])
                return pd.DataFrame(
                    {
                        "series_id": future_df["series_id"].tolist(),
                        "timestamp": future_df["timestamp"].tolist(),
                        "0.1": [4.0, 4.0],
                        "0.5": [5.0, 5.0],
                        "0.9": [6.0, 6.0],
                    }
                )

        context_frame = pd.DataFrame(
            {
                "series_id": [
                    "fresh_store__1",
                    "__praedixa_foundation_cold_start__",
                ],
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "target": [8.0, 8.0],
            }
        )
        future_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "product_id": ["BAGUETTE", "CROISSANT"],
                "dataset_source": ["bakery", "bakery"],
                "target_abs": [10.0, 20.0],
            }
        )
        model = Chronos2ZeroShotModel(
            pipeline=FakeChronosPipeline(),
            context_frame=context_frame,
            feature_cols=[],
            series_order=["fresh_store__1"],
            covariate_kinds={},
        )

        quantiles = predict_chronos2_quantiles(model, future_frame)

        self.assertEqual(quantiles["prediction_p50"].tolist(), [5.0, 5.0])

    def test_moirai_quantile_predictions_align_by_series_order(self) -> None:
        class FakeForecast:
            def __init__(self, values: tuple[float, float, float]) -> None:
                self.values = values

            def quantile(self, quantile: float) -> np.ndarray:
                if quantile == 0.1:
                    return np.asarray([self.values[0]])
                if quantile == 0.9:
                    return np.asarray([self.values[2]])
                return np.asarray([self.values[1]])

        class FakeMoiraiPredictor:
            def predict(self, inputs: object) -> list[FakeForecast]:
                _ = inputs
                return [
                    FakeForecast((9.0, 10.0, 11.0)),
                    FakeForecast((19.0, 20.0, 21.0)),
                ]

        context_frame = pd.DataFrame(
            {
                "series_id": ["store__A", "store__B"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "target_abs": [8.0, 18.0],
            }
        )
        future_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "product_id": ["A", "B"],
                "client_id": ["store__A", "store__B"],
                "target_abs": [10.0, 20.0],
            }
        )
        model = MoiraiZeroShotModel(
            predictor=FakeMoiraiPredictor(),
            context_frame=context_frame,
            series_order=["store__A", "store__B"],
            target_col="target_abs",
        )

        quantiles = predict_moirai_quantiles(model, future_frame)

        pd.testing.assert_frame_equal(
            quantiles,
            pd.DataFrame(
                {
                    "prediction_p10": [9.0, 19.0],
                    "prediction_p50": [10.0, 20.0],
                    "prediction_p90": [11.0, 21.0],
                }
            ),
        )

    def test_moirai_quantile_predictions_use_cold_start_context_for_unknown_series(
        self,
    ) -> None:
        class FakeForecast:
            def quantile(self, quantile: float) -> np.ndarray:
                values = {0.1: 4.0, 0.5: 5.0, 0.9: 6.0}
                return np.asarray([values[quantile]])

        class FakeMoiraiPredictor:
            def predict(self, inputs: object) -> list[FakeForecast]:
                return [FakeForecast() for _ in cast(list[object], inputs)]

        context_frame = pd.DataFrame(
            {
                "series_id": [
                    "fresh_store__1",
                    "__praedixa_foundation_cold_start__",
                ],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "target_abs": [8.0, 8.0],
            }
        )
        future_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "product_id": ["BAGUETTE", "CROISSANT"],
                "dataset_source": ["bakery", "bakery"],
                "target_abs": [10.0, 20.0],
            }
        )
        model = MoiraiZeroShotModel(
            predictor=FakeMoiraiPredictor(),
            context_frame=context_frame,
            series_order=["fresh_store__1"],
            target_col="target_abs",
        )

        quantiles = predict_moirai_quantiles(model, future_frame)

        self.assertEqual(quantiles["prediction_p50"].tolist(), [5.0, 5.0])

    def test_moirai_quantile_predictions_pass_dynamic_real_covariates(self) -> None:
        class FakeForecast:
            def quantile(self, quantile: float) -> np.ndarray:
                values = {0.1: 9.0, 0.5: 10.0, 0.9: 11.0}
                return np.asarray([values[quantile]])

        class FakeMoiraiPredictor:
            def __init__(self) -> None:
                self.inputs: list[dict[str, object]] = []

            def predict(self, inputs: object) -> list[FakeForecast]:
                self.inputs = list(cast(list[dict[str, object]], inputs))
                return [FakeForecast() for _ in self.inputs]

        context_frame = pd.DataFrame(
            {
                "series_id": ["store__A"],
                "date": pd.to_datetime(["2024-01-01"]),
                "target_abs": [8.0],
                "weather_temperature": [12.0],
                "lag_7": [7.0],
            }
        )
        future_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-02"]),
                "product_id": ["A"],
                "client_id": ["store__A"],
                "target_abs": [10.0],
                "weather_temperature": [14.0],
                "lag_7": [8.0],
            }
        )
        feature_cols = ["weather_temperature", "lag_7"]
        schema = build_covariate_schema(context_frame, feature_cols)
        predictor = FakeMoiraiPredictor()
        model = MoiraiZeroShotModel(
            predictor=predictor,
            context_frame=context_frame,
            series_order=["store__A"],
            target_col="target_abs",
            feature_cols=feature_cols,
            covariate_schema=schema,
        )

        predict_moirai_quantiles(model, future_frame)

        self.assertEqual(len(predictor.inputs), 1)
        dynamic_real = cast(np.ndarray, predictor.inputs[0]["feat_dynamic_real"])
        np.testing.assert_allclose(
            dynamic_real,
            np.asarray([[12.0, 14.0], [7.0, 8.0]], dtype=float),
        )

    def test_timesfm_quantile_predictions_align_by_series_order(self) -> None:
        class FakeTimesFMPipeline:
            def forecast(
                self, contexts: object, *, freq: list[int]
            ) -> tuple[np.ndarray, np.ndarray]:
                _ = contexts, freq
                return (
                    np.asarray([[10.0], [20.0]]),
                    np.asarray([[[9.0, 10.0, 11.0]], [[19.0, 20.0, 21.0]]]),
                )

        context_frame = pd.DataFrame(
            {
                "series_id": ["store__A", "store__B"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "target_abs": [8.0, 18.0],
            }
        )
        future_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "product_id": ["A", "B"],
                "client_id": ["store__A", "store__B"],
                "target_abs": [10.0, 20.0],
            }
        )
        model = TimesFMZeroShotModel(
            pipeline=FakeTimesFMPipeline(),
            context_frame=context_frame,
            series_order=["store__A", "store__B"],
            target_col="target_abs",
        )

        quantiles = predict_timesfm_quantiles(model, future_frame)

        pd.testing.assert_frame_equal(
            quantiles,
            pd.DataFrame(
                {
                    "prediction_p10": [9.0, 19.0],
                    "prediction_p50": [10.0, 20.0],
                    "prediction_p90": [11.0, 21.0],
                }
            ),
        )

    def test_timesfm_quantile_predictions_use_cold_start_context_for_unknown_series(
        self,
    ) -> None:
        class FakeTimesFMPipeline:
            def forecast(
                self, contexts: object, *, freq: list[int]
            ) -> tuple[np.ndarray, np.ndarray]:
                context_count = len(cast(list[np.ndarray], contexts))
                _ = freq
                return (
                    np.full((context_count, 1), 5.0),
                    np.full((context_count, 1, 3), 5.0),
                )

        context_frame = pd.DataFrame(
            {
                "series_id": [
                    "fresh_store__1",
                    "__praedixa_foundation_cold_start__",
                ],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "target_abs": [8.0, 8.0],
            }
        )
        future_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "product_id": ["BAGUETTE", "CROISSANT"],
                "dataset_source": ["bakery", "bakery"],
                "target_abs": [10.0, 20.0],
            }
        )
        model = TimesFMZeroShotModel(
            pipeline=FakeTimesFMPipeline(),
            context_frame=context_frame,
            series_order=["fresh_store__1"],
            target_col="target_abs",
        )

        quantiles = predict_timesfm_quantiles(model, future_frame)

        self.assertEqual(quantiles["prediction_p50"].tolist(), [5.0, 5.0])

    def test_timesfm_quantile_predictions_use_forecast_with_covariates(self) -> None:
        class FakeTimesFMPipeline:
            def __init__(self) -> None:
                self.kwargs: dict[str, object] = {}

            def forecast_with_covariates(
                self, **kwargs: object
            ) -> tuple[np.ndarray, np.ndarray]:
                self.kwargs = kwargs
                return (
                    np.asarray([[10.0], [20.0]]),
                    np.asarray([[[9.0, 10.0, 11.0]], [[19.0, 20.0, 21.0]]]),
                )

        context_frame = pd.DataFrame(
            {
                "series_id": ["store__A", "store__B"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "target_abs": [8.0, 18.0],
                "weather_temperature": [12.0, 13.0],
                "promo_type": ["none", "none"],
            }
        )
        future_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "product_id": ["A", "B"],
                "client_id": ["store__A", "store__B"],
                "target_abs": [10.0, 20.0],
                "weather_temperature": [14.0, 15.0],
                "promo_type": ["event", "none"],
            }
        )
        feature_cols = ["weather_temperature", "promo_type"]
        schema = build_covariate_schema(context_frame, feature_cols)
        pipeline = FakeTimesFMPipeline()
        model = TimesFMZeroShotModel(
            pipeline=pipeline,
            context_frame=context_frame,
            series_order=["store__A", "store__B"],
            target_col="target_abs",
            feature_cols=feature_cols,
            covariate_schema=schema,
        )

        quantiles = predict_timesfm_quantiles(model, future_frame)

        pd.testing.assert_frame_equal(
            quantiles,
            pd.DataFrame(
                {
                    "prediction_p10": [9.0, 19.0],
                    "prediction_p50": [10.0, 20.0],
                    "prediction_p90": [11.0, 21.0],
                }
            ),
        )
        dynamic_numeric = cast(
            dict[str, list[list[object]]],
            pipeline.kwargs["dynamic_numerical_covariates"],
        )
        dynamic_categorical = cast(
            dict[str, list[list[object]]],
            pipeline.kwargs["dynamic_categorical_covariates"],
        )
        self.assertEqual(
            dynamic_numeric["weather_temperature"],
            [[12.0, 14.0], [13.0, 15.0]],
        )
        self.assertEqual(
            dynamic_categorical["promo_type"],
            [["none", "event"], ["none", "none"]],
        )

    def test_daily_refit_predictions_keep_actuals_aligned_with_product_and_target_date(
        self,
    ) -> None:
        train_frame, valid_frame, test_frame, target_contract = _build_refit_frames()
        train_row_counts: list[int] = []
        train_fit_columns: list[list[str]] = []
        train_fit_weights: list[list[float] | None] = []

        def _record_fit_call(**kwargs: object) -> tuple[str, int]:
            train_fit_frame = cast(pd.DataFrame, kwargs["train_frame"])
            train_row_counts.append(len(train_fit_frame))
            train_fit_columns.append(list(train_fit_frame.columns))
            if "sample_weight_business" in train_fit_frame.columns:
                train_fit_weights.append(
                    train_fit_frame["sample_weight_business"].astype(float).tolist()
                )
            else:
                train_fit_weights.append(None)
            return "model", 7

        def _predict_absolute_stub(
            _model: object, frame: pd.DataFrame, **_: object
        ) -> np.ndarray:
            return cast(np.ndarray, frame["target_abs"].to_numpy())

        with (
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._fit_evaluation_model",
                side_effect=_record_fit_call,
            ),
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._predict_absolute",
                side_effect=_predict_absolute_stub,
            ),
        ):
            predictions_df, daily_report_df, best_iteration = (
                evaluate_daily_refit_predictions(
                    train_frame=train_frame,
                    valid_frame=valid_frame,
                    test_frame=test_frame,
                    feature_cols=["feat"],
                    target_contract=target_contract,
                    model_params={},
                )
            )

        self.assertEqual(best_iteration, 7)
        self.assertEqual(len(daily_report_df), 2)
        self.assertEqual(train_row_counts, [2, 4])
        self.assertEqual(train_fit_weights, [None, None])
        self.assertEqual(
            train_fit_columns,
            [list(train_frame.columns), list(train_frame.columns)],
        )
        self.assertEqual(daily_report_df["bakery_history_rows"].tolist(), [0, 2])
        pd.testing.assert_frame_equal(
            predictions_df.loc[:, ["product", "target_date", "actual"]].reset_index(
                drop=True
            ),
            pd.DataFrame(
                {
                    "product": ["A", "A", "B", "B"],
                    "target_date": [
                        "2024-02-01",
                        "2024-02-02",
                        "2024-02-01",
                        "2024-02-02",
                    ],
                    "actual": [20.0, 30.0, 10.0, 40.0],
                }
            ),
        )

    def test_daily_refit_predictions_can_downweight_non_bakery_history(
        self,
    ) -> None:
        train_frame, valid_frame, test_frame, target_contract = _build_refit_frames()
        train_fit_weights: list[list[float] | None] = []

        def _record_fit_call(**kwargs: object) -> tuple[str, int]:
            train_fit_frame = cast(pd.DataFrame, kwargs["train_frame"])
            if "sample_weight_business" in train_fit_frame.columns:
                train_fit_weights.append(
                    train_fit_frame["sample_weight_business"].astype(float).tolist()
                )
            else:
                train_fit_weights.append(None)
            return "model", 7

        def _predict_absolute_stub(
            _model: object, frame: pd.DataFrame, **_: object
        ) -> np.ndarray:
            return cast(np.ndarray, frame["target_abs"].to_numpy())

        with (
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._fit_evaluation_model",
                side_effect=_record_fit_call,
            ),
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._predict_absolute",
                side_effect=_predict_absolute_stub,
            ),
        ):
            evaluate_daily_refit_predictions(
                train_frame=train_frame,
                valid_frame=valid_frame,
                test_frame=test_frame,
                feature_cols=["feat"],
                target_contract=target_contract,
                model_params={
                    "daily_refit_bakery_sample_weight_multiplier": 10.0,
                    "daily_refit_non_bakery_sample_weight_multiplier": 0.1,
                    "daily_refit_bakery_min_sample_weight_multiplier": 10.0,
                },
            )

        self.assertEqual(train_fit_weights, [None, [0.1, 0.1, 10.0, 10.0]])

    def test_daily_refit_predictions_can_use_inverse_effective_progressive_weights(
        self,
    ) -> None:
        train_frame, valid_frame, test_frame, target_contract = _build_refit_frames()
        train_fit_weights: list[list[float] | None] = []

        def _record_fit_call(**kwargs: object) -> tuple[str, int]:
            train_fit_frame = cast(pd.DataFrame, kwargs["train_frame"])
            if "sample_weight_business" in train_fit_frame.columns:
                train_fit_weights.append(
                    train_fit_frame["sample_weight_business"].astype(float).tolist()
                )
            else:
                train_fit_weights.append(None)
            return "model", 7

        def _predict_absolute_stub(
            _model: object, frame: pd.DataFrame, **_: object
        ) -> np.ndarray:
            return cast(np.ndarray, frame["target_abs"].to_numpy())

        with (
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._fit_evaluation_model",
                side_effect=_record_fit_call,
            ),
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._predict_absolute",
                side_effect=_predict_absolute_stub,
            ),
        ):
            evaluate_daily_refit_predictions(
                train_frame=train_frame,
                valid_frame=valid_frame,
                test_frame=test_frame,
                feature_cols=["feat"],
                target_contract=target_contract,
                model_params={
                    "daily_refit_bakery_sample_weight_multiplier": 25.0,
                    "daily_refit_bakery_min_sample_weight_multiplier": 5.0,
                    "daily_refit_bakery_weight_warmup_days": 28,
                    "daily_refit_non_bakery_sample_weight_mode": "inverse_effective_bakery",
                },
            )

        effective_weight = 5.0 + (1.0 / 28.0) * (25.0 - 5.0)
        self.assertEqual(train_fit_weights[0], None)
        progressive_weights = cast(list[float], train_fit_weights[1])
        np.testing.assert_allclose(
            progressive_weights,
            [
                1.0 / effective_weight,
                1.0 / effective_weight,
                effective_weight,
                effective_weight,
            ],
        )

    def test_daily_refit_pretest_bakery_seed_does_not_add_weights_by_default(self) -> None:
        train_frame, valid_frame, test_frame, target_contract = _build_refit_frames()
        seed_frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-04", periods=28, freq="D"),
                "dataset_source": ["bakery"] * 28,
                "product": ["A"] * 28,
                "target": np.linspace(4.0, 31.0, 28),
                "target_abs": np.linspace(4.0, 31.0, 28),
                "feat": np.linspace(0.4, 3.1, 28),
                "bakery_refit_seed_flag": [True] * 28,
            }
        )
        train_frame = pd.concat([train_frame, seed_frame], ignore_index=True)
        train_fit_weights: list[list[float] | None] = []

        def _record_fit_call(**kwargs: object) -> tuple[str, int]:
            train_fit_frame = cast(pd.DataFrame, kwargs["train_frame"])
            if "sample_weight_business" in train_fit_frame.columns:
                train_fit_weights.append(
                    train_fit_frame["sample_weight_business"].astype(float).tolist()
                )
            else:
                train_fit_weights.append(None)
            return "model", 7

        def _predict_absolute_stub(
            _model: object, frame: pd.DataFrame, **_: object
        ) -> np.ndarray:
            return cast(np.ndarray, frame["target_abs"].to_numpy())

        with (
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._fit_evaluation_model",
                side_effect=_record_fit_call,
            ),
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._predict_absolute",
                side_effect=_predict_absolute_stub,
            ),
        ):
            evaluate_daily_refit_predictions(
                train_frame=train_frame,
                valid_frame=valid_frame,
                test_frame=test_frame,
                feature_cols=["feat"],
                target_contract=target_contract,
                model_params={},
            )

        self.assertEqual(train_fit_weights, [None, None])

    def test_daily_refit_predictions_can_apply_walk_forward_calibration(
        self,
    ) -> None:
        train_frame, valid_frame, test_frame, target_contract = _build_refit_frames()

        def _fit_stub(**_: object) -> tuple[str, int]:
            return "zero_shot_model", 1

        def _predict_underestimated(
            _model: object, frame: pd.DataFrame, **_: object
        ) -> np.ndarray:
            return cast(np.ndarray, frame["target_abs"].astype(float).to_numpy() - 10.0)

        with (
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._fit_evaluation_model",
                side_effect=_fit_stub,
            ),
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._predict_absolute",
                side_effect=_predict_underestimated,
            ),
        ):
            predictions_df, daily_report_df, _ = evaluate_daily_refit_predictions(
                train_frame=train_frame,
                valid_frame=valid_frame,
                test_frame=test_frame,
                feature_cols=["feat"],
                target_contract=target_contract,
                model_params={
                    "enable_daily_prediction_calibration": True,
                    "daily_prediction_calibration_min_rows": 2,
                },
            )

        self.assertEqual(
            daily_report_df["prediction_calibration_applied"].tolist(),
            [False, True],
        )
        self.assertEqual(
            daily_report_df["prediction_calibration_rows"].tolist(),
            [1, 3],
        )
        date_two_predictions = predictions_df.loc[
            predictions_df["target_date"] == "2024-02-02"
        ].sort_values("product")
        self.assertEqual(
            date_two_predictions["prediction_raw_uncalibrated"].tolist(),
            [20.0, 30.0],
        )
        np.testing.assert_allclose(
            date_two_predictions["prediction_raw"].to_numpy(dtype=float),
            np.array([30.0, 40.0], dtype=float),
        )
        self.assertTrue(date_two_predictions["prediction_calibration_applied"].all())

    def test_daily_refit_calibration_accepts_dt_and_product_id_frames(self) -> None:
        train_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "product_id": ["A", "A"],
                "target": [1.0, 2.0],
                "target_abs": [1.0, 2.0],
                "feat": [0.1, 0.2],
            }
        )
        valid_frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-03", "2024-01-04"]),
                "product_id": ["A", "A"],
                "target": [3.0, 4.0],
                "target_abs": [3.0, 4.0],
                "feat": [0.3, 0.4],
            }
        )
        test_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-02-01"]),
                "product": ["A"],
                "target": [5.0],
                "target_abs": [5.0],
                "feat": [1.0],
            }
        )
        contract = TargetContract("target", "target_abs", "identity", None)

        def _fit_stub(**_: object) -> tuple[str, int]:
            return "zero_shot_model", 1

        def _predict_identity(
            _model: object, frame: pd.DataFrame, **_: object
        ) -> np.ndarray:
            return cast(np.ndarray, frame["target_abs"].astype(float).to_numpy())

        with (
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._fit_evaluation_model",
                side_effect=_fit_stub,
            ),
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._predict_absolute",
                side_effect=_predict_identity,
            ),
        ):
            predictions_df, _, _ = evaluate_daily_refit_predictions(
                train_frame=train_frame,
                valid_frame=valid_frame,
                test_frame=test_frame,
                feature_cols=["feat"],
                target_contract=contract,
                model_params={
                    "enable_daily_prediction_calibration": True,
                    "initialize_daily_prediction_calibration_from_valid": True,
                    "daily_prediction_calibration_min_rows": 2,
                },
            )

        self.assertEqual(predictions_df["prediction_calibration_rows"].tolist(), [2])

    def test_daily_refit_applies_static_economic_calibration(self) -> None:
        train_frame, valid_frame, test_frame, target_contract = _build_refit_frames()

        def _fit_stub(**_: object) -> tuple[str, int]:
            return "xgboost_model", 1

        def _predict_overestimated(
            _model: object, frame: pd.DataFrame, **_: object
        ) -> np.ndarray:
            return cast(np.ndarray, frame["target_abs"].astype(float).to_numpy() + 2.0)

        with (
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._fit_evaluation_model",
                side_effect=_fit_stub,
            ),
            patch(
                "praedixa.demand_forecast.evaluation.pipeline._predict_absolute",
                side_effect=_predict_overestimated,
            ),
        ):
            predictions_df, daily_report_df, _ = evaluate_daily_refit_predictions(
                train_frame=train_frame,
                valid_frame=valid_frame,
                test_frame=test_frame,
                feature_cols=["feat"],
                target_contract=target_contract,
                model_params={
                    "economic_calibration": {
                        "version": "economic_decision_calibration_final",
                        "enabled": True,
                        "applied": True,
                        "global": {
                            "scope": "global",
                            "key": "__global__",
                            "rows": 100,
                            "slope": 1.0,
                            "intercept": -2.0,
                            "raw_decision_loss": 0.2,
                            "calibrated_decision_loss": 0.1,
                            "raw_wape": 0.2,
                            "calibrated_wape": 0.1,
                            "applied": True,
                        },
                        "families": {},
                        "products": {},
                    }
                },
            )

        np.testing.assert_allclose(
            predictions_df["prediction_raw"].to_numpy(dtype=float),
            predictions_df["actual"].to_numpy(dtype=float),
        )
        self.assertTrue(predictions_df["economic_calibration_applied"].all())
        self.assertEqual(
            daily_report_df["economic_calibration_scope"].tolist(),
            ["global", "global"],
        )

    def test_xgboost_refit_uses_all_cores_and_single_worker(self) -> None:
        train_frame, valid_frame, test_frame, target_contract = _build_refit_frames()
        recorded: dict[str, object] = {}

        def _record_refit_call(
            **kwargs: object,
        ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
            recorded["model_params"] = kwargs["model_params"]
            return pd.DataFrame(), pd.DataFrame(), 7

        with (
            patch.object(xgboost_runtime.os, "cpu_count", return_value=12),
            patch.object(
                xgboost_runtime,
                "evaluate_daily_refit_predictions",
                side_effect=_record_refit_call,
            ),
        ):
            _, _, best_iteration = (
                xgboost_runtime.evaluate_xgboost_daily_refit_predictions(
                    train_frame=train_frame,
                    valid_frame=valid_frame,
                    test_frame=test_frame,
                    feature_cols=["feat"],
                    target_contract=target_contract,
                    model_params={"model_backend": "xgboost", "n_jobs": 1},
                    logger=__import__("logging").getLogger(__name__),
                )
            )

        refit_params = cast(dict[str, object], recorded["model_params"])
        self.assertEqual(best_iteration, 7)
        self.assertEqual(refit_params["n_jobs"], 12)
        self.assertEqual(refit_params["evaluation_daily_refit_workers"], 1)

    def test_xgboost_evaluation_uses_all_detected_cores(self) -> None:
        train_frame, valid_frame, _, target_contract = _build_refit_frames()
        recorded: dict[str, object] = {}

        def _record_fit_call(*_: object, **kwargs: object) -> SimpleNamespace:
            recorded["model_params"] = kwargs["model_params"]
            return SimpleNamespace(
                model=SimpleNamespace(best_iteration=3),
                params={"n_estimators": 10},
            )

        with (
            patch.object(evaluation_modeling.os, "cpu_count", return_value=12),
            patch.object(
                evaluation_modeling,
                "fit_xgboost_model",
                side_effect=_record_fit_call,
            ),
        ):
            _, best_iteration = evaluation_modeling.fit_xgboost_evaluation_model(
                train_frame=train_frame,
                valid_frame=valid_frame,
                feature_cols=["feat"],
                target_contract=target_contract,
                model_params={
                    "model_backend": "xgboost",
                    "n_jobs": 12,
                    "enable_categorical": False,
                },
            )

        fit_params = cast(dict[str, object], recorded["model_params"])
        self.assertEqual(best_iteration, 4)
        self.assertEqual(fit_params["n_jobs"], 12)

    def test_xgboost_evaluation_forces_cpu_for_h100_params_on_macos(self) -> None:
        train_frame, valid_frame, _, target_contract = _build_refit_frames()
        recorded: dict[str, object] = {}

        def _record_fit_call(*_: object, **kwargs: object) -> SimpleNamespace:
            recorded["model_params"] = kwargs["model_params"]
            return SimpleNamespace(
                model=SimpleNamespace(best_iteration=3),
                params={"n_estimators": 10},
            )

        with (
            patch.object(evaluation_modeling.platform, "system", return_value="Darwin"),
            patch.object(
                evaluation_modeling,
                "fit_xgboost_model",
                side_effect=_record_fit_call,
            ),
        ):
            evaluation_modeling.fit_xgboost_evaluation_model(
                train_frame=train_frame,
                valid_frame=valid_frame,
                feature_cols=["feat"],
                target_contract=target_contract,
                model_params={
                    "model_backend": "xgboost",
                    "runtime_profile": "nvidia_h100",
                    "requested_runtime_profile": "nvidia_h100",
                    "device": "cuda",
                    "xgboost_matrix_type": "quantile",
                    "xgboost_gpu_input_backend": "auto",
                    "n_jobs": 1,
                },
            )

        fit_params = cast(dict[str, object], recorded["model_params"])
        self.assertEqual(fit_params["runtime_profile"], "local_cpu")
        self.assertEqual(fit_params["device"], "cpu")
        self.assertEqual(fit_params["xgboost_matrix_type"], "dmatrix")
        self.assertEqual(fit_params["xgboost_gpu_input_backend"], "cpu")
        self.assertFalse(fit_params["cuda_available"])

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

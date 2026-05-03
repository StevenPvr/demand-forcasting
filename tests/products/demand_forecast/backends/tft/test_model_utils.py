from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from typing import Any, cast
from unittest.mock import patch

import numpy as np
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

from praedixa.demand_forecast.backends.tft.model_utils import (  # noqa: E402
    DEFAULT_TFT_MODEL_PARAMS,
    fit_tft_model,
    predict_quantiles_with_tft_model,
    predict_with_tft_model,
    save_tft_model,
    select_tft_feature_columns,
)
from praedixa.demand_forecast.backends.tft.frame_utils import (  # noqa: E402
    attach_group_and_time_columns,
    resolve_layout,
)
import praedixa.demand_forecast.backends.tft.model_fit as model_fit_module  # noqa: E402


def _with_suffixed_tft_features(frame: pd.DataFrame) -> pd.DataFrame:
    suffixed = frame.copy()
    suffix_pairs = {
        "location_id": "location_id_static_cat",
        "product_id": "product_id_static_cat",
        "target_day_of_week": "target_day_of_week_known_cat",
        "target_holiday_flag": "target_holiday_flag_known_cat",
        "current_day_demand_qty": "current_day_demand_qty_known_real",
        "rolling_mean_7": "rolling_mean_7_known_real",
        "data_quality_pricing_promo_missing_count": (
            "data_quality_pricing_promo_missing_count_known_real"
        ),
        "data_quality_history_unavailable_count": (
            "data_quality_history_unavailable_count_known_real"
        ),
        "lag_1": "lag_1_unknown_real",
    }
    for source, target in suffix_pairs.items():
        if source in suffixed.columns:
            suffixed[target] = suffixed[source]
    return suffixed


class TFTModelUtilsTests(unittest.TestCase):
    def test_default_tft_model_params_enable_native_progress_bar(self) -> None:
        self.assertTrue(bool(DEFAULT_TFT_MODEL_PARAMS["enable_progress_bar"]))
        self.assertEqual(
            int(cast(Any, DEFAULT_TFT_MODEL_PARAMS["progress_bar_refresh_rate"])), 1
        )

    def test_business_validation_metrics_payload_scores_absolute_target_directly(
        self,
    ) -> None:
        train_frame = pd.DataFrame(
            {
                "target_demand_qty_d_plus_1": [10.0, 11.0],
            }
        )
        valid_frame = pd.DataFrame(
            {
                "target_demand_qty_d_plus_1": [12.0, 15.0],
            }
        )
        raw_predictions = np.asarray([11.0, 14.0], dtype=float)

        build_payload = cast(
            Callable[..., dict[str, float]],
            getattr(model_fit_module, "_business_validation_metrics_payload"),
        )
        payload = build_payload(
            valid_frame=valid_frame,
            raw_predictions=raw_predictions,
            target_col="target_demand_qty_d_plus_1",
            train_frame=train_frame,
        )

        self.assertAlmostEqual(float(payload["business_val_wape"]), 2.0 / 27.0)
        self.assertAlmostEqual(float(payload["business_val_abs_bias"]), 1.0)

    def test_fit_tft_model_skips_business_callback_when_business_metrics_disabled(
        self,
    ) -> None:
        train_frame = pd.DataFrame({"target": [1.0]})
        sentinel_model = object()
        mocked_callback_builder = object()
        dataset_artifacts = cast(
            Any,
            SimpleNamespace(
                layout={},
                training_dataset=SimpleNamespace(),
                training_slice=pd.DataFrame(),
                feature_scalers={},
                normalization_strategy={},
            ),
        )

        with (
            patch.object(
                model_fit_module,
                "_resolve_fit_request",
                return_value=(
                    train_frame,
                    ["feature_a"],
                    {"Callback": object},
                    {
                        "enable_business_validation_metrics": False,
                        "batch_size": 8,
                        "quantiles": [0.1, 0.5, 0.9],
                        "runtime_profile": "local_cpu",
                    },
                ),
            ),
            patch.object(
                model_fit_module,
                "suppress_tft_runtime_noise",
                return_value=nullcontext(),
            ),
            patch.object(model_fit_module, "seed_tft_runtime"),
            patch.object(
                model_fit_module,
                "_resolved_dataset_artifacts",
                return_value=dataset_artifacts,
            ),
            patch.object(
                model_fit_module,
                "_business_validation_logging_callback",
                side_effect=AssertionError("business callback should be skipped"),
            ),
            patch.object(
                model_fit_module,
                "_fit_resolved_model",
                return_value=(sentinel_model, None, 0, {}),
            ) as mocked_fit,
            patch.object(
                model_fit_module, "extract_interpretability_payload", return_value=None
            ),
            patch.object(
                model_fit_module,
                "build_fitted_tft_model",
                return_value=mocked_callback_builder,
            ),
        ):
            result = model_fit_module.fit_tft_model(
                train_frame=train_frame,
                feature_cols=["feature_a"],
                target_col="target",
            )

        self.assertIs(result, mocked_callback_builder)
        self.assertEqual(mocked_fit.call_args.kwargs["extra_callbacks"], [])

    def test_business_validation_callback_restores_parent_trainer_context(
        self,
    ) -> None:
        class _FakeLightningModule:
            def __init__(self, trainer: object) -> None:
                self._trainer = trainer
                self.training = True
                self.device = None
                self.train_calls = 0

            def train(self) -> "_FakeLightningModule":
                self.training = True
                self.train_calls += 1
                return self

            def eval(self) -> "_FakeLightningModule":
                self.training = False
                return self

        parent_trainer = SimpleNamespace(callback_metrics={}, current_epoch=1)
        predict_trainer = SimpleNamespace(callback_metrics={}, current_epoch=0)
        pl_module = _FakeLightningModule(parent_trainer)
        train_frame = pd.DataFrame({"target": [1.0]})
        valid_frame = pd.DataFrame({"target": [2.0]})
        dataset_artifacts = cast(
            Any,
            SimpleNamespace(
                training_dataset=SimpleNamespace(get_parameters=lambda: {}),
                training_slice=pd.DataFrame({"target": [1.0]}),
                feature_scalers={},
                normalization_strategy={},
            ),
        )

        callback = cast(
            Any,
            model_fit_module._business_validation_logging_callback(
                {"Callback": object},
                train_frame=train_frame,
                valid_frame=valid_frame,
                feature_cols=[],
                target_col="target",
                dataset_artifacts=dataset_artifacts,
                resolved_params={"batch_size": 8, "quantiles": [0.1, 0.5, 0.9]},
            ),
        )

        def _fake_predict(
            transient_model: object, *_: object, **__: object
        ) -> np.ndarray:
            self.assertIs(getattr(transient_model, "model"), pl_module)
            pl_module._trainer = predict_trainer
            pl_module.eval()
            return np.asarray([2.0], dtype=float)

        with (
            patch.object(
                model_fit_module,
                "predict_with_tft_model",
                side_effect=_fake_predict,
            ),
            patch.object(
                model_fit_module,
                "_business_validation_metrics_payload",
                return_value={
                    "business_val_wape": 0.25,
                    "business_val_abs_bias": 1.5,
                },
            ),
        ):
            callback.on_validation_epoch_end(parent_trainer, pl_module)

        self.assertIs(pl_module._trainer, parent_trainer)
        self.assertTrue(pl_module.training)
        self.assertEqual(pl_module.train_calls, 1)
        self.assertEqual(parent_trainer.callback_metrics["business_val_wape"], 0.25)
        self.assertEqual(
            parent_trainer.callback_metrics["business_val_abs_bias"],
            1.5,
        )

    def _build_training_frame(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = pd.date_range("2024-01-01", periods=32, freq="D")
        rows: list[dict[str, object]] = []
        for series_offset, series_id in enumerate(["store_1__sku_1", "store_2__sku_2"]):
            signal = np.sin(np.arange(len(dates)) / 4.0) + 3.0 + series_offset
            for index, dt in enumerate(dates):
                rows.append(
                    {
                        "client_id": series_id,
                        "dt": dt,
                        "target": float(signal[index]),
                        "location_id": f"store_{series_offset + 1}",
                        "product_id": f"sku_{series_offset + 1}",
                        "target_day_of_week": int(dt.dayofweek),
                        "target_holiday_flag": bool(dt.dayofweek >= 5),
                        "current_day_demand_qty": float(signal[index]),
                        "rolling_mean_7": float(signal[index]),
                    }
                )
        frame = (
            pd.DataFrame(rows).sort_values(["client_id", "dt"]).reset_index(drop=True)
        )
        frame = _with_suffixed_tft_features(frame)
        train_frame = frame[frame["dt"] < "2024-01-25"].copy().reset_index(drop=True)
        valid_frame = frame[frame["dt"] >= "2024-01-25"].copy().reset_index(drop=True)
        return train_frame, valid_frame

    def test_select_tft_feature_columns_excludes_requested_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=3, freq="D"),
                "target": [1.0, 2.0, 3.0],
                "client_id": ["store_1__sku_1"] * 3,
                "location_id": ["store_1"] * 3,
                "product_id": ["sku_1"] * 3,
                "rolling_mean_7": [10.0, 11.0, 12.0],
                "lag_1": [10.0, 11.0, 12.0],
            }
        )
        frame = _with_suffixed_tft_features(frame)

        feature_cols = select_tft_feature_columns(frame, excluded_cols={"dt", "target"})

        self.assertEqual(
            feature_cols,
            [
                "location_id_static_cat",
                "product_id_static_cat",
                "rolling_mean_7_known_real",
                "lag_1_unknown_real",
            ],
        )

    def test_select_tft_feature_columns_rejects_unmapped_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=3, freq="D"),
                "target": [1.0, 2.0, 3.0],
                "client_id": ["store_1__sku_1"] * 3,
                "feature_a": [10.0, 11.0, 12.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "feature_a"):
            select_tft_feature_columns(frame, excluded_cols={"dt", "target"})

    def test_fit_predict_and_save_tft_model(self) -> None:
        train_frame, valid_frame = self._build_training_frame()
        feature_cols = select_tft_feature_columns(
            train_frame, excluded_cols={"dt", "target"}
        )

        model = fit_tft_model(
            train_frame,
            feature_cols,
            target_col="target",
            valid_frame=valid_frame,
            model_params={
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 1,
                "batch_size": 8,
                "hidden_size": 8,
                "hidden_continuous_size": 4,
                "attention_head_size": 1,
                "dropout": 0.1,
                "learning_rate": 0.03,
                "max_encoder_length": 7,
                "patience": 1,
            },
            default_max_iter=1,
        )

        predictions = predict_with_tft_model(model, valid_frame, feature_cols)

        self.assertEqual(len(predictions), len(valid_frame))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertGreaterEqual(model.best_iteration, 0)
        self.assertGreater(model.runtime_metrics["rows_per_second"], 0.0)
        self.assertEqual(
            int(model.runtime_metrics["loader_worker_count"]),
            int(model.model_hyperparameters["num_workers"]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "foundation_tft_final_model.pt"
            saved_path = save_tft_model(model, output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_fit_tft_model_reaches_second_epoch_after_business_validation(
        self,
    ) -> None:
        train_frame, valid_frame = self._build_training_frame()
        feature_cols = select_tft_feature_columns(
            train_frame, excluded_cols={"dt", "target"}
        )

        model = fit_tft_model(
            train_frame,
            feature_cols,
            target_col="target",
            valid_frame=valid_frame,
            model_params={
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 2,
                "batch_size": 8,
                "hidden_size": 8,
                "hidden_continuous_size": 4,
                "attention_head_size": 1,
                "dropout": 0.1,
                "learning_rate": 0.03,
                "max_encoder_length": 7,
                "patience": 2,
                "loss_patience": 0,
                "enable_progress_bar": False,
                "enable_csv_logger": False,
                "enable_lr_monitor": False,
                "enable_validation_metric_logging": False,
                "enable_device_stats_monitor": False,
            },
            default_max_iter=2,
        )

        self.assertGreaterEqual(model.best_iteration, 0)
        self.assertTrue(
            np.isfinite(predict_with_tft_model(model, valid_frame, feature_cols)).all()
        )

    def test_predict_quantiles_with_tft_model_returns_aligned_quantiles(self) -> None:
        train_frame, valid_frame = self._build_training_frame()
        feature_cols = select_tft_feature_columns(
            train_frame, excluded_cols={"dt", "target"}
        )

        model = fit_tft_model(
            train_frame,
            feature_cols,
            target_col="target",
            valid_frame=valid_frame,
            model_params={
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 1,
                "batch_size": 8,
                "hidden_size": 8,
                "hidden_continuous_size": 4,
                "attention_head_size": 1,
                "dropout": 0.1,
                "learning_rate": 0.03,
                "max_encoder_length": 7,
                "patience": 1,
                "quantiles": [0.025, 0.1, 0.5, 0.9, 0.975],
            },
            default_max_iter=1,
        )

        quantile_predictions = predict_quantiles_with_tft_model(
            model, valid_frame, feature_cols
        )

        self.assertEqual(
            list(quantile_predictions.columns),
            [
                "prediction_p2_5",
                "prediction_p10",
                "prediction_p50",
                "prediction_p90",
                "prediction_p97_5",
            ],
        )
        self.assertEqual(len(quantile_predictions), len(valid_frame))
        self.assertTrue(np.isfinite(quantile_predictions.to_numpy(dtype=float)).all())

    def test_fit_tft_model_registers_train_only_real_feature_scalers(self) -> None:
        train_frame, valid_frame = self._build_training_frame()
        feature_cols = select_tft_feature_columns(
            train_frame, excluded_cols={"dt", "target"}
        )

        model = fit_tft_model(
            train_frame,
            feature_cols,
            target_col="target",
            valid_frame=valid_frame,
            model_params={
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 1,
                "batch_size": 8,
                "hidden_size": 8,
                "hidden_continuous_size": 4,
                "attention_head_size": 1,
                "dropout": 0.1,
                "learning_rate": 0.03,
                "max_encoder_length": 7,
                "patience": 1,
            },
            default_max_iter=1,
        )

        scalers = model.dataset_parameters.get("scalers", {})

        self.assertIn("rolling_mean_7_known_real", scalers)
        self.assertIn("current_day_demand_qty_known_real", scalers)
        self.assertNotIn("location_id_static_cat", scalers)
        self.assertIsNone(model.target_scaler)
        self.assertEqual(
            getattr(model.dataset_parameters.get("target_normalizer"), "method", None),
            "standard",
        )
        self.assertIn("peak_ram_mb", model.runtime_metrics)
        self.assertEqual(model.normalization_strategy["kind"], "group_normalizer")
        self.assertIn("variable_selection", model.interpretability_payload or {})

    def test_fit_tft_model_handles_unknown_validation_categories(self) -> None:
        train_frame, valid_frame = self._build_training_frame()
        valid_frame = valid_frame.copy()
        valid_frame.loc[valid_frame.index[0], "location_id_static_cat"] = "store_99"
        feature_cols = select_tft_feature_columns(
            train_frame, excluded_cols={"dt", "target"}
        )

        model = fit_tft_model(
            train_frame,
            feature_cols,
            target_col="target",
            valid_frame=valid_frame,
            model_params={
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 1,
                "batch_size": 8,
                "hidden_size": 8,
                "hidden_continuous_size": 4,
                "attention_head_size": 1,
                "dropout": 0.1,
                "learning_rate": 0.03,
                "max_encoder_length": 7,
                "patience": 1,
            },
            default_max_iter=1,
        )

        predictions = predict_with_tft_model(model, valid_frame, feature_cols)

        self.assertEqual(len(predictions), len(valid_frame))
        self.assertTrue(np.isfinite(predictions).all())

    def test_predict_with_tft_model_handles_weighted_training_contract(self) -> None:
        train_frame, valid_frame = self._build_training_frame()
        feature_cols = select_tft_feature_columns(
            train_frame, excluded_cols={"dt", "target"}
        )
        train_weights = np.linspace(0.8, 1.2, len(train_frame), dtype=float)
        valid_weights = np.linspace(0.9, 1.1, len(valid_frame), dtype=float)

        model = fit_tft_model(
            train_frame,
            feature_cols,
            target_col="target",
            valid_frame=valid_frame,
            train_weights=train_weights,
            valid_weights=valid_weights,
            model_params={
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 1,
                "batch_size": 8,
                "hidden_size": 8,
                "hidden_continuous_size": 4,
                "attention_head_size": 1,
                "dropout": 0.1,
                "learning_rate": 0.03,
                "max_encoder_length": 7,
                "patience": 1,
            },
            default_max_iter=1,
        )

        predictions = predict_with_tft_model(model, valid_frame, feature_cols)

        self.assertEqual(len(predictions), len(valid_frame))
        self.assertTrue(np.isfinite(predictions).all())

    def test_data_quality_counts_are_treated_as_known_reals(self) -> None:
        frame = pd.DataFrame(
            {
                "client_id": ["store_1__sku_1"] * 4,
                "location_id": ["store_1"] * 4,
                "product_id": ["sku_1"] * 4,
                "dt": pd.date_range("2024-01-01", periods=4, freq="D"),
                "target": [1.0, 2.0, 3.0, 4.0],
                "data_quality_pricing_promo_missing_count": [0.0, 1.0, 2.0, 0.0],
                "data_quality_history_unavailable_count": [0.0, 0.0, 1.0, 1.0],
                "rolling_mean_7": [1.0, 1.0, 2.0, 3.0],
                "lag_1": [1.0, 1.0, 2.0, 3.0],
            }
        )
        frame = _with_suffixed_tft_features(frame)
        feature_cols = select_tft_feature_columns(frame, excluded_cols={"dt", "target"})

        prepared = attach_group_and_time_columns(frame, feature_cols)
        layout = resolve_layout(prepared, feature_cols)

        self.assertIn(
            "data_quality_pricing_promo_missing_count_known_real",
            layout["time_varying_known_reals"],
        )
        self.assertIn(
            "data_quality_history_unavailable_count_known_real",
            layout["time_varying_known_reals"],
        )
        self.assertNotIn(
            "data_quality_pricing_promo_missing_count_known_real",
            layout["time_varying_known_categoricals"],
        )
        self.assertNotIn("lag_1", feature_cols)
        self.assertIn("lag_1_unknown_real", feature_cols)
        self.assertEqual(layout["time_varying_unknown_categoricals"], [])
        self.assertEqual(layout["time_varying_unknown_reals"], ["lag_1_unknown_real"])

    def test_precomputed_time_idx_is_rebased_after_slicing_gaps(self) -> None:
        frame = pd.DataFrame(
            {
                "client_id": ["store_1__sku_1"] * 4,
                "location_id": ["store_1"] * 4,
                "product_id": ["sku_1"] * 4,
                "dt": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06"]
                ),
                "__tft_group_id": ["store_1__sku_1"] * 4,
                "__tft_time_idx": [0, 1, 4, 5],
                "target": [1.0, 2.0, 3.0, 4.0],
                "rolling_mean_7": [1.0, 1.5, 2.0, 2.5],
            }
        )

        prepared = attach_group_and_time_columns(frame, ["rolling_mean_7"])

        self.assertEqual(prepared["__tft_time_idx"].tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
import tempfile
import unittest
from typing import Any, cast

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
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


class TFTModelUtilsTests(unittest.TestCase):
    def test_default_tft_model_params_enable_native_progress_bar(self) -> None:
        self.assertTrue(bool(DEFAULT_TFT_MODEL_PARAMS["enable_progress_bar"]))
        self.assertEqual(int(cast(Any, DEFAULT_TFT_MODEL_PARAMS["progress_bar_refresh_rate"])), 1)

    def _build_training_frame(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = pd.date_range("2024-01-01", periods=32, freq="D")
        rows: list[dict[str, object]] = []
        for series_offset, series_id in enumerate(["store_1__sku_1", "store_2__sku_2"]):
            signal = np.sin(np.arange(len(dates)) / 4.0) + 3.0 + series_offset
            for index, dt in enumerate(dates):
                rows.append(
                    {
                        "series_id": series_id,
                        "dt": dt,
                        "target": float(signal[index]),
                        "location_id": f"store_{series_offset + 1}",
                        "product_id": f"sku_{series_offset + 1}",
                        "target_day_of_week": int(dt.dayofweek),
                        "target_holiday_flag": bool(dt.dayofweek >= 5),
                        "current_day_demand_qty": float(signal[index]),
                        "avg_selling_price": 2.5 + (series_offset * 0.25),
                    }
                )
        frame = pd.DataFrame(rows).sort_values(["series_id", "dt"]).reset_index(drop=True)
        train_frame = frame[frame["dt"] < "2024-01-25"].copy().reset_index(drop=True)
        valid_frame = frame[frame["dt"] >= "2024-01-25"].copy().reset_index(drop=True)
        return train_frame, valid_frame

    def test_select_tft_feature_columns_excludes_requested_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=3, freq="D"),
                "target": [1.0, 2.0, 3.0],
                "series_id": ["store_1__sku_1"] * 3,
                "location_id": ["store_1"] * 3,
                "product_id": ["sku_1"] * 3,
                "rolling_mean_7": [10.0, 11.0, 12.0],
                "lag_1": [10.0, 11.0, 12.0],
            }
        )

        feature_cols = select_tft_feature_columns(frame, excluded_cols={"dt", "target"})

        self.assertEqual(feature_cols, ["location_id", "product_id", "rolling_mean_7"])

    def test_select_tft_feature_columns_rejects_unmapped_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=3, freq="D"),
                "target": [1.0, 2.0, 3.0],
                "series_id": ["store_1__sku_1"] * 3,
                "feature_a": [10.0, 11.0, 12.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "feature_a"):
            select_tft_feature_columns(frame, excluded_cols={"dt", "target"})

    def test_fit_predict_and_save_tft_model(self) -> None:
        train_frame, valid_frame = self._build_training_frame()
        feature_cols = select_tft_feature_columns(train_frame, excluded_cols={"dt", "target"})

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
        self.assertEqual(model.runtime_metrics["loader_worker_count"], 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "foundation_tft_final_model.pt"
            saved_path = save_tft_model(model, output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_predict_quantiles_with_tft_model_returns_aligned_quantiles(self) -> None:
        train_frame, valid_frame = self._build_training_frame()
        feature_cols = select_tft_feature_columns(train_frame, excluded_cols={"dt", "target"})

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

        quantile_predictions = predict_quantiles_with_tft_model(model, valid_frame, feature_cols)

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
        feature_cols = select_tft_feature_columns(train_frame, excluded_cols={"dt", "target"})

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

        self.assertIn("avg_selling_price", scalers)
        self.assertIn("current_day_demand_qty", scalers)
        self.assertNotIn("location_id", scalers)
        self.assertIsNone(model.target_scaler)
        self.assertEqual(getattr(model.dataset_parameters.get("target_normalizer"), "method", None), "standard")
        self.assertIn("peak_ram_mb", model.runtime_metrics)
        self.assertEqual(model.normalization_strategy["kind"], "group_normalizer")
        self.assertIn("variable_selection", model.interpretability_payload or {})

    def test_fit_tft_model_handles_unknown_validation_categories(self) -> None:
        train_frame, valid_frame = self._build_training_frame()
        valid_frame = valid_frame.copy()
        valid_frame.loc[valid_frame.index[0], "location_id"] = "store_99"
        feature_cols = select_tft_feature_columns(train_frame, excluded_cols={"dt", "target"})

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
        feature_cols = select_tft_feature_columns(train_frame, excluded_cols={"dt", "target"})
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

    def test_status_features_are_treated_as_known_categoricals(self) -> None:
        frame = pd.DataFrame(
            {
                "series_id": ["store_1__sku_1"] * 4,
                "location_id": ["store_1"] * 4,
                "product_id": ["sku_1"] * 4,
                "dt": pd.date_range("2024-01-01", periods=4, freq="D"),
                "target": [1.0, 2.0, 3.0, 4.0],
                "promo_flag_status": [2, 1, 2, 0],
                "rolling_mean_7_status": [2, 2, 1, 0],
                "rolling_mean_7": [1.0, 1.0, 2.0, 3.0],
                "lag_1": [1.0, 1.0, 2.0, 3.0],
            }
        )
        feature_cols = select_tft_feature_columns(frame, excluded_cols={"dt", "target"})

        prepared = attach_group_and_time_columns(frame, feature_cols)
        layout = resolve_layout(prepared, feature_cols)

        self.assertIn("promo_flag_status", layout["time_varying_known_categoricals"])
        self.assertIn("rolling_mean_7_status", layout["time_varying_known_categoricals"])
        self.assertNotIn("promo_flag_status", layout["time_varying_known_reals"])
        self.assertNotIn("lag_1", feature_cols)
        self.assertEqual(layout["time_varying_unknown_categoricals"], [])
        self.assertEqual(layout["time_varying_unknown_reals"], [])

    def test_precomputed_time_idx_is_rebased_after_slicing_gaps(self) -> None:
        frame = pd.DataFrame(
            {
                "series_id": ["store_1__sku_1"] * 4,
                "location_id": ["store_1"] * 4,
                "product_id": ["sku_1"] * 4,
                "dt": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06"]),
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

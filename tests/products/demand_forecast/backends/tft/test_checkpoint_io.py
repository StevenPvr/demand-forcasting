from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.backends.tft.model_utils import (  # noqa: E402
    fit_tft_model,
    load_tft_model,
    predict_with_tft_model,
    save_tft_model,
    select_tft_feature_columns,
)


class TFTCheckpointIOTests(unittest.TestCase):
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

    def test_tft_checkpoint_round_trip_preserves_predictions(self) -> None:
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
        original_predictions = predict_with_tft_model(model, valid_frame, feature_cols)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "foundation_tft_final_model.pt"
            save_tft_model(model, output_path)
            reloaded_model = load_tft_model(output_path)

        reloaded_predictions = predict_with_tft_model(reloaded_model, valid_frame, feature_cols)

        np.testing.assert_allclose(original_predictions, reloaded_predictions, atol=1e-6, rtol=1e-6)
        self.assertEqual(reloaded_model.runtime_profile, model.runtime_profile)
        self.assertEqual(reloaded_model.git_sha, model.git_sha)
        self.assertIn("runtime_profile", reloaded_model.system_info)
        self.assertIn("current_day_demand_qty", reloaded_model.feature_scalers)


if __name__ == "__main__":
    unittest.main()

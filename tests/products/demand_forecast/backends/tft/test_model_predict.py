from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.backends.tft.artifacts import FittedTFTModel  # noqa: E402
from praedixa.demand_forecast.backends.tft.model_predict import (  # noqa: E402
    predict_quantiles_with_tft_model,
    predict_with_tft_model,
)


class _DummyModel:
    def eval(self) -> None:
        return None


def _fitted_model() -> FittedTFTModel:
    return FittedTFTModel(
        model=_DummyModel(),
        dataset_parameters={},
        history_frame=pd.DataFrame(),
        group_col="__tft_group_id",
        time_idx_col="__tft_time_idx",
        target_col="target",
        prediction_row_id_col="__tft_prediction_row_id",
        batch_size=8,
        quantiles=[0.1, 0.5, 0.9],
        best_iteration=0,
        model_hyperparameters={
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        feature_scalers={},
        target_scaler=None,
        runtime_profile="local_cpu",
        system_info={},
        runtime_metrics={},
        git_sha=None,
        bundle_manifest=None,
        data_hashes={},
        normalization_strategy={"kind": "group_normalizer"},
        interpretability_payload=None,
        artifact_bundle_version=2,
    )


class TFTModelPredictTests(unittest.TestCase):
    def test_predict_with_tft_model_raises_by_default_when_contract_breaks(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "client_id": ["store_1__sku_1", "store_1__sku_1"],
                "location_id": ["store_1", "store_1"],
                "product_id": ["sku_1", "sku_1"],
                "dt": pd.to_datetime(["2024-02-01", "2024-02-02"]),
                "target_lag_7": [11.0, 13.0],
            }
        )

        with patch(
            "praedixa.demand_forecast.backends.tft.model_predict._prediction_dataset",
            side_effect=RuntimeError("broken contract"),
        ):
            with self.assertRaises(RuntimeError):
                predict_with_tft_model(
                    _fitted_model(), frame, ["location_id", "product_id"]
                )

    def test_predict_with_tft_model_falls_back_when_policy_allows_it(self) -> None:
        frame = pd.DataFrame(
            {
                "client_id": ["store_1__sku_1", "store_1__sku_1"],
                "location_id": ["store_1", "store_1"],
                "product_id": ["sku_1", "sku_1"],
                "dt": pd.to_datetime(["2024-02-01", "2024-02-02"]),
                "target_lag_7": [11.0, 13.0],
            }
        )
        fitted_model = _fitted_model()

        with patch(
            "praedixa.demand_forecast.backends.tft.model_predict._prediction_dataset",
            side_effect=RuntimeError("broken contract"),
        ):
            predictions = predict_with_tft_model(
                fitted_model,
                frame,
                ["location_id", "product_id"],
                fallback_policy="baseline",
            )

        self.assertEqual(predictions.tolist(), [11.0, 13.0])
        self.assertTrue(
            fitted_model.runtime_metrics["last_prediction_diagnostics"]["used_fallback"]
        )

    def test_predict_quantiles_with_tft_model_falls_back_when_policy_allows_it(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "client_id": ["store_1__sku_1"],
                "location_id": ["store_1"],
                "product_id": ["sku_1"],
                "dt": pd.to_datetime(["2024-02-01"]),
                "seasonal_naive_d7": [9.0],
            }
        )

        with patch(
            "praedixa.demand_forecast.backends.tft.model_predict._prediction_dataset",
            side_effect=RuntimeError("broken contract"),
        ):
            quantiles = predict_quantiles_with_tft_model(
                _fitted_model(),
                frame,
                ["location_id", "product_id"],
                fallback_policy="baseline",
            )

        self.assertEqual(
            list(quantiles.columns),
            ["prediction_p10", "prediction_p50", "prediction_p90"],
        )
        self.assertEqual(quantiles.iloc[0].tolist(), [9.0, 9.0, 9.0])

    def test_predict_with_tft_model_partially_falls_back_for_unsupported_groups(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "client_id": ["series_1", "series_2"],
                "location_id": ["store_1", "store_2"],
                "product_id": ["sku_1", "sku_2"],
                "dt": pd.to_datetime(["2024-02-01", "2024-02-01"]),
                "target_lag_7": [11.0, 13.0],
            }
        )
        fitted_model = _fitted_model()
        fitted_model.model_hyperparameters["max_encoder_length"] = 2
        fitted_model.history_frame = pd.DataFrame(
            {
                "__tft_group_id": ["series_1", "series_1"],
                "__tft_time_idx": [0, 1],
            }
        )

        def _fake_prediction_dataset(
            _: dict[str, object],
            *,
            fitted_model: FittedTFTModel,
            prepared_future: pd.DataFrame,
        ) -> object:
            self.assertEqual(
                prepared_future[fitted_model.group_col].tolist(), ["series_1"]
            )
            return object()

        with (
            patch(
                "praedixa.demand_forecast.backends.tft.model_predict._prediction_dataset",
                side_effect=_fake_prediction_dataset,
            ),
            patch(
                "praedixa.demand_forecast.backends.tft.model_predict._prediction_dataloader",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.backends.tft.model_predict._predict_payload",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.backends.tft.model_predict._prediction_index_frame",
                return_value=pd.DataFrame(
                    {
                        "__tft_group_id": ["series_1"],
                        "__tft_time_idx": [2],
                    }
                ),
            ),
            patch(
                "praedixa.demand_forecast.backends.tft.model_predict._prediction_output_to_numpy",
                return_value=np.array([42.0], dtype=float),
            ),
        ):
            predictions = predict_with_tft_model(
                fitted_model,
                frame,
                ["location_id", "product_id"],
                fallback_policy="baseline",
            )

        self.assertEqual(predictions.tolist(), [42.0, 13.0])

    def test_predict_quantiles_with_tft_model_partially_falls_back_for_unsupported_groups(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "client_id": ["series_1", "series_2"],
                "location_id": ["store_1", "store_2"],
                "product_id": ["sku_1", "sku_2"],
                "dt": pd.to_datetime(["2024-02-01", "2024-02-01"]),
                "target_lag_7": [11.0, 13.0],
            }
        )
        fitted_model = _fitted_model()
        fitted_model.model_hyperparameters["max_encoder_length"] = 2
        fitted_model.history_frame = pd.DataFrame(
            {
                "__tft_group_id": ["series_1", "series_1"],
                "__tft_time_idx": [0, 1],
            }
        )

        with (
            patch(
                "praedixa.demand_forecast.backends.tft.model_predict._prediction_dataset",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.backends.tft.model_predict._prediction_dataloader",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.backends.tft.model_predict._predict_payload",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.backends.tft.model_predict._prediction_index_frame",
                return_value=pd.DataFrame(
                    {
                        "__tft_group_id": ["series_1"],
                        "__tft_time_idx": [2],
                    }
                ),
            ),
            patch(
                "praedixa.demand_forecast.backends.tft.model_predict._prediction_output_to_numpy",
                return_value=np.array([[40.0, 42.0, 44.0]], dtype=float),
            ),
        ):
            quantiles = predict_quantiles_with_tft_model(
                fitted_model,
                frame,
                ["location_id", "product_id"],
                fallback_policy="baseline",
            )

        self.assertEqual(
            list(quantiles.columns),
            ["prediction_p10", "prediction_p50", "prediction_p90"],
        )
        self.assertEqual(quantiles.iloc[0].tolist(), [40.0, 42.0, 44.0])
        self.assertEqual(quantiles.iloc[1].tolist(), [13.0, 13.0, 13.0])


if __name__ == "__main__":
    unittest.main()

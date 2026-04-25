from __future__ import annotations

from pathlib import Path
import sys
import unittest

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

from praedixa.demand_forecast.backends.xgboost.model_fit import (  # noqa: E402
    fit_xgboost_model,
    predict_with_xgboost_model,
)
from praedixa.demand_forecast.backends.xgboost.model_common import (  # noqa: E402
    DEFAULT_XGBOOST_MODEL_PARAMS,
    xgboost_early_stopping_rounds,
    xgboost_num_boost_round,
    xgboost_train_params,
)
from praedixa.demand_forecast.backends.xgboost.preprocessing import (  # noqa: E402
    UNKNOWN_CATEGORY,
    prepare_xgboost_matrices,
    to_xgboost_feature_data,
)


class XGBoostModelFitTests(unittest.TestCase):
    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_frame = pd.DataFrame(
            {
                "client_id": ["a", "a", "b", "b", "c", "c"],
                "location_id": ["loc_1", "loc_1", "loc_2", "loc_2", "loc_3", "loc_3"],
                "target_day_of_week": [0, 1, 0, 1, 0, 1],
                "rolling_mean_7": [2.0, 2.1, 3.0, 3.1, 4.0, 4.1],
                "target_demand_qty_d_plus_1": [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            }
        )
        valid_frame = pd.DataFrame(
            {
                "client_id": ["a", "new_series"],
                "location_id": ["loc_1", "loc_new"],
                "target_day_of_week": [2, 2],
                "rolling_mean_7": [2.2, 5.0],
                "target_demand_qty_d_plus_1": [12.0, 40.0],
            }
        )
        return train_frame, valid_frame

    def test_preprocessing_defaults_to_stable_ordinal_categories_without_refitting(
        self,
    ) -> None:
        train_frame, valid_frame = self._frames()

        matrices = prepare_xgboost_matrices(
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=["client_id", "location_id", "target_day_of_week"],
            target_col="target_demand_qty_d_plus_1",
        )

        self.assertIn(UNKNOWN_CATEGORY, matrices.feature_spec.category_values["client_id"])
        self.assertFalse(matrices.feature_spec.native_categorical)
        unknown_code = matrices.feature_spec.category_values["client_id"].index(
            UNKNOWN_CATEGORY
        )
        encoded_series = matrices.valid_x["client_id"].to_numpy(
            dtype=np.float32,
            copy=False,
        )
        self.assertEqual(float(encoded_series[1]), float(unknown_code))
        self.assertEqual(matrices.valid_x["client_id"].dtype, np.dtype("float32"))
        self.assertFalse(isinstance(matrices.valid_x["client_id"].dtype, pd.CategoricalDtype))
        self.assertEqual(matrices.train_y.dtype, np.dtype("float32"))
        self.assertEqual(matrices.valid_y.dtype, np.dtype("float32"))
        self.assertTrue(bool(matrices.train_y.flags.c_contiguous))
        self.assertTrue(bool(matrices.valid_y.flags.c_contiguous))
        self.assertTrue(bool(matrices.train_y.flags.aligned))
        self.assertTrue(bool(matrices.valid_y.flags.aligned))

    def test_preprocessing_exports_contiguous_float32_arrays_for_native_dmatrix(
        self,
    ) -> None:
        train_frame, valid_frame = self._frames()
        matrices = prepare_xgboost_matrices(
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=["client_id", "location_id", "target_day_of_week"],
            target_col="target_demand_qty_d_plus_1",
        )

        train_data = to_xgboost_feature_data(matrices.train_x, matrices.feature_spec)

        self.assertIsInstance(train_data, np.ndarray)
        train_array = np.asarray(train_data)
        self.assertEqual(train_array.dtype, np.dtype("float32"))
        self.assertTrue(bool(train_array.flags.c_contiguous))
        self.assertTrue(bool(train_array.flags.aligned))

    def test_preprocessing_can_keep_native_categorical_when_requested(self) -> None:
        train_frame, valid_frame = self._frames()

        matrices = prepare_xgboost_matrices(
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=["client_id", "location_id", "target_day_of_week"],
            target_col="target_demand_qty_d_plus_1",
            native_categorical=True,
        )

        self.assertTrue(matrices.feature_spec.native_categorical)
        self.assertEqual(
            str(matrices.valid_x.loc[1, "client_id"]),
            UNKNOWN_CATEGORY,
        )
        self.assertTrue(isinstance(matrices.valid_x["client_id"].dtype, pd.CategoricalDtype))

    def test_fit_predict_returns_one_prediction_per_validation_row(self) -> None:
        train_frame, valid_frame = self._frames()

        model = fit_xgboost_model(
            train_frame,
            valid_frame,
            ["client_id", "location_id", "target_day_of_week", "rolling_mean_7"],
            target_col="target_demand_qty_d_plus_1",
            model_params={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "max_depth": 2,
                "n_jobs": 1,
                "random_state": 7,
            },
        )
        predictions = predict_with_xgboost_model(model, valid_frame)

        self.assertEqual(predictions.shape, (len(valid_frame),))
        self.assertTrue(bool(np.isfinite(predictions).all()))

    def test_train_params_default_to_native_categorical_mode(self) -> None:
        params = xgboost_train_params(DEFAULT_XGBOOST_MODEL_PARAMS)

        self.assertEqual(params["device"], "cpu")
        self.assertEqual(xgboost_num_boost_round(DEFAULT_XGBOOST_MODEL_PARAMS), 5000)
        self.assertEqual(xgboost_early_stopping_rounds(DEFAULT_XGBOOST_MODEL_PARAMS), 100)
        self.assertEqual(params["nthread"], 1)
        self.assertEqual(params["seed"], 42)
        self.assertEqual(params["alpha"], 1e-3)
        self.assertEqual(params["lambda"], 1.0)
        self.assertNotIn("enable_categorical", params)
        self.assertNotIn("n_estimators", params)
        self.assertNotIn("early_stopping_rounds", params)
        self.assertEqual(params["max_cat_to_onehot"], 16)
        self.assertEqual(params["max_cat_threshold"], 32)

    def test_train_params_keep_native_categorical_splits_when_enabled(self) -> None:
        params = xgboost_train_params(
            {**DEFAULT_XGBOOST_MODEL_PARAMS, "enable_categorical": True}
        )

        self.assertEqual(params["max_cat_to_onehot"], 16)
        self.assertEqual(params["max_cat_threshold"], 32)


if __name__ == "__main__":
    unittest.main()

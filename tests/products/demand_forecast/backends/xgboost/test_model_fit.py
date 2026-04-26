from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

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
    FittedXGBoostModel,
    fit_xgboost_model,
    fit_xgboost_model_and_predict_validation,
    fit_prepared_xgboost_matrices_and_predict_validation,
    predict_with_xgboost_model,
    prepare_xgboost_training_matrices,
)
from praedixa.demand_forecast.backends.xgboost.model_common import (  # noqa: E402
    DEFAULT_XGBOOST_MODEL_PARAMS,
    resolve_xgboost_model_params,
    xgboost_early_stopping_rounds,
    xgboost_num_boost_round,
    xgboost_train_params,
)
import praedixa.demand_forecast.backends.xgboost.model_fit as xgboost_model_fit_module  # noqa: E402
from praedixa.demand_forecast.backends.xgboost.runtime import (  # noqa: E402
    XGBoostRuntimeResolution,
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

        self.assertIn(
            UNKNOWN_CATEGORY, matrices.feature_spec.category_values["client_id"]
        )
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
        self.assertFalse(
            isinstance(matrices.valid_x["client_id"].dtype, pd.CategoricalDtype)
        )
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
        self.assertTrue(
            isinstance(matrices.valid_x["client_id"].dtype, pd.CategoricalDtype)
        )

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

    def test_fit_and_validation_predict_matches_public_predict_path(self) -> None:
        train_frame, valid_frame = self._frames()

        model, validation_predictions = fit_xgboost_model_and_predict_validation(
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
        public_predictions = predict_with_xgboost_model(model, valid_frame)

        np.testing.assert_allclose(
            validation_predictions,
            public_predictions,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_prebuilt_matrices_can_be_reused_for_trial_fit(self) -> None:
        train_frame, valid_frame = self._frames()
        base_params = {
            "n_estimators": 5,
            "early_stopping_rounds": 2,
            "max_depth": 2,
            "max_bin": 128,
            "n_jobs": 1,
            "random_state": 7,
        }
        prepared = prepare_xgboost_training_matrices(
            train_frame,
            valid_frame,
            ["client_id", "location_id", "target_day_of_week", "rolling_mean_7"],
            target_col="target_demand_qty_d_plus_1",
            model_params=base_params,
        )

        model, validation_predictions = (
            fit_prepared_xgboost_matrices_and_predict_validation(
                prepared,
                model_params={**base_params, "learning_rate": 0.05},
            )
        )
        public_predictions = predict_with_xgboost_model(model, valid_frame)

        self.assertEqual(validation_predictions.shape, (len(valid_frame),))
        np.testing.assert_allclose(
            validation_predictions,
            public_predictions,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_public_predict_falls_back_to_cpu_when_cuda_is_unavailable(self) -> None:
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
        cuda_model = FittedXGBoostModel(
            model=model.model,
            feature_spec=model.feature_spec,
            params={
                **model.params,
                "runtime_profile": "scaleway_l40s",
                "device": "cuda",
                "xgboost_matrix_type": "quantile",
            },
        )

        with (
            mock.patch(
                "praedixa.demand_forecast.backends.xgboost.model_fit.xgboost_cuda_preflight_available",
                return_value=False,
            ),
            self.assertLogs(
                "praedixa.demand_forecast.backends.xgboost.model_fit",
                level="WARNING",
            ),
        ):
            predictions = predict_with_xgboost_model(cuda_model, valid_frame)

        self.assertEqual(predictions.shape, (len(valid_frame),))
        self.assertTrue(bool(np.isfinite(predictions).all()))

    def test_train_params_default_to_native_categorical_mode(self) -> None:
        params = xgboost_train_params(DEFAULT_XGBOOST_MODEL_PARAMS)

        self.assertEqual(params["device"], "cpu")
        self.assertEqual(xgboost_num_boost_round(DEFAULT_XGBOOST_MODEL_PARAMS), 5000)
        self.assertEqual(
            xgboost_early_stopping_rounds(DEFAULT_XGBOOST_MODEL_PARAMS), 100
        )
        self.assertEqual(params["nthread"], 1)
        self.assertEqual(params["seed"], 42)
        self.assertEqual(params["max_depth"], 5)
        self.assertEqual(params["min_child_weight"], 64.0)
        self.assertEqual(params["alpha"], 1e-2)
        self.assertEqual(params["lambda"], 20.0)
        self.assertNotIn("enable_categorical", params)
        self.assertNotIn("n_estimators", params)
        self.assertNotIn("early_stopping_rounds", params)
        self.assertEqual(params["max_cat_to_onehot"], 8)
        self.assertEqual(params["max_cat_threshold"], 64)

    def test_train_params_keep_native_categorical_splits_when_enabled(self) -> None:
        params = xgboost_train_params(
            {**DEFAULT_XGBOOST_MODEL_PARAMS, "enable_categorical": True}
        )

        self.assertEqual(params["max_cat_to_onehot"], 8)
        self.assertEqual(params["max_cat_threshold"], 64)

    def test_resolved_cuda_params_force_single_gpu_worker_policy(self) -> None:
        with mock.patch(
            "praedixa.demand_forecast.backends.xgboost.model_common.resolve_xgboost_runtime_profile",
            return_value=XGBoostRuntimeResolution(
                requested_profile="scaleway_l40s",
                runtime_profile="scaleway_l40s",
                device="cuda",
                tree_method="hist",
                accelerator="gpu",
                devices=1,
                cuda_available=True,
                cuda_device_name="NVIDIA L40S",
            ),
        ):
            resolved = resolve_xgboost_model_params(
                {
                    "runtime_profile": "scaleway_l40s",
                    "device": "cuda",
                    "n_jobs": 8,
                    "max_parallel_fold_workers": 5,
                }
            )
        params = xgboost_train_params(resolved)

        self.assertEqual(resolved["device"], "cuda")
        self.assertEqual(resolved["tree_method"], "hist")
        self.assertEqual(resolved["xgboost_matrix_type"], "quantile")
        self.assertEqual(resolved["xgboost_gpu_input_backend"], "auto")
        self.assertEqual(resolved["n_jobs"], 1)
        self.assertEqual(resolved["max_parallel_fold_workers"], 1)
        self.assertEqual(params["device"], "cuda")
        self.assertEqual(params["nthread"], 1)

    def test_device_only_cuda_params_promote_runtime_profile(self) -> None:
        with mock.patch(
            "praedixa.demand_forecast.backends.xgboost.model_common.resolve_xgboost_runtime_profile",
            return_value=XGBoostRuntimeResolution(
                requested_profile="cuda",
                runtime_profile="cuda",
                device="cuda",
                tree_method="hist",
                accelerator="gpu",
                devices=1,
                cuda_available=True,
                cuda_device_name="NVIDIA L40S",
            ),
        ):
            resolved = resolve_xgboost_model_params({"device": "cuda", "n_jobs": 8})

        self.assertEqual(resolved["device"], "cuda")
        self.assertEqual(resolved["runtime_profile"], "cuda")
        self.assertEqual(resolved["xgboost_matrix_type"], "quantile")
        self.assertEqual(resolved["xgboost_gpu_input_backend"], "auto")

    def test_device_cuda_with_auto_profile_does_not_fall_back_to_cpu(self) -> None:
        with (
            mock.patch(
                "praedixa.demand_forecast.backends.xgboost.model_common.resolve_xgboost_runtime_profile",
                side_effect=RuntimeError("CUDA preflight failed"),
            ) as mocked_resolve_runtime,
            self.assertRaisesRegex(RuntimeError, "CUDA preflight failed"),
        ):
            resolve_xgboost_model_params(
                {"runtime_profile": "auto", "device": "cuda", "n_jobs": 8}
            )

        mocked_resolve_runtime.assert_called_once_with("cuda")

    def test_cuda_quantile_matrix_failure_is_not_silently_downgraded(self) -> None:
        class _FailingXGBoostModule:
            @staticmethod
            def QuantileDMatrix(**_: object) -> object:
                raise RuntimeError("quantile unavailable")

            @staticmethod
            def DMatrix(**_: object) -> object:
                return object()

        train_frame, valid_frame = self._frames()
        matrices = prepare_xgboost_matrices(
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=["client_id", "location_id", "target_day_of_week"],
            target_col="target_demand_qty_d_plus_1",
            native_categorical=True,
        )

        with self.assertRaisesRegex(RuntimeError, "requires QuantileDMatrix"):
            xgboost_model_fit_module._build_dmatrix(
                _FailingXGBoostModule(),
                matrices.train_x,
                matrices.feature_spec,
                params={
                    "runtime_profile": "scaleway_l40s",
                    "device": "cuda",
                    "xgboost_matrix_type": "quantile",
                    "max_bin": 128,
                    "n_jobs": 1,
                },
                label=matrices.train_y,
            )


if __name__ == "__main__":
    unittest.main()

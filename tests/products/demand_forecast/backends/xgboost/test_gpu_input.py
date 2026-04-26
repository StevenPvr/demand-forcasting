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

from praedixa.demand_forecast.backends.xgboost.gpu_input import (  # noqa: E402
    build_xgboost_matrix_input_payload,
    resolve_xgboost_gpu_input_backend,
)
from praedixa.demand_forecast.backends.xgboost.preprocessing import (  # noqa: E402
    prepare_xgboost_matrices,
    to_xgboost_feature_data,
)


class _FakeCupy:
    arrays: list[np.ndarray] = []

    @staticmethod
    def asarray(
        values: object, dtype: object | None = None
    ) -> tuple[str, tuple[int, ...], str]:
        array = np.asarray(values, dtype=dtype)
        _FakeCupy.arrays.append(array)
        return ("cupy", array.shape, str(array.dtype))


class _FakeCudfSeries:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)


class _FakeCudf:
    @staticmethod
    def from_pandas(frame: pd.DataFrame) -> tuple[str, dict[str, str]]:
        return ("cudf_frame", frame.dtypes.astype(str).to_dict())

    @staticmethod
    def DataFrame(
        values: object, columns: list[str]
    ) -> tuple[str, tuple[int, ...], tuple[str, ...]]:
        return ("cudf_matrix", np.asarray(values).shape, tuple(columns))

    Series = _FakeCudfSeries


class XGBoostGpuInputTests(unittest.TestCase):
    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_frame = pd.DataFrame(
            {
                "client_id": ["a", "a", "b", "b"],
                "location_id": ["loc_1", "loc_1", "loc_2", "loc_2"],
                "rolling_mean_7": [2.0, 2.1, 3.0, 3.1],
                "target_demand_qty_d_plus_1": [10.0, 11.0, 20.0, 21.0],
            }
        )
        valid_frame = train_frame.tail(1).copy()
        return train_frame, valid_frame

    def test_auto_cuda_uses_cupy_for_ordinal_encoded_features(self) -> None:
        train_frame, valid_frame = self._frames()
        matrices = prepare_xgboost_matrices(
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=["client_id", "location_id", "rolling_mean_7"],
            target_col="target_demand_qty_d_plus_1",
            native_categorical=False,
        )

        with mock.patch(
            "praedixa.demand_forecast.backends.xgboost.gpu_input.optional_gpu_module",
            side_effect=lambda name: _FakeCupy if name == "cupy" else None,
        ):
            resolution = resolve_xgboost_gpu_input_backend(
                {"runtime_profile": "scaleway_l40s", "device": "cuda"},
                matrices.feature_spec,
            )

        self.assertEqual(resolution.resolved_backend, "cupy")

    def test_auto_cuda_uses_cudf_for_native_categorical_features(self) -> None:
        train_frame, valid_frame = self._frames()
        matrices = prepare_xgboost_matrices(
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=["client_id", "location_id", "rolling_mean_7"],
            target_col="target_demand_qty_d_plus_1",
            native_categorical=True,
        )

        with mock.patch(
            "praedixa.demand_forecast.backends.xgboost.gpu_input.optional_gpu_module",
            side_effect=lambda name: _FakeCudf if name == "cudf" else None,
        ):
            resolution = resolve_xgboost_gpu_input_backend(
                {"runtime_profile": "scaleway_l40s", "device": "cuda"},
                matrices.feature_spec,
            )

        self.assertEqual(resolution.resolved_backend, "cudf")

    def test_explicit_cupy_rejects_native_categorical_features(self) -> None:
        train_frame, valid_frame = self._frames()
        matrices = prepare_xgboost_matrices(
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=["client_id", "location_id", "rolling_mean_7"],
            target_col="target_demand_qty_d_plus_1",
            native_categorical=True,
        )

        with self.assertRaisesRegex(RuntimeError, "CuPy input requires ordinal"):
            resolve_xgboost_gpu_input_backend(
                {
                    "runtime_profile": "scaleway_l40s",
                    "device": "cuda",
                    "xgboost_gpu_input_backend": "cupy",
                },
                matrices.feature_spec,
            )

    def test_cupy_payload_moves_features_label_and_weight_to_gpu_arrays(self) -> None:
        _FakeCupy.arrays = []
        train_frame, valid_frame = self._frames()
        matrices = prepare_xgboost_matrices(
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=["client_id", "location_id", "rolling_mean_7"],
            target_col="target_demand_qty_d_plus_1",
            native_categorical=False,
        )
        feature_data = to_xgboost_feature_data(matrices.train_x, matrices.feature_spec)
        sample_weight = np.ones(len(matrices.train_y), dtype=np.float32)

        with mock.patch(
            "praedixa.demand_forecast.backends.xgboost.gpu_input.optional_gpu_module",
            return_value=_FakeCupy,
        ):
            payload = build_xgboost_matrix_input_payload(
                data=feature_data,
                label=matrices.train_y,
                weight=sample_weight,
                feature_spec=matrices.feature_spec,
                model_params={
                    "runtime_profile": "scaleway_l40s",
                    "device": "cuda",
                    "xgboost_gpu_input_backend": "cupy",
                },
            )

        self.assertEqual(payload.resolution.resolved_backend, "cupy")
        self.assertEqual(payload.data, ("cupy", (4, 3), "float32"))
        self.assertEqual(payload.label, ("cupy", (4,), "float32"))
        self.assertEqual(payload.weight, ("cupy", (4,), "float32"))
        self.assertEqual(len(_FakeCupy.arrays), 3)

    def test_cudf_payload_preserves_native_categorical_dataframe(self) -> None:
        train_frame, valid_frame = self._frames()
        matrices = prepare_xgboost_matrices(
            train_frame=train_frame,
            valid_frame=valid_frame,
            feature_cols=["client_id", "location_id", "rolling_mean_7"],
            target_col="target_demand_qty_d_plus_1",
            native_categorical=True,
        )

        with mock.patch(
            "praedixa.demand_forecast.backends.xgboost.gpu_input.optional_gpu_module",
            return_value=_FakeCudf,
        ):
            payload = build_xgboost_matrix_input_payload(
                data=matrices.train_x,
                label=matrices.train_y,
                weight=None,
                feature_spec=matrices.feature_spec,
                model_params={
                    "runtime_profile": "scaleway_l40s",
                    "device": "cuda",
                    "xgboost_gpu_input_backend": "cudf",
                },
            )

        self.assertEqual(payload.resolution.resolved_backend, "cudf")
        self.assertEqual(payload.data[0], "cudf_frame")
        self.assertIn("category", payload.data[1]["client_id"])
        self.assertIsInstance(payload.label, _FakeCudfSeries)
        self.assertIsNone(payload.weight)


if __name__ == "__main__":
    unittest.main()

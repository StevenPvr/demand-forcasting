from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.xgboost.preprocessing import (
    XGBoostFeatureSpec,
)
from praedixa.demand_forecast.backends.xgboost.runtime import xgboost_uses_cuda
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_XGBOOST_GPU_INPUT_BACKEND,
    SUPPORTED_XGBOOST_GPU_INPUT_BACKENDS,
)


LOGGER = logging.getLogger(__name__)

XGBOOST_GPU_INPUT_AUTO = "auto"
XGBOOST_GPU_INPUT_CPU = "cpu"
XGBOOST_GPU_INPUT_CUPY = "cupy"
XGBOOST_GPU_INPUT_CUDF = "cudf"


@dataclass(frozen=True)
class XGBoostGpuInputResolution:
    requested_backend: str
    resolved_backend: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class XGBoostMatrixInputPayload:
    data: object
    label: object | None
    weight: object | None
    resolution: XGBoostGpuInputResolution


def optional_gpu_module(module_name: str) -> Any | None:
    if find_spec(module_name) is None:
        return None
    return import_module(module_name)


def gpu_input_backend_available(backend: str) -> bool:
    normalized = backend.strip().lower()
    if normalized in {XGBOOST_GPU_INPUT_AUTO, XGBOOST_GPU_INPUT_CPU}:
        return True
    if normalized not in SUPPORTED_XGBOOST_GPU_INPUT_BACKENDS:
        return False
    return optional_gpu_module(normalized) is not None


def resolve_xgboost_gpu_input_backend(
    model_params: Mapping[str, object],
    feature_spec: XGBoostFeatureSpec,
) -> XGBoostGpuInputResolution:
    requested_backend = (
        str(
            model_params.get(
                "xgboost_gpu_input_backend",
                DEFAULT_XGBOOST_GPU_INPUT_BACKEND,
            )
        )
        .strip()
        .lower()
    )
    if requested_backend not in SUPPORTED_XGBOOST_GPU_INPUT_BACKENDS:
        raise RuntimeError(
            f"Unknown XGBoost GPU input backend `{requested_backend}`. "
            f"Supported values: {SUPPORTED_XGBOOST_GPU_INPUT_BACKENDS}."
        )
    if not xgboost_uses_cuda(model_params):
        return XGBoostGpuInputResolution(
            requested_backend=requested_backend,
            resolved_backend=XGBOOST_GPU_INPUT_CPU,
        )
    if requested_backend == XGBOOST_GPU_INPUT_CPU:
        return XGBoostGpuInputResolution(
            requested_backend=requested_backend,
            resolved_backend=XGBOOST_GPU_INPUT_CPU,
        )
    if requested_backend == XGBOOST_GPU_INPUT_CUPY:
        _raise_if_cupy_cannot_handle_feature_spec(feature_spec)
        _require_gpu_module(XGBOOST_GPU_INPUT_CUPY)
        return XGBoostGpuInputResolution(
            requested_backend=requested_backend,
            resolved_backend=XGBOOST_GPU_INPUT_CUPY,
        )
    if requested_backend == XGBOOST_GPU_INPUT_CUDF:
        _require_gpu_module(XGBOOST_GPU_INPUT_CUDF)
        return XGBoostGpuInputResolution(
            requested_backend=requested_backend,
            resolved_backend=XGBOOST_GPU_INPUT_CUDF,
        )
    return _resolve_auto_gpu_input_backend(feature_spec, requested_backend)


def _resolve_auto_gpu_input_backend(
    feature_spec: XGBoostFeatureSpec,
    requested_backend: str,
) -> XGBoostGpuInputResolution:
    if feature_spec.native_categorical:
        if gpu_input_backend_available(XGBOOST_GPU_INPUT_CUDF):
            return XGBoostGpuInputResolution(
                requested_backend=requested_backend,
                resolved_backend=XGBOOST_GPU_INPUT_CUDF,
            )
        return XGBoostGpuInputResolution(
            requested_backend=requested_backend,
            resolved_backend=XGBOOST_GPU_INPUT_CPU,
            fallback_reason=(
                "cuDF is unavailable; keeping native categorical XGBoost inputs on CPU."
            ),
        )
    if gpu_input_backend_available(XGBOOST_GPU_INPUT_CUPY):
        return XGBoostGpuInputResolution(
            requested_backend=requested_backend,
            resolved_backend=XGBOOST_GPU_INPUT_CUPY,
        )
    if gpu_input_backend_available(XGBOOST_GPU_INPUT_CUDF):
        return XGBoostGpuInputResolution(
            requested_backend=requested_backend,
            resolved_backend=XGBOOST_GPU_INPUT_CUDF,
        )
    return XGBoostGpuInputResolution(
        requested_backend=requested_backend,
        resolved_backend=XGBOOST_GPU_INPUT_CPU,
        fallback_reason=(
            "Neither CuPy nor cuDF is available; keeping XGBoost inputs on CPU."
        ),
    )


def build_xgboost_matrix_input_payload(
    *,
    data: pd.DataFrame | np.ndarray,
    label: np.ndarray | None,
    weight: np.ndarray | None,
    feature_spec: XGBoostFeatureSpec,
    model_params: Mapping[str, object],
) -> XGBoostMatrixInputPayload:
    resolution = resolve_xgboost_gpu_input_backend(model_params, feature_spec)
    if resolution.fallback_reason is not None:
        LOGGER.debug("XGBoost GPU input fallback: %s", resolution.fallback_reason)
    if resolution.resolved_backend == XGBOOST_GPU_INPUT_CPU:
        return XGBoostMatrixInputPayload(
            data=data,
            label=label,
            weight=weight,
            resolution=resolution,
        )
    if resolution.resolved_backend == XGBOOST_GPU_INPUT_CUPY:
        return _cupy_payload(
            data=data,
            label=label,
            weight=weight,
            feature_spec=feature_spec,
            resolution=resolution,
        )
    return _cudf_payload(
        data=data,
        label=label,
        weight=weight,
        feature_spec=feature_spec,
        resolution=resolution,
    )


def _cupy_payload(
    *,
    data: pd.DataFrame | np.ndarray,
    label: np.ndarray | None,
    weight: np.ndarray | None,
    feature_spec: XGBoostFeatureSpec,
    resolution: XGBoostGpuInputResolution,
) -> XGBoostMatrixInputPayload:
    _raise_if_cupy_cannot_handle_feature_spec(feature_spec)
    cupy_module = _require_gpu_module(XGBOOST_GPU_INPUT_CUPY)
    return XGBoostMatrixInputPayload(
        data=_cupy_array(cupy_module, data),
        label=_cupy_array(cupy_module, label) if label is not None else None,
        weight=_cupy_array(cupy_module, weight) if weight is not None else None,
        resolution=resolution,
    )


def _cudf_payload(
    *,
    data: pd.DataFrame | np.ndarray,
    label: np.ndarray | None,
    weight: np.ndarray | None,
    feature_spec: XGBoostFeatureSpec,
    resolution: XGBoostGpuInputResolution,
) -> XGBoostMatrixInputPayload:
    cudf_module = _require_gpu_module(XGBOOST_GPU_INPUT_CUDF)
    return XGBoostMatrixInputPayload(
        data=_cudf_frame(cudf_module, data, feature_spec),
        label=_cudf_series(cudf_module, label) if label is not None else None,
        weight=_cudf_series(cudf_module, weight) if weight is not None else None,
        resolution=resolution,
    )


def _cupy_array(cupy_module: Any, values: pd.DataFrame | np.ndarray) -> object:
    asarray = getattr(cupy_module, "asarray")
    if isinstance(values, pd.DataFrame):
        values = values.to_numpy(dtype=np.float32, copy=True)
    return cast(object, asarray(values, dtype=np.float32))


def _cudf_frame(
    cudf_module: Any,
    values: pd.DataFrame | np.ndarray,
    feature_spec: XGBoostFeatureSpec,
) -> object:
    if isinstance(values, pd.DataFrame):
        from_pandas = getattr(cudf_module, "from_pandas")
        return cast(object, from_pandas(values))
    dataframe = getattr(cudf_module, "DataFrame")
    return cast(object, dataframe(values, columns=feature_spec.feature_cols))


def _cudf_series(cudf_module: Any, values: np.ndarray) -> object:
    series = getattr(cudf_module, "Series")
    return cast(object, series(values))


def _require_gpu_module(module_name: str) -> Any:
    module = optional_gpu_module(module_name)
    if module is None:
        raise RuntimeError(
            f"XGBoost GPU input backend `{module_name}` was requested, but `{module_name}` "
            "is not installed in this Python environment."
        )
    return module


def _raise_if_cupy_cannot_handle_feature_spec(
    feature_spec: XGBoostFeatureSpec,
) -> None:
    if feature_spec.native_categorical:
        raise RuntimeError(
            "XGBoost CuPy input requires ordinal-encoded features. "
            "Use `enable_categorical=False`, or use `xgboost_gpu_input_backend='cudf'` "
            "for native categorical features."
        )

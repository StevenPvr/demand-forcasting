from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
import logging
import subprocess
from typing import Any, Callable, cast

import numpy as np

from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_XGBOOST_CUDA_MATRIX_TYPE,
)


LOGGER = logging.getLogger(__name__)

XGBOOST_AUTO_RUNTIME_PROFILE = "auto"
XGBOOST_CPU_RUNTIME_PROFILE = "local_cpu"
XGBOOST_GENERIC_CUDA_RUNTIME_PROFILE = "cuda"
XGBOOST_CUDA_RUNTIME_PROFILES = frozenset(
    {XGBOOST_GENERIC_CUDA_RUNTIME_PROFILE, "scaleway_l40s", "nvidia_h100"}
)
SUPPORTED_XGBOOST_RUNTIME_PROFILES = frozenset(
    {
        XGBOOST_AUTO_RUNTIME_PROFILE,
        XGBOOST_CPU_RUNTIME_PROFILE,
        *XGBOOST_CUDA_RUNTIME_PROFILES,
    }
)


@dataclass(frozen=True)
class XGBoostRuntimeResolution:
    requested_profile: str
    runtime_profile: str
    device: str
    tree_method: str
    accelerator: str
    devices: int
    cuda_available: bool
    cuda_device_name: str | None
    fallback_reason: str | None = None


def _xgboost_module() -> Any | None:
    if find_spec("xgboost") is None:
        return None
    return import_module("xgboost")


def _torch_module() -> Any | None:
    if find_spec("torch") is None:
        return None
    return import_module("torch")


def _cuda_device_name() -> str | None:
    torch_module = _torch_module()
    if torch_module is not None:
        cuda = getattr(torch_module, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        if callable(is_available) and bool(is_available()):
            get_device_name = getattr(cuda, "get_device_name", None)
            if callable(get_device_name):
                try:
                    return str(get_device_name(0))
                except RuntimeError:
                    return None
    return _nvidia_smi_device_name()


def _nvidia_smi_device_name() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    return first_line or None


def _cuda_profile_from_device_name(device_name: str | None) -> str:
    if device_name is None:
        return XGBOOST_GENERIC_CUDA_RUNTIME_PROFILE
    normalized = device_name.upper()
    if "H100" in normalized:
        return "nvidia_h100"
    if "L40S" in normalized:
        return "scaleway_l40s"
    return XGBOOST_GENERIC_CUDA_RUNTIME_PROFILE


def _xgboost_build_has_cuda(xgb_module: Any) -> bool | None:
    build_info = getattr(xgb_module, "build_info", None)
    if not callable(build_info):
        return None
    try:
        raw_info = cast(Callable[[], object], build_info)()
    except Exception as exc:
        LOGGER.debug("XGBoost build_info unavailable: %s", exc)
        return None
    if not isinstance(raw_info, Mapping):
        return None
    info = cast(Mapping[str, object], raw_info)
    use_cuda = info.get("USE_CUDA")
    if isinstance(use_cuda, bool):
        return use_cuda
    if isinstance(use_cuda, int):
        return bool(use_cuda)
    if isinstance(use_cuda, str):
        return use_cuda.strip().lower() in {"1", "true", "yes", "on"}
    return None


@lru_cache(maxsize=1)
def xgboost_cuda_preflight_available() -> bool:
    """Retourne True seulement si le runtime XGBoost CUDA peut entraîner un mini booster."""

    xgb_module = _xgboost_module()
    if xgb_module is None:
        LOGGER.debug("XGBoost CUDA preflight skipped: xgboost is not installed.")
        return False
    device_name = _cuda_device_name()
    if device_name is None:
        LOGGER.debug("XGBoost CUDA preflight skipped: no visible CUDA GPU.")
        return False
    build_has_cuda = _xgboost_build_has_cuda(xgb_module)
    if build_has_cuda is False:
        LOGGER.debug(
            "XGBoost CUDA preflight skipped: libxgboost was built without CUDA."
        )
        return False
    data = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    label = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    try:
        matrix = xgb_module.QuantileDMatrix(
            data=data,
            label=label,
            feature_names=["cuda_probe"],
            max_bin=16,
            nthread=1,
        )
        xgb_module.train(
            params={
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 1,
                "max_bin": 16,
                "nthread": 1,
                "verbosity": 0,
            },
            dtrain=matrix,
            num_boost_round=1,
            verbose_eval=False,
        )
    except Exception as exc:
        LOGGER.debug("XGBoost CUDA preflight failed: %s", exc)
        return False
    LOGGER.debug("XGBoost CUDA preflight succeeded: device_name=%s", device_name)
    return True


def resolve_xgboost_runtime_profile(
    requested_profile: str = XGBOOST_AUTO_RUNTIME_PROFILE,
) -> XGBoostRuntimeResolution:
    normalized = requested_profile.strip().lower()
    if normalized == "mac_metal":
        raise RuntimeError(
            "XGBoost does not support the `mac_metal` runtime profile. "
            "Use `auto`, `local_cpu`, or a CUDA profile."
        )
    if normalized not in SUPPORTED_XGBOOST_RUNTIME_PROFILES:
        raise RuntimeError(f"Unknown XGBoost runtime profile `{requested_profile}`.")
    if normalized == XGBOOST_CPU_RUNTIME_PROFILE:
        return _cpu_resolution(requested_profile=requested_profile)
    cuda_available = xgboost_cuda_preflight_available()
    if normalized == XGBOOST_AUTO_RUNTIME_PROFILE and not cuda_available:
        return _cpu_resolution(
            requested_profile=requested_profile,
            fallback_reason="XGBoost CUDA preflight unavailable; falling back to CPU.",
        )
    if not cuda_available:
        raise RuntimeError(
            f"XGBoost runtime profile `{requested_profile}` requires a working "
            "XGBoost CUDA runtime, but the CUDA preflight failed."
        )
    device_name = _cuda_device_name()
    runtime_profile = (
        _cuda_profile_from_device_name(device_name)
        if normalized == XGBOOST_AUTO_RUNTIME_PROFILE
        else normalized
    )
    return XGBoostRuntimeResolution(
        requested_profile=requested_profile,
        runtime_profile=runtime_profile,
        device="cuda",
        tree_method="hist",
        accelerator="gpu",
        devices=1,
        cuda_available=True,
        cuda_device_name=device_name,
    )


def _cpu_resolution(
    *,
    requested_profile: str,
    fallback_reason: str | None = None,
) -> XGBoostRuntimeResolution:
    return XGBoostRuntimeResolution(
        requested_profile=requested_profile,
        runtime_profile=XGBOOST_CPU_RUNTIME_PROFILE,
        device="cpu",
        tree_method="hist",
        accelerator="cpu",
        devices=1,
        cuda_available=False,
        cuda_device_name=None,
        fallback_reason=fallback_reason,
    )


def xgboost_uses_cuda(model_params: Mapping[str, object]) -> bool:
    device = str(model_params.get("device", "")).lower()
    runtime_profile = str(model_params.get("runtime_profile", "")).lower()
    return device == "cuda" or runtime_profile in XGBOOST_CUDA_RUNTIME_PROFILES


def xgboost_matrix_type(model_params: Mapping[str, object]) -> str:
    if xgboost_uses_cuda(model_params):
        return str(
            model_params.get("xgboost_matrix_type", DEFAULT_XGBOOST_CUDA_MATRIX_TYPE)
        )
    return "dmatrix"

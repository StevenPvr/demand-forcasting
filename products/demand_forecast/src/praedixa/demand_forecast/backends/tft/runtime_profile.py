from __future__ import annotations

from typing import Any


DEFAULT_RUNTIME_PROFILE_NAME = "local_cpu"
DEFAULT_COMPILE_MODE = "off"
DEFAULT_DETERMINISM_MODE = "strict"


def _detect_bf16_support() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        return bool(is_bf16_supported())
    return False


def _detect_mps_support() -> bool:
    try:
        import torch
    except ImportError:
        return False
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    is_available = getattr(mps_backend, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def resolve_runtime_profile(
    profile_name: str,
    *,
    bf16_supported: bool | None = None,
) -> dict[str, Any]:
    resolved_bf16 = _detect_bf16_support() if bf16_supported is None else bf16_supported
    profiles: dict[str, dict[str, Any]] = {
        "local_cpu": {
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "compile_mode": DEFAULT_COMPILE_MODE,
            "determinism_mode": "strict",
            "matmul_precision": "highest",
        },
        "mac_metal": {
            "accelerator": "mps" if _detect_mps_support() else "cpu",
            "devices": 1,
            "precision": "32-true",
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "compile_mode": DEFAULT_COMPILE_MODE,
            "determinism_mode": "warn_only",
            "matmul_precision": "high",
        },
        "scaleway_l40s": {
            "accelerator": "gpu",
            "devices": 1,
            "precision": "bf16-mixed" if resolved_bf16 else "32-true",
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "compile_mode": DEFAULT_COMPILE_MODE,
            "determinism_mode": "warn_only",
            "matmul_precision": "high",
        },
    }
    if profile_name not in profiles:
        raise ValueError(f"Unknown TFT runtime profile `{profile_name}`.")
    resolved_profile = dict(profiles[profile_name])
    resolved_profile["torch_compile"] = resolved_profile["compile_mode"] != DEFAULT_COMPILE_MODE
    return resolved_profile

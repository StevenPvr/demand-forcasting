from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any


def resolve_git_sha(*, cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    resolved = result.stdout.strip()
    return resolved or None


def collect_tft_system_info(*, runtime_profile: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "host_name": socket.gethostname(),
        "instance_type": os.getenv("PRAEDIXA_INSTANCE_TYPE"),
        "runtime_profile": runtime_profile,
        "python_version": sys.version.split()[0],
        "gpu_name": None,
        "device_count": 0,
        "cuda_version": None,
        "torch_version": None,
        "lightning_version": None,
    }
    try:
        import lightning
        import torch
    except ImportError:
        return info
    info["torch_version"] = getattr(torch, "__version__", None)
    info["lightning_version"] = getattr(lightning, "__version__", None)
    if not torch.cuda.is_available():
        return info
    info["device_count"] = int(torch.cuda.device_count())
    info["gpu_name"] = str(torch.cuda.get_device_name(0))
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    return info

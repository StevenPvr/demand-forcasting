from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path
from typing import Any


def _stdlib_platform_module_spec() -> tuple[Path, Any]:
    stdlib_platform_path = Path(sysconfig.get_path("stdlib")) / "platform.py"
    spec = importlib.util.spec_from_file_location("_praedixa_stdlib_platform", stdlib_platform_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load stdlib platform module from {stdlib_platform_path}")
    return stdlib_platform_path, spec


def _load_stdlib_platform_attributes() -> None:
    _stdlib_platform_path, spec = _stdlib_platform_module_spec()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in dir(module):
        if name.startswith("__") and name not in {"__all__", "__doc__"}:
            continue
        globals().setdefault(name, getattr(module, name))


_load_stdlib_platform_attributes()

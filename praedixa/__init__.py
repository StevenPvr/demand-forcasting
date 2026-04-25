from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path
from typing import cast

_PACKAGE_PATH = cast(list[str], extend_path(__path__, __name__))
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

for _src_root in (
    _PROJECT_ROOT / "platform" / "python" / "src" / "praedixa",
    _PROJECT_ROOT / "products" / "demand_forecast" / "src" / "praedixa",
):
    if _src_root.exists():
        _PACKAGE_PATH.append(str(_src_root))

__path__ = _PACKAGE_PATH


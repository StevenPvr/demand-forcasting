from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXTRA_ROOTS = (
    _PROJECT_ROOT / "platform" / "python" / "src" / "praedixa",
    _PROJECT_ROOT / "products" / "demand_forecast" / "src" / "praedixa",
)

for _extra_root in _EXTRA_ROOTS:
    _extra_root_str = str(_extra_root)
    if _extra_root.exists() and _extra_root_str not in __path__:
        __path__.append(_extra_root_str)

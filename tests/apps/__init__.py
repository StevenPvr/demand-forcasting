from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path
from typing import cast

_PACKAGE_PATH = cast(list[str], extend_path(__path__, __name__))
_PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists()
)
_APP_PACKAGE_ROOT = _PROJECT_ROOT / "apps"
if _APP_PACKAGE_ROOT.exists():
    _PACKAGE_PATH.append(str(_APP_PACKAGE_ROOT))

__path__ = _PACKAGE_PATH

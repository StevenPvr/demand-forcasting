from __future__ import annotations

import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    current_file = Path(__file__).resolve()
    try:
        return next(
            parent for parent in current_file.parents if (parent / "AGENTS.md").exists()
        )
    except StopIteration as exc:  # pragma: no cover - repository invariant
        raise RuntimeError(
            f"Unable to resolve the Praedixa project root from {current_file}"
        ) from exc


PROJECT_ROOT = _resolve_project_root()
SEARCH_PATHS: tuple[Path, ...] = (
    PROJECT_ROOT,
    PROJECT_ROOT / "platform" / "python" / "src",
    PROJECT_ROOT / "products" / "demand_forecast" / "src",
)
for path in reversed(SEARCH_PATHS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    from praedixa.demand_forecast.feature_screening.main import (
        main as feature_screening_main,
    )

    feature_screening_main()


if __name__ == "__main__":
    main()

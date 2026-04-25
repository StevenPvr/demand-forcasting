from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_import_paths() -> None:
    if __package__ not in {None, ""}:
        return
    current_file = Path(__file__).resolve()
    try:
        project_root = next(
            parent for parent in current_file.parents if (parent / "AGENTS.md").exists()
        )
    except StopIteration as exc:  # pragma: no cover - repository invariant
        raise RuntimeError(
            f"Unable to resolve the Praedixa project root from {current_file}"
        ) from exc
    search_paths: tuple[Path, ...] = (
        project_root,
        project_root / "platform" / "python" / "src",
        project_root / "products" / "demand_forecast" / "src",
    )
    for path in reversed(search_paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from praedixa.demand_forecast.training.config.main_config import (  # noqa: E402
    OFFICIAL_OPTIMISATION_MAIN_CONFIG,
    OptimisationMainConfig,
    build_default_optimisation_main_config,
    build_tft_optimisation_main_config,
    build_xgboost_optimisation_main_config,
)
from praedixa.demand_forecast.training.entrypoints.main_runner import (  # noqa: E402
    main,
    run_optimisation_main,
)

__all__ = [
    "OFFICIAL_OPTIMISATION_MAIN_CONFIG",
    "OptimisationMainConfig",
    "build_default_optimisation_main_config",
    "build_tft_optimisation_main_config",
    "build_xgboost_optimisation_main_config",
    "main",
    "run_optimisation_main",
]


if __name__ == "__main__":
    main()

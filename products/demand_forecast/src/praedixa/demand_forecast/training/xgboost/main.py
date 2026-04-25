from __future__ import annotations

import faulthandler
import logging
import sys
from pathlib import Path


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

from praedixa.demand_forecast.training.main import (  # noqa: E402
    build_xgboost_optimisation_main_config,
    run_optimisation_main,
)


def main() -> None:
    faulthandler.enable(all_threads=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_optimisation_main(build_xgboost_optimisation_main_config())


if __name__ == "__main__":
    main()

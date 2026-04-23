from __future__ import annotations

import logging
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


def main() -> None:
    from praedixa.demand_forecast.backends.tft.backend import TFTBackendNotReadyError
    from praedixa.demand_forecast.evaluation.pipeline import (
        EvaluationBuildRequest,
        build_evaluation_outputs,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        build_evaluation_outputs(EvaluationBuildRequest())
    except TFTBackendNotReadyError:
        logging.getLogger(__name__).error(
            "Le backend TFT unique n'est pas encore branche : l'etape evaluation reste un placeholder structurel."
        )
        raise


if __name__ == "__main__":
    main()

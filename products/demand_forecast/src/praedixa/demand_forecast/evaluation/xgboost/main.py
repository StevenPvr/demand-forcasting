from __future__ import annotations

import logging
from pathlib import Path
import sys


def _bootstrap_import_paths() -> None:
    if __package__ not in {None, ""}:
        return
    current_file = Path(__file__).resolve()
    project_root = next(
        parent for parent in current_file.parents if (parent / "AGENTS.md").exists()
    )
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
    from praedixa.demand_forecast.evaluation.pipeline import (
        EvaluationBuildRequest,
        build_evaluation_outputs,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    build_evaluation_outputs(
        EvaluationBuildRequest(
            model_backend="xgboost",
        )
    )


if __name__ == "__main__":
    main()

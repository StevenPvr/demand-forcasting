from __future__ import annotations

import logging
from pathlib import Path
import sys
import warnings


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


def _configure_runtime_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The given NumPy array is not writable.*",
        category=UserWarning,
        module="chronos.chronos2.dataset",
    )


def main() -> None:
    from praedixa.demand_forecast.backends.chronos2.backend import (
        Chronos2BackendNotReadyError,
    )
    from praedixa.demand_forecast.evaluation.pipeline import (
        EvaluationBuildRequest,
        build_evaluation_outputs,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _configure_runtime_warnings()
    try:
        build_evaluation_outputs(
            EvaluationBuildRequest(
                model_backend="chronos2",
            )
        )
    except Chronos2BackendNotReadyError:
        logging.getLogger(__name__).error(
            "Le backend Chronos-2 n'est pas disponible : synchroniser l'environnement "
            "avec `uv sync --extra chronos2` puis relancer l'evaluation."
        )
        raise


if __name__ == "__main__":
    main()

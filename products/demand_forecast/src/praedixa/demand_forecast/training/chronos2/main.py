from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _force_single_job_runtime() -> None:
    thread_env_vars = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "PRAEDIXA_OPTIMISATION_SAMPLING_THREADS",
        "PRAEDIXA_STATISTICAL_BASELINE_WORKERS",
    )
    for name in thread_env_vars:
        os.environ[name] = "1"


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


_force_single_job_runtime()
_bootstrap_import_paths()

from praedixa.demand_forecast.training.main import (  # noqa: E402
    build_chronos2_finetune_optimisation_main_config,
    run_optimisation_main,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_optimisation_main(build_chronos2_finetune_optimisation_main_config())


if __name__ == "__main__":
    main()

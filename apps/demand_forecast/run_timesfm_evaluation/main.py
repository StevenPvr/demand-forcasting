from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

_TIMESFM_RUNTIME_ENV = "PRAEDIXA_TIMESFM_RUNTIME"


def _project_root() -> Path:
    current_file = Path(__file__).resolve()
    return next(
        parent for parent in current_file.parents if (parent / "AGENTS.md").exists()
    )


def _bootstrap_import_paths() -> None:
    if __package__ not in {None, ""}:
        return
    project_root = _project_root()
    search_paths: tuple[Path, ...] = (
        project_root,
        project_root / "platform" / "python" / "src",
        project_root / "products" / "demand_forecast" / "src",
    )
    for path in reversed(search_paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _reexec_with_timesfm_runtime_if_needed() -> None:
    if sys.version_info < (3, 12) or os.environ.get(_TIMESFM_RUNTIME_ENV) == "1":
        return
    project_root = _project_root()
    runtime_python = project_root / ".venv-timesfm" / "bin" / "python"
    if not runtime_python.exists():
        raise RuntimeError(
            "TimesFM requiert Python <3.12. Creer le runtime dedie avec: "
            "uv venv --python 3.11 .venv-timesfm && "
            ".venv-timesfm/bin/python -m pip install 'timesfm[torch]' jax pandas numpy duckdb pyarrow matplotlib scikit-learn pyyaml optuna xgboost"
        )
    env = os.environ.copy()
    env[_TIMESFM_RUNTIME_ENV] = "1"
    env.setdefault("MPLCONFIGDIR", str(project_root / "var" / "cache" / "matplotlib"))
    python_path_parts = [
        str(project_root),
        str(project_root / "platform" / "python" / "src"),
        str(project_root / "products" / "demand_forecast" / "src"),
    ]
    existing_python_path = env.get("PYTHONPATH")
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    os.execve(str(runtime_python), [str(runtime_python), *sys.argv], env)


_reexec_with_timesfm_runtime_if_needed()
_bootstrap_import_paths()


def main() -> None:
    from praedixa.demand_forecast.backends.timesfm.backend import (
        TimesFMBackendNotReadyError,
    )
    from praedixa.demand_forecast.evaluation.pipeline import (
        EvaluationBuildRequest,
        build_evaluation_outputs,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        build_evaluation_outputs(EvaluationBuildRequest(model_backend="timesfm"))
    except TimesFMBackendNotReadyError:
        logging.getLogger(__name__).error(
            "Le backend TimesFM requiert le runtime dedie `.venv-timesfm` en Python 3.11."
        )
        raise


if __name__ == "__main__":
    main()

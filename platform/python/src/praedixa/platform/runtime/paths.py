from __future__ import annotations

import os
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


PROJECT_ROOT: Path = _resolve_project_root()
PLATFORM_ROOT: Path = PROJECT_ROOT / "platform"
PLATFORM_PYTHON_ROOT: Path = PLATFORM_ROOT / "python"
PLATFORM_SRC_ROOT: Path = PLATFORM_PYTHON_ROOT / "src"
PRODUCTS_ROOT: Path = PROJECT_ROOT / "products"
DEMAND_FORECAST_ROOT: Path = PRODUCTS_ROOT / "demand_forecast"
DEMAND_FORECAST_SRC_ROOT: Path = DEMAND_FORECAST_ROOT / "src"
APPS_ROOT: Path = PROJECT_ROOT / "apps"
DOCS_ROOT: Path = PROJECT_ROOT / "docs"
OPS_ROOT: Path = PROJECT_ROOT / "ops"
WAREHOUSE_PROJECT_DIR: Path = PLATFORM_ROOT / "warehouse"

_var_dir_env = os.getenv("PRAEDIXA_VAR_DIR")
_resolved_var_dir = Path(_var_dir_env) if _var_dir_env else (PROJECT_ROOT / "var")
if not _resolved_var_dir.is_absolute():
    _resolved_var_dir = PROJECT_ROOT / _resolved_var_dir
VAR_DIR: Path = _resolved_var_dir

SOURCES_DIR: Path = VAR_DIR / "sources"
DATASETS_DIR: Path = VAR_DIR / "datasets"
EXPERIMENTS_DIR: Path = VAR_DIR / "experiments"
CACHE_DIR: Path = VAR_DIR / "cache"
LOGS_DIR: Path = VAR_DIR / "logs"
WAREHOUSE_RUNTIME_DIR: Path = VAR_DIR / "warehouse"
DBT_TARGET_DIR: Path = WAREHOUSE_RUNTIME_DIR / "target"
DBT_LOG_DIR: Path = WAREHOUSE_RUNTIME_DIR / "logs"

GLOBAL_DATASET_DIR: Path = DATASETS_DIR / "global_dataset"
EXTERNAL_OPEN_DIR: Path = DATASETS_DIR / "open_exogenous"
FEATURE_SELECTION_DIR: Path = EXPERIMENTS_DIR / "demand_forecast" / "feature_screening"
TRAINING_BUNDLE_DIR: Path = EXPERIMENTS_DIR / "demand_forecast" / "training_bundle"
OPTIMISATION_DIR: Path = EXPERIMENTS_DIR / "demand_forecast" / "training"
EVALUATION_DIR: Path = EXPERIMENTS_DIR / "demand_forecast" / "evaluation"

LOCAL_DUCKDB_PATH: Path = WAREHOUSE_RUNTIME_DIR / "praedixa.duckdb"

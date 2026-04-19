from __future__ import annotations

"""Deprecated compatibility shim for the former ClickHouse bronze loader."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.load_bronze_duckdb import (  # noqa: F401
    BronzeLoadRuntimeConfig,
    BronzeTableSpec,
    DEFAULT_BATCH_ROWS,
    backup_local_bronze_sources,
    build_cloud_duckdb_config_from_env,
    build_local_duckdb_config_from_env,
    build_runtime_config_from_env,
    default_bronze_specs,
    ensure_table,
    load_all_bronze_tables,
    load_bronze_tables_into_target,
    main,
    truncate_table,
)

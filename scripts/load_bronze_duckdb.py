from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Protocol

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bronze_duckdb_specs import BronzeTableSpec, default_active_bronze_specs, default_bronze_specs

logger = logging.getLogger(__name__)
DEFAULT_BATCH_ROWS = 100_000
DEFAULT_LOCAL_BACKUP_DIR = Path("data/local_backup")
DEFAULT_LOCAL_DUCKDB_PATH = Path("data/warehouse/praedixa.duckdb")
DEFAULT_CLOUD_DUCKDB_PATH = "md:praedixa"
DEFAULT_BRONZE_SCHEMA = "bronze"


class DuckDBConnectionProtocol(Protocol):
    """Minimal DuckDB connection protocol used by the bronze loader."""

    def execute(self, query: str, parameters: object | None = None) -> object:
        """Execute one SQL statement."""

    def register(self, view_name: str, python_object: object) -> object:
        """Register a Python object as a DuckDB relation."""

    def unregister(self, view_name: str) -> object:
        """Drop a previously registered relation."""

    def close(self) -> object:
        """Close the DuckDB connection."""


@dataclass(frozen=True)
class DuckDBTargetConfig:
    """Connection settings for one DuckDB target."""

    database_path: str
    schema_name: str
    is_cloud: bool
    token: str | None = None


@dataclass(frozen=True)
class BronzeLoadRuntimeConfig:
    """Runtime switches controlling local/cloud DuckDB sync and local backup."""

    local_warehouse_enabled: bool
    cloud_warehouse_enabled: bool
    local_backup_enabled: bool
    local_backup_dir: Path


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def build_runtime_config_from_env() -> BronzeLoadRuntimeConfig:
    """Build runtime switches for local DuckDB, cloud DuckDB, and backups."""

    return BronzeLoadRuntimeConfig(
        local_warehouse_enabled=_env_flag("PRAEDIXA_ENABLE_LOCAL_WAREHOUSE", default=True),
        cloud_warehouse_enabled=_env_flag("PRAEDIXA_ENABLE_CLOUD_WAREHOUSE", default=False),
        local_backup_enabled=_env_flag("PRAEDIXA_ENABLE_LOCAL_BACKUP", default=True),
        local_backup_dir=_env_path("PRAEDIXA_LOCAL_BACKUP_DIR", DEFAULT_LOCAL_BACKUP_DIR),
    )


def build_local_duckdb_config_from_env() -> DuckDBTargetConfig:
    """Build the local DuckDB target configuration from the environment."""

    return DuckDBTargetConfig(
        database_path=os.environ.get("PRAEDIXA_DUCKDB_LOCAL_PATH", str(DEFAULT_LOCAL_DUCKDB_PATH)),
        schema_name=os.environ.get("PRAEDIXA_DUCKDB_BRONZE_SCHEMA", DEFAULT_BRONZE_SCHEMA),
        is_cloud=False,
    )


def build_cloud_duckdb_config_from_env() -> DuckDBTargetConfig:
    """Build the cloud DuckDB target configuration from the environment."""

    token = os.environ.get("PRAEDIXA_DUCKDB_CLOUD_TOKEN") or os.environ.get("motherduck_token")
    return DuckDBTargetConfig(
        database_path=os.environ.get("PRAEDIXA_DUCKDB_CLOUD_PATH", DEFAULT_CLOUD_DUCKDB_PATH),
        schema_name=os.environ.get("PRAEDIXA_DUCKDB_BRONZE_SCHEMA", DEFAULT_BRONZE_SCHEMA),
        is_cloud=True,
        token=token,
    )


def _duckdb_connection_string(config: DuckDBTargetConfig) -> str:
    if not config.is_cloud or config.token is None or "motherduck_token=" in config.database_path:
        return config.database_path
    separator = "&" if "?" in config.database_path else "?"
    return f"{config.database_path}{separator}motherduck_token={config.token}"


def build_duckdb_client(config: DuckDBTargetConfig) -> DuckDBConnectionProtocol:
    """Create a DuckDB connection for one target."""

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "duckdb is required to load bronze tables. Install project dependencies before running."
        ) from exc

    if not config.is_cloud:
        Path(config.database_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(database=_duckdb_connection_string(config))


def _backup_target_path(backup_root: Path, spec: BronzeTableSpec) -> Path:
    return backup_root / spec.source_name / spec.source_path.name


def _write_backup_manifest(backup_root: Path, payload: dict[str, object]) -> Path:
    manifest_path = backup_root / "bronze_backup_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest_path


def backup_local_bronze_sources(
    specs: list[BronzeTableSpec],
    *,
    backup_root: str | Path,
) -> dict[str, object]:
    """Copy local bronze source files into a local backup directory with a manifest."""

    target_root = Path(backup_root)
    copied_sources: list[dict[str, object]] = []
    for spec in specs:
        if not spec.source_path.exists():
            continue
        target_path = _backup_target_path(target_root, spec)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec.source_path, target_path)
        copied_sources.append(
            {
                "source_name": spec.source_name,
                "source_path": str(spec.source_path),
                "backup_path": str(target_path),
                "size_bytes": int(target_path.stat().st_size),
            }
        )

    manifest_payload = {
        "created_at": _utc_now().isoformat(),
        "backup_root": str(target_root),
        "sources": copied_sources,
    }
    manifest_path = _write_backup_manifest(target_root, manifest_payload)
    return {
        "backup_root": str(target_root),
        "manifest_path": str(manifest_path),
        "source_count": len(copied_sources),
    }


def _utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(tzinfo=None)


def _iter_parquet_batches(path: Path, batch_rows: int) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_rows):
        yield batch.to_pandas()


def _iter_csv_batches(path: Path, batch_rows: int) -> pd.DataFrame:
    yield from pd.read_csv(path, chunksize=batch_rows)


def _iter_m5_long_batches(path: Path, batch_rows: int) -> pd.DataFrame:
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    for chunk in pd.read_csv(path, chunksize=batch_rows):
        demand_columns = [column for column in chunk.columns if column.startswith("d_")]
        yield chunk.melt(
            id_vars=id_columns,
            value_vars=demand_columns,
            var_name="d",
            value_name="sale_qty",
        )


def _prepare_bronze_batch(spec: BronzeTableSpec, batch: pd.DataFrame) -> pd.DataFrame:
    prepared = batch.copy()
    prepared["source_file_path"] = str(spec.source_path)
    prepared["loaded_at"] = _utc_now()

    if spec.table_name == "bronze_bakery_order_lines":
        prepared = prepared.rename(
            columns={
                "Unnamed: 0": "row_index",
                "date": "sale_date_raw",
                "time": "sale_time_raw",
                "ticket_number": "ticket_number_raw",
                "article": "article_raw",
                "Quantity": "quantity_raw",
                "unit_price": "unit_price_raw",
            }
        )
        prepared["source_partition"] = "historical"
    elif spec.table_name == "bronze_freshretail_daily":
        prepared["source_partition"] = "train" if "data_train" in spec.source_path.name else "val"
    elif spec.table_name == "bronze_m5_sales_long":
        prepared["source_partition"] = "historical"

    return prepared


def _iter_prepared_batches(spec: BronzeTableSpec, batch_rows: int) -> pd.DataFrame:
    if spec.table_name == "bronze_m5_sales_long":
        iterator = _iter_m5_long_batches(spec.source_path, batch_rows=batch_rows)
    elif spec.source_path.suffix.lower() == ".parquet":
        iterator = _iter_parquet_batches(spec.source_path, batch_rows=batch_rows)
    else:
        iterator = _iter_csv_batches(spec.source_path, batch_rows=batch_rows)

    for batch in iterator:
        yield _prepare_bronze_batch(spec, batch)


def _quote_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def ensure_table(client: DuckDBConnectionProtocol, spec: BronzeTableSpec) -> None:
    """Create one bronze table if it does not yet exist."""

    client.execute(spec.ddl)


def truncate_table(client: DuckDBConnectionProtocol, *, schema_name: str, table_name: str) -> None:
    """Delete existing rows from one bronze table before a full replacement load."""

    client.execute(f"DELETE FROM {schema_name}.{table_name}")


def _insert_batch(
    client: DuckDBConnectionProtocol,
    *,
    schema_name: str,
    table_name: str,
    batch: pd.DataFrame,
) -> None:
    temp_view_name = "__praedixa_bronze_batch"
    column_list = ", ".join(_quote_identifier(column_name) for column_name in batch.columns)
    client.register(temp_view_name, batch)
    try:
        client.execute(
            f"""
            INSERT INTO {schema_name}.{table_name} ({column_list})
            SELECT {column_list}
            FROM {temp_view_name}
            """
        )
    finally:
        client.unregister(temp_view_name)


def _group_specs_by_table(specs: list[BronzeTableSpec]) -> dict[str, list[BronzeTableSpec]]:
    grouped_specs: dict[str, list[BronzeTableSpec]] = defaultdict(list)
    for spec in specs:
        grouped_specs[spec.table_name].append(spec)
    return dict(grouped_specs)


def load_bronze_tables_into_target(
    client: DuckDBConnectionProtocol,
    *,
    specs: list[BronzeTableSpec],
    schema_name: str,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> dict[str, int]:
    """Load all local bronze sources into one DuckDB target."""

    row_counts: dict[str, int] = {}
    for table_name, table_specs in _group_specs_by_table(specs).items():
        ensure_table(client, table_specs[0])
        truncate_table(client, schema_name=schema_name, table_name=table_name)

        for spec in table_specs:
            if not spec.source_path.exists():
                continue

            inserted_rows = 0
            for batch in _iter_prepared_batches(spec, batch_rows=batch_rows):
                _insert_batch(client, schema_name=schema_name, table_name=table_name, batch=batch)
                inserted_rows += len(batch)
            row_counts[spec.source_name] = inserted_rows
            logger.info(
                "Loaded %s rows into %s from %s",
                inserted_rows,
                f"{schema_name}.{table_name}",
                spec.source_path,
            )
    return row_counts


def _load_target(
    *,
    target_name: str,
    target_config: DuckDBTargetConfig,
    specs: list[BronzeTableSpec],
    batch_rows: int,
) -> dict[str, int]:
    client = build_duckdb_client(target_config)
    try:
        return load_bronze_tables_into_target(
            client,
            specs=specs,
            schema_name=target_config.schema_name,
            batch_rows=batch_rows,
        )
    finally:
        client.close()
        logger.info("Closed DuckDB target %s (%s).", target_name, target_config.database_path)


def load_all_bronze_tables(
    *,
    data_dir: str | Path,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> dict[str, object]:
    """Load local bronze sources into local/cloud DuckDB targets and/or local backup."""

    schema_name = os.environ.get("PRAEDIXA_DUCKDB_BRONZE_SCHEMA", DEFAULT_BRONZE_SCHEMA)
    specs = default_bronze_specs(
        data_dir,
        schema_name=schema_name,
        open_exogenous_dir=os.environ.get("PRAEDIXA_OPEN_EXOGENOUS_DIR"),
    )
    return load_selected_bronze_specs(specs=specs, batch_rows=batch_rows)


def load_selected_bronze_specs(
    *,
    specs: list[BronzeTableSpec],
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> dict[str, object]:
    """Load a selected subset of bronze specs into local/cloud DuckDB targets and/or local backup."""

    runtime_config = build_runtime_config_from_env()
    result: dict[str, object] = {
        "local_warehouse_enabled": runtime_config.local_warehouse_enabled,
        "cloud_warehouse_enabled": runtime_config.cloud_warehouse_enabled,
        "local_backup_enabled": runtime_config.local_backup_enabled,
        "local_warehouse_row_counts": {},
        "cloud_warehouse_row_counts": {},
    }

    if runtime_config.local_backup_enabled:
        result["local_backup"] = backup_local_bronze_sources(specs, backup_root=runtime_config.local_backup_dir)

    if runtime_config.local_warehouse_enabled:
        local_config = build_local_duckdb_config_from_env()
        result["local_warehouse_row_counts"] = _load_target(
            target_name="local",
            target_config=local_config,
            specs=specs,
            batch_rows=batch_rows,
        )
    else:
        logger.info("Local DuckDB warehouse disabled.")

    if runtime_config.cloud_warehouse_enabled:
        cloud_config = build_cloud_duckdb_config_from_env()
        result["cloud_warehouse_row_counts"] = _load_target(
            target_name="cloud",
            target_config=cloud_config,
            specs=specs,
            batch_rows=batch_rows,
        )
    else:
        logger.info("Cloud DuckDB warehouse disabled.")

    return result


def main() -> None:
    """Load all local bronze replacement datasets into DuckDB targets."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    data_dir = os.environ.get("PRAEDIXA_BRONZE_DATA_DIR", "data")
    load_all_bronze_tables(data_dir=data_dir)


if __name__ == "__main__":
    main()

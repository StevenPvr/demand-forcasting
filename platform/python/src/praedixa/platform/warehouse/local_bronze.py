from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Protocol, cast

import pandas as pd
import pyarrow.parquet as pq

from praedixa.platform.runtime.paths import (
    LOCAL_DUCKDB_PATH,
    PROJECT_ROOT,
    SOURCES_DIR,
    WAREHOUSE_RUNTIME_DIR,
)
from praedixa.platform.warehouse.bronze_specs import (
    BronzeTableSpec,
    bronze_source_manifest_ddl,
    default_active_bronze_specs,
    default_bronze_specs,
)

logger = logging.getLogger(__name__)
DEFAULT_BATCH_ROWS = 100_000
DEFAULT_LOCAL_BACKUP_DIR = SOURCES_DIR / "local_backup"
DEFAULT_LOCAL_DUCKDB_PATH = LOCAL_DUCKDB_PATH
DEFAULT_CLOUD_DUCKDB_PATH = "md:praedixa"
DEFAULT_BRONZE_SCHEMA = "bronze"
PARQUET_API: Any = pq

__all__ = [
    "BronzeTableSpec",
    "EmptyRequiredBronzeSourceError",
    "MissingBronzeColumnsError",
    "MissingRequiredBronzeSourceError",
    "prepare_bronze_batch",
    "backup_local_bronze_sources",
    "build_cloud_duckdb_config_from_env",
    "build_local_duckdb_config_from_env",
    "build_runtime_config_from_env",
    "default_active_bronze_specs",
    "default_bronze_specs",
    "load_inline_bronze_frame",
    "load_all_bronze_tables",
    "validate_bronze_sources",
]


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


class MissingRequiredBronzeSourceError(FileNotFoundError):
    """Raised before warehouse mutation when a required bronze source is absent."""


class EmptyRequiredBronzeSourceError(ValueError):
    """Raised before warehouse mutation when a required bronze source file is empty."""


class MissingBronzeColumnsError(ValueError):
    """Raised when a bronze source is missing columns required by its contract."""


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = Path(os.environ.get(name, str(default)))
    return value if value.is_absolute() else PROJECT_ROOT / value


def build_runtime_config_from_env() -> BronzeLoadRuntimeConfig:
    """Build runtime switches for local DuckDB, cloud DuckDB, and backups."""

    return BronzeLoadRuntimeConfig(
        local_warehouse_enabled=_env_flag(
            "PRAEDIXA_ENABLE_LOCAL_WAREHOUSE", default=True
        ),
        cloud_warehouse_enabled=_env_flag(
            "PRAEDIXA_ENABLE_CLOUD_WAREHOUSE", default=False
        ),
        local_backup_enabled=_env_flag("PRAEDIXA_ENABLE_LOCAL_BACKUP", default=False),
        local_backup_dir=_env_path(
            "PRAEDIXA_LOCAL_BACKUP_DIR", DEFAULT_LOCAL_BACKUP_DIR
        ),
    )


def build_local_duckdb_config_from_env() -> DuckDBTargetConfig:
    """Build the local DuckDB target configuration from the environment."""

    return DuckDBTargetConfig(
        database_path=os.environ.get(
            "PRAEDIXA_DUCKDB_LOCAL_PATH", str(DEFAULT_LOCAL_DUCKDB_PATH)
        ),
        schema_name=os.environ.get(
            "PRAEDIXA_DUCKDB_BRONZE_SCHEMA", DEFAULT_BRONZE_SCHEMA
        ),
        is_cloud=False,
    )


def build_cloud_duckdb_config_from_env() -> DuckDBTargetConfig:
    """Build the cloud DuckDB target configuration from the environment."""

    token = os.environ.get("PRAEDIXA_DUCKDB_CLOUD_TOKEN") or os.environ.get(
        "motherduck_token"
    )
    return DuckDBTargetConfig(
        database_path=os.environ.get(
            "PRAEDIXA_DUCKDB_CLOUD_PATH", DEFAULT_CLOUD_DUCKDB_PATH
        ),
        schema_name=os.environ.get(
            "PRAEDIXA_DUCKDB_BRONZE_SCHEMA", DEFAULT_BRONZE_SCHEMA
        ),
        is_cloud=True,
        token=token,
    )


def _duckdb_connection_string(config: DuckDBTargetConfig) -> str:
    if (
        not config.is_cloud
        or config.token is None
        or "motherduck_token=" in config.database_path
    ):
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
    client = duckdb.connect(database=_duckdb_connection_string(config))
    _configure_duckdb_client(client)
    return client


def _configure_duckdb_client(client: DuckDBConnectionProtocol) -> None:
    """Apply conservative DuckDB runtime settings shared by bronze loads."""

    client.execute("SET preserve_insertion_order = false")
    if threads := os.environ.get("PRAEDIXA_DUCKDB_THREADS"):
        client.execute(f"SET threads = {int(threads)}")
    if memory_limit := os.environ.get("PRAEDIXA_DUCKDB_MEMORY_LIMIT"):
        escaped_memory_limit = memory_limit.replace("'", "''")
        client.execute(f"SET memory_limit = '{escaped_memory_limit}'")
    if temp_directory := os.environ.get("PRAEDIXA_DUCKDB_TEMP_DIRECTORY"):
        temp_path = Path(temp_directory)
        temp_path.mkdir(parents=True, exist_ok=True)
        escaped_temp_directory = str(temp_path).replace("'", "''")
        client.execute(f"SET temp_directory = '{escaped_temp_directory}'")


def validate_bronze_sources(specs: list[BronzeTableSpec]) -> None:
    """Fail before any warehouse mutation when required bronze inputs are unsafe."""

    missing_required = [
        f"{spec.source_name}={spec.source_path}"
        for spec in specs
        if spec.required and not spec.source_path.exists()
    ]
    if missing_required:
        raise MissingRequiredBronzeSourceError(
            "Missing required bronze sources: " + ", ".join(missing_required)
        )

    empty_required = [
        f"{spec.source_name}={spec.source_path}"
        for spec in specs
        if spec.required
        and not spec.allow_empty
        and spec.source_path.exists()
        and spec.source_path.stat().st_size == 0
    ]
    if empty_required:
        raise EmptyRequiredBronzeSourceError(
            "Empty required bronze sources: " + ", ".join(empty_required)
        )


def _backup_target_path(backup_root: Path, spec: BronzeTableSpec) -> Path:
    return backup_root / spec.source_name / spec.source_path.name


def _backup_is_current(source_path: Path, backup_path: Path) -> bool:
    if not backup_path.exists():
        return False
    source_stat = source_path.stat()
    backup_stat = backup_path.stat()
    return (
        source_stat.st_size == backup_stat.st_size
        and source_stat.st_mtime_ns == backup_stat.st_mtime_ns
    )


def _write_backup_manifest(backup_root: Path, payload: dict[str, object]) -> Path:
    manifest_path = backup_root / "bronze_backup_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return manifest_path


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest_entry(
    spec: BronzeTableSpec,
    *,
    row_counts: dict[str, int],
) -> dict[str, object]:
    exists = spec.source_path.exists()
    stat = spec.source_path.stat() if exists else None
    return {
        "source_name": spec.source_name,
        "table_name": spec.table_name,
        "source_path": str(spec.source_path),
        "required": spec.required,
        "allow_empty": spec.allow_empty,
        "partition_keys": list(spec.partition_keys),
        "source_policy_id": spec.source_policy_id,
        "exists": exists,
        "size_bytes": int(stat.st_size) if stat is not None else None,
        "mtime_ns": int(stat.st_mtime_ns) if stat is not None else None,
        "sha256": _source_sha256(spec.source_path) if exists else None,
        "loaded_rows": row_counts.get(spec.source_name),
    }


def write_bronze_source_manifest(
    *,
    specs: list[BronzeTableSpec],
    row_counts: dict[str, int],
) -> Path:
    """Write a deterministic source manifest for the bronze replacement load."""

    manifest_path = WAREHOUSE_RUNTIME_DIR / "bronze_source_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": _utc_now().isoformat(),
        "sources": [
            _source_manifest_entry(spec, row_counts=row_counts) for spec in specs
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return manifest_path


def backup_local_bronze_sources(
    specs: list[BronzeTableSpec],
    *,
    backup_root: str | Path,
) -> dict[str, object]:
    """Copy local bronze source files into a local backup directory with a manifest."""

    target_root = Path(backup_root)
    source_entries: list[dict[str, object]] = []
    copied_count = 0
    for spec in specs:
        if not spec.source_path.exists():
            continue
        target_path = _backup_target_path(target_root, spec)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        copied = not _backup_is_current(spec.source_path, target_path)
        if copied:
            shutil.copy2(spec.source_path, target_path)
            copied_count += 1
        source_entries.append(
            {
                "source_name": spec.source_name,
                "source_path": str(spec.source_path),
                "backup_path": str(target_path),
                "size_bytes": int(target_path.stat().st_size),
                "copied": copied,
            }
        )

    manifest_payload: dict[str, object] = {
        "created_at": _utc_now().isoformat(),
        "backup_root": str(target_root),
        "sources": source_entries,
    }
    manifest_path = _write_backup_manifest(target_root, manifest_payload)
    return {
        "backup_root": str(target_root),
        "manifest_path": str(manifest_path),
        "source_count": len(source_entries),
        "copied_count": copied_count,
    }


def _utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(tzinfo=None)


def _iter_parquet_batches(path: Path, batch_rows: int) -> Iterator[pd.DataFrame]:
    parquet_file: Any = PARQUET_API.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_rows):
        yield cast(pd.DataFrame, batch.to_pandas())


def _iter_csv_batches(path: Path, batch_rows: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, chunksize=batch_rows)


def prepare_bronze_batch(spec: BronzeTableSpec, batch: pd.DataFrame) -> pd.DataFrame:
    prepared = batch.copy(deep=False)
    if "source_name" not in prepared.columns:
        prepared["source_name"] = spec.source_name
    if "source_policy_id" not in prepared.columns:
        prepared["source_policy_id"] = spec.source_policy_id or spec.source_name
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

    return prepared


def _validate_expected_columns(spec: BronzeTableSpec, prepared: pd.DataFrame) -> None:
    if not spec.expected_columns:
        return
    missing_columns = [
        column for column in spec.expected_columns if column not in prepared.columns
    ]
    if missing_columns:
        raise MissingBronzeColumnsError(
            f"Bronze source `{spec.source_name}` is missing required columns after preparation: {missing_columns}"
        )


def _iter_prepared_batches(
    spec: BronzeTableSpec, batch_rows: int
) -> Iterator[pd.DataFrame]:
    if spec.source_path.suffix.lower() == ".parquet":
        iterator = _iter_parquet_batches(spec.source_path, batch_rows=batch_rows)
    else:
        iterator = _iter_csv_batches(spec.source_path, batch_rows=batch_rows)

    for batch in iterator:
        prepared = prepare_bronze_batch(spec, batch)
        _validate_expected_columns(spec, prepared)
        yield prepared


def _quote_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _table_column_names(
    client: DuckDBConnectionProtocol,
    *,
    schema_name: str,
    table_name: str,
) -> set[str]:
    result: Any = client.execute(f"PRAGMA table_info('{schema_name}.{table_name}')")
    return {str(row[1]) for row in result.fetchall()}


def recreate_table(
    client: DuckDBConnectionProtocol, *, schema_name: str, table_name: str
) -> None:
    """Drop one bronze table before a full replacement load."""

    client.execute(f"DROP TABLE IF EXISTS {schema_name}.{table_name}")


def ensure_table(client: DuckDBConnectionProtocol, spec: BronzeTableSpec) -> None:
    """Create one bronze table from the current DDL definition."""

    client.execute(spec.ddl)


def _insert_batch(
    client: DuckDBConnectionProtocol,
    *,
    schema_name: str,
    table_name: str,
    batch: pd.DataFrame,
) -> None:
    temp_view_name = "__praedixa_bronze_batch"
    table_columns = _table_column_names(
        client,
        schema_name=schema_name,
        table_name=table_name,
    )
    insert_columns = [column for column in batch.columns if column in table_columns]
    if not insert_columns:
        raise MissingBronzeColumnsError(
            f"No insertable columns found for bronze table `{schema_name}.{table_name}`."
        )
    ignored_columns = sorted(set(batch.columns).difference(insert_columns))
    if ignored_columns:
        logger.debug(
            "Ignoring source columns absent from bronze table: table=%s ignored_columns=%s",
            table_name,
            ignored_columns,
        )
    column_list = ", ".join(
        _quote_identifier(column_name) for column_name in insert_columns
    )
    client.register(temp_view_name, batch.loc[:, insert_columns])
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


def _prepare_inline_bronze_batch(
    frame: pd.DataFrame,
    *,
    source_label: str,
    source_policy_id: str,
) -> pd.DataFrame:
    prepared = frame.copy(deep=False)
    prepared["source_name"] = source_label
    prepared["source_policy_id"] = source_policy_id
    prepared["source_file_path"] = source_label
    prepared["loaded_at"] = _utc_now()
    return prepared


def _load_inline_frame_target(
    *,
    target_name: str,
    target_config: DuckDBTargetConfig,
    table_name: str,
    ddl: str,
    frame: pd.DataFrame,
    source_name: str,
    source_policy_id: str,
) -> int:
    client = build_duckdb_client(target_config)
    try:
        loaded_rows = 0
        spec = BronzeTableSpec(
            table_name=table_name,
            source_path=Path(source_name),
            ddl=ddl,
            source_name=source_name,
            source_policy_id=source_policy_id,
        )

        def operation() -> None:
            nonlocal loaded_rows
            recreate_table(client, schema_name=target_config.schema_name, table_name=table_name)
            ensure_table(client, spec)
            prepared = _prepare_inline_bronze_batch(
                frame,
                source_label=source_name,
                source_policy_id=source_policy_id,
            )
            _insert_batch(
                client,
                schema_name=target_config.schema_name,
                table_name=table_name,
                batch=prepared,
            )
            loaded_rows = len(prepared)
            _persist_manifest_to_target(
                client,
                specs=[spec],
                row_counts={source_name: loaded_rows},
                schema_name=target_config.schema_name,
            )

        _run_transaction(client, operation)
        return loaded_rows
    finally:
        client.close()
        logger.info(
            "Closed DuckDB target %s (%s).", target_name, target_config.database_path
        )


def load_inline_bronze_frame(
    *,
    table_name: str,
    ddl: str,
    frame: pd.DataFrame,
    source_name: str,
    source_policy_id: str | None = None,
) -> dict[str, int]:
    """Load one in-memory bronze frame into the enabled DuckDB targets without persisting a source file."""

    runtime_config = build_runtime_config_from_env()
    row_counts: dict[str, int] = {}
    resolved_source_policy_id = source_policy_id or source_name
    if runtime_config.local_warehouse_enabled:
        local_config = build_local_duckdb_config_from_env()
        row_counts["local"] = _load_inline_frame_target(
            target_name="local",
            target_config=local_config,
            table_name=table_name,
            ddl=ddl,
            frame=frame,
            source_name=source_name,
            source_policy_id=resolved_source_policy_id,
        )
    if runtime_config.cloud_warehouse_enabled:
        cloud_config = build_cloud_duckdb_config_from_env()
        row_counts["cloud"] = _load_inline_frame_target(
            target_name="cloud",
            target_config=cloud_config,
            table_name=table_name,
            ddl=ddl,
            frame=frame,
            source_name=source_name,
            source_policy_id=resolved_source_policy_id,
        )
    return row_counts


def _group_specs_by_table(
    specs: list[BronzeTableSpec],
) -> dict[str, list[BronzeTableSpec]]:
    grouped_specs: dict[str, list[BronzeTableSpec]] = defaultdict(list)
    for spec in specs:
        grouped_specs[spec.table_name].append(spec)
    return dict(grouped_specs)


def _run_transaction(
    client: DuckDBConnectionProtocol,
    operation: Callable[[], None],
) -> None:
    client.execute("BEGIN TRANSACTION")
    try:
        operation()
    except Exception:
        client.execute("ROLLBACK")
        raise
    client.execute("COMMIT")


def _load_bronze_table_group(
    client: DuckDBConnectionProtocol,
    *,
    table_name: str,
    table_specs: list[BronzeTableSpec],
    schema_name: str,
    batch_rows: int,
) -> dict[str, int]:
    row_counts: dict[str, int] = {}

    def operation() -> None:
        existing_specs = [spec for spec in table_specs if spec.source_path.exists()]
        recreate_table(client, schema_name=schema_name, table_name=table_name)
        ensure_table(client, table_specs[0])

        if not existing_specs:
            for spec in table_specs:
                row_counts[spec.source_name] = 0
            logger.warning(
                "No local files for optional bronze table %s.%s; recreated empty table.",
                schema_name,
                table_name,
            )
            return

        for spec in table_specs:
            if not spec.source_path.exists():
                row_counts[spec.source_name] = 0
                logger.warning(
                    "Skipping missing optional bronze source %s at %s",
                    spec.source_name,
                    spec.source_path,
                )
                continue

            inserted_rows = 0
            for batch in _iter_prepared_batches(spec, batch_rows=batch_rows):
                _insert_batch(
                    client,
                    schema_name=schema_name,
                    table_name=table_name,
                    batch=batch,
                )
                inserted_rows += len(batch)
            row_counts[spec.source_name] = inserted_rows
            logger.info(
                "Loaded %s rows into %s from %s",
                inserted_rows,
                f"{schema_name}.{table_name}",
                spec.source_path,
            )

    operation()
    return row_counts


def _persist_manifest_to_target(
    client: DuckDBConnectionProtocol,
    *,
    specs: list[BronzeTableSpec],
    row_counts: dict[str, int],
    schema_name: str,
) -> None:
    client.execute(bronze_source_manifest_ddl(schema_name))
    loaded_at = _utc_now()
    source_run_id = f"bronze_{loaded_at.strftime('%Y%m%d_%H%M%S')}"
    rows: list[dict[str, object]] = []
    for spec in specs:
        exists = spec.source_path.exists()
        stat = spec.source_path.stat() if exists else None
        rows.append(
            {
                "source_run_id": source_run_id,
                "source_name": spec.source_name,
                "source_policy_id": spec.source_policy_id or spec.source_name,
                "table_name": spec.table_name,
                "source_path": str(spec.source_path),
                "required": spec.required,
                "allow_empty": spec.allow_empty,
                "size_bytes": int(stat.st_size) if stat is not None else None,
                "mtime_ns": int(stat.st_mtime_ns) if stat is not None else None,
                "sha256": _source_sha256(spec.source_path) if exists else None,
                "loaded_rows": row_counts.get(spec.source_name, 0),
                "loaded_at": loaded_at,
            }
        )
    if not rows:
        return
    manifest_frame = pd.DataFrame(rows)
    temp_view_name = "__praedixa_bronze_source_manifest"
    client.register(temp_view_name, manifest_frame)
    try:
        client.execute(
            f"""
            INSERT INTO {schema_name}.bronze_source_manifest
            SELECT *
            FROM {temp_view_name}
            """
        )
    finally:
        client.unregister(temp_view_name)


def load_bronze_tables_into_target(
    client: DuckDBConnectionProtocol,
    *,
    specs: list[BronzeTableSpec],
    schema_name: str,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> dict[str, int]:
    """Load all local bronze sources into one DuckDB target."""

    row_counts: dict[str, int] = {}

    def operation() -> None:
        for table_name, table_specs in _group_specs_by_table(specs).items():
            row_counts.update(
                _load_bronze_table_group(
                    client,
                    table_name=table_name,
                    table_specs=table_specs,
                    schema_name=schema_name,
                    batch_rows=batch_rows,
                )
            )
        _persist_manifest_to_target(
            client,
            specs=specs,
            row_counts=row_counts,
            schema_name=schema_name,
        )

    _run_transaction(client, operation)
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
        logger.info(
            "Closed DuckDB target %s (%s).", target_name, target_config.database_path
        )


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

    validate_bronze_sources(specs)
    runtime_config = build_runtime_config_from_env()
    result: dict[str, object] = {
        "local_warehouse_enabled": runtime_config.local_warehouse_enabled,
        "cloud_warehouse_enabled": runtime_config.cloud_warehouse_enabled,
        "local_backup_enabled": runtime_config.local_backup_enabled,
        "local_warehouse_row_counts": {},
        "cloud_warehouse_row_counts": {},
    }

    if runtime_config.local_backup_enabled:
        result["local_backup"] = backup_local_bronze_sources(
            specs, backup_root=runtime_config.local_backup_dir
        )

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

    local_row_counts = cast(dict[str, int], result["local_warehouse_row_counts"])
    manifest_path = write_bronze_source_manifest(
        specs=specs,
        row_counts=local_row_counts,
    )
    result["source_manifest_path"] = str(manifest_path)

    return result


def main() -> None:
    """Load all local bronze replacement datasets into DuckDB targets."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    data_dir = os.environ.get("PRAEDIXA_BRONZE_DATA_DIR", str(SOURCES_DIR))
    load_all_bronze_tables(
        data_dir=data_dir,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence, cast

import duckdb
import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.tft.frame_utils import (
    GROUP_COL,
    TIME_IDX_COL,
    build_group_identifier,
)
from praedixa.demand_forecast.backends.tft.feature_mapping_spec import (
    TFT_GROUP_ID_COLUMNS,
)
from praedixa.demand_forecast.backends.tft.feature_mapping import (
    TFT_EXPLICIT_ROLE_BY_COLUMN,
)
from praedixa.demand_forecast.backends.tft.feature_mapping import (
    select_explicit_tft_group_id_columns,
)
from praedixa.demand_forecast.backends.tft.feature_contract import (
    build_feature_contract,
)
from praedixa.demand_forecast.backends.tft.model_utils import select_tft_feature_columns
from praedixa.demand_forecast.contracts.targets import (
    DEFAULT_ABSOLUTE_TARGET_COL,
    TargetContract,
    build_target_contract_metadata,
    ensure_learning_target_column,
    resolve_target_contract,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_DATE_COL,
    DEFAULT_DUCKDB_PATH,
    DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
    DEFAULT_GOLD_TABLE,
    DEFAULT_IDENTIFIER_FEATURE_COLS,
    DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES,
    DEFAULT_REMOVED_MODEL_INPUT_COLS,
)
from praedixa.demand_forecast.training.sampling.dataset_filters import (
    dataset_source_not_in_filter,
)
from praedixa.demand_forecast.training.validation.eligibility import (
    DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE,
    GOLD_TRAINING_ELIGIBILITY,
    filter_training_eligible_rows,
    training_eligibility_mask,
)
from praedixa.platform.runtime.paths import TRAINING_BUNDLE_DIR
from praedixa.demand_forecast.training_bundle.manifest import sha256_file


DEFAULT_TRAIN_INPUT_PATH: Path | None = None
DEFAULT_TUNING_INPUT_PATH: Path | None = None
DEFAULT_VALID_INPUT_PATH: Path | None = None
DEFAULT_OUTPUT_DIR = TRAINING_BUNDLE_DIR
DEFAULT_SMOKE_SERIES_LIMIT: int | None = None
DEFAULT_SMOKE_DATASET_SOURCE: str | None = None
DEFAULT_SMOKE_MIN_TRAIN_ROWS = 56
DEFAULT_SMOKE_MIN_TUNING_ROWS = 28
DEFAULT_SMOKE_MIN_VALID_ROWS = 0
BASELINE_SUPPORT_COLUMNS: tuple[str, ...] = ("lag_1",)
SAMPLING_SUPPORT_COLUMNS: tuple[str, ...] = ("location_id", "product_id")
OPTIMISATION_SUPPORT_COLUMNS: tuple[str, ...] = (GROUP_COL, TIME_IDX_COL)
CANONICAL_GROUP_ID_COL: str = TFT_GROUP_ID_COLUMNS[0]


@dataclass(frozen=True)
class BundleProjectionMetadata:
    feature_cols: list[str]
    group_id_columns: list[str]
    projection_columns: list[str]
    projection_dtypes: dict[str, str]
    target_contract: TargetContract
    train_group_sizes: dict[str, int]


@dataclass(frozen=True)
class BundleSplitMetadata:
    train_path: str | Path
    tuning_path: str | Path
    valid_path: str | Path | None
    train_rows: int
    tuning_rows: int
    valid_rows: int
    excluded_optimisation_dataset_sources: tuple[str, ...]


def _sanitize_model_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized = frame.copy()
    removed_columns = [
        column
        for column in DEFAULT_REMOVED_MODEL_INPUT_COLS
        if column in sanitized.columns
    ]
    if removed_columns:
        sanitized = sanitized.drop(columns=removed_columns)
    return sanitized


def _exclude_dataset_sources(
    frame: pd.DataFrame,
    *,
    excluded_dataset_sources: tuple[str, ...],
) -> pd.DataFrame:
    if not excluded_dataset_sources or DEFAULT_DATASET_SOURCE_COL not in frame.columns:
        return frame
    excluded = set(excluded_dataset_sources)
    mask = ~frame[DEFAULT_DATASET_SOURCE_COL].astype(str).isin(excluded)
    return frame.loc[mask].reset_index(drop=True)


def _json_dump(path: Path, payload: Mapping[str, object | None]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame[DEFAULT_DATE_COL] = pd.to_datetime(frame[DEFAULT_DATE_COL])
    frame = _sanitize_model_input_frame(frame)
    return frame.sort_values(DEFAULT_DATE_COL).reset_index(drop=True)


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _gold_schema_columns(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
) -> list[str]:
    rows = connection.execute(f"describe select * from {gold_table} limit 0").fetchall()
    columns = [str(row[0]) for row in rows]
    if not columns:
        raise ValueError(f"Gold table `{gold_table}` exposes no columns.")
    return columns


def _gold_schema_types(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
) -> dict[str, str]:
    rows = connection.execute(f"describe select * from {gold_table} limit 0").fetchall()
    return {str(row[0]): str(row[1]).upper() for row in rows}


def _empty_schema_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _gold_training_eligibility_filter(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    split_bucket: str,
    table_alias: str | None = None,
) -> str:
    if split_bucket not in {"train", "val"}:
        return "true"
    schema_preview = connection.execute(f"select * from {gold_table} limit 0").fetchdf()
    return GOLD_TRAINING_ELIGIBILITY.sql_filter(
        available_columns=set(schema_preview.columns),
        table_alias=table_alias,
    )


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _smoke_series_selection_query(
    *,
    gold_table: str,
    series_limit: int,
    dataset_source: str | None,
    min_train_rows: int,
    min_tuning_rows: int,
    min_valid_rows: int,
    excluded_dataset_sources: tuple[str, ...],
    train_eligibility_filter: str,
    tuning_eligibility_filter: str,
) -> str:
    filters: list[str] = []
    if dataset_source is not None:
        filters.append(f"dataset_source = {_sql_literal(dataset_source)}")
    exclusion_filter = dataset_source_not_in_filter(
        DEFAULT_DATASET_SOURCE_COL,
        excluded_dataset_sources,
    )
    if exclusion_filter != "true":
        filters.append(exclusion_filter)
    where_clause = "" if not filters else f"where {' and '.join(filters)}"
    return f"""
    with eligible as (
        select
            dataset_source,
            {CANONICAL_GROUP_ID_COL},
            sum(case when split_bucket = 'train' and {train_eligibility_filter} then 1 else 0 end) as train_rows,
            sum(case when split_bucket = 'val' and {tuning_eligibility_filter} then 1 else 0 end) as tuning_rows,
            sum(case when split_bucket = 'test' then 1 else 0 end) as valid_rows
        from {gold_table}
        {where_clause}
        group by 1, 2
        having train_rows >= {int(min_train_rows)}
           and tuning_rows >= {int(min_tuning_rows)}
           and valid_rows >= {int(min_valid_rows)}
    )
    select dataset_source, {CANONICAL_GROUP_ID_COL}
    from eligible
    order by train_rows desc, tuning_rows desc, dataset_source, {CANONICAL_GROUP_ID_COL}
    limit {int(series_limit)}
    """


def _load_explicit_frames(
    *,
    train_input_path: str | Path,
    tuning_input_path: str | Path,
    valid_input_path: str | Path | None,
) -> tuple[Path, Path, Path | None, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train_path = Path(train_input_path)
    tuning_path = Path(tuning_input_path)
    valid_path = None if valid_input_path is None else Path(valid_input_path)
    train_frame = _read_frame(train_path)
    tuning_frame = _read_frame(tuning_path)
    valid_frame = None if valid_path is None else _read_frame(valid_path)
    return train_path, tuning_path, valid_path, train_frame, tuning_frame, valid_frame


def _bundle_output_paths(output_dir: Path, *, include_valid: bool) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "train": output_dir / "train.parquet",
        "tuning": output_dir / "tuning.parquet",
        "optimisation_train": output_dir / "optimisation_train.parquet",
        "optimisation_tuning": output_dir / "optimisation_tuning.parquet",
        "feature_manifest": output_dir / "feature_manifest.json",
        "feature_roles": output_dir / "feature_roles.json",
        "split_manifest": output_dir / "split_manifest.json",
        "target_contract": output_dir / "target_contract.json",
        "bundle_manifest": output_dir / "bundle_manifest.json",
        "optimisation_manifest": output_dir / "optimisation_manifest.json",
        "training_exclusion_report": output_dir / "training_exclusion_report.json",
    }
    if include_valid:
        paths["valid"] = output_dir / "valid.parquet"
        paths["optimisation_valid"] = output_dir / "optimisation_valid.parquet"
    return paths


def _training_exclusion_report(
    frame: pd.DataFrame,
    *,
    label: str,
) -> dict[str, object]:
    if frame.empty:
        return {"label": label, "input_rows": 0, "kept_rows": 0, "dropped_rows": 0}
    mask = training_eligibility_mask(frame)
    report: dict[str, object] = {
        "label": label,
        "input_rows": int(len(frame)),
        "kept_rows": int(mask.sum()),
        "dropped_rows": int((~mask).sum()),
    }
    if "censor_flag" in frame.columns:
        report["censored_rows"] = int(
            frame["censor_flag"].fillna(False).astype(bool).sum()
        )
    if "label_quality_score" in frame.columns:
        label_quality = pd.to_numeric(frame["label_quality_score"], errors="coerce")
        report["low_quality_rows"] = int(
            label_quality.lt(DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE)
            .fillna(True)
            .sum()
        )
    if "usable_for_training_flag" in frame.columns:
        report["not_usable_rows"] = int(
            (~frame["usable_for_training_flag"].fillna(False).astype(bool)).sum()
        )
    if "target_source" in frame.columns:
        target_source = frame["target_source"].astype("string")
        report["closed_or_dense_zero_rows"] = int(
            target_source.isin(
                ["closed_or_missing_observation", "dense_calendar_zero_fill"]
            ).sum()
        )
    return report


def _dataset_source_exclusion_report(
    frame: pd.DataFrame,
    *,
    label: str,
    excluded_dataset_sources: tuple[str, ...],
) -> dict[str, object]:
    report: dict[str, object] = {
        "label": label,
        "input_rows": int(len(frame)),
        "excluded_dataset_sources": list(excluded_dataset_sources),
        "excluded_rows": 0,
        "kept_rows": int(len(frame)),
    }
    if not excluded_dataset_sources or DEFAULT_DATASET_SOURCE_COL not in frame.columns:
        return report
    excluded = set(excluded_dataset_sources)
    excluded_mask = frame[DEFAULT_DATASET_SOURCE_COL].astype(str).isin(excluded)
    report["excluded_rows"] = int(excluded_mask.sum())
    report["kept_rows"] = int((~excluded_mask).sum())
    return report


def _filter_bundle_training_frames(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    train_frame_before_holdout: pd.DataFrame,
    tuning_frame_before_holdout: pd.DataFrame,
    excluded_dataset_sources: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    report: dict[str, object] = {
        "min_label_quality_score": DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE,
        "holdout_exclusions": [
            _dataset_source_exclusion_report(
                train_frame_before_holdout,
                label="train",
                excluded_dataset_sources=excluded_dataset_sources,
            ),
            _dataset_source_exclusion_report(
                tuning_frame_before_holdout,
                label="tuning",
                excluded_dataset_sources=excluded_dataset_sources,
            ),
        ],
        "splits": [
            _training_exclusion_report(train_frame, label="train"),
            _training_exclusion_report(tuning_frame, label="tuning"),
        ],
    }
    return (
        filter_training_eligible_rows(train_frame, label="train_bundle"),
        filter_training_eligible_rows(tuning_frame, label="tuning_bundle"),
        report,
    )


def _gold_count_where(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    where_clause: str,
) -> int:
    row = connection.execute(
        f"select count(*) from {gold_table} where {where_clause}"
    ).fetchone()
    if row is None:
        return 0
    return int(cast(int, row[0]))


def _gold_holdout_exclusion_report(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    split_bucket: str,
    label: str,
    excluded_dataset_sources: tuple[str, ...],
) -> dict[str, object]:
    split_filter = f"split_bucket = {_sql_literal(split_bucket)}"
    input_rows = _gold_count_where(
        connection,
        gold_table=gold_table,
        where_clause=split_filter,
    )
    if not excluded_dataset_sources:
        return {
            "label": label,
            "input_rows": input_rows,
            "excluded_dataset_sources": [],
            "excluded_rows": 0,
            "kept_rows": input_rows,
        }
    kept_filter = dataset_source_not_in_filter(
        DEFAULT_DATASET_SOURCE_COL,
        excluded_dataset_sources,
    )
    kept_rows = _gold_count_where(
        connection,
        gold_table=gold_table,
        where_clause=f"{split_filter} and {kept_filter}",
    )
    return {
        "label": label,
        "input_rows": input_rows,
        "excluded_dataset_sources": list(excluded_dataset_sources),
        "excluded_rows": int(input_rows - kept_rows),
        "kept_rows": kept_rows,
    }


def _gold_training_split_exclusion_report(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    split_bucket: str,
    label: str,
    excluded_dataset_sources: tuple[str, ...],
    available_columns: set[str],
) -> dict[str, object]:
    split_filter = f"split_bucket = {_sql_literal(split_bucket)}"
    scope_filter = dataset_source_not_in_filter(
        DEFAULT_DATASET_SOURCE_COL,
        excluded_dataset_sources,
    )
    scoped_where = f"{split_filter} and {scope_filter}"
    eligibility_filter = _gold_training_eligibility_filter(
        connection,
        gold_table=gold_table,
        split_bucket=split_bucket,
    )
    input_rows = _gold_count_where(
        connection,
        gold_table=gold_table,
        where_clause=scoped_where,
    )
    kept_rows = _gold_count_where(
        connection,
        gold_table=gold_table,
        where_clause=f"{scoped_where} and {eligibility_filter}",
    )
    report: dict[str, object] = {
        "label": label,
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "dropped_rows": int(input_rows - kept_rows),
    }
    if "censor_flag" in available_columns:
        report["censored_rows"] = _gold_count_where(
            connection,
            gold_table=gold_table,
            where_clause=f"{scoped_where} and coalesce(censor_flag, false)",
        )
    if "label_quality_score" in available_columns:
        report["low_quality_rows"] = _gold_count_where(
            connection,
            gold_table=gold_table,
            where_clause=(
                f"{scoped_where} and "
                f"coalesce(label_quality_score < {DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE:.6f}, true)"
            ),
        )
    if "usable_for_training_flag" in available_columns:
        report["not_usable_rows"] = _gold_count_where(
            connection,
            gold_table=gold_table,
            where_clause=(
                f"{scoped_where} and not coalesce(usable_for_training_flag, false)"
            ),
        )
    if "target_source" in available_columns:
        report["closed_or_dense_zero_rows"] = _gold_count_where(
            connection,
            gold_table=gold_table,
            where_clause=(
                f"{scoped_where} and target_source in "
                "('closed_or_missing_observation', 'dense_calendar_zero_fill')"
            ),
        )
    return report


def _gold_training_exclusion_report(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    excluded_dataset_sources: tuple[str, ...],
    available_columns: set[str],
) -> dict[str, object]:
    return {
        "min_label_quality_score": DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE,
        "holdout_exclusions": [
            _gold_holdout_exclusion_report(
                connection,
                gold_table=gold_table,
                split_bucket="train",
                label="train",
                excluded_dataset_sources=excluded_dataset_sources,
            ),
            _gold_holdout_exclusion_report(
                connection,
                gold_table=gold_table,
                split_bucket="val",
                label="tuning",
                excluded_dataset_sources=excluded_dataset_sources,
            ),
        ],
        "splits": [
            _gold_training_split_exclusion_report(
                connection,
                gold_table=gold_table,
                split_bucket="train",
                label="train",
                excluded_dataset_sources=excluded_dataset_sources,
                available_columns=available_columns,
            ),
            _gold_training_split_exclusion_report(
                connection,
                gold_table=gold_table,
                split_bucket="val",
                label="tuning",
                excluded_dataset_sources=excluded_dataset_sources,
                available_columns=available_columns,
            ),
        ],
    }


def _bundle_projection_columns(
    feature_cols: list[str],
    *,
    group_id_cols: list[str],
    learning_target_col: str,
    absolute_target_col: str,
    reconstruction_anchor_col: str | None,
    available_columns: set[str],
) -> list[str]:
    ordered: list[str] = [DEFAULT_DATE_COL]
    for column in [
        DEFAULT_DATASET_SOURCE_COL,
        *group_id_cols,
        absolute_target_col,
        learning_target_col,
        reconstruction_anchor_col,
        *BASELINE_SUPPORT_COLUMNS,
        *SAMPLING_SUPPORT_COLUMNS,
        *feature_cols,
    ]:
        if column is None or column not in available_columns or column in ordered:
            continue
        ordered.append(column)
    return ordered


def _bool_like_duckdb_type(type_name: str) -> bool:
    return type_name.upper() in {"BOOLEAN", "BOOL"}


def _integer_like_duckdb_type(type_name: str) -> bool:
    normalized = type_name.upper()
    return any(
        token in normalized
        for token in (
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
        )
    )


def _float32_projection_columns(
    *,
    feature_cols: list[str],
    learning_target_col: str,
    absolute_target_col: str,
    reconstruction_anchor_col: str | None,
) -> set[str]:
    return {
        column
        for column in [
            learning_target_col,
            absolute_target_col,
            reconstruction_anchor_col,
            *BASELINE_SUPPORT_COLUMNS,
            *feature_cols,
        ]
        if column is not None
        and (
            column
            in {
                learning_target_col,
                absolute_target_col,
                reconstruction_anchor_col,
                *BASELINE_SUPPORT_COLUMNS,
            }
            or TFT_EXPLICIT_ROLE_BY_COLUMN.get(column, "").endswith("_real")
        )
    }


def _categorical_projection_columns(
    *,
    feature_cols: list[str],
    group_id_columns: list[str],
) -> set[str]:
    return set(group_id_columns).union(
        {
            column
            for column in feature_cols
            if not TFT_EXPLICIT_ROLE_BY_COLUMN.get(column, "").endswith("_real")
        }
    )


def _gold_projection_expression(
    column: str,
    *,
    table_alias: str,
    column_types: Mapping[str, str],
    float32_columns: set[str],
    categorical_columns: set[str],
) -> str:
    source = f"{table_alias}.{_quote_identifier(column)}"
    output = _quote_identifier(column)
    column_type = column_types.get(column, "")
    if column == DEFAULT_DATE_COL:
        return f"cast({source} as date) as {output}"
    if column in float32_columns:
        return f"try_cast({source} as real) as {output}"
    if _bool_like_duckdb_type(column_type):
        return f"cast(try_cast({source} as boolean) as tinyint) as {output}"
    if column.endswith("_status") and _integer_like_duckdb_type(column_type):
        return f"try_cast({source} as tinyint) as {output}"
    if column in categorical_columns:
        return f"cast({source} as varchar) as {output}"
    return f"{source} as {output}"


def _gold_projection_select_sql(
    *,
    projection_columns: list[str],
    table_alias: str,
    column_types: Mapping[str, str],
    feature_cols: list[str],
    group_id_columns: list[str],
    target_contract: TargetContract,
) -> str:
    float32_columns = _float32_projection_columns(
        feature_cols=feature_cols,
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=target_contract.absolute_target_col,
        reconstruction_anchor_col=target_contract.reconstruction_anchor_col,
    )
    categorical_columns = _categorical_projection_columns(
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
    )
    return ",\n                ".join(
        _gold_projection_expression(
            column,
            table_alias=table_alias,
            column_types=column_types,
            float32_columns=float32_columns,
            categorical_columns=categorical_columns,
        )
        for column in projection_columns
    )


def _sql_group_identifier_expr(
    *,
    table_alias: str,
    group_id_columns: list[str],
) -> str:
    if not group_id_columns:
        return "'global_series'"
    parts = [
        f"coalesce(cast({table_alias}.{_quote_identifier(column)} as varchar), '<NA>')"
        for column in group_id_columns
    ]
    return " || '__' || ".join(parts)


def _optimize_projection_frame(
    projected: pd.DataFrame,
    *,
    feature_cols: list[str],
    group_id_columns: list[str],
    learning_target_col: str,
    absolute_target_col: str,
    reconstruction_anchor_col: str | None,
) -> pd.DataFrame:
    float32_columns = {
        column
        for column in [
            learning_target_col,
            absolute_target_col,
            reconstruction_anchor_col,
            *BASELINE_SUPPORT_COLUMNS,
            *feature_cols,
        ]
        if column is not None
        and (
            column
            in {
                learning_target_col,
                absolute_target_col,
                reconstruction_anchor_col,
                *BASELINE_SUPPORT_COLUMNS,
            }
            or TFT_EXPLICIT_ROLE_BY_COLUMN.get(column, "").endswith("_real")
        )
    }
    categorical_columns = set(group_id_columns).union(
        {
            column
            for column in feature_cols
            if not TFT_EXPLICIT_ROLE_BY_COLUMN.get(column, "").endswith("_real")
        }
    )
    optimized = projected.copy()
    for column in optimized.columns:
        if column == DEFAULT_DATE_COL:
            continue
        if column in float32_columns:
            optimized[column] = pd.to_numeric(
                optimized[column], errors="coerce"
            ).astype("float32")
            continue
        if pd.api.types.is_bool_dtype(optimized[column]):
            optimized[column] = optimized[column].astype("int8")
            continue
        if column.endswith("_status") and pd.api.types.is_numeric_dtype(
            optimized[column]
        ):
            optimized[column] = pd.to_numeric(
                optimized[column], errors="coerce"
            ).astype("Int8")
            continue
        if column in categorical_columns:
            optimized[column] = optimized[column].astype("string")
    return optimized


def _projected_frame(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    feature_cols: list[str],
    group_id_columns: list[str],
    learning_target_col: str,
    absolute_target_col: str,
    reconstruction_anchor_col: str | None,
) -> pd.DataFrame:
    projected = (
        frame.loc[:, columns]
        .copy()
        .sort_values(DEFAULT_DATE_COL)
        .reset_index(drop=True)
    )
    return _optimize_projection_frame(
        projected,
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        learning_target_col=learning_target_col,
        absolute_target_col=absolute_target_col,
        reconstruction_anchor_col=reconstruction_anchor_col,
    )


def _build_optimisation_projection(
    frame: pd.DataFrame,
    *,
    group_id_columns: list[str],
    train_group_sizes: dict[str, int] | None,
) -> pd.DataFrame:
    projected = frame.copy()
    projected[DEFAULT_DATE_COL] = pd.to_datetime(projected[DEFAULT_DATE_COL])
    projected[GROUP_COL] = build_group_identifier(projected, group_id_columns)
    projected = projected.sort_values([GROUP_COL, DEFAULT_DATE_COL]).reset_index(
        drop=True
    )
    time_idx = projected.groupby(GROUP_COL, sort=False).cumcount().astype(np.int32)
    if train_group_sizes is not None:
        offsets = projected[GROUP_COL].map(train_group_sizes).fillna(0).astype(np.int32)
        time_idx = (time_idx + offsets).astype(np.int32)
    projected[TIME_IDX_COL] = time_idx
    return projected


def _optimisation_manifest_payload(
    *,
    output_paths: dict[str, Path],
    train_group_sizes: dict[str, int],
    projection_columns: list[str],
    group_id_columns: list[str],
    include_valid: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "bundle_version": 1,
        "train_path": str(output_paths["optimisation_train"]),
        "tuning_path": str(output_paths["optimisation_tuning"]),
        "feature_manifest_path": str(output_paths["feature_manifest"]),
        "target_contract_path": str(output_paths["target_contract"]),
        "projection_columns": [*projection_columns, *OPTIMISATION_SUPPORT_COLUMNS],
        "group_id_columns": group_id_columns,
        "support_columns": list(OPTIMISATION_SUPPORT_COLUMNS),
        "train_group_count": len(train_group_sizes),
        "precomputed_tft_support": True,
    }
    if include_valid:
        payload["valid_path"] = str(output_paths["optimisation_valid"])
    return payload


def _feature_manifest_payload(
    feature_cols: list[str],
    *,
    group_id_columns: list[str],
    projection_columns: list[str],
    projection_dtypes: dict[str, str],
) -> dict[str, object]:
    feature_contract = build_feature_contract(feature_cols)
    return {
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "group_id_columns": group_id_columns,
        "projection_columns": projection_columns,
        "projection_dtypes": projection_dtypes,
        "feature_roles": {
            column: TFT_EXPLICIT_ROLE_BY_COLUMN[column] for column in feature_cols
        },
        "feature_contract": feature_contract,
    }


def _load_bundle_frames(
    *,
    train_input_path: str | Path | None,
    tuning_input_path: str | Path | None,
    valid_input_path: str | Path | None,
) -> tuple[Path, Path, Path | None, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if train_input_path is None or tuning_input_path is None:
        raise ValueError(
            "train_input_path and tuning_input_path must either both be provided or both be omitted."
        )
    return _load_explicit_frames(
        train_input_path=train_input_path,
        tuning_input_path=tuning_input_path,
        valid_input_path=valid_input_path,
    )


def _resolve_bundle_contract(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, TargetContract]:
    target_contract = resolve_target_contract(
        train_frame, tuning_frame, requested_target_col=target_col
    )
    train_frame = ensure_learning_target_column(train_frame, target_contract)
    tuning_frame = ensure_learning_target_column(tuning_frame, target_contract)
    if valid_frame is not None:
        valid_frame = ensure_learning_target_column(valid_frame, target_contract)
    return train_frame, tuning_frame, valid_frame, target_contract


def _resolve_bundle_features(
    *,
    train_frame: pd.DataFrame,
    target_contract: TargetContract,
) -> tuple[list[str], list[str], list[str]]:
    feature_cols = select_tft_feature_columns(
        train_frame,
        excluded_cols={
            DEFAULT_DATE_COL,
            target_contract.learning_target_col,
            target_contract.absolute_target_col,
            *BASELINE_SUPPORT_COLUMNS,
            *DEFAULT_IDENTIFIER_FEATURE_COLS,
            *DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
            *DEFAULT_REMOVED_MODEL_INPUT_COLS,
        },
    )
    group_id_columns = select_explicit_tft_group_id_columns(train_frame)
    projection_columns = _bundle_projection_columns(
        feature_cols,
        group_id_cols=group_id_columns,
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=target_contract.absolute_target_col,
        reconstruction_anchor_col=target_contract.reconstruction_anchor_col,
        available_columns=set(train_frame.columns),
    )
    return feature_cols, group_id_columns, projection_columns


def _resolve_bundle_projection_from_columns(
    *,
    columns: Sequence[str],
    target_col: str,
) -> tuple[TargetContract, list[str], list[str], list[str]]:
    schema_columns = [
        column for column in columns if column not in DEFAULT_REMOVED_MODEL_INPUT_COLS
    ]
    schema_frame = _empty_schema_frame(schema_columns)
    target_contract = resolve_target_contract(
        schema_frame,
        schema_frame,
        requested_target_col=target_col,
    )
    feature_cols, group_id_columns, projection_columns = _resolve_bundle_features(
        train_frame=schema_frame,
        target_contract=target_contract,
    )
    return target_contract, feature_cols, group_id_columns, projection_columns


def _gold_split_source_sql(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    split_bucket: str,
    excluded_dataset_sources: tuple[str, ...],
    smoke_series_limit: int | None,
    smoke_dataset_source: str | None,
    smoke_min_train_rows: int,
    smoke_min_tuning_rows: int,
    smoke_min_valid_rows: int,
) -> str:
    training_eligibility_filter = _gold_training_eligibility_filter(
        connection,
        gold_table=gold_table,
        split_bucket=split_bucket,
        table_alias="gold",
    )
    if smoke_series_limit is not None:
        train_eligibility_filter = _gold_training_eligibility_filter(
            connection,
            gold_table=gold_table,
            split_bucket="train",
        )
        tuning_eligibility_filter = _gold_training_eligibility_filter(
            connection,
            gold_table=gold_table,
            split_bucket="val",
        )
        selected_series_query = _smoke_series_selection_query(
            gold_table=gold_table,
            series_limit=smoke_series_limit,
            dataset_source=smoke_dataset_source,
            min_train_rows=smoke_min_train_rows,
            min_tuning_rows=smoke_min_tuning_rows,
            min_valid_rows=smoke_min_valid_rows,
            excluded_dataset_sources=excluded_dataset_sources,
            train_eligibility_filter=train_eligibility_filter,
            tuning_eligibility_filter=tuning_eligibility_filter,
        )
        return f"""
        with selected_series as (
            {selected_series_query}
        )
        select gold.*
        from {gold_table} as gold
        inner join selected_series
            on gold.dataset_source = selected_series.dataset_source
           and gold.{CANONICAL_GROUP_ID_COL} = selected_series.{CANONICAL_GROUP_ID_COL}
        where gold.split_bucket = {_sql_literal(split_bucket)}
          and {training_eligibility_filter}
        """
    dataset_scope_filter = dataset_source_not_in_filter(
        f"gold.{DEFAULT_DATASET_SOURCE_COL}",
        excluded_dataset_sources,
    )
    return f"""
    select gold.*
    from {gold_table} as gold
    where gold.split_bucket = {_sql_literal(split_bucket)}
      and {dataset_scope_filter}
      and {training_eligibility_filter}
    """


def _count_query_rows(
    connection: duckdb.DuckDBPyConnection,
    query: str,
) -> int:
    row = connection.execute(
        f"select count(*) from ({query}) as counted_rows"
    ).fetchone()
    if row is None:
        return 0
    return int(cast(int, row[0]))


def _copy_query_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    *,
    query: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        copy (
            {query}
        ) to ? (
            format parquet,
            compression zstd,
            row_group_size 122880
        )
        """,
        [str(output_path)],
    )


def _regular_gold_projection_query(
    *,
    source_sql: str,
    projection_columns: list[str],
    column_types: Mapping[str, str],
    feature_cols: list[str],
    group_id_columns: list[str],
    target_contract: TargetContract,
) -> str:
    projection_sql = _gold_projection_select_sql(
        projection_columns=projection_columns,
        table_alias="source",
        column_types=column_types,
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        target_contract=target_contract,
    )
    return f"""
    select
        {projection_sql}
    from ({source_sql}) as source
    """


def _train_group_sizes_sql(
    *,
    train_source_sql: str,
    group_id_columns: list[str],
) -> str:
    group_expr = _sql_group_identifier_expr(
        table_alias="train_source",
        group_id_columns=group_id_columns,
    )
    return f"""
    select
        {group_expr} as {_quote_identifier(GROUP_COL)},
        count(*) as train_rows
    from ({train_source_sql}) as train_source
    group by 1
    """


def _optimisation_gold_projection_query(
    *,
    source_sql: str,
    train_source_sql: str,
    projection_columns: list[str],
    column_types: Mapping[str, str],
    feature_cols: list[str],
    group_id_columns: list[str],
    target_contract: TargetContract,
    offset_with_train_sizes: bool,
) -> str:
    projection_sql = _gold_projection_select_sql(
        projection_columns=projection_columns,
        table_alias="source",
        column_types=column_types,
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        target_contract=target_contract,
    )
    group_expr = _sql_group_identifier_expr(
        table_alias="source",
        group_id_columns=group_id_columns,
    )
    local_time_idx = (
        f"row_number() over (partition by {group_expr} "
        f"order by source.{_quote_identifier(DEFAULT_DATE_COL)}) - 1"
    )
    if not offset_with_train_sizes:
        return f"""
        select
            {projection_sql},
            {group_expr} as {_quote_identifier(GROUP_COL)},
            cast({local_time_idx} as integer) as {_quote_identifier(TIME_IDX_COL)}
        from ({source_sql}) as source
        """
    base_columns_sql = ",\n            ".join(
        f"split_projected.{_quote_identifier(column)}"
        for column in [*projection_columns, GROUP_COL]
    )
    train_group_sizes_sql = _train_group_sizes_sql(
        train_source_sql=train_source_sql,
        group_id_columns=group_id_columns,
    )
    return f"""
    with train_group_sizes as (
        {train_group_sizes_sql}
    ),
    split_projected as (
        select
            {projection_sql},
            {group_expr} as {_quote_identifier(GROUP_COL)},
            cast({local_time_idx} as integer) as "__tft_local_time_idx"
        from ({source_sql}) as source
    )
    select
        {base_columns_sql},
        cast(
            split_projected."__tft_local_time_idx" + coalesce(train_group_sizes.train_rows, 0)
            as integer
        ) as {_quote_identifier(TIME_IDX_COL)}
    from split_projected
    left join train_group_sizes using ({_quote_identifier(GROUP_COL)})
    """


def _fetch_train_group_sizes(
    connection: duckdb.DuckDBPyConnection,
    *,
    train_source_sql: str,
    group_id_columns: list[str],
) -> dict[str, int]:
    rows = connection.execute(
        _train_group_sizes_sql(
            train_source_sql=train_source_sql,
            group_id_columns=group_id_columns,
        )
    ).fetchall()
    return {str(row[0]): int(cast(int, row[1])) for row in rows}


def _parquet_projection_dtypes(
    connection: duckdb.DuckDBPyConnection,
    *,
    path: Path,
) -> dict[str, str]:
    preview = connection.execute(
        "select * from read_parquet(?) limit 0",
        [str(path)],
    ).fetchdf()
    return {str(column): str(dtype) for column, dtype in preview.dtypes.items()}


def _write_bundle_frames(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    output_paths: dict[str, Path],
    feature_cols: list[str],
    group_id_columns: list[str],
    projection_columns: list[str],
    target_contract: TargetContract,
) -> tuple[int, int, int, dict[str, str], dict[str, int]]:
    train_projected = _projected_frame(
        train_frame,
        columns=projection_columns,
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=target_contract.absolute_target_col,
        reconstruction_anchor_col=target_contract.reconstruction_anchor_col,
    )
    tuning_projected = _projected_frame(
        tuning_frame,
        columns=projection_columns,
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=target_contract.absolute_target_col,
        reconstruction_anchor_col=target_contract.reconstruction_anchor_col,
    )
    train_projected.to_parquet(output_paths["train"], index=False)
    tuning_projected.to_parquet(output_paths["tuning"], index=False)
    train_rows = int(len(train_projected))
    tuning_rows = int(len(tuning_projected))
    valid_rows = 0
    valid_projected: pd.DataFrame | None = None
    if valid_frame is not None:
        valid_projected = _projected_frame(
            valid_frame,
            columns=projection_columns,
            feature_cols=feature_cols,
            group_id_columns=group_id_columns,
            learning_target_col=target_contract.learning_target_col,
            absolute_target_col=target_contract.absolute_target_col,
            reconstruction_anchor_col=target_contract.reconstruction_anchor_col,
        )
        valid_projected.to_parquet(output_paths["valid"], index=False)
        valid_rows = int(len(valid_projected))
    train_group_sizes = {
        str(group_id): int(size)
        for group_id, size in _build_optimisation_projection(
            train_projected,
            group_id_columns=group_id_columns,
            train_group_sizes=None,
        )
        .groupby(GROUP_COL, sort=False)
        .size()
        .items()
    }
    optimisation_train = _build_optimisation_projection(
        train_projected,
        group_id_columns=group_id_columns,
        train_group_sizes=None,
    )
    optimisation_tuning = _build_optimisation_projection(
        tuning_projected,
        group_id_columns=group_id_columns,
        train_group_sizes=train_group_sizes,
    )
    optimisation_train.to_parquet(output_paths["optimisation_train"], index=False)
    optimisation_tuning.to_parquet(output_paths["optimisation_tuning"], index=False)
    if valid_projected is not None:
        optimisation_valid = _build_optimisation_projection(
            valid_projected,
            group_id_columns=group_id_columns,
            train_group_sizes=train_group_sizes,
        )
        optimisation_valid.to_parquet(output_paths["optimisation_valid"], index=False)
    raw_projection_dtypes = (
        pd.read_parquet(output_paths["train"]).dtypes.astype(str).to_dict()
    )
    projection_dtypes = {
        str(column): str(dtype) for column, dtype in raw_projection_dtypes.items()
    }
    return train_rows, tuning_rows, valid_rows, projection_dtypes, train_group_sizes


def _bundle_manifest_payload(
    *,
    train_path: str | Path,
    tuning_path: str | Path,
    valid_path: str | Path | None,
    output_paths: dict[str, Path],
    train_rows: int,
    tuning_rows: int,
    valid_rows: int,
    feature_count: int,
    excluded_optimisation_dataset_sources: tuple[str, ...],
) -> dict[str, object]:
    bundle_manifest: dict[str, object] = {
        "bundle_version": 2,
        "train_input_path": str(train_path),
        "tuning_input_path": str(tuning_path),
        "valid_input_path": str(valid_path) if valid_path is not None else None,
        "optimisation_train_path": str(output_paths["optimisation_train"]),
        "optimisation_tuning_path": str(output_paths["optimisation_tuning"]),
        "optimisation_manifest_path": str(output_paths["optimisation_manifest"]),
        "train_rows": train_rows,
        "tuning_rows": tuning_rows,
        "valid_rows": valid_rows,
        "feature_count": feature_count,
        "excluded_optimisation_dataset_sources": list(
            excluded_optimisation_dataset_sources
        ),
        "feature_manifest_path": str(output_paths["feature_manifest"]),
        "feature_roles_path": str(output_paths["feature_roles"]),
        "split_manifest_path": str(output_paths["split_manifest"]),
        "target_contract_path": str(output_paths["target_contract"]),
        "training_exclusion_report_path": str(
            output_paths["training_exclusion_report"]
        ),
        "train_sha256": sha256_file(output_paths["train"]),
        "tuning_sha256": sha256_file(output_paths["tuning"]),
    }
    if valid_path is not None:
        bundle_manifest["valid_sha256"] = sha256_file(output_paths["valid"])
        bundle_manifest["optimisation_valid_path"] = str(
            output_paths["optimisation_valid"]
        )
    return bundle_manifest


def _split_manifest_payload(
    *,
    train_rows: int,
    tuning_rows: int,
    valid_rows: int,
    excluded_optimisation_dataset_sources: tuple[str, ...],
) -> dict[str, object]:
    return {
        "train": {
            "rows": train_rows,
            "excluded_dataset_sources": list(excluded_optimisation_dataset_sources),
        },
        "tuning": {
            "rows": tuning_rows,
            "excluded_dataset_sources": list(excluded_optimisation_dataset_sources),
        },
        "valid": {"rows": valid_rows},
    }


def _persist_bundle_metadata(
    *,
    output_paths: dict[str, Path],
    projection: BundleProjectionMetadata,
    splits: BundleSplitMetadata,
    training_exclusion_report: dict[str, object],
) -> None:
    feature_contract = build_feature_contract(projection.feature_cols)
    _json_dump(
        output_paths["feature_manifest"],
        _feature_manifest_payload(
            projection.feature_cols,
            group_id_columns=projection.group_id_columns,
            projection_columns=projection.projection_columns,
            projection_dtypes=projection.projection_dtypes,
        ),
    )
    _json_dump(output_paths["feature_roles"], feature_contract)
    _json_dump(
        output_paths["split_manifest"],
        _split_manifest_payload(
            train_rows=splits.train_rows,
            tuning_rows=splits.tuning_rows,
            valid_rows=splits.valid_rows,
            excluded_optimisation_dataset_sources=splits.excluded_optimisation_dataset_sources,
        ),
    )
    _json_dump(
        output_paths["target_contract"],
        build_target_contract_metadata(projection.target_contract),
    )
    _json_dump(output_paths["training_exclusion_report"], training_exclusion_report)
    _json_dump(
        output_paths["bundle_manifest"],
        _bundle_manifest_payload(
            train_path=splits.train_path,
            tuning_path=splits.tuning_path,
            valid_path=splits.valid_path,
            output_paths=output_paths,
            train_rows=splits.train_rows,
            tuning_rows=splits.tuning_rows,
            valid_rows=splits.valid_rows,
            feature_count=len(projection.feature_cols),
            excluded_optimisation_dataset_sources=splits.excluded_optimisation_dataset_sources,
        ),
    )
    _json_dump(
        output_paths["optimisation_manifest"],
        _optimisation_manifest_payload(
            output_paths=output_paths,
            train_group_sizes=projection.train_group_sizes,
            projection_columns=projection.projection_columns,
            group_id_columns=projection.group_id_columns,
            include_valid=splits.valid_path is not None,
        ),
    )


def _gold_input_descriptor(
    *,
    duckdb_path: str | Path,
    gold_table: str,
    split_bucket: str,
) -> str:
    return f"duckdb://{Path(duckdb_path).as_posix()}#{gold_table}[split={split_bucket}]"


def _copy_regular_and_optimisation_gold_splits(
    connection: duckdb.DuckDBPyConnection,
    *,
    output_paths: dict[str, Path],
    train_source_sql: str,
    tuning_source_sql: str,
    valid_source_sql: str | None,
    projection_columns: list[str],
    column_types: Mapping[str, str],
    feature_cols: list[str],
    group_id_columns: list[str],
    target_contract: TargetContract,
) -> None:
    split_specs: tuple[tuple[str, str, str, bool], ...] = (
        ("train", "optimisation_train", train_source_sql, False),
        ("tuning", "optimisation_tuning", tuning_source_sql, True),
    )
    if valid_source_sql is not None:
        split_specs = (
            *split_specs,
            ("valid", "optimisation_valid", valid_source_sql, True),
        )
    for regular_key, optimisation_key, source_sql, offset in split_specs:
        regular_query = _regular_gold_projection_query(
            source_sql=source_sql,
            projection_columns=projection_columns,
            column_types=column_types,
            feature_cols=feature_cols,
            group_id_columns=group_id_columns,
            target_contract=target_contract,
        )
        _copy_query_to_parquet(
            connection,
            query=regular_query,
            output_path=output_paths[regular_key],
        )
        optimisation_query = _optimisation_gold_projection_query(
            source_sql=source_sql,
            train_source_sql=train_source_sql,
            projection_columns=projection_columns,
            column_types=column_types,
            feature_cols=feature_cols,
            group_id_columns=group_id_columns,
            target_contract=target_contract,
            offset_with_train_sizes=offset,
        )
        _copy_query_to_parquet(
            connection,
            query=optimisation_query,
            output_path=output_paths[optimisation_key],
        )


def _build_training_bundle_from_gold(
    *,
    output_dir: Path,
    duckdb_path: str | Path,
    gold_table: str,
    smoke_series_limit: int | None,
    smoke_dataset_source: str | None,
    smoke_min_train_rows: int,
    smoke_min_tuning_rows: int,
    smoke_min_valid_rows: int,
    target_col: str,
    excluded_optimisation_dataset_sources: tuple[str, ...],
) -> dict[str, Path]:
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        columns = _gold_schema_columns(connection, gold_table=gold_table)
        column_types = _gold_schema_types(connection, gold_table=gold_table)
        target_contract, feature_cols, group_id_columns, projection_columns = (
            _resolve_bundle_projection_from_columns(
                columns=columns,
                target_col=target_col,
            )
        )
        train_source_sql = _gold_split_source_sql(
            connection,
            gold_table=gold_table,
            split_bucket="train",
            excluded_dataset_sources=excluded_optimisation_dataset_sources,
            smoke_series_limit=smoke_series_limit,
            smoke_dataset_source=smoke_dataset_source,
            smoke_min_train_rows=smoke_min_train_rows,
            smoke_min_tuning_rows=smoke_min_tuning_rows,
            smoke_min_valid_rows=smoke_min_valid_rows,
        )
        tuning_source_sql = _gold_split_source_sql(
            connection,
            gold_table=gold_table,
            split_bucket="val",
            excluded_dataset_sources=excluded_optimisation_dataset_sources,
            smoke_series_limit=smoke_series_limit,
            smoke_dataset_source=smoke_dataset_source,
            smoke_min_train_rows=smoke_min_train_rows,
            smoke_min_tuning_rows=smoke_min_tuning_rows,
            smoke_min_valid_rows=smoke_min_valid_rows,
        )
        valid_source_sql = _gold_split_source_sql(
            connection,
            gold_table=gold_table,
            split_bucket="test",
            excluded_dataset_sources=(),
            smoke_series_limit=smoke_series_limit,
            smoke_dataset_source=smoke_dataset_source,
            smoke_min_train_rows=smoke_min_train_rows,
            smoke_min_tuning_rows=smoke_min_tuning_rows,
            smoke_min_valid_rows=smoke_min_valid_rows,
        )
        train_rows = _count_query_rows(connection, train_source_sql)
        tuning_rows = _count_query_rows(connection, tuning_source_sql)
        valid_rows = _count_query_rows(connection, valid_source_sql)
        if train_rows == 0 or tuning_rows == 0:
            raise ValueError(
                "Training bundle train/tuning splits are empty after excluding "
                "holdout dataset sources and non-trainable labels."
            )
        include_valid = valid_rows > 0
        output_paths = _bundle_output_paths(output_dir, include_valid=include_valid)
        _copy_regular_and_optimisation_gold_splits(
            connection,
            output_paths=output_paths,
            train_source_sql=train_source_sql,
            tuning_source_sql=tuning_source_sql,
            valid_source_sql=valid_source_sql if include_valid else None,
            projection_columns=projection_columns,
            column_types=column_types,
            feature_cols=feature_cols,
            group_id_columns=group_id_columns,
            target_contract=target_contract,
        )
        projection_dtypes = _parquet_projection_dtypes(
            connection,
            path=output_paths["train"],
        )
        train_group_sizes = _fetch_train_group_sizes(
            connection,
            train_source_sql=train_source_sql,
            group_id_columns=group_id_columns,
        )
        training_exclusion_report = _gold_training_exclusion_report(
            connection,
            gold_table=gold_table,
            excluded_dataset_sources=excluded_optimisation_dataset_sources,
            available_columns=set(columns),
        )
    finally:
        connection.close()

    valid_input_path: str | None = None
    if include_valid:
        valid_input_path = _gold_input_descriptor(
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            split_bucket="test",
        )
    _persist_bundle_metadata(
        output_paths=output_paths,
        projection=BundleProjectionMetadata(
            feature_cols=feature_cols,
            group_id_columns=group_id_columns,
            projection_columns=projection_columns,
            projection_dtypes=projection_dtypes,
            target_contract=target_contract,
            train_group_sizes=train_group_sizes,
        ),
        splits=BundleSplitMetadata(
            train_path=_gold_input_descriptor(
                duckdb_path=duckdb_path,
                gold_table=gold_table,
                split_bucket="train",
            ),
            tuning_path=_gold_input_descriptor(
                duckdb_path=duckdb_path,
                gold_table=gold_table,
                split_bucket="val",
            ),
            valid_path=valid_input_path,
            train_rows=train_rows,
            tuning_rows=tuning_rows,
            valid_rows=valid_rows,
            excluded_optimisation_dataset_sources=excluded_optimisation_dataset_sources,
        ),
        training_exclusion_report=training_exclusion_report,
    )
    return output_paths


def build_training_bundle(
    *,
    train_input_path: str | Path | None = DEFAULT_TRAIN_INPUT_PATH,
    tuning_input_path: str | Path | None = DEFAULT_TUNING_INPUT_PATH,
    valid_input_path: str | Path | None = DEFAULT_VALID_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    duckdb_path: str | Path = DEFAULT_DUCKDB_PATH,
    gold_table: str = DEFAULT_GOLD_TABLE,
    smoke_series_limit: int | None = DEFAULT_SMOKE_SERIES_LIMIT,
    smoke_dataset_source: str | None = DEFAULT_SMOKE_DATASET_SOURCE,
    smoke_min_train_rows: int = DEFAULT_SMOKE_MIN_TRAIN_ROWS,
    smoke_min_tuning_rows: int = DEFAULT_SMOKE_MIN_TUNING_ROWS,
    smoke_min_valid_rows: int = DEFAULT_SMOKE_MIN_VALID_ROWS,
    target_col: str = DEFAULT_ABSOLUTE_TARGET_COL,
    excluded_optimisation_dataset_sources: tuple[
        str, ...
    ] = DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES,
) -> dict[str, Path]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if train_input_path is None and tuning_input_path is None:
        return _build_training_bundle_from_gold(
            output_dir=resolved_output_dir,
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            smoke_series_limit=smoke_series_limit,
            smoke_dataset_source=smoke_dataset_source,
            smoke_min_train_rows=smoke_min_train_rows,
            smoke_min_tuning_rows=smoke_min_tuning_rows,
            smoke_min_valid_rows=smoke_min_valid_rows,
            target_col=target_col,
            excluded_optimisation_dataset_sources=excluded_optimisation_dataset_sources,
        )
    train_path, tuning_path, valid_path, train_frame, tuning_frame, valid_frame = (
        _load_bundle_frames(
            train_input_path=train_input_path,
            tuning_input_path=tuning_input_path,
            valid_input_path=valid_input_path,
        )
    )
    train_frame_before_holdout = train_frame
    tuning_frame_before_holdout = tuning_frame
    train_frame = _exclude_dataset_sources(
        train_frame,
        excluded_dataset_sources=excluded_optimisation_dataset_sources,
    )
    tuning_frame = _exclude_dataset_sources(
        tuning_frame,
        excluded_dataset_sources=excluded_optimisation_dataset_sources,
    )
    train_frame, tuning_frame, training_exclusion_report = (
        _filter_bundle_training_frames(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            train_frame_before_holdout=train_frame_before_holdout,
            tuning_frame_before_holdout=tuning_frame_before_holdout,
            excluded_dataset_sources=excluded_optimisation_dataset_sources,
        )
    )
    if train_frame.empty or tuning_frame.empty:
        raise ValueError(
            "Training bundle train/tuning splits are empty after excluding holdout "
            f"dataset sources: {list(excluded_optimisation_dataset_sources)}."
        )
    train_frame, tuning_frame, valid_frame, target_contract = _resolve_bundle_contract(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        valid_frame=valid_frame,
        target_col=target_col,
    )
    feature_cols, group_id_columns, projection_columns = _resolve_bundle_features(
        train_frame=train_frame, target_contract=target_contract
    )
    output_paths = _bundle_output_paths(
        resolved_output_dir, include_valid=valid_frame is not None
    )
    train_rows, tuning_rows, valid_rows, projection_dtypes, train_group_sizes = (
        _write_bundle_frames(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            valid_frame=valid_frame,
            output_paths=output_paths,
            feature_cols=feature_cols,
            group_id_columns=group_id_columns,
            projection_columns=projection_columns,
            target_contract=target_contract,
        )
    )
    _persist_bundle_metadata(
        output_paths=output_paths,
        projection=BundleProjectionMetadata(
            feature_cols=feature_cols,
            group_id_columns=group_id_columns,
            projection_columns=projection_columns,
            projection_dtypes=projection_dtypes,
            target_contract=target_contract,
            train_group_sizes=train_group_sizes,
        ),
        splits=BundleSplitMetadata(
            train_path=train_path,
            tuning_path=tuning_path,
            valid_path=valid_path,
            train_rows=train_rows,
            tuning_rows=tuning_rows,
            valid_rows=valid_rows,
            excluded_optimisation_dataset_sources=excluded_optimisation_dataset_sources,
        ),
        training_exclusion_report=training_exclusion_report,
    )
    return output_paths

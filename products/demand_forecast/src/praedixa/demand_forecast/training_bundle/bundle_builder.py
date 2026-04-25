from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, cast

import duckdb
import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.tft.frame_utils import (
    GROUP_COL,
    TIME_IDX_COL,
    build_group_identifier,
)
from praedixa.demand_forecast.backends.tft.feature_mapping_spec import TFT_GROUP_ID_COLUMNS
from praedixa.demand_forecast.backends.tft.feature_mapping import TFT_EXPLICIT_ROLE_BY_COLUMN
from praedixa.demand_forecast.backends.tft.feature_mapping import select_explicit_tft_group_id_columns
from praedixa.demand_forecast.backends.tft.feature_contract import build_feature_contract
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
    NON_TRAINABLE_TARGET_SOURCES,
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
    train_path: Path
    tuning_path: Path
    valid_path: Path | None
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


def _materialized_gold_split_path(cache_dir: Path, split_bucket: str) -> Path:
    return cache_dir / f"{split_bucket}.parquet"


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _gold_projection_sql(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    table_alias: str | None = None,
) -> str:
    schema_preview = connection.execute(f"select * from {gold_table} limit 0").fetchdf()
    if schema_preview.empty and not list(schema_preview.columns):
        raise ValueError(f"Gold table `{gold_table}` exposes no columns.")
    prefix = "" if table_alias is None else f"{table_alias}."
    return ", ".join(f"{prefix}{_quote_identifier(column)}" for column in schema_preview.columns)


def _gold_split_row_count(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    split_bucket: str,
    excluded_dataset_sources: tuple[str, ...],
) -> int:
    dataset_scope_filter = dataset_source_not_in_filter(
        DEFAULT_DATASET_SOURCE_COL,
        excluded_dataset_sources,
    )
    query = (
        f"select count(*) from {gold_table} "
        f"where split_bucket = ? and {dataset_scope_filter}"
    )
    row = connection.execute(query, [split_bucket]).fetchone()
    if row is None:
        raise RuntimeError(
            f"Unable to count rows for split bucket `{split_bucket}` in `{gold_table}`."
        )
    return int(cast(int, row[0]))


def _gold_training_eligibility_filter(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    split_bucket: str,
) -> str:
    if split_bucket not in {"train", "val"}:
        return "true"
    schema_preview = connection.execute(f"select * from {gold_table} limit 0").fetchdf()
    available_columns = set(schema_preview.columns)
    required_columns = {"usable_for_training_flag", "censor_flag", "label_quality_score"}
    if not required_columns.issubset(available_columns):
        return "true"
    target_source_filter = ""
    if "target_source" in available_columns:
        target_source_filter = (
            f"and coalesce(target_source, '') not in {_sql_tuple(NON_TRAINABLE_TARGET_SOURCES)}"
        )
    return f"""
(
    coalesce(usable_for_training_flag, false)
    and not coalesce(censor_flag, false)
    and coalesce(label_quality_score, 0.0) >= {DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE:.6f}
    {target_source_filter}
)""".strip()


def _materialize_gold_split(
    *,
    cache_dir: Path,
    duckdb_path: str | Path,
    gold_table: str,
    split_bucket: str,
    excluded_dataset_sources: tuple[str, ...] = (),
) -> Path | None:
    output_path = _materialized_gold_split_path(cache_dir, split_bucket)
    dataset_scope_filter = dataset_source_not_in_filter(
        DEFAULT_DATASET_SOURCE_COL,
        excluded_dataset_sources,
    )
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        projection_sql = _gold_projection_sql(connection, gold_table=gold_table)
        training_eligibility_filter = _gold_training_eligibility_filter(
            connection,
            gold_table=gold_table,
            split_bucket=split_bucket,
        )
        if (
            _gold_split_row_count(
                connection,
                gold_table=gold_table,
                split_bucket=split_bucket,
                excluded_dataset_sources=excluded_dataset_sources,
            )
            == 0
        ):
            return None
        connection.execute(
            f"""
            copy (
                select {projection_sql}
                from {gold_table}
                where split_bucket = '{split_bucket}'
                  and {dataset_scope_filter}
                  and {training_eligibility_filter}
                order by {DEFAULT_DATE_COL}, {CANONICAL_GROUP_ID_COL}
            ) to '{output_path.as_posix()}' (
                format parquet,
                compression zstd,
                row_group_size 122880
            )
            """
        )
    finally:
        connection.close()
    return output_path


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_tuple(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(_sql_literal(value) for value in values) + ")"


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


def _materialize_smoke_gold_split(
    *,
    cache_dir: Path,
    duckdb_path: str | Path,
    gold_table: str,
    split_bucket: str,
    series_limit: int,
    dataset_source: str | None,
    min_train_rows: int,
    min_tuning_rows: int,
    min_valid_rows: int,
    excluded_dataset_sources: tuple[str, ...] = (),
) -> Path | None:
    output_path = _materialized_gold_split_path(cache_dir, split_bucket)
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        projection_sql = _gold_projection_sql(
            connection,
            gold_table=gold_table,
            table_alias="gold",
        )
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
            series_limit=series_limit,
            dataset_source=dataset_source,
            min_train_rows=min_train_rows,
            min_tuning_rows=min_tuning_rows,
            min_valid_rows=min_valid_rows,
            excluded_dataset_sources=excluded_dataset_sources,
            train_eligibility_filter=train_eligibility_filter,
            tuning_eligibility_filter=tuning_eligibility_filter,
        )
        training_eligibility_filter = _gold_training_eligibility_filter(
            connection,
            gold_table=gold_table,
            split_bucket=split_bucket,
        )
        row = connection.execute(
            f"select count(*) from ({selected_series_query}) as selected_series"
        ).fetchone()
        selected_series_count = 0 if row is None else int(cast(int, row[0]))
        if selected_series_count == 0:
            return None
        connection.execute(
            f"""
            copy (
                with selected_series as (
                    {selected_series_query}
                )
                select {projection_sql}
                from {gold_table} as gold
                inner join selected_series
                    on gold.dataset_source = selected_series.dataset_source
                   and gold.{CANONICAL_GROUP_ID_COL} = selected_series.{CANONICAL_GROUP_ID_COL}
                where gold.split_bucket = '{split_bucket}'
                  and {training_eligibility_filter}
                order by gold.{DEFAULT_DATE_COL}, gold.{CANONICAL_GROUP_ID_COL}
            ) to '{output_path.as_posix()}' (
                format parquet,
                compression zstd,
                row_group_size 122880
            )
            """
        )
    finally:
        connection.close()
    return output_path if output_path.exists() and output_path.stat().st_size > 0 else None


def _load_materialized_gold_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return _read_frame(path)


def _load_frames_from_gold(
    *,
    output_dir: Path,
    duckdb_path: str | Path,
    gold_table: str,
    smoke_series_limit: int | None,
    smoke_dataset_source: str | None,
    smoke_min_train_rows: int,
    smoke_min_tuning_rows: int,
    smoke_min_valid_rows: int,
    excluded_optimisation_dataset_sources: tuple[str, ...],
) -> tuple[Path, Path, Path | None, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    cache_dir = output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if smoke_series_limit is not None:
        train_path = _materialize_smoke_gold_split(
            cache_dir=cache_dir,
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            split_bucket="train",
            series_limit=smoke_series_limit,
            dataset_source=smoke_dataset_source,
            min_train_rows=smoke_min_train_rows,
            min_tuning_rows=smoke_min_tuning_rows,
            min_valid_rows=smoke_min_valid_rows,
            excluded_dataset_sources=excluded_optimisation_dataset_sources,
        )
        tuning_path = _materialize_smoke_gold_split(
            cache_dir=cache_dir,
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            split_bucket="val",
            series_limit=smoke_series_limit,
            dataset_source=smoke_dataset_source,
            min_train_rows=smoke_min_train_rows,
            min_tuning_rows=smoke_min_tuning_rows,
            min_valid_rows=smoke_min_valid_rows,
            excluded_dataset_sources=excluded_optimisation_dataset_sources,
        )
        valid_path = _materialize_smoke_gold_split(
            cache_dir=cache_dir,
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            split_bucket="test",
            series_limit=smoke_series_limit,
            dataset_source=smoke_dataset_source,
            min_train_rows=smoke_min_train_rows,
            min_tuning_rows=smoke_min_tuning_rows,
            min_valid_rows=smoke_min_valid_rows,
            excluded_dataset_sources=(),
        )
    else:
        train_path = _materialize_gold_split(
            cache_dir=cache_dir,
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            split_bucket="train",
            excluded_dataset_sources=excluded_optimisation_dataset_sources,
        )
        tuning_path = _materialize_gold_split(
            cache_dir=cache_dir,
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            split_bucket="val",
            excluded_dataset_sources=excluded_optimisation_dataset_sources,
        )
        valid_path = _materialize_gold_split(
            cache_dir=cache_dir,
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            split_bucket="test",
            excluded_dataset_sources=(),
        )
    if train_path is None or tuning_path is None:
        raise FileNotFoundError(
            f"Gold table `{gold_table}` must expose non-empty `train` and `val` split buckets."
        )
    train_frame = _read_frame(train_path)
    tuning_frame = _read_frame(tuning_path)
    valid_frame = _load_materialized_gold_frame(valid_path)
    return train_path, tuning_path, valid_path, train_frame, tuning_frame, valid_frame


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
        report["censored_rows"] = int(frame["censor_flag"].fillna(False).astype(bool).sum())
    if "label_quality_score" in frame.columns:
        label_quality = pd.to_numeric(frame["label_quality_score"], errors="coerce")
        report["low_quality_rows"] = int(
            label_quality.lt(DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE).fillna(True).sum()
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
            column in {
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
            optimized[column] = pd.to_numeric(optimized[column], errors="coerce").astype("float32")
            continue
        if pd.api.types.is_bool_dtype(optimized[column]):
            optimized[column] = optimized[column].astype("int8")
            continue
        if column.endswith("_status") and pd.api.types.is_numeric_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], errors="coerce").astype("Int8")
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
    projected = frame.loc[:, columns].copy().sort_values(DEFAULT_DATE_COL).reset_index(drop=True)
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
    projected = projected.sort_values([GROUP_COL, DEFAULT_DATE_COL]).reset_index(drop=True)
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
        "feature_roles": {column: TFT_EXPLICIT_ROLE_BY_COLUMN[column] for column in feature_cols},
        "feature_contract": feature_contract,
    }


def _load_bundle_frames(
    *,
    output_dir: Path,
    train_input_path: str | Path | None,
    tuning_input_path: str | Path | None,
    valid_input_path: str | Path | None,
    duckdb_path: str | Path,
    gold_table: str,
    smoke_series_limit: int | None,
    smoke_dataset_source: str | None,
    smoke_min_train_rows: int,
    smoke_min_tuning_rows: int,
    smoke_min_valid_rows: int,
    excluded_optimisation_dataset_sources: tuple[str, ...],
) -> tuple[Path, Path, Path | None, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if train_input_path is None and tuning_input_path is None:
        return _load_frames_from_gold(
            output_dir=output_dir,
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            smoke_series_limit=smoke_series_limit,
            smoke_dataset_source=smoke_dataset_source,
            smoke_min_train_rows=smoke_min_train_rows,
            smoke_min_tuning_rows=smoke_min_tuning_rows,
            smoke_min_valid_rows=smoke_min_valid_rows,
            excluded_optimisation_dataset_sources=excluded_optimisation_dataset_sources,
        )
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
    target_contract = resolve_target_contract(train_frame, tuning_frame, requested_target_col=target_col)
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
        ).groupby(GROUP_COL, sort=False).size().items()
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
    raw_projection_dtypes = pd.read_parquet(output_paths["train"]).dtypes.astype(str).to_dict()
    projection_dtypes = {str(column): str(dtype) for column, dtype in raw_projection_dtypes.items()}
    return train_rows, tuning_rows, valid_rows, projection_dtypes, train_group_sizes


def _bundle_manifest_payload(
    *,
    train_path: Path,
    tuning_path: Path,
    valid_path: Path | None,
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
        bundle_manifest["optimisation_valid_path"] = str(output_paths["optimisation_valid"])
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
    train_path, tuning_path, valid_path, train_frame, tuning_frame, valid_frame = _load_bundle_frames(
        output_dir=resolved_output_dir,
        train_input_path=train_input_path,
        tuning_input_path=tuning_input_path,
        valid_input_path=valid_input_path,
        duckdb_path=duckdb_path,
        gold_table=gold_table,
        smoke_series_limit=smoke_series_limit,
        smoke_dataset_source=smoke_dataset_source,
        smoke_min_train_rows=smoke_min_train_rows,
        smoke_min_tuning_rows=smoke_min_tuning_rows,
        smoke_min_valid_rows=smoke_min_valid_rows,
        excluded_optimisation_dataset_sources=excluded_optimisation_dataset_sources,
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
    train_frame, tuning_frame, training_exclusion_report = _filter_bundle_training_frames(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        train_frame_before_holdout=train_frame_before_holdout,
        tuning_frame_before_holdout=tuning_frame_before_holdout,
        excluded_dataset_sources=excluded_optimisation_dataset_sources,
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
    feature_cols, group_id_columns, projection_columns = _resolve_bundle_features(train_frame=train_frame, target_contract=target_contract)
    output_paths = _bundle_output_paths(resolved_output_dir, include_valid=valid_frame is not None)
    train_rows, tuning_rows, valid_rows, projection_dtypes, train_group_sizes = _write_bundle_frames(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        valid_frame=valid_frame,
        output_paths=output_paths,
        feature_cols=feature_cols,
        group_id_columns=group_id_columns,
        projection_columns=projection_columns,
        target_contract=target_contract,
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

from __future__ import annotations

from dataclasses import replace
import logging
import os
from pathlib import Path

import duckdb
import pandas as pd

from praedixa.platform.utils.memory import downcast_pandas_frame
from praedixa.demand_forecast.training.sampling.execution import (
    configure_sampling_connection,
    full_relation_metadata,
    load_full_relation_frame,
    sample_relation_frame,
    sampling_metadata_for_relation,
)
from praedixa.demand_forecast.training.sampling.loader_context import (
    parquet_relations_with_dataset_source,
    parquet_relation_sql,
    parquet_target_filter_context,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_DATE_COL,
    DEFAULT_DUCKDB_PATH,
    DEFAULT_GOLD_TABLE,
    DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES,
    DEFAULT_TRAIN_INPUT_PATH,
    DEFAULT_TRAIN_SAMPLE_FRACTION,
    DEFAULT_TUNING_INPUT_PATH,
    DEFAULT_TUNING_SAMPLE_FRACTION,
)
from praedixa.demand_forecast.training.sampling.dataset_filters import (
    dataset_source_not_in_filter,
)
from praedixa.demand_forecast.training.sampling import (
    GoldSplitSamplingSpec,
    RelationSamplingSpec,
    build_gold_split_sampling_query,
    load_gold_split_sampling_metadata,
    log_sampling_summary,
    relation_sampling_requires_top_up,
    refresh_sampling_metadata_from_sampled_frame,
    resolve_gold_projection_columns,
    resolve_sampling_store_col,
)
from praedixa.demand_forecast.training.validation.eligibility import (
    filter_training_eligible_rows,
)


def _gold_sampling_context(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    date_col: str,
    dataset_source_col: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    excluded_dataset_sources: tuple[str, ...],
) -> tuple[str, list[str], GoldSplitSamplingSpec, GoldSplitSamplingSpec]:
    schema_preview = connection.execute(f"select * from {gold_table} limit 0").fetchdf()
    sample_store_col = resolve_sampling_store_col(schema_preview)
    projection_columns = resolve_gold_projection_columns(
        schema_preview,
        date_col=date_col,
        dataset_source_col=dataset_source_col,
        sample_store_col=sample_store_col,
    )
    return (
        sample_store_col,
        projection_columns,
        GoldSplitSamplingSpec(
            gold_table=gold_table,
            split_bucket="train",
            date_col=date_col,
            dataset_source_col=dataset_source_col,
            sample_store_col=sample_store_col,
            sample_fraction=train_sample_fraction,
            excluded_dataset_sources=excluded_dataset_sources,
        ),
        GoldSplitSamplingSpec(
            gold_table=gold_table,
            split_bucket="val",
            date_col=date_col,
            dataset_source_col=dataset_source_col,
            sample_store_col=sample_store_col,
            sample_fraction=tuning_sample_fraction,
            excluded_dataset_sources=excluded_dataset_sources,
        ),
    )


def _load_gold_sampling_metadata(
    connection: duckdb.DuckDBPyConnection,
    *,
    train_spec: GoldSplitSamplingSpec,
    tuning_spec: GoldSplitSamplingSpec,
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        load_gold_split_sampling_metadata(connection, spec=train_spec),
        load_gold_split_sampling_metadata(connection, spec=tuning_spec),
    )


def _load_gold_sampled_frames(
    connection: duckdb.DuckDBPyConnection,
    *,
    logger: logging.Logger,
    train_spec: GoldSplitSamplingSpec,
    tuning_spec: GoldSplitSamplingSpec,
    projection_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Submitting sampled gold train split query.")
    train_frame = connection.execute(
        build_gold_split_sampling_query(
            gold_table=train_spec.gold_table,
            split_bucket=train_spec.split_bucket,
            date_col=train_spec.date_col,
            dataset_source_col=train_spec.dataset_source_col,
            sample_store_col=train_spec.sample_store_col,
            sample_fraction=train_spec.sample_fraction,
            selected_columns=projection_columns,
            excluded_dataset_sources=train_spec.excluded_dataset_sources,
        )
    ).fetchdf()
    logger.info("Submitting sampled gold validation split query.")
    tuning_frame = connection.execute(
        build_gold_split_sampling_query(
            gold_table=tuning_spec.gold_table,
            split_bucket=tuning_spec.split_bucket,
            date_col=tuning_spec.date_col,
            dataset_source_col=tuning_spec.dataset_source_col,
            sample_store_col=tuning_spec.sample_store_col,
            sample_fraction=tuning_spec.sample_fraction,
            selected_columns=projection_columns,
            excluded_dataset_sources=tuning_spec.excluded_dataset_sources,
        )
    ).fetchdf()
    return train_frame, tuning_frame


def _load_gold_from_connection(
    connection: duckdb.DuckDBPyConnection,
    *,
    gold_table: str,
    logger: logging.Logger,
    date_col: str,
    dataset_source_col: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    excluded_dataset_sources: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object], dict[str, object]]:
    sample_store_col, projection_columns, train_spec, tuning_spec = (
        _gold_sampling_context(
            connection,
            gold_table=gold_table,
            date_col=date_col,
            dataset_source_col=dataset_source_col,
            train_sample_fraction=train_sample_fraction,
            tuning_sample_fraction=tuning_sample_fraction,
            excluded_dataset_sources=excluded_dataset_sources,
        )
    )
    train_sampling_metadata, tuning_sampling_metadata = _load_gold_sampling_metadata(
        connection,
        train_spec=train_spec,
        tuning_spec=tuning_spec,
    )
    log_sampling_summary("Planned train", train_sampling_metadata, logger=logger)
    log_sampling_summary("Planned validation", tuning_sampling_metadata, logger=logger)
    train_frame, tuning_frame = _load_gold_sampled_frames(
        connection,
        logger=logger,
        train_spec=train_spec,
        tuning_spec=tuning_spec,
        projection_columns=projection_columns,
    )
    return (
        train_frame,
        tuning_frame,
        sample_store_col,
        train_sampling_metadata,
        tuning_sampling_metadata,
    )


def _resolved_parquet_paths(
    *,
    train_input_path: str | Path | None,
    tuning_input_path: str | Path | None,
) -> tuple[Path, Path]:
    train_path = Path(train_input_path) if train_input_path is not None else None
    tuning_path = Path(tuning_input_path) if tuning_input_path is not None else None
    if train_path is None or tuning_path is None:
        raise ValueError(
            "train_input_path and tuning_input_path must be provided for parquet loading."
        )
    return train_path, tuning_path


def _parquet_sampling_context(
    *,
    train_connection: duckdb.DuckDBPyConnection,
    tuning_connection: duckdb.DuckDBPyConnection,
    train_path: Path,
    tuning_path: Path,
    dataset_source_col: str,
    target_col: str,
    date_col: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    excluded_dataset_sources: tuple[str, ...],
) -> tuple[str, list[str], list[str], RelationSamplingSpec, RelationSamplingSpec]:
    train_base_relation_sql = parquet_relation_sql(train_path)
    tuning_base_relation_sql = parquet_relation_sql(tuning_path)
    train_schema_preview = train_connection.execute(
        f"{train_base_relation_sql} limit 0"
    ).fetchdf()
    tuning_schema_preview = tuning_connection.execute(
        f"{tuning_base_relation_sql} limit 0"
    ).fetchdf()
    train_relation_sql, train_columns, tuning_relation_sql, tuning_columns = (
        parquet_relations_with_dataset_source(
            train_base_relation_sql=train_base_relation_sql,
            tuning_base_relation_sql=tuning_base_relation_sql,
            train_columns=list(train_schema_preview.columns),
            tuning_columns=list(tuning_schema_preview.columns),
            dataset_source_col=dataset_source_col,
        )
    )
    sample_store_col, train_target_filter_col, tuning_target_filter_col = (
        parquet_target_filter_context(
            train_schema_preview=train_schema_preview,
            tuning_schema_preview=tuning_schema_preview,
            dataset_source_col=dataset_source_col,
            target_col=target_col,
        )
    )
    exclusion_filter = dataset_source_not_in_filter(
        dataset_source_col,
        excluded_dataset_sources,
    )
    return (
        sample_store_col,
        train_columns,
        tuning_columns,
        RelationSamplingSpec(
            relation_sql=(
                f"{train_relation_sql} where {train_target_filter_col} is not null "
                f"and {exclusion_filter}"
            ),
            date_col=date_col,
            dataset_source_col=dataset_source_col,
            sample_store_col=sample_store_col,
            sample_fraction=train_sample_fraction,
        ),
        RelationSamplingSpec(
            relation_sql=(
                f"{tuning_relation_sql} where {tuning_target_filter_col} is not null "
                f"and {exclusion_filter}"
            ),
            date_col=date_col,
            dataset_source_col=dataset_source_col,
            sample_store_col=sample_store_col,
            sample_fraction=tuning_sample_fraction,
        ),
    )


def _load_parquet_from_connections(
    *,
    train_connection: duckdb.DuckDBPyConnection,
    tuning_connection: duckdb.DuckDBPyConnection,
    train_path: Path,
    tuning_path: Path,
    logger: logging.Logger,
    date_col: str,
    dataset_source_col: str,
    target_col: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    excluded_dataset_sources: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object], dict[str, object]]:
    (
        sample_store_col,
        train_columns,
        tuning_columns,
        train_sampling_spec,
        tuning_sampling_spec,
    ) = _parquet_sampling_context(
        train_connection=train_connection,
        tuning_connection=tuning_connection,
        train_path=train_path,
        tuning_path=tuning_path,
        dataset_source_col=dataset_source_col,
        target_col=target_col,
        date_col=date_col,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
        excluded_dataset_sources=excluded_dataset_sources,
    )
    train_full_load = train_sampling_spec.sample_fraction >= 1.0
    tuning_full_load = tuning_sampling_spec.sample_fraction >= 1.0
    if train_full_load:
        train_sampling_spec = replace(train_sampling_spec, allow_top_up=False)
        logger.info(
            "Bypassing parquet train sampling because sample_fraction=1.0; loading full relation directly."
        )
        train_frame = load_full_relation_frame(
            connection=train_connection,
            relation_sql=train_sampling_spec.relation_sql,
            selected_columns=train_columns,
        )
        train_sampling_metadata = full_relation_metadata(
            frame=train_frame,
            spec=train_sampling_spec,
        )
    else:
        train_sampling_metadata = sampling_metadata_for_relation(
            connection=train_connection,
            spec=train_sampling_spec,
            available_columns=train_columns,
        )
        train_sampling_spec = replace(
            train_sampling_spec,
            allow_top_up=relation_sampling_requires_top_up(train_sampling_metadata),
        )
        logger.info(
            "Executing sampled parquet train query: source=%s original_rows=%s planned_sampled_rows=%s columns=%s",
            train_path,
            train_sampling_metadata["original_rows"],
            train_sampling_metadata["sampled_rows"],
            len(train_columns),
        )
        train_frame = sample_relation_frame(
            connection=train_connection,
            spec=train_sampling_spec,
            selected_columns=train_columns,
        )
    if tuning_full_load:
        tuning_sampling_spec = replace(tuning_sampling_spec, allow_top_up=False)
        logger.info(
            "Bypassing parquet validation sampling because sample_fraction=1.0; loading full relation directly."
        )
        tuning_frame = load_full_relation_frame(
            connection=tuning_connection,
            relation_sql=tuning_sampling_spec.relation_sql,
            selected_columns=tuning_columns,
        )
        tuning_sampling_metadata = full_relation_metadata(
            frame=tuning_frame,
            spec=tuning_sampling_spec,
        )
    else:
        tuning_sampling_metadata = sampling_metadata_for_relation(
            connection=tuning_connection,
            spec=tuning_sampling_spec,
            available_columns=tuning_columns,
        )
        tuning_sampling_spec = replace(
            tuning_sampling_spec,
            allow_top_up=relation_sampling_requires_top_up(tuning_sampling_metadata),
        )
        logger.info(
            "Executing sampled parquet validation query: source=%s original_rows=%s planned_sampled_rows=%s columns=%s",
            tuning_path,
            tuning_sampling_metadata["original_rows"],
            tuning_sampling_metadata["sampled_rows"],
            len(tuning_columns),
        )
        tuning_frame = sample_relation_frame(
            connection=tuning_connection,
            spec=tuning_sampling_spec,
            selected_columns=tuning_columns,
        )
    log_sampling_summary("Planned train", train_sampling_metadata, logger=logger)
    log_sampling_summary("Planned validation", tuning_sampling_metadata, logger=logger)
    logger.info(
        "Parquet loading execution plan: train_full_load=%s train_top_up=%s tuning_full_load=%s tuning_top_up=%s",
        train_full_load,
        train_sampling_spec.allow_top_up,
        tuning_full_load,
        tuning_sampling_spec.allow_top_up,
    )
    return (
        train_frame,
        tuning_frame,
        sample_store_col,
        train_sampling_metadata,
        tuning_sampling_metadata,
    )


def load_gold_train_tuning_frames(
    duckdb_path: str | Path = DEFAULT_DUCKDB_PATH,
    gold_table: str = DEFAULT_GOLD_TABLE,
    *,
    logger: logging.Logger,
    date_col: str = DEFAULT_DATE_COL,
    dataset_source_col: str = DEFAULT_DATASET_SOURCE_COL,
    train_sample_fraction: float = DEFAULT_TRAIN_SAMPLE_FRACTION,
    tuning_sample_fraction: float = DEFAULT_TUNING_SAMPLE_FRACTION,
    excluded_dataset_sources: tuple[str, ...] = DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object], dict[str, object]]:
    logger.info(
        "Loading sampled gold train/validation splits: duckdb_path=%s gold_table=%s train_split=train tuning_split=val train_sample_fraction=%.4f tuning_sample_fraction=%.4f",
        duckdb_path,
        gold_table,
        train_sample_fraction,
        tuning_sample_fraction,
    )
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        (
            train_frame,
            tuning_frame,
            sample_store_col,
            train_sampling_metadata,
            tuning_sampling_metadata,
        ) = _load_gold_from_connection(
            connection,
            gold_table=gold_table,
            logger=logger,
            date_col=date_col,
            dataset_source_col=dataset_source_col,
            train_sample_fraction=train_sample_fraction,
            tuning_sample_fraction=tuning_sample_fraction,
            excluded_dataset_sources=excluded_dataset_sources,
        )
    finally:
        connection.close()
    return _finalize_loaded_frames(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        sample_store_col=sample_store_col,
        train_sampling_metadata=train_sampling_metadata,
        tuning_sampling_metadata=tuning_sampling_metadata,
        dataset_source_col=dataset_source_col,
        date_col=date_col,
        logger=logger,
        loaded_label="gold splits",
    )


def load_parquet_train_tuning_frames(
    *,
    train_input_path: str | Path | None = DEFAULT_TRAIN_INPUT_PATH,
    tuning_input_path: str | Path | None = DEFAULT_TUNING_INPUT_PATH,
    logger: logging.Logger,
    date_col: str = DEFAULT_DATE_COL,
    dataset_source_col: str = DEFAULT_DATASET_SOURCE_COL,
    target_col: str,
    train_sample_fraction: float = DEFAULT_TRAIN_SAMPLE_FRACTION,
    tuning_sample_fraction: float = DEFAULT_TUNING_SAMPLE_FRACTION,
    excluded_dataset_sources: tuple[str, ...] = DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object], dict[str, object]]:
    train_path, tuning_path = _resolved_parquet_paths(
        train_input_path=train_input_path,
        tuning_input_path=tuning_input_path,
    )
    train_connection = duckdb.connect()
    tuning_connection = duckdb.connect()
    sampling_threads = max(1, os.cpu_count() or 1)
    try:
        for connection in (train_connection, tuning_connection):
            configure_sampling_connection(connection, threads=sampling_threads)
        logger.info(
            "Configured DuckDB parquet sampling runtime: threads_per_query=%s preserve_insertion_order=false",
            sampling_threads,
        )
        (
            train_frame,
            tuning_frame,
            sample_store_col,
            train_sampling_metadata,
            tuning_sampling_metadata,
        ) = _load_parquet_from_connections(
            train_connection=train_connection,
            tuning_connection=tuning_connection,
            train_path=train_path,
            tuning_path=tuning_path,
            logger=logger,
            date_col=date_col,
            dataset_source_col=dataset_source_col,
            target_col=target_col,
            train_sample_fraction=train_sample_fraction,
            tuning_sample_fraction=tuning_sample_fraction,
            excluded_dataset_sources=excluded_dataset_sources,
        )
    finally:
        train_connection.close()
        tuning_connection.close()
    return _finalize_loaded_frames(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        sample_store_col=sample_store_col,
        train_sampling_metadata=train_sampling_metadata,
        tuning_sampling_metadata=tuning_sampling_metadata,
        dataset_source_col=dataset_source_col,
        date_col=date_col,
        logger=logger,
        loaded_label="parquet splits",
    )


def _finalize_loaded_frames(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    sample_store_col: str,
    train_sampling_metadata: dict[str, object],
    tuning_sampling_metadata: dict[str, object],
    dataset_source_col: str,
    date_col: str,
    logger: logging.Logger,
    loaded_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object], dict[str, object]]:
    train_frame = filter_training_eligible_rows(
        train_frame,
        label=f"{loaded_label}:train",
        logger=logger,
    )
    resolved_train_frame = downcast_pandas_frame(train_frame)
    resolved_tuning_frame = downcast_pandas_frame(tuning_frame)
    if resolved_train_frame.empty or resolved_tuning_frame.empty:
        raise ValueError(
            "Gold dataset must contain non-empty train and val splits for optimisation."
        )
    refreshed_train_metadata, refreshed_tuning_metadata = (
        _refresh_loaded_sampling_metadata(
            train_sampling_metadata=train_sampling_metadata,
            tuning_sampling_metadata=tuning_sampling_metadata,
            train_frame=resolved_train_frame,
            tuning_frame=resolved_tuning_frame,
            dataset_source_col=dataset_source_col,
            sample_store_col=sample_store_col,
            date_col=date_col,
        )
    )
    logger.info(
        "Sampled %s loaded: train_rows=%s tuning_rows=%s columns=%s sample_store_col=%s",
        loaded_label,
        len(resolved_train_frame),
        len(resolved_tuning_frame),
        len(resolved_train_frame.columns),
        sample_store_col,
    )
    return (
        resolved_train_frame,
        resolved_tuning_frame,
        sample_store_col,
        refreshed_train_metadata,
        refreshed_tuning_metadata,
    )


def _refresh_loaded_sampling_metadata(
    *,
    train_sampling_metadata: dict[str, object],
    tuning_sampling_metadata: dict[str, object],
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    dataset_source_col: str,
    sample_store_col: str,
    date_col: str,
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        refresh_sampling_metadata_from_sampled_frame(
            train_sampling_metadata,
            train_frame,
            dataset_source_col=dataset_source_col,
            sample_store_col=sample_store_col,
            date_col=date_col,
        ),
        refresh_sampling_metadata_from_sampled_frame(
            tuning_sampling_metadata,
            tuning_frame,
            dataset_source_col=dataset_source_col,
            sample_store_col=sample_store_col,
            date_col=date_col,
        ),
    )

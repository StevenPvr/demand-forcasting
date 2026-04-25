from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd
import polars as pl

from praedixa.demand_forecast.evaluation.bakery_reference_dataset import (
    build_bakery_reference_splits_from_gold,
)
from praedixa.demand_forecast.evaluation.bakery_metrics import (
    REFERENCE_DATE_COL,
    REFERENCE_PRODUCT_COL,
    REFERENCE_TARGET_COL,
)
from praedixa.demand_forecast.evaluation.reference import build_bakery_reference_feature_frame
from praedixa.platform.utils.memory import downcast_pandas_frame, read_parquet_projected
from praedixa.demand_forecast.training.config.constants import DEFAULT_DATASET_SOURCE_COL, DEFAULT_DATE_COL
from praedixa.demand_forecast.training.sampling.loaders import (
    load_gold_train_tuning_frames,
    load_parquet_train_tuning_frames,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES,
)
from praedixa.demand_forecast.contracts.targets import (
    TargetContract,
    ensure_learning_target_column,
    resolve_target_contract,
)


def resolve_reference_product_col(frame: pd.DataFrame) -> str:
    for candidate in ("product_id", "product", "series_id", "sku_id", "client_id"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Unable to resolve a product identifier column for evaluation.")


def resolve_reference_date_col(frame: pd.DataFrame) -> str:
    for candidate in ("target_dt", "dt", "date"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Unable to resolve a date column for evaluation.")


def normalize_reference_split_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = (
        pl.from_pandas(frame, include_index=False)
        .with_columns(
            [
                pl.col(REFERENCE_DATE_COL).cast(pl.Datetime, strict=False),
                pl.col(REFERENCE_PRODUCT_COL).cast(pl.Utf8, strict=False),
                pl.col(REFERENCE_TARGET_COL).cast(pl.Float64, strict=False),
            ]
        )
        .sort([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL])
    )
    return downcast_pandas_frame(normalized.to_pandas())


def to_reference_frame(
    frame: pd.DataFrame,
    *,
    target_col: str,
) -> pd.DataFrame:
    date_col = resolve_reference_date_col(frame)
    product_col = resolve_reference_product_col(frame)
    reference = (
        pl.from_pandas(frame, include_index=False)
        .select(
            [
                pl.col(date_col).cast(pl.Datetime, strict=False).alias(REFERENCE_DATE_COL),
                pl.col(product_col).cast(pl.Utf8, strict=False).alias(REFERENCE_PRODUCT_COL),
                pl.col(target_col).cast(pl.Float64, strict=False).alias(REFERENCE_TARGET_COL),
            ]
        )
        .sort([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL])
    )
    return downcast_pandas_frame(reference.to_pandas())


def _load_gold_bakery_feature_frame(
    *,
    duckdb_path: str | Path,
    gold_table: str,
) -> pd.DataFrame:
    table_sql = _safe_table_identifier(gold_table)
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        return connection.execute(
            f"""
            select *
            from {table_sql}
            where dataset_source = 'bakery'
            """
        ).fetchdf()
    finally:
        connection.close()


def _safe_table_identifier(table_name: str) -> str:
    parts = table_name.split(".")
    if not parts or any(not part.replace("_", "").isalnum() for part in parts):
        raise ValueError(f"Unsafe DuckDB table identifier: {table_name!r}")
    return ".".join(f'"{part}"' for part in parts)


def _reference_full_frame(
    reference_train: pd.DataFrame,
    reference_val: pd.DataFrame,
    reference_test: pd.DataFrame,
) -> pd.DataFrame:
    return downcast_pandas_frame(
        pl.concat(
            [
                pl.from_pandas(reference_train, include_index=False),
                pl.from_pandas(reference_val, include_index=False),
                pl.from_pandas(reference_test, include_index=False),
            ],
            how="diagonal_relaxed",
            rechunk=True,
        ).to_pandas()
    )


def _dataset_sources(frame: pd.DataFrame) -> list[str]:
    if DEFAULT_DATASET_SOURCE_COL not in frame.columns:
        return []
    values = frame[DEFAULT_DATASET_SOURCE_COL].dropna().astype(str).unique().tolist()
    return sorted(str(value) for value in values)


def _uses_transfer_training(model_backend: str) -> bool:
    return model_backend.strip().lower() == "xgboost"


def _load_transfer_training_frames(
    *,
    duckdb_path: str | Path,
    gold_table: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame, valid_frame, _, _, _ = load_gold_train_tuning_frames(
        duckdb_path=duckdb_path,
        gold_table=gold_table,
        logger=logger,
        date_col=DEFAULT_DATE_COL,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
        excluded_dataset_sources=DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES,
    )
    return train_frame, valid_frame


def _materialize_bakery_reference_feature_frame(
    *,
    reference_full_df: pd.DataFrame,
    reference_split_df: pd.DataFrame,
    gold_base_df: pd.DataFrame,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = build_bakery_reference_feature_frame(reference_full_df, reference_split_df, gold_base_df)
    if feature_df.empty:
        raise ValueError("No bakery reference rows were materialized for evaluation.")

    scored_reference = normalize_reference_split_frame(reference_split_df)
    if len(feature_df) != len(scored_reference):
        raise ValueError(
            f"Materialized bakery reference {split_name} rows do not match the reference split "
            f"(materialized_rows={len(feature_df)} reference_rows={len(scored_reference)})."
        )
    actual_diff = float(
        abs(
            scored_reference[REFERENCE_TARGET_COL].astype(float).to_numpy()
            - feature_df["target_demand_qty_d_plus_1"].astype(float).to_numpy()
        ).max()
    )
    if actual_diff > 0.0:
        raise ValueError(
            f"Materialized bakery reference {split_name} target does not match bakery_sales actuals "
            f"(max_abs_diff={actual_diff})."
        )
    return feature_df, scored_reference


def load_gold_bakery_overlap_test_frame(
    *,
    duckdb_path: str | Path,
    reference_full_df: pd.DataFrame,
    reference_test_df: pd.DataFrame,
    gold_table: str,
    gold_feature_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    resolved_gold_feature_df = (
        gold_feature_df
        if gold_feature_df is not None
        else _load_gold_bakery_feature_frame(duckdb_path=duckdb_path, gold_table=gold_table)
    )
    overlap_df, scored_reference_test = _materialize_bakery_reference_feature_frame(
        reference_full_df=reference_full_df,
        reference_split_df=reference_test_df,
        gold_base_df=resolved_gold_feature_df,
        split_name="test",
    )
    overlap_start_date = pd.Timestamp(scored_reference_test[REFERENCE_DATE_COL].min())
    metadata: dict[str, object] = {
        "reference_test_rows": int(len(reference_test_df)),
        "overlap_test_rows": int(len(scored_reference_test)),
        "missing_reference_test_rows": 0,
        "overlap_start_date": overlap_start_date.strftime("%Y-%m-%d"),
        "overlap_end_date": pd.Timestamp(scored_reference_test[REFERENCE_DATE_COL].max()).strftime("%Y-%m-%d"),
        "product_count": int(scored_reference_test[REFERENCE_PRODUCT_COL].nunique()),
    }
    return overlap_df, scored_reference_test, metadata


def prepare_training_target_frame(
    frame: pd.DataFrame,
    *,
    target_contract: TargetContract,
) -> pd.DataFrame:
    prepared = ensure_learning_target_column(frame.copy(), target_contract)
    filtered = pl.from_pandas(prepared, include_index=False).filter(
        pl.col(target_contract.learning_target_col).is_not_null()
    )
    return downcast_pandas_frame(filtered.to_pandas())


def prepare_scored_test_frame(
    test_frame: pd.DataFrame,
    scored_reference_test: pd.DataFrame,
    *,
    target_contract: TargetContract,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    prepared = ensure_learning_target_column(test_frame.copy(), target_contract)
    prepared_pl = pl.from_pandas(prepared, include_index=False).with_row_index("__row_nr")
    valid_row_idx = (
        prepared_pl.filter(pl.col(target_contract.learning_target_col).is_not_null())
        .get_column("__row_nr")
        .to_numpy()
    )
    filtered_test = downcast_pandas_frame(
        prepared_pl.filter(pl.col(target_contract.learning_target_col).is_not_null()).drop("__row_nr").to_pandas()
    )
    filtered_reference = downcast_pandas_frame(
        pl.from_pandas(scored_reference_test, include_index=False)
        .with_row_index("__row_nr")
        .filter(pl.col("__row_nr").is_in(valid_row_idx))
        .drop("__row_nr")
        .to_pandas()
    )
    dropped_rows = int(len(prepared) - len(filtered_test))
    return filtered_test, filtered_reference, dropped_rows


def _read_local_test_frame(
    path: str | Path,
    *,
    evaluation_dataset_source: str | None,
) -> pd.DataFrame:
    if evaluation_dataset_source is None:
        return read_parquet_projected(Path(path))
    connection = duckdb.connect()
    try:
        return connection.execute(
            """
            select *
            from read_parquet(?)
            where dataset_source = ?
            """,
            [str(path), evaluation_dataset_source],
        ).fetchdf()
    finally:
        connection.close()


def _reference_history_frame(
    frame: pd.DataFrame,
    *,
    evaluation_dataset_source: str | None,
) -> pd.DataFrame:
    if evaluation_dataset_source is None or DEFAULT_DATASET_SOURCE_COL not in frame.columns:
        return frame
    return frame[frame[DEFAULT_DATASET_SOURCE_COL] == evaluation_dataset_source].copy()


def load_local_mode_frames(
    *,
    train_selection_input_path: str | Path,
    train_tuning_input_path: str | Path,
    val_input_path: str | Path,
    requested_target_col: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    evaluation_dataset_source: str | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object] | None]:
    train_frame, valid_frame, _, _, _ = load_parquet_train_tuning_frames(
        train_input_path=train_selection_input_path,
        tuning_input_path=train_tuning_input_path,
        logger=logger,
        date_col=DEFAULT_DATE_COL,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        target_col=requested_target_col,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
    )
    test_frame = _read_local_test_frame(
        val_input_path,
        evaluation_dataset_source=evaluation_dataset_source,
    )
    combined_history = downcast_pandas_frame(
        pl.concat(
            [
                pl.from_pandas(train_frame, include_index=False),
                pl.from_pandas(valid_frame, include_index=False),
            ],
            how="diagonal_relaxed",
            rechunk=True,
        ).to_pandas()
    )
    target_contract = resolve_target_contract(combined_history, test_frame, requested_target_col=requested_target_col)
    absolute_target_col = target_contract.absolute_target_col
    history_reference = to_reference_frame(
        _reference_history_frame(
            combined_history,
            evaluation_dataset_source=evaluation_dataset_source,
        ),
        target_col=absolute_target_col,
    )
    test_reference = to_reference_frame(test_frame, target_col=absolute_target_col)
    test_frame = test_frame.copy()
    test_frame[REFERENCE_DATE_COL] = test_reference[REFERENCE_DATE_COL].to_numpy()
    test_frame[REFERENCE_PRODUCT_COL] = test_reference[REFERENCE_PRODUCT_COL].to_numpy()
    return train_frame, valid_frame, test_frame, history_reference, test_reference, None


def load_gold_reference_mode_frames(
    *,
    duckdb_path: str | Path,
    gold_table: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    train_frame: pd.DataFrame | None = None,
    valid_frame: pd.DataFrame | None = None,
    model_backend: str = "tft",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    logger = logging.getLogger(__name__)
    transfer_training = _uses_transfer_training(model_backend)
    reference_bundle = build_bakery_reference_splits_from_gold(
        duckdb_path=duckdb_path,
        logger=logger,
    )
    reference_train = reference_bundle.train_df
    reference_val = reference_bundle.val_df
    reference_test = reference_bundle.test_df
    reference_full = _reference_full_frame(reference_train, reference_val, reference_test)
    gold_feature_df = _load_gold_bakery_feature_frame(duckdb_path=duckdb_path, gold_table=gold_table)
    logger.info(
        "Loaded bakery reference feature table from gold: table=%s rows=%s columns=%s",
        gold_table,
        len(gold_feature_df),
        len(gold_feature_df.columns),
    )
    scored_reference_train = normalize_reference_split_frame(reference_train)
    scored_reference_val = normalize_reference_split_frame(reference_val)
    if transfer_training:
        resolved_train_frame, resolved_valid_frame = (
            (train_frame, valid_frame)
            if train_frame is not None and valid_frame is not None
            else _load_transfer_training_frames(
                duckdb_path=duckdb_path,
                gold_table=gold_table,
                train_sample_fraction=train_sample_fraction,
                tuning_sample_fraction=tuning_sample_fraction,
                logger=logger,
            )
        )
    else:
        resolved_train_frame, scored_reference_train = (
            (train_frame, scored_reference_train)
            if train_frame is not None
            else _materialize_bakery_reference_feature_frame(
                reference_full_df=reference_full,
                reference_split_df=reference_train,
                gold_base_df=gold_feature_df,
                split_name="train",
            )
        )
        resolved_valid_frame, scored_reference_val = (
            (valid_frame, scored_reference_val)
            if valid_frame is not None
            else _materialize_bakery_reference_feature_frame(
                reference_full_df=reference_full,
                reference_split_df=reference_val,
                gold_base_df=gold_feature_df,
                split_name="val",
            )
        )
    overlap_test_frame, scored_reference_test, overlap_metadata = load_gold_bakery_overlap_test_frame(
        duckdb_path=duckdb_path,
        reference_full_df=reference_full,
        reference_test_df=reference_test,
        gold_table=gold_table,
        gold_feature_df=gold_feature_df,
    )
    training_protocol = (
        "freshretail_transfer_holdout_with_bakery_test_refits"
        if transfer_training
        else "bakery_supervised_reference_train_val"
    )
    logger.info(
        "Bakery reference evaluation uses ARIMA-comparable test/metrics with model_training_protocol=%s train_rows=%s val_rows=%s test_rows=%s train_sources=%s val_sources=%s reference_protocol=%s",
        training_protocol,
        int(len(resolved_train_frame)),
        int(len(resolved_valid_frame)),
        int(cast(Any, reference_bundle.metadata["reference_test_rows"])),
        _dataset_sources(resolved_train_frame),
        _dataset_sources(resolved_valid_frame),
        str(reference_bundle.metadata["reference_protocol"]),
    )
    history_reference = downcast_pandas_frame(
        pl.concat(
            [
                pl.from_pandas(scored_reference_train, include_index=False),
                pl.from_pandas(scored_reference_val, include_index=False),
            ],
            how="diagonal_relaxed",
            rechunk=True,
        )
        .sort([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL])
        .to_pandas()
    )
    overlap_metadata = {
        **reference_bundle.metadata,
        **overlap_metadata,
        "model_training_protocol": training_protocol,
        "model_training_dataset_sources": _dataset_sources(resolved_train_frame),
        "model_validation_dataset_sources": _dataset_sources(resolved_valid_frame),
        "model_training_excluded_dataset_sources": list(
            DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES
        )
        if transfer_training
        else [],
        "bakery_pretest_rows_used_for_model_training": 0
        if transfer_training
        else int(len(resolved_train_frame) + len(resolved_valid_frame)),
        "bakery_history_rows_used_for_baselines": int(
            len(scored_reference_train) + len(scored_reference_val)
        ),
        "bakery_refit_policy": "test_window_history_only",
    }
    return (
        resolved_train_frame,
        resolved_valid_frame,
        overlap_test_frame,
        history_reference,
        scored_reference_test,
        overlap_metadata,
    )

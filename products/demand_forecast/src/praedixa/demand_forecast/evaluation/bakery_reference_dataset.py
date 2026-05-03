from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd
import polars as pl

from praedixa.demand_forecast.evaluation.bakery_metrics import (
    REFERENCE_DATE_COL,
    REFERENCE_PRODUCT_COL,
    REFERENCE_TARGET_COL,
)
from praedixa.demand_forecast.evaluation.constants import (
    DEFAULT_BAKERY_REFERENCE_DATASET_SOURCE,
    DEFAULT_BAKERY_REFERENCE_MAX_ZERO_DAY_RATIO,
    DEFAULT_BAKERY_REFERENCE_PROTOCOL,
    DEFAULT_BAKERY_REFERENCE_TEST_RATIO,
    DEFAULT_BAKERY_REFERENCE_TRAIN_RATIO,
    DEFAULT_BAKERY_REFERENCE_VAL_RATIO,
)
from praedixa.platform.utils.memory import downcast_pandas_frame


@dataclass(frozen=True)
class BakeryReferenceSplitBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    metadata: dict[str, object]


def _load_bakery_reference_source_frame(
    *,
    duckdb_path: str | Path,
) -> pd.DataFrame:
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        return connection.execute(
            """
            select
                dt as date,
                product_id as product,
                current_day_demand_qty as quantity,
                is_observed_row
            from gold.gold_base_panel_d1
            where dataset_source = ?
            """
            ,
            [DEFAULT_BAKERY_REFERENCE_DATASET_SOURCE],
        ).fetchdf()
    finally:
        connection.close()


def _normalized_reference_source_frame(source_frame: pd.DataFrame) -> pl.DataFrame:
    return (
        pl.from_pandas(source_frame, include_index=False)
        .select(
            [
                pl.col(REFERENCE_DATE_COL).cast(pl.Datetime, strict=False),
                pl.col(REFERENCE_PRODUCT_COL).cast(pl.Utf8, strict=False),
                pl.col(REFERENCE_TARGET_COL).cast(pl.Float64, strict=False),
                (
                    pl.col("is_observed_row").cast(pl.Boolean, strict=False)
                    if "is_observed_row" in source_frame.columns
                    else pl.lit(True, dtype=pl.Boolean)
                ).fill_null(False).alias("__is_observed_row"),
            ]
        )
        .sort([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL])
    )


def _dense_reference_frame(source_frame: pl.DataFrame) -> pl.DataFrame:
    source_frame = source_frame.with_columns(
        pl.col(REFERENCE_DATE_COL).cast(pl.Datetime("ns"))
    )
    min_date = source_frame.get_column(REFERENCE_DATE_COL).min()
    max_date = source_frame.get_column(REFERENCE_DATE_COL).max()
    if min_date is None or max_date is None:
        raise ValueError("Bakery reference source is empty.")
    date_frame = pl.DataFrame(
        {
            REFERENCE_DATE_COL: pd.date_range(
                pd.Timestamp(cast(Any, min_date)),
                pd.Timestamp(cast(Any, max_date)),
                freq="D",
            )
        }
    ).with_columns(
        pl.col(REFERENCE_DATE_COL).cast(pl.Datetime("ns"))
    )
    product_frame = source_frame.select(pl.col(REFERENCE_PRODUCT_COL).unique().sort())
    observed_dates = (
        source_frame.filter(pl.col("__is_observed_row"))
        .select(pl.col(REFERENCE_DATE_COL).unique())
        .get_column(REFERENCE_DATE_COL)
    )
    return (
        date_frame.join(product_frame, how="cross")
        .join(
            source_frame,
            how="left",
            on=[REFERENCE_DATE_COL, REFERENCE_PRODUCT_COL],
        )
        .with_columns(
            [
                pl.col(REFERENCE_TARGET_COL).fill_null(0.0),
                (~pl.col(REFERENCE_DATE_COL).is_in(observed_dates.implode())).cast(pl.Int8).alias("is_missing_day"),
                pl.col("__is_observed_row").fill_null(False).alias("is_observed_row"),
            ]
        )
        .sort([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL])
    )


def _filtered_reference_products(reference_frame: pl.DataFrame) -> pl.DataFrame:
    kept_products = (
        reference_frame.group_by(REFERENCE_PRODUCT_COL, maintain_order=True)
        .agg(
            pl.col(REFERENCE_TARGET_COL).eq(0.0).mean().alias("__zero_day_ratio"),
        )
        .filter(pl.col("__zero_day_ratio") <= DEFAULT_BAKERY_REFERENCE_MAX_ZERO_DAY_RATIO)
        .select(REFERENCE_PRODUCT_COL)
    )
    filtered = reference_frame.join(kept_products, how="inner", on=REFERENCE_PRODUCT_COL)
    if filtered.height == 0:
        raise ValueError("No bakery products remained after applying the zero-day ratio filter.")
    return filtered


def _split_reference_frame(reference_frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    indexed = (
        reference_frame.sort([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL])
        .with_columns(
            [
                pl.int_range(0, pl.len()).over(REFERENCE_PRODUCT_COL).alias("__row_idx"),
                pl.len().over(REFERENCE_PRODUCT_COL).alias("__row_count"),
            ]
        )
        .with_columns(
            [
                (pl.col("__row_count") * DEFAULT_BAKERY_REFERENCE_TRAIN_RATIO).floor().cast(pl.Int64).alias("__train_end"),
                (pl.col("__row_count") * DEFAULT_BAKERY_REFERENCE_VAL_RATIO).floor().cast(pl.Int64).alias("__val_len"),
            ]
        )
        .with_columns(
            (pl.col("__train_end") + pl.col("__val_len")).alias("__val_end")
        )
    )
    train_df = indexed.filter(pl.col("__row_idx") < pl.col("__train_end"))
    val_df = indexed.filter((pl.col("__row_idx") >= pl.col("__train_end")) & (pl.col("__row_idx") < pl.col("__val_end")))
    test_df = indexed.filter(pl.col("__row_idx") >= pl.col("__val_end"))
    drop_cols = ["__row_idx", "__row_count", "__train_end", "__val_len", "__val_end"]
    return (train_df.drop(drop_cols), val_df.drop(drop_cols), test_df.drop(drop_cols))


def _reference_metadata(
    *,
    dense_frame: pl.DataFrame,
    filtered_frame: pl.DataFrame,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> dict[str, object]:
    return {
        "reference_protocol": DEFAULT_BAKERY_REFERENCE_PROTOCOL,
        "reference_dataset_source": DEFAULT_BAKERY_REFERENCE_DATASET_SOURCE,
        "reference_full_rows": int(filtered_frame.height),
        "reference_product_count": int(filtered_frame.get_column(REFERENCE_PRODUCT_COL).n_unique()),
        "reference_full_date_count": int(filtered_frame.get_column(REFERENCE_DATE_COL).n_unique()),
        "reference_dense_product_candidates": int(dense_frame.get_column(REFERENCE_PRODUCT_COL).n_unique()),
        "reference_train_rows": int(train_df.height),
        "reference_val_rows": int(val_df.height),
        "reference_test_rows": int(test_df.height),
        "reference_train_ratio": float(DEFAULT_BAKERY_REFERENCE_TRAIN_RATIO),
        "reference_val_ratio": float(DEFAULT_BAKERY_REFERENCE_VAL_RATIO),
        "reference_test_ratio": float(DEFAULT_BAKERY_REFERENCE_TEST_RATIO),
        "reference_max_zero_day_ratio": float(DEFAULT_BAKERY_REFERENCE_MAX_ZERO_DAY_RATIO),
        "reference_start_date": pd.Timestamp(cast(Any, filtered_frame.get_column(REFERENCE_DATE_COL).min())).strftime(
            "%Y-%m-%d"
        ),
        "reference_end_date": pd.Timestamp(cast(Any, filtered_frame.get_column(REFERENCE_DATE_COL).max())).strftime(
            "%Y-%m-%d"
        ),
    }


def build_bakery_reference_splits_from_source_frame(
    source_frame: pd.DataFrame,
) -> BakeryReferenceSplitBundle:
    normalized = _normalized_reference_source_frame(source_frame)
    dense_frame = _dense_reference_frame(normalized)
    filtered_frame = _filtered_reference_products(dense_frame)
    train_df, val_df, test_df = _split_reference_frame(filtered_frame)
    metadata = _reference_metadata(
        dense_frame=dense_frame,
        filtered_frame=filtered_frame,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
    return BakeryReferenceSplitBundle(
        train_df=downcast_pandas_frame(train_df.to_pandas()),
        val_df=downcast_pandas_frame(val_df.to_pandas()),
        test_df=downcast_pandas_frame(test_df.to_pandas()),
        metadata=metadata,
    )


def build_bakery_reference_splits_from_gold(
    *,
    duckdb_path: str | Path,
    logger: logging.Logger,
) -> BakeryReferenceSplitBundle:
    logger.info(
        "Materializing bakery reference splits from gold_base_panel_d1: protocol=%s dataset_source=%s",
        DEFAULT_BAKERY_REFERENCE_PROTOCOL,
        DEFAULT_BAKERY_REFERENCE_DATASET_SOURCE,
    )
    bundle = build_bakery_reference_splits_from_source_frame(
        _load_bakery_reference_source_frame(duckdb_path=duckdb_path)
    )
    logger.info(
        "Materialized bakery reference splits from gold: products=%s full_rows=%s train_rows=%s val_rows=%s test_rows=%s",
        int(cast(Any, bundle.metadata["reference_product_count"])),
        int(cast(Any, bundle.metadata["reference_full_rows"])),
        int(cast(Any, bundle.metadata["reference_train_rows"])),
        int(cast(Any, bundle.metadata["reference_val_rows"])),
        int(cast(Any, bundle.metadata["reference_test_rows"])),
    )
    return bundle

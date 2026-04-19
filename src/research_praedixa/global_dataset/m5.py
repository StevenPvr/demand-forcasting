from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from research_praedixa.global_dataset.schema import align_lazy_frame_to_canonical_schema, build_series_id_expr


DEFAULT_CALENDAR_PATH = Path("data/m5/calendar.csv")
DEFAULT_SELL_PRICES_PATH = Path("data/m5/sell_prices.csv")
DEFAULT_SALES_PATH = Path("data/m5/sales_train_validation.csv")
DEFAULT_OUTPUT_PATH = Path("data/global_dataset/m5_daily.parquet")
DEFAULT_SOURCE_RUN_ID = "manual_local"
DEFAULT_SILVER_RUN_ID = "manual_local"
DEFAULT_CHUNK_ROWS = 256


def load_m5_calendar_frame(path: str | Path) -> pl.DataFrame:
    """Load the M5 calendar reference file."""

    return pl.read_csv(path, try_parse_dates=True)


def load_m5_sell_prices_frame(path: str | Path) -> pl.DataFrame:
    """Load the M5 sell prices reference file."""

    return pl.read_csv(path)


def _m5_snap_flag_expr() -> pl.Expr:
    return (
        pl.when(pl.col("state_id") == "CA")
        .then(pl.col("snap_CA").cast(pl.Boolean))
        .when(pl.col("state_id") == "TX")
        .then(pl.col("snap_TX").cast(pl.Boolean))
        .when(pl.col("state_id") == "WI")
        .then(pl.col("snap_WI").cast(pl.Boolean))
        .otherwise(None)
    )


def _m5_enrichment_expressions(
    *,
    source_partition: str,
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.Expr]:
    return [
        pl.lit("m5").alias("dataset_source"),
        pl.lit(source_partition).alias("source_partition"),
        pl.lit(source_run_id).alias("source_run_id"),
        pl.lit(silver_run_id).alias("silver_run_id"),
        pl.col("date").cast(pl.Date).alias("dt"),
        pl.col("store_id").cast(pl.Utf8).alias("location_id"),
        pl.col("item_id").cast(pl.Utf8).alias("product_id"),
        pl.col("state_id").cast(pl.Utf8).alias("region_id"),
        pl.lit(None).cast(pl.Utf8).alias("org_group_id"),
        pl.col("cat_id").cast(pl.Utf8).alias("category_level_1"),
        pl.col("dept_id").cast(pl.Utf8).alias("category_level_2"),
        pl.lit(None).cast(pl.Utf8).alias("category_level_3"),
        pl.col("observed_demand_qty").cast(pl.Float32).alias("observed_demand_qty"),
        (pl.col("observed_demand_qty").cast(pl.Float32) * pl.col("sell_price").cast(pl.Float32)).alias(
            "observed_revenue_net"
        ),
        pl.lit(None).cast(pl.Float32).alias("observed_discount_amount"),
        pl.col("sell_price").cast(pl.Float32).alias("avg_selling_price"),
        pl.lit(None).cast(pl.Boolean).alias("promo_flag"),
        (pl.col("event_name_1").is_not_null() | pl.col("event_name_2").is_not_null()).alias("holiday_flag"),
        pl.lit(None).cast(pl.Boolean).alias("activity_flag"),
        pl.lit(None).cast(pl.Boolean).alias("observed_stockout_flag"),
        pl.lit(False).alias("observed_stockout_available"),
        pl.lit(None).cast(pl.Float32).alias("observed_stockout_intensity"),
        pl.lit(True).alias("location_open_flag"),
        pl.lit(True).alias("day_complete_flag"),
        pl.lit(False).alias("missing_sales_flag"),
        pl.col("weekday").cast(pl.Utf8).alias("calendar_weekday_name"),
        (pl.col("wday").cast(pl.Int8) - 1).alias("calendar_day_of_week"),
        pl.col("month").cast(pl.Int8).alias("calendar_month"),
        pl.col("year").cast(pl.Int16).alias("calendar_year"),
        pl.col("wm_yr_wk").cast(pl.Int32).alias("calendar_week_key"),
        pl.col("event_name_1").cast(pl.Utf8).alias("event_name_1"),
        pl.col("event_type_1").cast(pl.Utf8).alias("event_type_1"),
        pl.col("event_name_2").cast(pl.Utf8).alias("event_name_2"),
        pl.col("event_type_2").cast(pl.Utf8).alias("event_type_2"),
        _m5_snap_flag_expr().alias("snap_flag"),
        pl.lit(None).cast(pl.Float32).alias("weather_precipitation"),
        pl.lit(None).cast(pl.Float32).alias("weather_temperature"),
        pl.lit(None).cast(pl.Float32).alias("weather_humidity"),
        pl.lit(None).cast(pl.Float32).alias("weather_wind_level"),
        pl.lit(False).alias("anomaly_flag"),
    ]


def standardize_m5_sales_chunk(
    sales_chunk: pd.DataFrame,
    *,
    calendar_frame: pl.DataFrame,
    sell_prices_frame: pl.DataFrame,
    source_partition: str = "historical",
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.DataFrame:
    """Convert a wide M5 sales chunk into the canonical daily demand contract."""

    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    demand_columns = [column for column in sales_chunk.columns if column.startswith("d_")]
    long_chunk = sales_chunk.melt(
        id_vars=id_columns,
        value_vars=demand_columns,
        var_name="d",
        value_name="observed_demand_qty",
    )
    long_frame = pl.from_pandas(long_chunk)
    joined = (
        long_frame.lazy()
        .join(calendar_frame.lazy(), on="d", how="left")
        .join(
            sell_prices_frame.lazy(),
            left_on=["store_id", "item_id", "wm_yr_wk"],
            right_on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )
        .with_columns(
            _m5_enrichment_expressions(
                source_partition=source_partition,
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )
        .with_columns(build_series_id_expr("location_id", "product_id").alias("series_id"))
    )
    return align_lazy_frame_to_canonical_schema(joined).collect()


def _iter_m5_sales_chunks(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, chunksize=chunk_rows)


def build_m5_standardized_dataset(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    sales_input_path: str | Path = DEFAULT_SALES_PATH,
    calendar_input_path: str | Path = DEFAULT_CALENDAR_PATH,
    sell_prices_input_path: str | Path = DEFAULT_SELL_PRICES_PATH,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Path:
    """Write the canonical M5 daily dataset as a parquet file."""

    sales_path = Path(sales_input_path)
    calendar_frame = load_m5_calendar_frame(calendar_input_path)
    sell_prices_frame = load_m5_sell_prices_frame(sell_prices_input_path)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    try:
        for chunk in _iter_m5_sales_chunks(sales_path, chunk_rows=chunk_rows):
            standardized = standardize_m5_sales_chunk(
                chunk,
                calendar_frame=calendar_frame,
                sell_prices_frame=sell_prices_frame,
                source_partition="historical",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
            table = pa.Table.from_pandas(standardized.to_pandas(use_pyarrow_extension_array=True), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(target_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return target_path

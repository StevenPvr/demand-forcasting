from __future__ import annotations

from pathlib import Path

import polars as pl

from praedixa.platform.datasets.standardization.supplemental_corpus import freshretail_promo_flag_expr
from praedixa.platform.datasets.standardization.schema import align_lazy_frame_to_canonical_schema, build_series_id_expr
from praedixa.platform.runtime.paths import GLOBAL_DATASET_DIR
from praedixa.platform.runtime.paths import SOURCES_DIR


DEFAULT_TRAIN_INPUT_PATH = SOURCES_DIR / "bakery_sales" / "data_train.parquet"
DEFAULT_VAL_INPUT_PATH = SOURCES_DIR / "bakery_sales" / "data_val.parquet"
DEFAULT_OUTPUT_PATH = GLOBAL_DATASET_DIR / "freshretail_daily.parquet"
DEFAULT_SOURCE_RUN_ID = "manual_local"
DEFAULT_SILVER_RUN_ID = "manual_local"


def _freshretail_canonical_expressions(
    *,
    stockout_flag_expr: pl.Expr,
    source_partition: str,
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.Expr]:
    return [
        pl.col("dt").cast(pl.Date).alias("dt"),
        pl.col("store_id").cast(pl.Utf8).alias("location_id"),
        pl.col("product_id").cast(pl.Utf8).alias("product_id"),
        pl.col("city_id").cast(pl.Utf8).alias("region_id"),
        pl.col("management_group_id").cast(pl.Utf8).alias("org_group_id"),
        pl.col("first_category_id").cast(pl.Utf8).alias("category_level_1"),
        pl.col("second_category_id").cast(pl.Utf8).alias("category_level_2"),
        pl.col("third_category_id").cast(pl.Utf8).alias("category_level_3"),
        pl.col("sale_amount").cast(pl.Float32).alias("observed_demand_qty"),
        pl.lit("observed_sales").alias("target_semantics"),
        pl.coalesce([stockout_flag_expr, pl.lit(False)]).cast(pl.Boolean).alias("censor_flag"),
        pl.lit("observed_sales").alias("target_source"),
        pl.when(pl.coalesce([stockout_flag_expr, pl.lit(False)]))
        .then(pl.lit(0.5))
        .otherwise(pl.lit(1.0))
        .cast(pl.Float32)
        .alias("label_quality_score"),
        (~pl.coalesce([stockout_flag_expr, pl.lit(False)])).alias("usable_for_training_flag"),
        pl.lit(None).cast(pl.Float32).alias("observed_revenue_net"),
        pl.col("discount").cast(pl.Float32).alias("observed_discount_amount"),
        freshretail_promo_flag_expr().alias("promo_flag"),
        pl.col("holiday_flag").cast(pl.Boolean).alias("holiday_flag"),
        pl.col("activity_flag").cast(pl.Boolean).alias("activity_flag"),
        stockout_flag_expr.alias("observed_stockout_flag"),
        pl.lit(True).alias("observed_stockout_available"),
        pl.col("stock_hour6_22_cnt").cast(pl.Float32).alias("observed_stockout_intensity"),
        pl.lit(True).alias("day_complete_flag"),
        pl.col("dt").cast(pl.Date).dt.strftime("%A").alias("calendar_weekday_name"),
        (pl.col("dt").cast(pl.Date).dt.weekday() - 1).cast(pl.Int8).alias("calendar_day_of_week"),
        pl.col("dt").cast(pl.Date).dt.month().cast(pl.Int8).alias("calendar_month"),
        pl.col("dt").cast(pl.Date).dt.year().cast(pl.Int16).alias("calendar_year"),
        pl.col("dt").cast(pl.Date).dt.week().cast(pl.Int32).alias("calendar_week_key"),
        pl.lit(None).cast(pl.Utf8).alias("event_name_1"),
        pl.lit(None).cast(pl.Utf8).alias("event_type_1"),
        pl.lit(None).cast(pl.Utf8).alias("event_name_2"),
        pl.lit(None).cast(pl.Utf8).alias("event_type_2"),
        pl.col("precpt").cast(pl.Float32).alias("weather_precipitation"),
        pl.col("avg_temperature").cast(pl.Float32).alias("weather_temperature"),
        pl.col("avg_humidity").cast(pl.Float32).alias("weather_humidity"),
        pl.col("avg_wind_level").cast(pl.Float32).alias("weather_wind_level"),
        pl.lit("freshretail").alias("dataset_source"),
        pl.lit(source_partition).alias("source_partition"),
        pl.lit(source_run_id).alias("source_run_id"),
        pl.lit(silver_run_id).alias("silver_run_id"),
    ]


def standardize_freshretail_lazy_frame(
    frame: pl.LazyFrame,
    *,
    source_partition: str,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.LazyFrame:
    """Map FreshRetail daily data into the canonical daily demand contract."""

    available_columns = set(frame.collect_schema().names())
    if "is_censored" in available_columns:
        stockout_flag_expr = pl.col("is_censored").cast(pl.Boolean)
    else:
        stockout_flag_expr = (pl.col("stock_hour6_22_cnt").cast(pl.Float32) > 0)

    normalized = frame.with_columns(
        _freshretail_canonical_expressions(
            stockout_flag_expr=stockout_flag_expr,
            source_partition=source_partition,
            source_run_id=source_run_id,
            silver_run_id=silver_run_id,
        )
    ).with_columns(build_series_id_expr("location_id", "product_id").alias("series_id"))
    return align_lazy_frame_to_canonical_schema(normalized)


def build_freshretail_standardized_frame(
    *,
    train_input_path: str | Path = DEFAULT_TRAIN_INPUT_PATH,
    val_input_path: str | Path = DEFAULT_VAL_INPUT_PATH,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.DataFrame:
    """Build the canonical FreshRetail frame in memory."""

    lazy_frames: list[pl.LazyFrame] = []
    train_path = Path(train_input_path)
    val_path = Path(val_input_path)
    if train_path.exists():
        lazy_frames.append(
            standardize_freshretail_lazy_frame(
                pl.scan_parquet(train_path),
                source_partition="train",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )
    if val_path.exists():
        lazy_frames.append(
            standardize_freshretail_lazy_frame(
                pl.scan_parquet(val_path),
                source_partition="val",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )
    if not lazy_frames:
        raise FileNotFoundError("No FreshRetail parquet inputs were found for canonical dataset generation.")
    return pl.concat(lazy_frames, how="vertical_relaxed").collect()


def build_freshretail_standardized_dataset(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    train_input_path: str | Path = DEFAULT_TRAIN_INPUT_PATH,
    val_input_path: str | Path = DEFAULT_VAL_INPUT_PATH,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> Path:
    """Write the canonical FreshRetail daily dataset as a parquet file."""

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    combined = build_freshretail_standardized_frame(
        train_input_path=train_input_path,
        val_input_path=val_input_path,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
    combined.write_parquet(target_path)
    return target_path

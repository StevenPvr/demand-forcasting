from __future__ import annotations

import re

import polars as pl

from praedixa.platform.datasets.standardization.schema import (
    align_lazy_frame_to_canonical_schema,
    build_series_id_expr,
)


DEFAULT_SOURCE_RUN_ID = "manual_local"
DEFAULT_SILVER_RUN_ID = "manual_local"
M5_DAY_COLUMN_PATTERN = re.compile(r"^d_\d+$")


def freshretail_promo_flag_expr(discount_column: str = "discount") -> pl.Expr:
    """Infer a promo flag from the FreshRetail discount field without forcing one encoding."""

    discount = pl.col(discount_column).cast(pl.Float32)
    return (
        pl.when(discount.is_null())
        .then(None)
        .when(discount <= 0)
        .then(False)
        .when(discount == 1)
        .then(False)
        .when(discount < 1)
        .then(True)
        .otherwise(True)
    )


def standardize_freshretail_lt_lazy_frame(
    frame: pl.LazyFrame,
    *,
    source_partition: str,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.LazyFrame:
    """Map FreshRetail-LT daily data into the canonical demand contract."""

    available_columns = set(frame.collect_schema().names())
    if "is_censored" in available_columns:
        stockout_flag_expr = pl.col("is_censored").cast(pl.Boolean)
    else:
        stockout_flag_expr = pl.col("stock_hour6_22_cnt").cast(pl.Float32) > 0

    normalized = frame.with_columns(
        _freshretail_lt_projection_columns(
            stockout_flag_expr=stockout_flag_expr,
            source_partition=source_partition,
            source_run_id=source_run_id,
            silver_run_id=silver_run_id,
        )
    ).with_columns(build_series_id_expr("location_id", "product_id").alias("series_id"))
    return align_lazy_frame_to_canonical_schema(normalized)


def standardize_m5_sales_lazy_frame(
    sales_frame: pl.LazyFrame,
    calendar_frame: pl.LazyFrame,
    *,
    source_partition: str,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.LazyFrame:
    """Map M5 wide daily unit sales into the canonical daily demand contract."""

    day_columns = [
        column_name
        for column_name in sales_frame.collect_schema().names()
        if M5_DAY_COLUMN_PATTERN.match(column_name)
    ]
    if not day_columns:
        raise ValueError("M5 sales frame does not expose any d_### daily columns.")

    calendar = _standardize_m5_calendar(calendar_frame)
    long_sales = sales_frame.unpivot(
        index=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        on=day_columns,
        variable_name="m5_day_key",
        value_name="observed_demand_qty",
    )
    normalized = (
        long_sales.join(calendar, on="m5_day_key", how="inner")
        .with_columns(
            [
                pl.col("store_id").cast(pl.Utf8).alias("location_id"),
                pl.col("item_id").cast(pl.Utf8).alias("product_id"),
                pl.col("state_id").cast(pl.Utf8).alias("region_id"),
                pl.col("cat_id").cast(pl.Utf8).alias("category_level_1"),
                pl.col("dept_id").cast(pl.Utf8).alias("category_level_2"),
                pl.col("observed_demand_qty").cast(pl.Float32),
                pl.lit("m5_unit_sales").alias("target_source"),
            ]
        )
        .filter(
            pl.col("dt").is_not_null() & pl.col("observed_demand_qty").is_not_null()
        )
    )
    return _complete_canonical_daily_frame(
        normalized,
        dataset_source="m5_forecasting_accuracy",
        source_partition=source_partition,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


def _freshretail_lt_projection_columns(
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
        pl.coalesce([stockout_flag_expr, pl.lit(False)])
        .cast(pl.Boolean)
        .alias("censor_flag"),
        pl.lit("observed_sales").alias("target_source"),
        pl.lit(1.0).cast(pl.Float32).alias("label_quality_score"),
        pl.lit(True).alias("usable_for_training_flag"),
        pl.lit(None).cast(pl.Float32).alias("observed_revenue_net"),
        pl.col("discount").cast(pl.Float32).alias("observed_discount_amount"),
        freshretail_promo_flag_expr().alias("promo_flag"),
        pl.col("holiday_flag").cast(pl.Boolean).alias("holiday_flag"),
        pl.col("activity_flag").cast(pl.Boolean).alias("activity_flag"),
        stockout_flag_expr.alias("observed_stockout_flag"),
        pl.lit(True).alias("observed_stockout_available"),
        pl.col("stock_hour6_22_cnt")
        .cast(pl.Float32)
        .alias("observed_stockout_intensity"),
        pl.lit(None).cast(pl.Boolean).alias("day_complete_flag"),
        pl.col("dt").cast(pl.Date).dt.strftime("%A").alias("calendar_weekday_name"),
        (pl.col("dt").cast(pl.Date).dt.weekday() - 1)
        .cast(pl.Int8)
        .alias("calendar_day_of_week"),
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
        pl.lit("freshretail_lt").alias("dataset_source"),
        pl.lit(source_partition).alias("source_partition"),
        pl.lit(source_run_id).alias("source_run_id"),
        pl.lit(silver_run_id).alias("silver_run_id"),
    ]


def _standardize_m5_calendar(calendar_frame: pl.LazyFrame) -> pl.LazyFrame:
    available_columns = set(calendar_frame.collect_schema().names())
    event_name_1 = _expr_or_default(
        available_columns, "event_name_1", pl.lit(None).cast(pl.Utf8)
    )
    event_type_1 = _expr_or_default(
        available_columns, "event_type_1", pl.lit(None).cast(pl.Utf8)
    )
    event_name_2 = _expr_or_default(
        available_columns, "event_name_2", pl.lit(None).cast(pl.Utf8)
    )
    event_type_2 = _expr_or_default(
        available_columns, "event_type_2", pl.lit(None).cast(pl.Utf8)
    )
    return calendar_frame.select(
        [
            pl.col("d").cast(pl.Utf8).alias("m5_day_key"),
            pl.col("date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            .alias("dt"),
            event_name_1.cast(pl.Utf8, strict=False).alias("event_name_1"),
            event_type_1.cast(pl.Utf8, strict=False).alias("event_type_1"),
            event_name_2.cast(pl.Utf8, strict=False).alias("event_name_2"),
            event_type_2.cast(pl.Utf8, strict=False).alias("event_type_2"),
            (event_name_1.is_not_null() | event_name_2.is_not_null()).alias(
                "holiday_flag"
            ),
        ]
    )


def _expr_or_default(
    available_columns: set[str],
    column_name: str,
    default_expr: pl.Expr,
) -> pl.Expr:
    if column_name in available_columns:
        return pl.col(column_name)
    return default_expr


def _complete_canonical_daily_frame(
    frame: pl.LazyFrame,
    *,
    dataset_source: str,
    source_partition: str,
    source_run_id: str,
    silver_run_id: str,
) -> pl.LazyFrame:
    available_columns = set(frame.collect_schema().names())
    normalized = frame.with_columns(
        [
            pl.lit(dataset_source).alias("dataset_source"),
            pl.lit(source_partition).alias("source_partition"),
            pl.lit(source_run_id).alias("source_run_id"),
            pl.lit(silver_run_id).alias("silver_run_id"),
            _expr_or_default(
                available_columns, "target_semantics", pl.lit("observed_sales")
            ).alias("target_semantics"),
            _expr_or_default(available_columns, "censor_flag", pl.lit(False)).alias(
                "censor_flag"
            ),
            _expr_or_default(
                available_columns, "target_source", pl.lit("observed_sales")
            ).alias("target_source"),
            _expr_or_default(
                available_columns, "label_quality_score", pl.lit(1.0)
            ).alias("label_quality_score"),
            _expr_or_default(
                available_columns, "usable_for_training_flag", pl.lit(True)
            ).alias("usable_for_training_flag"),
            _expr_or_default(
                available_columns, "observed_stockout_available", pl.lit(False)
            ).alias("observed_stockout_available"),
            _expr_or_default(
                available_columns, "day_complete_flag", pl.lit(True)
            ).alias("day_complete_flag"),
            _expr_or_default(available_columns, "activity_flag", pl.lit(True)).alias(
                "activity_flag"
            ),
            _expr_or_default(available_columns, "holiday_flag", pl.lit(False)).alias(
                "holiday_flag"
            ),
            pl.col("dt").cast(pl.Date).dt.strftime("%A").alias("calendar_weekday_name"),
            (pl.col("dt").cast(pl.Date).dt.weekday() - 1)
            .cast(pl.Int8)
            .alias("calendar_day_of_week"),
            pl.col("dt").cast(pl.Date).dt.month().cast(pl.Int8).alias("calendar_month"),
            pl.col("dt").cast(pl.Date).dt.year().cast(pl.Int16).alias("calendar_year"),
            pl.col("dt")
            .cast(pl.Date)
            .dt.week()
            .cast(pl.Int32)
            .alias("calendar_week_key"),
        ]
    )
    if "series_id" not in available_columns:
        normalized = normalized.with_columns(
            build_series_id_expr("location_id", "product_id").alias("series_id")
        )
    return align_lazy_frame_to_canonical_schema(normalized)

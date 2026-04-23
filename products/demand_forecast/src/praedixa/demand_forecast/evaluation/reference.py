from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl

from praedixa.demand_forecast.evaluation.bakery_style import REFERENCE_DATE_COL
from praedixa.demand_forecast.evaluation.bakery_style import REFERENCE_PRODUCT_COL
from praedixa.demand_forecast.evaluation.bakery_style import REFERENCE_TARGET_COL
from praedixa.platform.utils.memory import downcast_pandas_frame


def _to_polars_frame(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(frame, include_index=False)


def _normalized_reference_split_frame_polars(frame: pd.DataFrame) -> pl.DataFrame:
    return (
        _to_polars_frame(frame)
        .with_columns(
            [
                pl.col(REFERENCE_DATE_COL).cast(pl.Datetime, strict=False),
                pl.col(REFERENCE_PRODUCT_COL).cast(pl.Utf8, strict=False),
                pl.col(REFERENCE_TARGET_COL).cast(pl.Float64, strict=False),
            ]
        )
        .sort([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL])
    )

def _reference_base_frame(reference_full_df: pd.DataFrame) -> pl.DataFrame:
    return _normalized_reference_split_frame_polars(reference_full_df).select(
        [
            pl.col(REFERENCE_DATE_COL).alias("dt"),
            pl.col(REFERENCE_PRODUCT_COL).alias("product_id"),
            pl.col(REFERENCE_TARGET_COL).alias("current_day_demand_qty"),
        ]
    )


def _normalized_gold_base_frame(gold_base_panel_df: pd.DataFrame) -> pl.DataFrame:
    return _to_polars_frame(gold_base_panel_df).with_columns(
        [
            pl.col("dt").cast(pl.Datetime, strict=False),
            pl.col("target_dt").cast(pl.Datetime, strict=False),
            pl.col("product_id").cast(pl.Utf8, strict=False),
        ]
    )


def _reference_enrichment_columns(gold_base: pl.DataFrame) -> list[str]:
    return [
        column
        for column in gold_base.columns
        if column not in {"dt", "target_dt", "product_id", "current_day_demand_qty"}
    ]


def _merged_reference_frame(base: pl.DataFrame, gold_base: pl.DataFrame) -> pl.DataFrame:
    enrichment_cols = _reference_enrichment_columns(gold_base)
    return base.join(
        gold_base.select(["dt", "product_id", *enrichment_cols]),
        how="left",
        on=["dt", "product_id"],
    )


def _reference_templates(gold_base: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    date_template = gold_base.sort(["dt", "product_id"]).group_by("dt", maintain_order=True).first()
    product_template = gold_base.sort("dt").group_by("product_id", maintain_order=True).last()
    price_template = gold_base.group_by("product_id", maintain_order=True).agg(
        pl.col("avg_selling_price").median().alias("avg_selling_price_template")
    )
    return date_template, product_template, price_template


def _fill_template_columns(
    merged: pl.DataFrame,
    *,
    template: pl.DataFrame,
    fill_columns: list[str],
    key_column: str,
    suffix: str,
) -> pl.DataFrame:
    available = [column for column in fill_columns if column in merged.columns and column in template.columns]
    if not available:
        return merged
    renamed = {column: f"{column}{suffix}" for column in available}
    return (
        merged.join(
            template.select([key_column, *available]).rename(renamed),
            how="left",
            on=key_column,
        )
        .with_columns([pl.coalesce(pl.col(column), pl.col(renamed[column])).alias(column) for column in available])
        .drop(list(renamed.values()))
    )


def _fill_reference_templates(
    merged: pl.DataFrame,
    *,
    date_template: pl.DataFrame,
    product_template: pl.DataFrame,
) -> pl.DataFrame:
    with_dates = _fill_template_columns(
        merged,
        template=date_template,
        fill_columns=_DATE_TEMPLATE_FILL_COLUMNS,
        key_column="dt",
        suffix="__date_template",
    )
    return _fill_template_columns(
        with_dates,
        template=product_template,
        fill_columns=_PRODUCT_TEMPLATE_FILL_COLUMNS,
        key_column="product_id",
        suffix="__product_template",
    )


def _resolved_column_expr(
    frame: pl.DataFrame,
    column: str,
    default: object,
    *,
    dtype: Any,
) -> pl.Expr:
    base_expr = pl.col(column) if column in frame.columns else pl.lit(default, dtype=dtype)
    return base_expr.cast(dtype, strict=False).fill_null(default)


def _optional_float_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column in frame.columns:
        return pl.col(column).cast(pl.Float64, strict=False)
    return pl.lit(None, dtype=pl.Float64)


def _weather_available_expr() -> pl.Expr:
    return pl.any_horizontal(
        [
            pl.col("weather_temperature").is_not_null(),
            pl.col("weather_precipitation").is_not_null(),
            pl.col("weather_humidity").is_not_null(),
            pl.col("weather_wind_level").is_not_null(),
        ]
    ).alias("weather_available")


def _macro_available_expr() -> pl.Expr:
    return pl.any_horizontal(
        [
            pl.col("gdp_growth_latest").is_not_null(),
            pl.col("gdp_current_usd_latest").is_not_null(),
            pl.col("lending_interest_rate_latest").is_not_null(),
            pl.col("government_debt_pct_gdp_latest").is_not_null(),
        ]
    ).alias("macro_available")


def _apply_reference_defaults(merged: pl.DataFrame, *, price_template: pl.DataFrame) -> pl.DataFrame:
    base = merged.join(price_template, how="left", on="product_id").with_columns(
        [
            pl.lit("bakery").alias("dataset_source"),
            _resolved_column_expr(merged, "location_id", "bakery_store_1", dtype=pl.Utf8).alias("location_id"),
            _resolved_column_expr(merged, "product_active_flag", True, dtype=pl.Boolean).alias("product_active_flag"),
            _resolved_column_expr(merged, "location_open_flag", True, dtype=pl.Boolean).alias("location_open_flag"),
            _resolved_column_expr(merged, "day_complete_flag", True, dtype=pl.Boolean).alias("day_complete_flag"),
            _resolved_column_expr(merged, "is_missing_day", 0, dtype=pl.Int64)
            .cast(pl.Boolean)
            .alias("missing_sales_flag"),
            _resolved_column_expr(merged, "observed_stockout_flag", False, dtype=pl.Boolean).alias(
                "observed_stockout_flag"
            ),
            _resolved_column_expr(merged, "observed_stockout_available", False, dtype=pl.Boolean).alias(
                "observed_stockout_available"
            ),
            _resolved_column_expr(merged, "anomaly_flag", False, dtype=pl.Boolean).alias("anomaly_flag"),
            _resolved_column_expr(merged, "location_closed_flag", False, dtype=pl.Boolean).alias(
                "location_closed_flag"
            ),
            pl.coalesce(
                _optional_float_expr(merged, "avg_selling_price"),
                pl.col("avg_selling_price_template"),
            ).alias("avg_selling_price"),
            _resolved_column_expr(merged, "observed_discount_amount", 0.0, dtype=pl.Float64).alias(
                "observed_discount_amount"
            ),
            _resolved_column_expr(merged, "promo_flag", False, dtype=pl.Boolean).alias("promo_flag"),
            (pl.col("dt") + pl.duration(days=1)).alias("target_dt"),
        ]
    )
    revenue_fill = pl.col("current_day_demand_qty").fill_null(0.0) * pl.col("avg_selling_price").fill_null(0.0)
    return (
        base.with_columns(
            [
                (~pl.col("missing_sales_flag")).alias("is_observed_row"),
                pl.col("current_day_demand_qty").fill_null(0.0).eq(0.0).alias("true_zero_demand_flag"),
                pl.coalesce(_optional_float_expr(base, "observed_revenue_net"), revenue_fill).alias(
                    "observed_revenue_net"
                ),
                pl.col("avg_selling_price").is_not_null().alias("price_available"),
                _weather_available_expr(),
                _macro_available_expr(),
            ]
        )
        .drop("avg_selling_price_template")
        .sort(["product_id", "dt"])
    )


def _build_bakery_overlap_feature_block(merged: pl.DataFrame) -> pl.DataFrame:
    with_target_dow = merged.with_columns(((pl.col("target_dt").dt.weekday() - 1).cast(pl.Int8)).alias("__target_dow"))
    return with_target_dow.with_columns(
        [
            pl.col("current_day_demand_qty").shift(-1).over("product_id").alias("target_demand_qty_d_plus_1"),
            pl.col("current_day_demand_qty").shift(1).over("product_id").alias("lag_1"),
            pl.col("current_day_demand_qty").shift(7).over("product_id").alias("lag_7"),
            pl.col("current_day_demand_qty").shift(14).over("product_id").alias("lag_14"),
            pl.col("current_day_demand_qty").shift(21).over("product_id").alias("lag_21"),
            pl.col("current_day_demand_qty").shift(28).over("product_id").alias("lag_28"),
            pl.col("current_day_demand_qty").shift(6).over("product_id").alias("target_lag_7"),
            pl.col("avg_selling_price").shift(1).over("product_id").alias("avg_selling_price_lag_1"),
            pl.col("promo_flag").shift(1).over("product_id").alias("promo_flag_lag_1"),
            pl.col("promo_flag")
            .cast(pl.Float64)
            .rolling_mean(window_size=7, min_samples=1)
            .over("product_id")
            .alias("promo_rate_7"),
            pl.col("activity_flag")
            .cast(pl.Float64)
            .rolling_mean(window_size=7, min_samples=1)
            .over("product_id")
            .alias("activity_rate_7"),
            pl.col("weather_temperature").alias("weather_temperature_lag_0"),
            pl.col("weather_temperature").shift(1).over("product_id").alias("weather_temperature_lag_1"),
            pl.col("weather_precipitation").alias("weather_precipitation_lag_0"),
            pl.col("weather_humidity").alias("weather_humidity_lag_0"),
            pl.col("weather_wind_level").alias("weather_wind_level_lag_0"),
            pl.col("current_day_demand_qty")
            .shift(7)
            .rolling_mean(window_size=4, min_samples=1)
            .over(["product_id", "__target_dow"])
            .alias("target_same_dow_mean_4w"),
            (
                pl.col("gdp_growth_latest")
                - pl.col("gdp_growth_latest").shift(28).over("product_id")
            ).alias("gdp_growth_latest_delta_28"),
            (
                pl.col("gdp_current_usd_latest")
                - pl.col("gdp_current_usd_latest").shift(28).over("product_id")
            ).alias("gdp_current_usd_latest_delta_28"),
            (
                pl.col("lending_interest_rate_latest")
                - pl.col("lending_interest_rate_latest").shift(28).over("product_id")
            ).alias("lending_interest_rate_latest_delta_28"),
            (
                pl.col("government_debt_pct_gdp_latest")
                - pl.col("government_debt_pct_gdp_latest").shift(28).over("product_id")
            ).alias("government_debt_pct_gdp_latest_delta_28"),
        ]
    ).drop("__target_dow")


def _target_rows_from_reference(merged: pl.DataFrame, reference_test: pl.DataFrame) -> pd.DataFrame:
    target_rows = (
        reference_test.join(
            merged,
            how="left",
            left_on=[REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL],
            right_on=["product_id", "target_dt"],
        )
        .with_columns(
            [
                pl.col(REFERENCE_PRODUCT_COL).cast(pl.Utf8, strict=False).alias("product_id"),
                pl.col(REFERENCE_DATE_COL).cast(pl.Datetime, strict=False).alias("target_dt"),
                pl.col(REFERENCE_TARGET_COL).cast(pl.Float64, strict=False).alias("target_demand_qty_d_plus_1"),
                pl.when(pl.col("target_lag_7").is_not_null() & (pl.col("target_lag_7") >= 0.0))
                .then(pl.col("target_demand_qty_d_plus_1").log1p() - pl.col("target_lag_7").log1p())
                .otherwise(None)
                .alias("target_delta_log_wow_d_plus_1"),
            ]
        )
        .sort([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL])
    )
    return downcast_pandas_frame(target_rows.to_pandas())


def build_bakery_reference_feature_frame(
    reference_full_df: pd.DataFrame,
    reference_test_df: pd.DataFrame,
    gold_base_panel_df: pd.DataFrame,
) -> pd.DataFrame:
    reference_test = _normalized_reference_split_frame_polars(reference_test_df)
    gold_base = _normalized_gold_base_frame(gold_base_panel_df)
    merged = _merged_reference_frame(_reference_base_frame(reference_full_df), gold_base)
    date_template, product_template, price_template = _reference_templates(gold_base)
    merged = _fill_reference_templates(
        merged,
        date_template=date_template,
        product_template=product_template,
    )
    prepared = _apply_reference_defaults(merged, price_template=price_template)
    return _target_rows_from_reference(_build_bakery_overlap_feature_block(prepared), reference_test)


_DATE_TEMPLATE_FILL_COLUMNS = [
    "source_partition",
    "source_run_id",
    "location_open_flag",
    "day_complete_flag",
    "holiday_flag",
    "activity_flag",
    "school_holiday_flag",
    "bridge_day_flag",
    "pre_holiday_flag",
    "post_holiday_flag",
    "weather_precipitation",
    "weather_temperature",
    "weather_humidity",
    "weather_wind_level",
    "gdp_growth_latest",
    "gdp_current_usd_latest",
    "lending_interest_rate_latest",
    "government_debt_pct_gdp_latest",
    "client_id",
    "vertical_level_1",
    "vertical_level_2",
    "country_code",
    "region_code",
    "city_name",
    "freshretail_rescaled_flag",
    "target_scale_assumption",
    "gold_run_id",
]

_PRODUCT_TEMPLATE_FILL_COLUMNS = [
    "series_id",
    "location_id",
    "region_id",
    "org_group_id",
    "category_level_1",
    "category_level_2",
    "category_level_3",
]

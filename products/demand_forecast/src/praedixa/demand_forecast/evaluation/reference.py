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


def _merged_reference_frame(
    base: pl.DataFrame, gold_base: pl.DataFrame
) -> pl.DataFrame:
    enrichment_cols = _reference_enrichment_columns(gold_base)
    return base.join(
        gold_base.select(["dt", "product_id", *enrichment_cols]),
        how="left",
        on=["dt", "product_id"],
    )


def _reference_templates(gold_base: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    date_template = (
        gold_base.sort(["dt", "product_id"]).group_by("dt", maintain_order=True).first()
    )
    product_template = (
        gold_base.sort("dt").group_by("product_id", maintain_order=True).last()
    )
    return date_template, product_template


def _fill_template_columns(
    merged: pl.DataFrame,
    *,
    template: pl.DataFrame,
    fill_columns: list[str],
    key_column: str,
    suffix: str,
) -> pl.DataFrame:
    available = [
        column
        for column in fill_columns
        if column in merged.columns and column in template.columns
    ]
    if not available:
        return merged
    renamed = {column: f"{column}{suffix}" for column in available}
    return (
        merged.join(
            template.select([key_column, *available]).rename(renamed),
            how="left",
            on=key_column,
        )
        .with_columns(
            [
                pl.coalesce(pl.col(column), pl.col(renamed[column])).alias(column)
                for column in available
            ]
        )
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
    base_expr = (
        pl.col(column) if column in frame.columns else pl.lit(default, dtype=dtype)
    )
    return base_expr.cast(dtype, strict=False).fill_null(default)


def _optional_float_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column in frame.columns:
        return pl.col(column).cast(pl.Float64, strict=False)
    return pl.lit(None, dtype=pl.Float64)


def _optional_source_expr(frame: pl.DataFrame, *columns: str) -> pl.Expr:
    for column in columns:
        if column in frame.columns:
            return pl.col(column)
    return pl.lit(None)


def _coalesced_feature_expr(
    frame: pl.DataFrame,
    column: str,
    fallback: pl.Expr,
) -> pl.Expr:
    if column in frame.columns:
        return pl.coalesce(pl.col(column), fallback).alias(column)
    return fallback.alias(column)


def _client_id_expr(frame: pl.DataFrame) -> pl.Expr:
    default_client_id = pl.concat_str(
        [
            pl.col("location_id").cast(pl.Utf8, strict=False),
            pl.lit("__"),
            pl.col("product_id").cast(pl.Utf8, strict=False),
        ]
    )
    if "client_id" not in frame.columns:
        return default_client_id
    return pl.coalesce(
        pl.col("client_id").cast(pl.Utf8, strict=False),
        default_client_id,
    )


def _apply_reference_defaults(merged: pl.DataFrame) -> pl.DataFrame:
    base = merged.with_columns(
        [
            pl.lit("bakery").alias("dataset_source"),
            _resolved_column_expr(
                merged, "location_id", "bakery_store_1", dtype=pl.Utf8
            ).alias("location_id"),
            _resolved_column_expr(
                merged, "day_complete_flag", True, dtype=pl.Boolean
            ).alias("day_complete_flag"),
            _resolved_column_expr(
                merged, "observed_stockout_flag", False, dtype=pl.Boolean
            ).alias("observed_stockout_flag"),
            _resolved_column_expr(
                merged, "observed_stockout_available", False, dtype=pl.Boolean
            ).alias("observed_stockout_available"),
            _resolved_column_expr(
                merged, "observed_discount_amount", 0.0, dtype=pl.Float64
            ).alias("observed_discount_amount"),
            _resolved_column_expr(merged, "promo_flag", False, dtype=pl.Boolean).alias(
                "promo_flag"
            ),
            _resolved_column_expr(
                merged, "source_role", "benchmark", dtype=pl.Utf8
            ).alias("source_role"),
            _resolved_column_expr(
                merged, "is_synthetic_source", False, dtype=pl.Boolean
            ).alias("is_synthetic_source"),
            (pl.col("dt") + pl.duration(days=1)).alias("target_dt"),
        ]
    ).with_columns(_client_id_expr(merged).alias("client_id"))
    return base.with_columns(
        [
            _resolved_column_expr(
                base, "is_observed_row", True, dtype=pl.Boolean
            ).alias("is_observed_row"),
            pl.col("current_day_demand_qty")
            .fill_null(0.0)
            .eq(0.0)
            .alias("true_zero_demand_flag"),
            _optional_float_expr(base, "observed_revenue_net").alias(
                "observed_revenue_net"
            ),
        ]
    ).sort(["product_id", "dt"])


def _build_bakery_overlap_feature_block(merged: pl.DataFrame) -> pl.DataFrame:
    with_target_dow = merged.with_columns(
        ((pl.col("target_dt").dt.weekday() - 1).cast(pl.Int8)).alias("__target_dow")
    )
    with_features = with_target_dow.with_columns(
        [
            _coalesced_feature_expr(
                with_target_dow,
                "target_demand_qty_d_plus_1",
                pl.col("current_day_demand_qty").shift(-1).over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "lag_1",
                pl.col("current_day_demand_qty").shift(1).over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "lag_7",
                pl.col("current_day_demand_qty").shift(7).over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "lag_14",
                pl.col("current_day_demand_qty").shift(14).over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "lag_21",
                pl.col("current_day_demand_qty").shift(21).over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "lag_28",
                pl.col("current_day_demand_qty").shift(28).over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "target_lag_1",
                pl.col("current_day_demand_qty"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "target_lag_7",
                pl.col("current_day_demand_qty").shift(6).over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "promo_flag_lag_1",
                pl.col("promo_flag").shift(1).over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "promo_rate_7",
                pl.col("promo_flag")
                .cast(pl.Float64)
                .rolling_mean(window_size=7, min_samples=1)
                .over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "activity_rate_7",
                pl.col("activity_flag")
                .cast(pl.Float64)
                .rolling_mean(window_size=7, min_samples=1)
                .over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "weather_temperature_lag_0",
                _optional_source_expr(
                    with_target_dow, "weather_temperature", "weather_temperature_lag_0"
                ),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "weather_temperature_lag_1",
                _optional_source_expr(
                    with_target_dow, "weather_temperature", "weather_temperature_lag_0"
                )
                .shift(1)
                .over("product_id"),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "weather_precipitation_lag_0",
                _optional_source_expr(
                    with_target_dow,
                    "weather_precipitation",
                    "weather_precipitation_lag_0",
                ),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "weather_humidity_lag_0",
                _optional_source_expr(
                    with_target_dow, "weather_humidity", "weather_humidity_lag_0"
                ),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "weather_wind_level_lag_0",
                _optional_source_expr(
                    with_target_dow, "weather_wind_level", "weather_wind_level_lag_0"
                ),
            ),
            _coalesced_feature_expr(
                with_target_dow,
                "target_same_dow_mean_4w",
                pl.col("current_day_demand_qty")
                .shift(7)
                .rolling_mean(window_size=4, min_samples=1)
                .over(["product_id", "__target_dow"]),
            ),
        ]
    )
    field_baseline = (
        pl.when(
            pl.col("target_lag_1").is_not_null() & pl.col("target_lag_7").is_not_null()
        )
        .then((pl.col("target_lag_1") * 0.5) + (pl.col("target_lag_7") * 0.5))
        .otherwise(pl.coalesce(pl.col("target_lag_1"), pl.col("target_lag_7")))
    )
    with_baseline = with_features.with_columns(
        _coalesced_feature_expr(
            with_features,
            "field_baseline_blend_lag_1_lag_7_d_plus_1",
            field_baseline,
        )
    )
    return with_baseline.with_columns(
        _coalesced_feature_expr(
            with_baseline,
            "target_residual_field_blend_lag_1_lag_7_d_plus_1",
            pl.col("target_demand_qty_d_plus_1")
            - pl.col("field_baseline_blend_lag_1_lag_7_d_plus_1"),
        )
    ).drop("__target_dow")


def _reference_bool_expr(
    frame: pl.DataFrame,
    column: str,
    default: bool,
) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(default, dtype=pl.Boolean)
    return pl.col(column).cast(pl.Boolean, strict=False).fill_null(default)


def _target_source_expr(target_rows: pl.DataFrame) -> pl.Expr:
    missing_day = _reference_bool_expr(target_rows, "is_missing_day", False)
    observed_row = _reference_bool_expr(target_rows, "is_observed_row", True)
    zero_quantity = (
        pl.col(REFERENCE_TARGET_COL)
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .eq(0.0)
    )
    return (
        pl.when(missing_day)
        .then(pl.lit("closed_or_missing_observation"))
        .when((~observed_row) & zero_quantity)
        .then(pl.lit("dense_calendar_zero_fill"))
        .otherwise(pl.lit("observed_sales"))
    )


def _apply_reference_target_metadata(target_rows: pl.DataFrame) -> pl.DataFrame:
    target_source = _target_source_expr(target_rows)
    trainable_target = target_source.eq(pl.lit("observed_sales"))
    return target_rows.with_columns(
        [
            pl.lit("observed_sales").alias("target_semantics"),
            pl.lit(False, dtype=pl.Boolean).alias("censor_flag"),
            target_source.alias("target_source"),
            (
                pl.when(trainable_target)
                .then(pl.lit(1.0))
                .when(target_source.eq(pl.lit("dense_calendar_zero_fill")))
                .then(pl.lit(0.8))
                .otherwise(pl.lit(0.0))
            ).alias("label_quality_score"),
            trainable_target.alias("usable_for_training_flag"),
            pl.col(REFERENCE_TARGET_COL)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .eq(0.0)
            .alias("target_true_zero_demand_flag"),
        ]
    )


def _target_rows_from_reference(
    merged: pl.DataFrame, reference_test: pl.DataFrame
) -> pd.DataFrame:
    target_rows = _apply_reference_target_metadata(
        reference_test.join(
            merged,
            how="left",
            left_on=[REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL],
            right_on=["product_id", "target_dt"],
        ).with_columns(
            [
                pl.col(REFERENCE_PRODUCT_COL)
                .cast(pl.Utf8, strict=False)
                .alias("product_id"),
                pl.col(REFERENCE_DATE_COL)
                .cast(pl.Datetime, strict=False)
                .alias("target_dt"),
                pl.col(REFERENCE_TARGET_COL)
                .cast(pl.Float64, strict=False)
                .alias("target_demand_qty_d_plus_1"),
            ]
        )
    ).sort([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL])
    return downcast_pandas_frame(target_rows.to_pandas())


def build_bakery_reference_feature_frame(
    reference_full_df: pd.DataFrame,
    reference_test_df: pd.DataFrame,
    gold_base_panel_df: pd.DataFrame,
) -> pd.DataFrame:
    reference_test = _normalized_reference_split_frame_polars(reference_test_df)
    gold_base = _normalized_gold_base_frame(gold_base_panel_df)
    merged = _merged_reference_frame(
        _reference_base_frame(reference_full_df), gold_base
    )
    date_template, product_template = _reference_templates(gold_base)
    merged = _fill_reference_templates(
        merged,
        date_template=date_template,
        product_template=product_template,
    )
    prepared = _apply_reference_defaults(merged)
    return _target_rows_from_reference(
        _build_bakery_overlap_feature_block(prepared), reference_test
    )


_DATE_TEMPLATE_FILL_COLUMNS = [
    "source_partition",
    "source_run_id",
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
    "lending_interest_rate_latest",
    "vertical_level_1",
    "vertical_level_2",
    "country_code",
    "region_code",
    "city_name",
    "gold_run_id",
]

_PRODUCT_TEMPLATE_FILL_COLUMNS = [
    "client_id",
    "location_id",
    "region_id",
    "org_group_id",
    "category_level_1",
    "category_level_2",
    "category_level_3",
]

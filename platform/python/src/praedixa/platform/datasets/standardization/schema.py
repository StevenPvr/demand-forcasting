from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import cast

import polars as pl


CANONICAL_SCHEMA: OrderedDict[str, pl.DataType] = cast(
    OrderedDict[str, pl.DataType],
    OrderedDict(
    [
        ("dataset_source", pl.Utf8),
        ("source_partition", pl.Utf8),
        ("source_run_id", pl.Utf8),
        ("series_id", pl.Utf8),
        ("dt", pl.Date),
        ("location_id", pl.Utf8),
        ("product_id", pl.Utf8),
        ("region_id", pl.Utf8),
        ("org_group_id", pl.Utf8),
        ("category_level_1", pl.Utf8),
        ("category_level_2", pl.Utf8),
        ("category_level_3", pl.Utf8),
        ("observed_demand_qty", pl.Float32),
        ("observed_revenue_net", pl.Float32),
        ("observed_discount_amount", pl.Float32),
        ("avg_selling_price", pl.Float32),
        ("promo_flag", pl.Boolean),
        ("holiday_flag", pl.Boolean),
        ("activity_flag", pl.Boolean),
        ("observed_stockout_flag", pl.Boolean),
        ("observed_stockout_available", pl.Boolean),
        ("observed_stockout_intensity", pl.Float32),
        ("location_open_flag", pl.Boolean),
        ("day_complete_flag", pl.Boolean),
        ("missing_sales_flag", pl.Boolean),
        ("calendar_weekday_name", pl.Utf8),
        ("calendar_day_of_week", pl.Int8),
        ("calendar_month", pl.Int8),
        ("calendar_year", pl.Int16),
        ("calendar_week_key", pl.Int32),
        ("event_name_1", pl.Utf8),
        ("event_type_1", pl.Utf8),
        ("event_name_2", pl.Utf8),
        ("event_type_2", pl.Utf8),
        ("weather_precipitation", pl.Float32),
        ("weather_temperature", pl.Float32),
        ("weather_humidity", pl.Float32),
        ("weather_wind_level", pl.Float32),
        ("anomaly_flag", pl.Boolean),
        ("silver_run_id", pl.Utf8),
    ]
    ),
)

CANONICAL_COLUMNS: list[str] = list(CANONICAL_SCHEMA.keys())


@dataclass(frozen=True)
class CanonicalDatasetContract:
    """Canonical contract for daily silver-like demand datasets."""

    grain: str = "dt x location_id x product_id"
    cadence: str = "daily"
    target: str = "observed_demand_qty"


def build_series_id_expr(location_col: str, product_col: str) -> pl.Expr:
    """Build a stable series identifier from location and product identifiers."""

    return pl.concat_str(
        [pl.col(location_col).cast(pl.Utf8), pl.col(product_col).cast(pl.Utf8)],
        separator="__",
    )


def align_lazy_frame_to_canonical_schema(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Cast and reorder a lazy frame to the canonical global dataset schema."""

    available_columns = set(frame.collect_schema().names())
    projected_columns: list[pl.Expr] = []
    for column_name, dtype in CANONICAL_SCHEMA.items():
        if column_name in available_columns:
            projected_columns.append(pl.col(column_name).cast(dtype, strict=False).alias(column_name))
        else:
            projected_columns.append(pl.lit(None).cast(dtype).alias(column_name))
    return frame.select(projected_columns)

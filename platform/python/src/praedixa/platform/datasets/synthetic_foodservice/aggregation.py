"""Aggregate daily or ticket-level data into coarser/finer time granularities.

Produces separate DataFrames for: 15min, hourly, half-day, weekly, monthly.
"""

from __future__ import annotations

import pandas as pd

_NUMERIC_AGG_COLUMNS: tuple[str, ...] = (
    "observed_demand_qty", "observed_revenue_net", "observed_discount_amount",
)


def aggregate_daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows to ISO-week level."""

    if daily.empty:
        return daily
    df = daily.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df["week_start"] = df["dt"].dt.to_period("W").dt.start_time.dt.date
    return _group_aggregate(df, group_keys=["dataset_source", "location_id", "product_id", "week_start"])


def aggregate_daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows to calendar-month level."""

    if daily.empty:
        return daily
    df = daily.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df["month_start"] = df["dt"].dt.to_period("M").dt.start_time.dt.date
    return _group_aggregate(df, group_keys=["dataset_source", "location_id", "product_id", "month_start"])


def aggregate_tickets_to_15min(lines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticket lines to 15-minute slot level."""

    if lines.empty:
        return lines
    return _group_aggregate(
        lines,
        group_keys=["dataset_source", "location_id", "product_id", "dt", "time_slot"],
        qty_col="quantity",
        revenue_col="line_amount_ttc",
        discount_col="discount_amount",
    )


def aggregate_tickets_to_hourly(lines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticket lines to hourly level."""

    if lines.empty:
        return lines
    df = lines.copy()
    df["hour"] = df["time_slot"].astype(int) // 4
    return _group_aggregate(
        df,
        group_keys=["dataset_source", "location_id", "product_id", "dt", "hour"],
        qty_col="quantity",
        revenue_col="line_amount_ttc",
        discount_col="discount_amount",
    )


def aggregate_tickets_to_halfday(lines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticket lines to half-day level (AM/PM, split at 14:00)."""

    if lines.empty:
        return lines
    df = lines.copy()
    time_slot = pd.to_numeric(df["time_slot"], errors="coerce").fillna(0).astype(int)
    df["halfday"] = pd.Series("PM", index=df.index)
    df.loc[time_slot < 56, "halfday"] = "AM"
    return _group_aggregate(
        df,
        group_keys=["dataset_source", "location_id", "product_id", "dt", "halfday"],
        qty_col="quantity",
        revenue_col="line_amount_ttc",
        discount_col="discount_amount",
    )


def _group_aggregate(
    df: pd.DataFrame,
    *,
    group_keys: list[str],
    qty_col: str = "observed_demand_qty",
    revenue_col: str = "observed_revenue_net",
    discount_col: str = "observed_discount_amount",
) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {}
    if qty_col in df.columns:
        agg_spec[qty_col] = (qty_col, "sum")
    if revenue_col in df.columns:
        agg_spec[revenue_col] = (revenue_col, "sum")
    if discount_col in df.columns:
        agg_spec[discount_col] = (discount_col, "sum")
    if "censor_flag" in df.columns:
        agg_spec["censor_flag"] = ("censor_flag", "max")
    if "promo_flag" in df.columns:
        agg_spec["promo_flag"] = ("promo_flag", "max")

    numeric_df = df.copy()
    for col in (qty_col, revenue_col, discount_col):
        if col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce").fillna(0.0)

    result = numeric_df.groupby(group_keys, as_index=False).agg(**agg_spec)
    return result

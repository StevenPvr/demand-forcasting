"""POS data quality corruption layer.

Applies MCAR/MAR missing values, duplicate tickets, timestamp errors,
price inconsistencies, promotion tag drops, and ID churn based on per-site
quality profiles (high / medium / medium_low / low).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityProfile:
    """Corruption rates for one data_quality_level."""

    missing_rate_mcar: float
    missing_rate_mar: float
    duplicate_ticket_rate: float
    timestamp_error_rate: float
    price_inconsistency_rate: float
    promotion_tag_missing_rate: float
    id_churn_rate: float
    partial_export_day_rate: float
    cancelled_lines_remain_rate: float
    line_total_mismatch_rate: float


QUALITY_PROFILES: dict[str, QualityProfile] = {
    "high": QualityProfile(
        0.001, 0.002, 0.0002, 0.002, 0.001, 0.02, 0.01, 0.005, 0.005, 0.003
    ),
    "medium": QualityProfile(
        0.008, 0.012, 0.002, 0.015, 0.008, 0.08, 0.04, 0.02, 0.02, 0.015
    ),
    "medium_low": QualityProfile(
        0.02, 0.05, 0.006, 0.04, 0.02, 0.15, 0.12, 0.06, 0.05, 0.04
    ),
    "low": QualityProfile(0.06, 0.12, 0.02, 0.12, 0.08, 0.35, 0.25, 0.18, 0.12, 0.12),
}


def assign_quality_levels(
    locations: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Series:
    """Assign a data_quality_level to each location."""

    levels: tuple[str, ...] = ("high", "medium", "medium_low", "low")
    weights = np.array([0.20, 0.35, 0.30, 0.15])
    indices = rng.choice(len(levels), size=len(locations), p=weights)
    selected_levels: list[str] = [levels[int(index)] for index in indices.tolist()]
    return pd.Series(selected_levels, index=locations.index, name="data_quality_level")


def corrupt_daily_frame(
    daily: pd.DataFrame,
    quality_levels: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply corruption to a model-facing daily frame based on site quality."""

    if daily.empty:
        return daily
    df = daily.copy()
    merged = df.merge(
        quality_levels[["location_id", "data_quality_level"]],
        on="location_id",
        how="left",
    )
    merged["data_quality_level"] = merged["data_quality_level"].fillna("medium")

    for level, profile in QUALITY_PROFILES.items():
        mask = merged["data_quality_level"] == level
        if not mask.any():
            continue
        idx = merged.index[mask]
        df = _apply_mcar_missing(df, idx, profile.missing_rate_mcar, rng)
        df = _apply_promo_drop(df, idx, profile.promotion_tag_missing_rate, rng)
        df = _apply_price_noise(df, idx, profile.price_inconsistency_rate, rng)
        df = _apply_partial_export(df, idx, profile.partial_export_day_rate, rng)

    return df


def corrupt_tickets_frame(
    tickets: pd.DataFrame,
    quality_levels: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply ticket-level corruption (duplicates, timestamp errors)."""

    if tickets.empty:
        return tickets
    df = tickets.copy()
    merged = df.merge(
        quality_levels[["location_id", "data_quality_level"]],
        on="location_id",
        how="left",
    )
    merged["data_quality_level"] = merged["data_quality_level"].fillna("medium")

    for level, profile in QUALITY_PROFILES.items():
        mask = merged["data_quality_level"] == level
        if not mask.any():
            continue
        idx = merged.index[mask]
        df = _apply_duplicate_tickets(df, idx, profile.duplicate_ticket_rate, rng)
        df = _apply_timestamp_error(df, idx, profile.timestamp_error_rate, rng)

    return df


def _apply_mcar_missing(
    df: pd.DataFrame,
    idx: pd.Index,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Set observed sales labels to missing and make rows non-trainable."""

    hit = rng.random(len(idx)) < rate
    target = idx[hit]
    if len(target) > 0 and "observed_demand_qty" in df.columns:
        df = _mark_unusable_observation(df, target, "missing_observed_sales")
    return df


def _apply_promo_drop(
    df: pd.DataFrame,
    idx: pd.Index,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Drop promo_flag for some rows."""

    promo_rows = idx[df.loc[idx, "promo_flag"].astype(bool)]
    if len(promo_rows) == 0:
        return df
    hit = rng.random(len(promo_rows)) < rate
    df.loc[promo_rows[hit], "promo_flag"] = False
    return df


def _apply_price_noise(
    df: pd.DataFrame,
    idx: pd.Index,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject small price inconsistencies."""

    hit = rng.random(len(idx)) < rate
    target = idx[hit]
    if len(target) > 0 and "observed_revenue_net" in df.columns:
        noise = rng.normal(1.0, 0.05, size=len(target))
        df.loc[target, "observed_revenue_net"] = (
            df.loc[target, "observed_revenue_net"].astype(float) * noise
        ).round(2)
    return df


def _apply_partial_export(
    df: pd.DataFrame,
    idx: pd.Index,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Mark whole site-days as incomplete exports."""

    if "dt" not in df.columns or "location_id" not in df.columns:
        return df
    site_days = df.loc[idx, ["location_id", "dt"]].drop_duplicates()
    if site_days.empty:
        return df
    hit = rng.random(len(site_days)) < rate
    selected = site_days.loc[hit]
    if selected.empty:
        return df
    key = pd.MultiIndex.from_frame(df[["location_id", "dt"]])
    selected_key = pd.MultiIndex.from_frame(selected[["location_id", "dt"]])
    target = df.index[key.isin(selected_key)]
    if len(target) > 0:
        df = _mark_unusable_observation(df, target, "partial_export_day")
    return df


def _mark_unusable_observation(
    df: pd.DataFrame,
    target: pd.Index,
    reason: str,
) -> pd.DataFrame:
    df.loc[target, "observed_demand_qty"] = np.nan
    if "observed_revenue_net" in df.columns:
        df.loc[target, "observed_revenue_net"] = np.nan
    if "observed_discount_amount" in df.columns:
        df.loc[target, "observed_discount_amount"] = np.nan
    if "day_complete_flag" in df.columns:
        df.loc[target, "day_complete_flag"] = False
    if "usable_for_training_flag" in df.columns:
        df.loc[target, "usable_for_training_flag"] = False
    if "label_quality_score" in df.columns:
        df.loc[target, "label_quality_score"] = 0.0
    if "target_source" in df.columns:
        df.loc[target, "target_source"] = reason
    return df


def _apply_duplicate_tickets(
    df: pd.DataFrame,
    idx: pd.Index,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Duplicate a small fraction of tickets."""

    hit = rng.random(len(idx)) < rate
    target = idx[hit]
    if len(target) == 0:
        return df
    duplicates = df.loc[target].copy()
    duplicates["ticket_id"] = duplicates["ticket_id"].astype(str) + "_dup"
    return pd.concat([df, duplicates], ignore_index=True)


def _apply_timestamp_error(
    df: pd.DataFrame,
    idx: pd.Index,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shift time_slot by a random amount for some tickets."""

    if "time_slot" not in df.columns:
        return df
    hit = rng.random(len(idx)) < rate
    target = idx[hit]
    if len(target) > 0:
        shift = rng.integers(-8, 9, size=len(target))
        df.loc[target, "time_slot"] = np.clip(
            df.loc[target, "time_slot"].astype(int) + shift,
            0,
            95,
        )
    return df

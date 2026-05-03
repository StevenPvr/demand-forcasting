"""Statistical validation for generated synthetic data (Gate 2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationReport:
    """Summary of statistical checks on generated data."""

    total_rows: int
    series_count: int
    zero_rate: float
    censored_rate: float
    promo_rate: float
    mean_demand: float
    median_demand: float
    cv_demand: float
    autocorr_lag1: float
    autocorr_lag7: float
    dow_lift_range: tuple[float, float]
    source_stats: dict[str, dict[str, float]]
    warnings: list[str]


def validate_generated_dataset(daily: pd.DataFrame) -> ValidationReport:
    """Run statistical plausibility checks on the generated data."""
    warnings: list[str] = []
    qty = pd.to_numeric(daily["observed_demand_qty"], errors="coerce").fillna(0.0)
    total = len(daily)
    zero_rate = float((qty == 0).mean()) if total > 0 else 0.0
    censored_rate = (
        float(daily["censor_flag"].astype(bool).mean()) if total > 0 else 0.0
    )
    promo_rate = float(daily["promo_flag"].astype(bool).mean()) if total > 0 else 0.0
    mean_d = float(qty.mean())
    median_d = float(qty.median())
    std_d = float(qty.std()) if total > 1 else 0.0
    cv = std_d / max(mean_d, 1e-6)

    ac1 = _per_series_autocorr(daily, qty, 1)
    ac7 = _per_series_autocorr(daily, qty, 7)
    dow_range = _dow_lift_range(daily, qty)

    if zero_rate > 0.70:
        warnings.append(f"Very high zero rate: {zero_rate:.2%}")
    if censored_rate > 0.50:
        warnings.append(f"Censored rate suspiciously high: {censored_rate:.2%}")
    if cv < 0.5:
        warnings.append(f"Low coefficient of variation: {cv:.2f}")
    usable = daily["usable_for_training_flag"].fillna(False).astype(bool)
    missing_label = pd.to_numeric(daily["observed_demand_qty"], errors="coerce").isna()
    incomplete = ~daily["day_complete_flag"].fillna(False).astype(bool)
    if bool((usable & missing_label).any()):
        warnings.append("Training-usable rows with missing observed_demand_qty")
    if bool((usable & incomplete).any()):
        warnings.append("Training-usable rows marked day_complete_flag=false")

    source_stats = _per_source_stats(daily, qty)

    _log_report(
        total, zero_rate, censored_rate, promo_rate, mean_d, cv, ac1, ac7, warnings
    )

    return ValidationReport(
        total_rows=total,
        series_count=int(daily["series_id"].nunique()) if total > 0 else 0,
        zero_rate=zero_rate,
        censored_rate=censored_rate,
        promo_rate=promo_rate,
        mean_demand=mean_d,
        median_demand=median_d,
        cv_demand=cv,
        autocorr_lag1=ac1,
        autocorr_lag7=ac7,
        dow_lift_range=dow_range,
        source_stats=source_stats,
        warnings=warnings,
    )


def _per_series_autocorr(daily: pd.DataFrame, qty: pd.Series, lag: int) -> float:
    if len(qty) < lag + 10 or "series_id" not in daily.columns:
        return 0.0
    df = daily.loc[:, ["series_id", "dt"]].copy()
    df["_qty"] = qty
    df["dt"] = pd.to_datetime(df["dt"])
    values: list[float] = []
    for _, group in df.sort_values(["series_id", "dt"]).groupby("series_id"):
        series = group["_qty"]
        if len(series) < lag + 10 or series.nunique(dropna=True) <= 1:
            continue
        corr = series.autocorr(lag=lag)
        if pd.notna(corr):
            values.append(float(corr))
    if not values:
        return 0.0
    return float(pd.Series(values).median())


def _dow_lift_range(daily: pd.DataFrame, qty: pd.Series) -> tuple[float, float]:
    if daily.empty:
        return (1.0, 1.0)
    df = daily.copy()
    df["_qty"] = qty
    if "calendar_day_of_week" not in df.columns:
        return (1.0, 1.0)
    dow_means = df.groupby("calendar_day_of_week")["_qty"].mean()
    global_mean = max(float(qty.mean()), 1e-6)
    lifts = dow_means / global_mean
    return (float(lifts.min()), float(lifts.max()))


def _per_source_stats(
    daily: pd.DataFrame, qty: pd.Series
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    df = daily.copy()
    df["_qty"] = qty
    for source, grp in df.groupby("dataset_source"):
        sq = grp["_qty"]
        result[str(source)] = {
            "rows": float(len(grp)),
            "mean": float(sq.mean()),
            "zero_rate": float((sq == 0).mean()),
            "censored_rate": float(grp["censor_flag"].astype(bool).mean()),
        }
    return result


def _log_report(
    total: int,
    zero_rate: float,
    censored_rate: float,
    promo_rate: float,
    mean_d: float,
    cv: float,
    ac1: float,
    ac7: float,
    warnings: list[str],
) -> None:
    LOGGER.info("=== Validation report ===")
    LOGGER.info(
        "  rows=%d zero=%.2f%% censored=%.2f%% promo=%.2f%%",
        total,
        100 * zero_rate,
        100 * censored_rate,
        100 * promo_rate,
    )
    LOGGER.info(
        "  mean=%.2f cv=%.2f autocorr(1)=%.3f autocorr(7)=%.3f", mean_d, cv, ac1, ac7
    )
    for w in warnings:
        LOGGER.warning("  VALIDATION WARNING: %s", w)

from __future__ import annotations

import numpy as np
import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.synthetic_calendar import (
    SyntheticCalendar,
)
from praedixa.platform.datasets.synthetic_foodservice.taxonomy import (
    SYNTHETIC_BAKERY_SOURCE,
    SYNTHETIC_QSR_SOURCE,
    SYNTHETIC_RESTAURANT_SOURCE,
    VERTICAL_SPEC_BY_SOURCE,
)


def scaled_product_base(
    raw_base: np.ndarray,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    normalized = raw_base / max(float(np.nanmedian(raw_base)), 1e-6)
    scaled = rng.uniform(low, high, size=len(raw_base)) * normalized
    return np.clip(scaled, low * 0.25, high * 1.8)


def dow_lift(source: str, day_of_week: np.ndarray) -> np.ndarray:
    if source == SYNTHETIC_BAKERY_SOURCE:
        lifts = np.array([1.08, 0.96, 0.98, 1.02, 1.16, 1.38, 0.74])
    elif source == SYNTHETIC_QSR_SOURCE:
        lifts = np.array([0.90, 0.94, 1.00, 1.06, 1.26, 1.34, 1.10])
    else:
        lifts = np.array([0.72, 0.78, 0.88, 1.02, 1.34, 1.48, 1.12])
    return lifts[day_of_week.astype(int)]


def seasonal_lift(source: str, month: np.ndarray) -> np.ndarray:
    annual = np.sin(2.0 * np.pi * (month.astype(float) - 1.0) / 12.0)
    if source == SYNTHETIC_BAKERY_SOURCE:
        return 1.0 + 0.10 * annual + np.where(np.isin(month, [1, 12]), 0.12, 0.0)
    if source == SYNTHETIC_QSR_SOURCE:
        return 1.0 + 0.06 * annual + np.where(np.isin(month, [7, 8]), 0.08, 0.0)
    return 1.0 + 0.08 * annual + np.where(np.isin(month, [6, 7, 8, 12]), 0.10, 0.0)


def weather_lift(source: str, weather: pd.DataFrame) -> np.ndarray:
    temperature = weather["weather_temperature"].to_numpy(dtype=float)
    precipitation = weather["weather_precipitation"].to_numpy(dtype=float)
    if source == SYNTHETIC_BAKERY_SOURCE:
        return np.clip(
            1.0 - 0.006 * precipitation + 0.004 * np.maximum(12.0 - temperature, 0.0),
            0.78,
            1.22,
        )
    if source == SYNTHETIC_QSR_SOURCE:
        return np.clip(
            1.0 + 0.005 * np.maximum(temperature - 18.0, 0.0) - 0.003 * precipitation,
            0.82,
            1.24,
        )
    return np.clip(
        1.0 - 0.004 * precipitation + 0.003 * np.maximum(temperature - 16.0, 0.0),
        0.80,
        1.20,
    )


def holiday_lift(source: str) -> float:
    if source == SYNTHETIC_BAKERY_SOURCE:
        return 0.70
    if source == SYNTHETIC_QSR_SOURCE:
        return 0.92
    return 0.82


def negative_binomial(*, mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dispersion = 7.0
    probability = dispersion / (dispersion + np.maximum(mu, 0.01))
    return rng.negative_binomial(dispersion, probability).astype(float)


def active_matrix(
    assortment: pd.DataFrame,
    calendar: SyntheticCalendar,
) -> np.ndarray:
    dates = pd.to_datetime(calendar.dates)
    starts = pd.to_datetime(assortment["active_start"]).to_numpy(dtype="datetime64[ns]")
    ends = pd.to_datetime(assortment["active_end"]).to_numpy(dtype="datetime64[ns]")
    return (dates.to_numpy()[None, :] >= starts[:, None]) & (
        dates.to_numpy()[None, :] <= ends[:, None]
    )


def closed_day_mask(
    source: str,
    calendar: SyntheticCalendar,
    rng: np.random.Generator,
) -> np.ndarray:
    base_probability = 0.015 if source == SYNTHETIC_QSR_SOURCE else 0.035
    regular_closed = rng.random(len(calendar.dates)) < base_probability
    sunday_bakery = (
        (source == SYNTHETIC_BAKERY_SOURCE)
        & (calendar.day_of_week == 6)
        & (rng.random(len(calendar.dates)) < 0.34)
    )
    return regular_closed | sunday_bakery


def promo_matrix(
    source: str,
    product_count: int,
    date_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = VERTICAL_SPEC_BY_SOURCE[source].promo_probability
    raw = rng.random((product_count, date_count)) < probability
    burst = rng.random((product_count, 1)) < 0.22
    return raw | (
        burst & (rng.random((product_count, date_count)) < probability * 0.45)
    )


def zero_inflation_mask(
    source: str,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    probability = VERTICAL_SPEC_BY_SOURCE[source].zero_inflation
    return rng.random(shape) < probability


def available_quantity(
    source: str,
    mu: np.ndarray,
    latent: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    tightness = VERTICAL_SPEC_BY_SOURCE[source].stockout_tightness
    planning_noise = rng.normal(loc=1.0, scale=0.22 + tightness, size=mu.shape)
    if source == SYNTHETIC_BAKERY_SOURCE:
        production = mu * planning_noise
    else:
        production = np.maximum(
            mu * planning_noise, latent * rng.uniform(0.75, 1.20, size=mu.shape)
        )
    return np.maximum(np.floor(production), 0.0)


def capacity_factor(
    source: str,
    date_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    saturation_probability = 0.09 if source == SYNTHETIC_QSR_SOURCE else 0.055
    saturation = rng.random(date_count) < saturation_probability
    constrained = rng.uniform(0.72, 0.94, size=date_count)
    return np.where(saturation, constrained, 1.0)


def waste_multiplier(source: str) -> float:
    return 0.72 if source == SYNTHETIC_BAKERY_SOURCE else 0.18


def day_complete_mask(date_count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(date_count) > 0.012


def staffing_hours(latent: np.ndarray, source: str) -> np.ndarray:
    if source == SYNTHETIC_QSR_SOURCE:
        return 18.0 + latent * 0.045
    if source == SYNTHETIC_BAKERY_SOURCE:
        return 12.0 + latent * 0.055
    if source == SYNTHETIC_RESTAURANT_SOURCE:
        return 20.0 + latent * 0.075
    raise ValueError(f"Unsupported synthetic source: {source}")

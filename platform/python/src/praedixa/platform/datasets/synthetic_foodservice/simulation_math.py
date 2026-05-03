from __future__ import annotations

import numpy as np
import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.synthetic_calendar import (
    SyntheticCalendar,
)
from praedixa.platform.datasets.synthetic_foodservice.taxonomy import (
    SYNTHETIC_BAKERY_SOURCE,
    SYNTHETIC_BRASSERIE_SOURCE,
    SYNTHETIC_COFFEE_SHOP_SOURCE,
    SYNTHETIC_DARK_KITCHEN_SOURCE,
    SYNTHETIC_QSR_SOURCE,
    SYNTHETIC_RESTAURANT_SOURCE,
    SYNTHETIC_SANDWICH_SHOP_SOURCE,
    SYNTHETIC_SEASONAL_TOURIST_SOURCE,
    SYNTHETIC_SNACK_SOURCE,
    VERTICAL_SPEC_BY_SOURCE,
)

# ---------- Day-of-week profiles (Mon=0 .. Sun=6) ----------

_DOW_PROFILES: dict[str, np.ndarray] = {
    SYNTHETIC_QSR_SOURCE: np.array([0.90, 0.94, 1.00, 1.06, 1.26, 1.34, 1.10]),
    SYNTHETIC_BAKERY_SOURCE: np.array([1.08, 0.96, 0.98, 1.02, 1.16, 1.38, 0.74]),
    SYNTHETIC_RESTAURANT_SOURCE: np.array([0.72, 0.78, 0.88, 1.02, 1.34, 1.48, 1.12]),
    SYNTHETIC_BRASSERIE_SOURCE: np.array([0.82, 0.86, 0.92, 1.04, 1.28, 1.38, 1.04]),
    SYNTHETIC_COFFEE_SHOP_SOURCE: np.array([1.10, 1.08, 1.06, 1.04, 1.02, 0.78, 0.62]),
    SYNTHETIC_SNACK_SOURCE: np.array([0.85, 0.88, 0.92, 1.00, 1.22, 1.36, 1.18]),
    SYNTHETIC_DARK_KITCHEN_SOURCE: np.array([0.82, 0.86, 0.90, 1.02, 1.22, 1.42, 1.30]),
    SYNTHETIC_SANDWICH_SHOP_SOURCE: np.array(
        [1.12, 1.08, 1.06, 1.04, 1.02, 0.72, 0.48]
    ),
    SYNTHETIC_SEASONAL_TOURIST_SOURCE: np.array(
        [0.70, 0.72, 0.80, 0.95, 1.30, 1.52, 1.40]
    ),
}

# ---------- Holiday multipliers ----------

_HOLIDAY_LIFT: dict[str, float] = {
    SYNTHETIC_QSR_SOURCE: 0.92,
    SYNTHETIC_BAKERY_SOURCE: 0.70,
    SYNTHETIC_RESTAURANT_SOURCE: 0.82,
    SYNTHETIC_BRASSERIE_SOURCE: 0.80,
    SYNTHETIC_COFFEE_SHOP_SOURCE: 0.65,
    SYNTHETIC_SNACK_SOURCE: 0.78,
    SYNTHETIC_DARK_KITCHEN_SOURCE: 0.88,
    SYNTHETIC_SANDWICH_SHOP_SOURCE: 0.60,
    SYNTHETIC_SEASONAL_TOURIST_SOURCE: 1.15,
}

# ---------- Waste multipliers ----------

_WASTE_MULTIPLIER: dict[str, float] = {
    SYNTHETIC_QSR_SOURCE: 0.18,
    SYNTHETIC_BAKERY_SOURCE: 0.72,
    SYNTHETIC_RESTAURANT_SOURCE: 0.18,
    SYNTHETIC_BRASSERIE_SOURCE: 0.20,
    SYNTHETIC_COFFEE_SHOP_SOURCE: 0.35,
    SYNTHETIC_SNACK_SOURCE: 0.22,
    SYNTHETIC_DARK_KITCHEN_SOURCE: 0.15,
    SYNTHETIC_SANDWICH_SHOP_SOURCE: 0.40,
    SYNTHETIC_SEASONAL_TOURIST_SOURCE: 0.25,
}

# ---------- Staffing base hours + per-unit cost ----------

_STAFFING_BASE: dict[str, float] = {
    SYNTHETIC_QSR_SOURCE: 18.0,
    SYNTHETIC_BAKERY_SOURCE: 12.0,
    SYNTHETIC_RESTAURANT_SOURCE: 20.0,
    SYNTHETIC_BRASSERIE_SOURCE: 18.0,
    SYNTHETIC_COFFEE_SHOP_SOURCE: 10.0,
    SYNTHETIC_SNACK_SOURCE: 14.0,
    SYNTHETIC_DARK_KITCHEN_SOURCE: 16.0,
    SYNTHETIC_SANDWICH_SHOP_SOURCE: 10.0,
    SYNTHETIC_SEASONAL_TOURIST_SOURCE: 22.0,
}

_STAFFING_PER_UNIT: dict[str, float] = {
    SYNTHETIC_QSR_SOURCE: 0.045,
    SYNTHETIC_BAKERY_SOURCE: 0.055,
    SYNTHETIC_RESTAURANT_SOURCE: 0.075,
    SYNTHETIC_BRASSERIE_SOURCE: 0.065,
    SYNTHETIC_COFFEE_SHOP_SOURCE: 0.035,
    SYNTHETIC_SNACK_SOURCE: 0.050,
    SYNTHETIC_DARK_KITCHEN_SOURCE: 0.040,
    SYNTHETIC_SANDWICH_SHOP_SOURCE: 0.038,
    SYNTHETIC_SEASONAL_TOURIST_SOURCE: 0.070,
}

# ---------- Closed-day base probabilities ----------

_CLOSED_DAY_PROBABILITY: dict[str, float] = {
    SYNTHETIC_QSR_SOURCE: 0.015,
    SYNTHETIC_BAKERY_SOURCE: 0.035,
    SYNTHETIC_RESTAURANT_SOURCE: 0.035,
    SYNTHETIC_BRASSERIE_SOURCE: 0.030,
    SYNTHETIC_COFFEE_SHOP_SOURCE: 0.020,
    SYNTHETIC_SNACK_SOURCE: 0.025,
    SYNTHETIC_DARK_KITCHEN_SOURCE: 0.012,
    SYNTHETIC_SANDWICH_SHOP_SOURCE: 0.040,
    SYNTHETIC_SEASONAL_TOURIST_SOURCE: 0.045,
}

# ---------- Capacity saturation probabilities ----------

_SATURATION_PROBABILITY: dict[str, float] = {
    SYNTHETIC_QSR_SOURCE: 0.090,
    SYNTHETIC_BAKERY_SOURCE: 0.055,
    SYNTHETIC_RESTAURANT_SOURCE: 0.055,
    SYNTHETIC_BRASSERIE_SOURCE: 0.060,
    SYNTHETIC_COFFEE_SHOP_SOURCE: 0.045,
    SYNTHETIC_SNACK_SOURCE: 0.070,
    SYNTHETIC_DARK_KITCHEN_SOURCE: 0.080,
    SYNTHETIC_SANDWICH_SHOP_SOURCE: 0.050,
    SYNTHETIC_SEASONAL_TOURIST_SOURCE: 0.100,
}

# ---------- Sunday bakery-like closure sources ----------

_SUNDAY_CLOSURE_SOURCES: frozenset[str] = frozenset(
    {
        SYNTHETIC_BAKERY_SOURCE,
        SYNTHETIC_SANDWICH_SHOP_SOURCE,
    }
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
    lifts = _DOW_PROFILES.get(source, _DOW_PROFILES[SYNTHETIC_RESTAURANT_SOURCE])
    return lifts[day_of_week.astype(int)]


def seasonal_lift(source: str, month: np.ndarray) -> np.ndarray:
    annual = np.cos(2.0 * np.pi * (month.astype(float) - 7.0) / 12.0)
    base = 1.0 + _seasonal_amplitude(source) * annual
    return base + _seasonal_bonus(source, month)


def _seasonal_amplitude(source: str) -> float:
    amplitudes: dict[str, float] = {
        SYNTHETIC_BAKERY_SOURCE: -0.04,
        SYNTHETIC_QSR_SOURCE: 0.06,
        SYNTHETIC_RESTAURANT_SOURCE: 0.08,
        SYNTHETIC_BRASSERIE_SOURCE: 0.08,
        SYNTHETIC_COFFEE_SHOP_SOURCE: -0.08,
        SYNTHETIC_SNACK_SOURCE: 0.07,
        SYNTHETIC_DARK_KITCHEN_SOURCE: -0.04,
        SYNTHETIC_SANDWICH_SHOP_SOURCE: 0.08,
        SYNTHETIC_SEASONAL_TOURIST_SOURCE: 0.35,
    }
    return amplitudes.get(source, 0.08)


def _seasonal_bonus(source: str, month: np.ndarray) -> np.ndarray:
    bonus_months: dict[str, tuple[tuple[int, ...], float]] = {
        SYNTHETIC_BAKERY_SOURCE: ((1, 12), 0.12),
        SYNTHETIC_QSR_SOURCE: ((7, 8), 0.08),
        SYNTHETIC_RESTAURANT_SOURCE: ((6, 7, 8, 12), 0.10),
        SYNTHETIC_BRASSERIE_SOURCE: ((6, 7, 8), 0.10),
        SYNTHETIC_COFFEE_SHOP_SOURCE: ((11, 12, 1, 2), 0.08),
        SYNTHETIC_SNACK_SOURCE: ((9, 10, 11), 0.06),
        SYNTHETIC_DARK_KITCHEN_SOURCE: ((11, 12, 1), 0.06),
        SYNTHETIC_SANDWICH_SHOP_SOURCE: ((6, 7), 0.05),
        SYNTHETIC_SEASONAL_TOURIST_SOURCE: ((6, 7, 8), 0.30),
    }
    months_set, bonus = bonus_months.get(source, ((), 0.0))
    return np.where(np.isin(month, list(months_set)), bonus, 0.0)


def weather_lift(source: str, weather: pd.DataFrame) -> np.ndarray:
    temperature = weather["weather_temperature"].to_numpy(dtype=float)
    precipitation = weather["weather_precipitation"].to_numpy(dtype=float)
    params = _weather_params(source)
    effect = (
        1.0
        + params["temp_coeff"] * np.maximum(temperature - params["temp_ref"], 0.0)
        + params["rain_coeff"] * precipitation
    )
    return np.clip(effect, params["clip_low"], params["clip_high"])


def _weather_params(source: str) -> dict[str, float]:
    profiles: dict[str, dict[str, float]] = {
        SYNTHETIC_BAKERY_SOURCE: {
            "temp_coeff": 0.003,
            "temp_ref": 12.0,
            "rain_coeff": -0.006,
            "clip_low": 0.78,
            "clip_high": 1.22,
        },
        SYNTHETIC_QSR_SOURCE: {
            "temp_coeff": 0.005,
            "temp_ref": 18.0,
            "rain_coeff": -0.003,
            "clip_low": 0.82,
            "clip_high": 1.24,
        },
        SYNTHETIC_COFFEE_SHOP_SOURCE: {
            "temp_coeff": -0.004,
            "temp_ref": 12.0,
            "rain_coeff": 0.002,
            "clip_low": 0.80,
            "clip_high": 1.20,
        },
        SYNTHETIC_DARK_KITCHEN_SOURCE: {
            "temp_coeff": 0.002,
            "temp_ref": 20.0,
            "rain_coeff": 0.004,
            "clip_low": 0.85,
            "clip_high": 1.25,
        },
        SYNTHETIC_SEASONAL_TOURIST_SOURCE: {
            "temp_coeff": 0.008,
            "temp_ref": 18.0,
            "rain_coeff": -0.008,
            "clip_low": 0.65,
            "clip_high": 1.40,
        },
    }
    default: dict[str, float] = {
        "temp_coeff": 0.003,
        "temp_ref": 16.0,
        "rain_coeff": -0.004,
        "clip_low": 0.80,
        "clip_high": 1.20,
    }
    return profiles.get(source, default)


def holiday_lift(source: str) -> float:
    return _HOLIDAY_LIFT.get(source, 0.82)


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
    base_probability = _CLOSED_DAY_PROBABILITY.get(source, 0.030)
    regular_closed = rng.random(len(calendar.dates)) < base_probability
    sunday_closed = (
        (source in _SUNDAY_CLOSURE_SOURCES)
        & (calendar.day_of_week == 6)
        & (rng.random(len(calendar.dates)) < 0.34)
    )
    return regular_closed | sunday_closed


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
    rng: np.random.Generator,
) -> np.ndarray:
    tightness = VERTICAL_SPEC_BY_SOURCE[source].stockout_tightness
    planning_noise = rng.normal(loc=1.0, scale=0.22 + tightness, size=mu.shape)
    bakery_like = source in {
        SYNTHETIC_BAKERY_SOURCE,
        SYNTHETIC_SANDWICH_SHOP_SOURCE,
    }
    if bakery_like:
        production = mu * planning_noise
    else:
        production = mu * planning_noise * rng.uniform(0.92, 1.22, size=mu.shape)
    return np.maximum(np.floor(production), 0.0)


def capacity_factor(
    source: str,
    date_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    saturation_probability = _SATURATION_PROBABILITY.get(source, 0.055)
    saturation = rng.random(date_count) < saturation_probability
    constrained = rng.uniform(0.72, 0.94, size=date_count)
    return np.where(saturation, constrained, 1.0)


def waste_multiplier(source: str) -> float:
    return _WASTE_MULTIPLIER.get(source, 0.18)


def day_complete_mask(date_count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(date_count) > 0.012


def staffing_hours(expected_units: np.ndarray, source: str) -> np.ndarray:
    base = _STAFFING_BASE.get(source, 16.0)
    per_unit = _STAFFING_PER_UNIT.get(source, 0.050)
    return base + expected_units * per_unit

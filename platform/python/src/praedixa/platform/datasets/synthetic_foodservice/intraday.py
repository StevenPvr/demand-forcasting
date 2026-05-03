"""Intraday demand distribution profiles for ticket-level simulation.

Each vertical has an intraday profile defining the fraction of daily demand
falling into each 15-minute slot.  Profiles are normalised to sum to 1.0.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

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
)

SLOTS_PER_DAY: int = 96  # 24h × 4 slots/h
SLOT_MINUTES: int = 15


def intraday_profile(source: str) -> np.ndarray:
    """Return a normalised 96-slot intensity vector for the source vertical."""

    builder = _PROFILE_BUILDERS.get(source, _dual_peak_generic)
    raw: np.ndarray = builder()
    return raw / raw.sum()


def slot_to_time_label(slot_index: int) -> str:
    """Return 'HH:MM' for a 0-based 15-min slot index."""

    hours: int = slot_index // 4
    minutes: int = (slot_index % 4) * SLOT_MINUTES
    return f"{hours:02d}:{minutes:02d}"


def _gaussian_peak(slots: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((slots - center) / width) ** 2)


def _bakery_classic() -> np.ndarray:
    slots = np.arange(SLOTS_PER_DAY, dtype=float)
    morning = _gaussian_peak(slots, center=32.0, width=5.0)  # 08:00
    lunch = _gaussian_peak(slots, center=50.0, width=4.0)     # 12:30
    afternoon = _gaussian_peak(slots, center=62.0, width=4.5)  # 15:30
    return morning * 1.6 + lunch * 1.0 + afternoon * 0.4


def _dual_peak_qsr() -> np.ndarray:
    slots = np.arange(SLOTS_PER_DAY, dtype=float)
    lunch = _gaussian_peak(slots, center=49.0, width=5.0)   # 12:15
    dinner = _gaussian_peak(slots, center=76.0, width=5.5)   # 19:00
    late = _gaussian_peak(slots, center=86.0, width=3.0)      # 21:30
    return lunch * 1.3 + dinner * 1.5 + late * 0.3


def _lunch_dinner_gap() -> np.ndarray:
    slots = np.arange(SLOTS_PER_DAY, dtype=float)
    lunch = _gaussian_peak(slots, center=50.0, width=4.5)   # 12:30
    dinner = _gaussian_peak(slots, center=80.0, width=5.0)   # 20:00
    return lunch * 1.2 + dinner * 1.4


def _long_day_apero() -> np.ndarray:
    slots = np.arange(SLOTS_PER_DAY, dtype=float)
    lunch = _gaussian_peak(slots, center=50.0, width=5.0)
    apero = _gaussian_peak(slots, center=72.0, width=4.0)   # 18:00
    dinner = _gaussian_peak(slots, center=80.0, width=4.5)
    return lunch * 1.0 + apero * 0.6 + dinner * 1.2


def _morning_continuous() -> np.ndarray:
    slots = np.arange(SLOTS_PER_DAY, dtype=float)
    morning = _gaussian_peak(slots, center=34.0, width=4.0)  # 08:30
    midmorning = _gaussian_peak(slots, center=42.0, width=4.0)  # 10:30
    lunch = _gaussian_peak(slots, center=50.0, width=4.5)
    return morning * 1.5 + midmorning * 0.8 + lunch * 1.0


def _evening_peak() -> np.ndarray:
    slots = np.arange(SLOTS_PER_DAY, dtype=float)
    lunch = _gaussian_peak(slots, center=50.0, width=4.0)
    evening = _gaussian_peak(slots, center=78.0, width=6.0)  # 19:30
    late = _gaussian_peak(slots, center=88.0, width=3.5)
    return lunch * 0.7 + evening * 1.5 + late * 0.5


def _evening_platform_peak() -> np.ndarray:
    slots = np.arange(SLOTS_PER_DAY, dtype=float)
    lunch = _gaussian_peak(slots, center=49.0, width=4.5)
    evening = _gaussian_peak(slots, center=78.0, width=5.5)
    late = _gaussian_peak(slots, center=87.0, width=4.0)
    return lunch * 0.8 + evening * 1.6 + late * 0.6


def _lunch_dominant() -> np.ndarray:
    slots = np.arange(SLOTS_PER_DAY, dtype=float)
    lunch = _gaussian_peak(slots, center=49.0, width=4.0)
    afternoon = _gaussian_peak(slots, center=58.0, width=3.5)
    return lunch * 2.0 + afternoon * 0.3


def _dual_peak_generic() -> np.ndarray:
    slots = np.arange(SLOTS_PER_DAY, dtype=float)
    lunch = _gaussian_peak(slots, center=50.0, width=5.0)
    dinner = _gaussian_peak(slots, center=78.0, width=5.5)
    return lunch * 1.0 + dinner * 1.2


_PROFILE_BUILDERS: dict[str, Callable[[], np.ndarray]] = {
    SYNTHETIC_QSR_SOURCE: _dual_peak_qsr,
    SYNTHETIC_BAKERY_SOURCE: _bakery_classic,
    SYNTHETIC_RESTAURANT_SOURCE: _lunch_dinner_gap,
    SYNTHETIC_BRASSERIE_SOURCE: _long_day_apero,
    SYNTHETIC_COFFEE_SHOP_SOURCE: _morning_continuous,
    SYNTHETIC_SNACK_SOURCE: _evening_peak,
    SYNTHETIC_DARK_KITCHEN_SOURCE: _evening_platform_peak,
    SYNTHETIC_SANDWICH_SHOP_SOURCE: _lunch_dominant,
    SYNTHETIC_SEASONAL_TOURIST_SOURCE: _dual_peak_generic,
}

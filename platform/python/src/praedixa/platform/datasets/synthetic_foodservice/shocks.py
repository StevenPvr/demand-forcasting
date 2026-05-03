"""Extreme shock injection from the scenario catalog.

Applies time-bounded demand multipliers on the daily latent mean matrix
before NB draws.  Each shock type is a simple multiplicative or additive
modifier with configurable duration and intensity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from praedixa.platform.datasets.synthetic_foodservice.synthetic_calendar import (
    SyntheticCalendar,
)


@dataclass(frozen=True)
class ShockEvent:
    """One injected shock episode on a specific site."""

    shock_id: str
    category: str
    start_day_index: int
    end_day_index: int
    intensity: float
    recovery_halflife_days: float


@dataclass(frozen=True)
class ShockTemplate:
    """Typed template used to sample one shock event."""

    shock_id: str
    category: str
    duration_days: tuple[int, int]
    intensity_range: tuple[float, float]
    recovery_halflife_days: float


def generate_shock_schedule(
    *,
    calendar: SyntheticCalendar,
    site_count: int,
    seed: int,
    shocks_per_site_year: float = 2.5,
) -> list[tuple[int, ShockEvent]]:
    """Generate random shock events for all sites across the horizon."""

    rng = np.random.default_rng(seed + 9999)
    n_days = len(calendar.dates)
    years = max(n_days / 365.0, 0.1)
    schedule: list[tuple[int, ShockEvent]] = []

    for site_idx in range(site_count):
        n_shocks = rng.poisson(shocks_per_site_year * years)
        for shock_num in range(n_shocks):
            shock = _random_shock(rng, n_days, site_idx, shock_num)
            schedule.append((site_idx, shock))

    return schedule


def apply_shocks_to_mu(
    mu: np.ndarray,
    site_shocks: list[ShockEvent],
    n_dates: int,
) -> np.ndarray:
    """Multiply latent mean matrix by shock factors."""

    if not site_shocks:
        return mu
    factor = np.ones(n_dates, dtype=float)
    for shock in site_shocks:
        start = max(0, shock.start_day_index)
        end = min(n_dates, shock.end_day_index)
        for d in range(start, end):
            elapsed = d - start
            decay = _decay_factor(elapsed, shock.recovery_halflife_days)
            factor[d] *= 1.0 + shock.intensity * decay
    factor = np.clip(factor, 0.05, 5.0)
    return mu * factor[None, :]


def _decay_factor(elapsed_days: int, halflife: float) -> float:
    if halflife <= 0.0:
        return 1.0
    return float(0.5 ** (elapsed_days / halflife))


_SHOCK_CATALOG: tuple[ShockTemplate, ...] = (
    ShockTemplate("heatwave", "weather", (3, 14), (-0.10, 0.15), 2.0),
    ShockTemplate("continuous_rain", "weather", (2, 7), (-0.25, -0.05), 1.0),
    ShockTemplate("cold_wave", "weather", (4, 12), (-0.15, 0.05), 2.0),
    ShockTemplate("transport_strike", "societal", (1, 5), (-0.35, -0.10), 0.5),
    ShockTemplate("supplier_shutdown", "supply", (2, 21), (-0.30, -0.05), 3.0),
    ShockTemplate("oven_failure", "ops", (1, 3), (-0.50, -0.20), 0.5),
    ShockTemplate("pos_outage", "data", (1, 2), (-0.15, -0.02), 0.0),
    ShockTemplate("staff_absence", "ops", (1, 3), (-0.30, -0.10), 1.0),
    ShockTemplate("local_buzz", "market", (7, 45), (0.05, 0.25), 14.0),
    ShockTemplate("bad_review", "market", (14, 90), (-0.35, -0.05), 30.0),
    ShockTemplate("competitor_open", "market", (30, 120), (-0.20, -0.05), 30.0),
    ShockTemplate("competitor_close", "market", (30, 120), (0.05, 0.25), 45.0),
    ShockTemplate("construction", "local", (30, 180), (-0.40, -0.10), 0.0),
    ShockTemplate("chef_change", "offer", (7, 30), (-0.15, 0.10), 7.0),
    ShockTemplate("menu_reset", "offer", (14, 45), (-0.20, -0.05), 14.0),
    ShockTemplate("inflation_shock", "economics", (30, 90), (-0.15, -0.05), 0.0),
    ShockTemplate("platform_promo", "promo", (3, 14), (0.10, 0.40), 3.0),
    ShockTemplate("sporting_event", "event", (1, 2), (0.15, 0.55), 0.5),
    ShockTemplate("local_festival", "event", (2, 7), (0.10, 0.35), 1.0),
    ShockTemplate("forced_closure", "regulatory", (3, 30), (-0.95, -0.80), 0.0),
    ShockTemplate("pos_migration", "data", (2, 14), (-0.08, -0.02), 0.0),
    ShockTemplate("data_loss", "data", (1, 7), (-0.10, -0.03), 0.0),
)


def _random_shock(
    rng: np.random.Generator,
    n_days: int,
    site_idx: int,
    shock_num: int,
) -> ShockEvent:
    template = _SHOCK_CATALOG[rng.integers(0, len(_SHOCK_CATALOG))]
    dur_lo, dur_hi = template.duration_days
    duration = int(rng.integers(dur_lo, dur_hi + 1))
    int_lo, int_hi = template.intensity_range
    intensity = float(rng.uniform(int_lo, int_hi))
    start = int(rng.integers(0, max(1, n_days - duration)))
    return ShockEvent(
        shock_id=f"{template.shock_id}_{site_idx}_{shock_num}",
        category=template.category,
        start_day_index=start,
        end_day_index=start + duration,
        intensity=intensity,
        recovery_halflife_days=template.recovery_halflife_days,
    )

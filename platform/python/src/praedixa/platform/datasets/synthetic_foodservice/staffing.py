"""Staff schedule generation for capacity and workforce planning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.simulation_math import (
    staffing_hours,
)

STAFF_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "schedule_id",
    "dataset_source",
    "location_id",
    "dt",
    "staff_role",
    "planned_hours",
    "actual_hours",
    "absence_flag",
    "delay_minutes",
    "temporary_worker_flag",
    "productivity_level",
    "headcount_planned",
    "headcount_actual",
)

_ROLES: tuple[str, ...] = ("kitchen", "service", "manager", "cleaning")
_ROLE_WEIGHTS: tuple[float, ...] = (0.35, 0.40, 0.10, 0.15)


def generate_staff_schedules(
    daily: pd.DataFrame,
    _oracle: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Generate daily staff schedules from lagged visible POS demand only."""
    if daily.empty:
        return pd.DataFrame(columns=STAFF_SCHEDULE_COLUMNS)
    rng = np.random.default_rng(seed + 55555)
    merged = _build_planning_frame(daily)
    rows: list[dict[str, object]] = []
    counter = 0
    for _, row in merged.iterrows():
        counter = _add_site_day(rows, row, rng, counter)
    return pd.DataFrame(rows, columns=STAFF_SCHEDULE_COLUMNS)


def _build_planning_frame(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["dt"] = pd.to_datetime(frame["dt"])
    frame["observed_demand_qty"] = pd.to_numeric(
        frame["observed_demand_qty"], errors="coerce"
    ).fillna(0.0)
    grouped = frame.groupby(
        ["dataset_source", "location_id", "dt"],
        as_index=False,
    ).agg(
        observed_sum=("observed_demand_qty", "sum"),
        activity_flag=("activity_flag", "max"),
    )
    grouped = grouped.sort_values(["dataset_source", "location_id", "dt"])
    grouped["planned_units"] = grouped.groupby(
        ["dataset_source", "location_id"],
        group_keys=False,
    )["observed_sum"].apply(_lagged_planning_units)
    grouped["planned_units"] = grouped["planned_units"].fillna(0.0)
    return grouped


def _lagged_planning_units(series: pd.Series) -> pd.Series:
    lagged = series.shift(1)
    return lagged.rolling(window=7, min_periods=1).mean()


def _add_site_day(
    rows: list[dict[str, object]],
    row: pd.Series,
    rng: np.random.Generator,
    counter: int,
) -> int:
    planned_units = float(row["planned_units"])
    source = str(row["dataset_source"])
    needed = float(staffing_hours(np.array([planned_units]), source)[0])
    active = bool(row.get("activity_flag", True))
    for role_idx, role in enumerate(_ROLES):
        counter += 1
        share = _ROLE_WEIGHTS[role_idx]
        planned = round(needed * share, 2) if active else 0.0
        absent = bool(rng.random() < 0.04)
        delay = int(rng.integers(0, 30)) if rng.random() < 0.08 else 0
        temp = bool(rng.random() < 0.06)
        prod = _productivity(rng, absent, temp)
        actual = 0.0 if absent else max(0.0, round((planned - delay / 60.0) * prod, 2))
        hc_plan = 0 if planned <= 0.0 else max(1, int(round(planned / 7.5)))
        rows.append(
            {
                "schedule_id": f"sched_{counter:010d}",
                "dataset_source": source,
                "location_id": row["location_id"],
                "dt": row["dt"],
                "staff_role": role,
                "planned_hours": planned,
                "actual_hours": actual,
                "absence_flag": absent,
                "delay_minutes": delay,
                "temporary_worker_flag": temp,
                "productivity_level": round(prod, 3),
                "headcount_planned": hc_plan,
                "headcount_actual": 0 if absent else hc_plan,
            }
        )
    return counter


def _productivity(rng: np.random.Generator, absent: bool, temp: bool) -> float:
    if absent:
        return 0.0
    base = float(rng.beta(a=8.0, b=2.0))
    if temp:
        base *= 0.75
    return min(base, 1.0)

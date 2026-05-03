from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from typing import Protocol, cast

import numpy as np
import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.config import (
    SyntheticFoodserviceConfig,
)


@dataclass(frozen=True)
class SyntheticCalendar:
    """Date-level exogenous signals used by the simulator."""

    frame: pd.DataFrame
    dates: np.ndarray
    day_of_week: np.ndarray
    month: np.ndarray
    holiday_flag: np.ndarray
    event_type: np.ndarray
    event_name: np.ndarray
    bridge_day_flag: np.ndarray
    school_holiday_zone_a: np.ndarray
    school_holiday_zone_b: np.ndarray
    school_holiday_zone_c: np.ndarray


class WeatherCityRecord(Protocol):
    """Typed view of a pandas ``itertuples`` city row."""

    city_id: str
    latitude: float


def build_synthetic_calendar(config: SyntheticFoodserviceConfig) -> SyntheticCalendar:
    """Build deterministic calendar/event signals for the synthetic horizon."""

    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    frame = pd.DataFrame({"dt": dates.date})
    frame["calendar_day_of_week"] = dates.dayofweek.astype("int16")
    frame["calendar_month"] = dates.month.astype("int16")
    frame["calendar_year"] = dates.year.astype("int16")
    frame["calendar_week_key"] = dates.isocalendar().week.astype("int16").to_numpy()
    frame["calendar_weekday_name"] = dates.day_name()
    frame["holiday_flag"] = [_is_french_public_holiday(day) for day in dates]
    frame["bridge_day_flag"] = _bridge_day_series(dates, frame["holiday_flag"])
    frame["school_holiday_zone_a"] = _school_holiday_series(dates, "A")
    frame["school_holiday_zone_b"] = _school_holiday_series(dates, "B")
    frame["school_holiday_zone_c"] = _school_holiday_series(dates, "C")
    frame["event_type_1"] = _event_type_series(dates, frame["bridge_day_flag"])
    frame["event_name_1"] = [
        _event_name_from_type(event_type) for event_type in frame["event_type_1"]
    ]
    frame["activity_flag"] = True
    return SyntheticCalendar(
        frame=frame,
        dates=frame["dt"].to_numpy(),
        day_of_week=frame["calendar_day_of_week"].to_numpy(dtype=np.int16),
        month=frame["calendar_month"].to_numpy(dtype=np.int16),
        holiday_flag=frame["holiday_flag"].to_numpy(dtype=bool),
        event_type=frame["event_type_1"].to_numpy(dtype=object),
        event_name=frame["event_name_1"].to_numpy(dtype=object),
        bridge_day_flag=frame["bridge_day_flag"].to_numpy(dtype=bool),
        school_holiday_zone_a=frame["school_holiday_zone_a"].to_numpy(dtype=bool),
        school_holiday_zone_b=frame["school_holiday_zone_b"].to_numpy(dtype=bool),
        school_holiday_zone_c=frame["school_holiday_zone_c"].to_numpy(dtype=bool),
    )


def _is_french_public_holiday(timestamp: pd.Timestamp) -> bool:
    date_value = timestamp.date()
    easter = _easter_sunday(timestamp.year)
    moving_days = {
        easter + dt.timedelta(days=1),
        easter + dt.timedelta(days=39),
        easter + dt.timedelta(days=50),
    }
    if date_value in moving_days:
        return True
    fixed_days = {
        (1, 1),
        (5, 1),
        (5, 8),
        (7, 14),
        (8, 15),
        (11, 1),
        (11, 11),
        (12, 25),
    }
    return (int(timestamp.month), int(timestamp.day)) in fixed_days


def _easter_sunday(year: int) -> dt.date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    correction = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * correction) // 451
    month = (h + correction - 7 * m + 114) // 31
    day = ((h + correction - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def _bridge_day_series(
    dates: pd.DatetimeIndex,
    holiday_flag: pd.Series,
) -> list[bool]:
    holidays = set(dates[holiday_flag.astype(bool)].date)
    result: list[bool] = []
    for timestamp in dates:
        date_value = timestamp.date()
        monday_before_tuesday_holiday = (
            timestamp.dayofweek == 0 and date_value + dt.timedelta(days=1) in holidays
        )
        friday_after_thursday_holiday = (
            timestamp.dayofweek == 4 and date_value - dt.timedelta(days=1) in holidays
        )
        result.append(monday_before_tuesday_holiday or friday_after_thursday_holiday)
    return result


def _event_type_series(
    dates: pd.DatetimeIndex,
    bridge_day_flag: pd.Series,
) -> list[str]:
    event_types: list[str] = []
    for index, timestamp in enumerate(dates):
        if bool(bridge_day_flag.iloc[index]):
            event_types.append("bridge_day")
        elif timestamp.month == 12 and timestamp.day >= 15:
            event_types.append("holiday_season")
        elif timestamp.month in {6, 7, 8} and timestamp.dayofweek in {4, 5}:
            event_types.append("tourism_weekend")
        elif timestamp.month == 2 and timestamp.day == 14:
            event_types.append("local_special_day")
        elif timestamp.day <= 3 and timestamp.dayofweek in {0, 1, 2, 3, 4}:
            event_types.append("payday_period")
        else:
            event_types.append("nothing")
    return event_types


def _event_name_from_type(event_type: str | None) -> str:
    if event_type is None or pd.isna(event_type) or event_type == "nothing":
        return "nothing"
    names = {
        "holiday_season": "winter_holiday_season",
        "tourism_weekend": "summer_tourism_weekend",
        "local_special_day": "valentines_day",
        "payday_period": "early_month_payday_period",
        "bridge_day": "french_bridge_day",
    }
    return names.get(event_type, "nothing")


def _school_holiday_series(dates: pd.DatetimeIndex, zone: str) -> list[bool]:
    periods = _school_holiday_periods()[zone]
    result: list[bool] = []
    for timestamp in dates:
        date_value = timestamp.date()
        result.append(any(start <= date_value <= end for start, end in periods))
    return result


def _school_holiday_periods() -> dict[str, tuple[tuple[dt.date, dt.date], ...]]:
    common_2024_2025 = (
        (dt.date(2024, 10, 19), dt.date(2024, 11, 4)),
        (dt.date(2024, 12, 21), dt.date(2025, 1, 6)),
        (dt.date(2025, 7, 5), dt.date(2025, 8, 31)),
    )
    common_2025_2026 = (
        (dt.date(2025, 10, 18), dt.date(2025, 11, 3)),
        (dt.date(2025, 12, 20), dt.date(2026, 1, 5)),
        (dt.date(2026, 7, 4), dt.date(2026, 8, 31)),
    )
    common_2026_2027 = (
        (dt.date(2026, 10, 17), dt.date(2026, 11, 2)),
        (dt.date(2026, 12, 19), dt.date(2027, 1, 4)),
        (dt.date(2027, 7, 3), dt.date(2027, 8, 31)),
    )
    return {
        "A": common_2024_2025
        + (
            (dt.date(2025, 2, 22), dt.date(2025, 3, 10)),
            (dt.date(2025, 4, 19), dt.date(2025, 5, 5)),
        )
        + common_2025_2026
        + (
            (dt.date(2026, 2, 7), dt.date(2026, 2, 23)),
            (dt.date(2026, 4, 4), dt.date(2026, 4, 20)),
        )
        + common_2026_2027
        + (
            (dt.date(2027, 2, 13), dt.date(2027, 3, 1)),
            (dt.date(2027, 4, 10), dt.date(2027, 4, 26)),
        ),
        "B": common_2024_2025
        + (
            (dt.date(2025, 2, 8), dt.date(2025, 2, 24)),
            (dt.date(2025, 4, 5), dt.date(2025, 4, 22)),
        )
        + common_2025_2026
        + (
            (dt.date(2026, 2, 14), dt.date(2026, 3, 2)),
            (dt.date(2026, 4, 11), dt.date(2026, 4, 27)),
        )
        + common_2026_2027
        + (
            (dt.date(2027, 2, 20), dt.date(2027, 3, 8)),
            (dt.date(2027, 4, 17), dt.date(2027, 5, 3)),
        ),
        "C": common_2024_2025
        + (
            (dt.date(2025, 2, 15), dt.date(2025, 3, 3)),
            (dt.date(2025, 4, 12), dt.date(2025, 4, 28)),
        )
        + common_2025_2026
        + (
            (dt.date(2026, 2, 21), dt.date(2026, 3, 9)),
            (dt.date(2026, 4, 18), dt.date(2026, 5, 4)),
        )
        + common_2026_2027
        + (
            (dt.date(2027, 2, 6), dt.date(2027, 2, 22)),
            (dt.date(2027, 4, 3), dt.date(2027, 4, 19)),
        ),
    }


def build_city_weather_frame(
    *,
    cities: pd.DataFrame,
    calendar: SyntheticCalendar,
    seed: int,
) -> pd.DataFrame:
    """Generate city-date weather observations available in the source feed."""

    rng = np.random.default_rng(seed + 17)
    frames = [
        _city_weather_rows(cast(WeatherCityRecord, raw_city), calendar, rng)
        for raw_city in cities.itertuples(index=False)
    ]
    return pd.concat(frames, ignore_index=True)


def _city_weather_rows(
    city: WeatherCityRecord,
    calendar: SyntheticCalendar,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_dates = len(calendar.dates)
    seasonal = np.cos(2.0 * np.pi * (calendar.month.astype(float) - 7.0) / 12.0)
    latitude_adjustment = (46.5 - float(city.latitude)) * 0.55
    temperature = (
        14.0 + 9.5 * seasonal + latitude_adjustment + rng.normal(0.0, 2.1, n_dates)
    )
    precipitation = rng.gamma(shape=0.7, scale=2.2, size=n_dates)
    rain_mask = rng.random(n_dates) < (
        0.18 + 0.08 * np.cos(2.0 * np.pi * calendar.month / 12.0)
    )
    humidity = np.clip(
        62.0 + precipitation * 3.8 + rng.normal(0.0, 8.0, n_dates), 25.0, 99.0
    )
    wind = np.clip(rng.gamma(shape=2.2, scale=1.4, size=n_dates), 0.0, 12.0)
    return pd.DataFrame(
        {
            "city_id": city.city_id,
            "dt": calendar.dates,
            "weather_temperature": temperature.round(2),
            "weather_precipitation": np.where(rain_mask, precipitation, 0.0).round(2),
            "weather_humidity": humidity.round(2),
            "weather_wind_level": wind.round(2),
        }
    )

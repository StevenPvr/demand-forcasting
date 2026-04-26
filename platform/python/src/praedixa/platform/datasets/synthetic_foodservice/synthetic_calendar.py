from __future__ import annotations

from dataclasses import dataclass
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
    frame["event_type_1"] = _event_type_series(dates)
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
        event_type=frame["event_type_1"].fillna("").to_numpy(dtype=object),
        event_name=frame["event_name_1"].fillna("").to_numpy(dtype=object),
    )


def _is_french_public_holiday(timestamp: pd.Timestamp) -> bool:
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


def _event_type_series(dates: pd.DatetimeIndex) -> list[str | None]:
    event_types: list[str | None] = []
    for timestamp in dates:
        if timestamp.month == 12 and timestamp.day >= 15:
            event_types.append("holiday_season")
        elif timestamp.month in {6, 7, 8} and timestamp.dayofweek in {4, 5}:
            event_types.append("tourism_weekend")
        elif timestamp.month == 2 and timestamp.day == 14:
            event_types.append("local_special_day")
        elif timestamp.day <= 3 and timestamp.dayofweek in {0, 1, 2, 3, 4}:
            event_types.append("payday_period")
        else:
            event_types.append(None)
    return event_types


def _event_name_from_type(event_type: str | None) -> str | None:
    if event_type is None or pd.isna(event_type):
        return None
    names = {
        "holiday_season": "winter_holiday_season",
        "tourism_weekend": "summer_tourism_weekend",
        "local_special_day": "valentines_day",
        "payday_period": "early_month_payday_period",
    }
    return names[event_type]


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
    seasonal = np.sin(2.0 * np.pi * (calendar.month.astype(float) - 1.0) / 12.0)
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.simulation_math import (
    active_matrix,
    available_quantity,
    capacity_factor,
    closed_day_mask,
    day_complete_mask,
    dow_lift,
    holiday_lift,
    negative_binomial,
    promo_matrix,
    scaled_product_base,
    seasonal_lift,
    staffing_hours,
    waste_multiplier,
    weather_lift,
    zero_inflation_mask,
)
from praedixa.platform.datasets.synthetic_foodservice.synthetic_calendar import (
    SyntheticCalendar,
)
from praedixa.platform.datasets.synthetic_foodservice.taxonomy import (
    VERTICAL_SPEC_BY_SOURCE,
)


MODEL_FACING_COLUMNS: tuple[str, ...] = (
    "dataset_source",
    "source_partition",
    "source_run_id",
    "series_id",
    "dt",
    "location_id",
    "product_id",
    "region_id",
    "org_group_id",
    "category_level_1",
    "category_level_2",
    "category_level_3",
    "observed_demand_qty",
    "target_semantics",
    "censor_flag",
    "target_source",
    "label_quality_score",
    "usable_for_training_flag",
    "observed_revenue_net",
    "observed_discount_amount",
    "promo_flag",
    "holiday_flag",
    "activity_flag",
    "observed_stockout_flag",
    "observed_stockout_available",
    "observed_stockout_intensity",
    "day_complete_flag",
    "calendar_weekday_name",
    "calendar_day_of_week",
    "calendar_month",
    "calendar_year",
    "calendar_week_key",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "weather_precipitation",
    "weather_temperature",
    "weather_humidity",
    "weather_wind_level",
    "silver_run_id",
    "source_name",
    "source_policy_id",
)

ORACLE_COLUMNS: tuple[str, ...] = (
    "dataset_source",
    "dt",
    "location_id",
    "product_id",
    "latent_demand_qty_debug",
    "lost_sales_qty_debug",
    "waste_qty_debug",
    "production_qty_debug",
    "staffing_required_hours_debug",
)


@dataclass(frozen=True)
class BatchSimulationResult:
    """One generated source batch and its non-model oracle sidecar."""

    daily: pd.DataFrame
    oracle: pd.DataFrame


class DemandLocationRecord(Protocol):
    """Typed view of a pandas ``itertuples`` location row."""

    dataset_source: str
    location_id: str
    city_id: str
    region_id: str
    org_group_id: str


def generate_location_batch(
    *,
    locations: pd.DataFrame,
    assortments: pd.DataFrame,
    calendar: SyntheticCalendar,
    city_weather: pd.DataFrame,
    seed: int,
    source_run_id: str,
) -> BatchSimulationResult:
    """Generate observed source rows for a batch of sites."""

    daily_frames: list[pd.DataFrame] = []
    oracle_frames: list[pd.DataFrame] = []
    for raw_location in locations.itertuples(index=False):
        location = cast(DemandLocationRecord, raw_location)
        rng = np.random.default_rng(
            seed + _stable_location_offset(str(location.location_id))
        )
        location_assortment = assortments.loc[
            assortments["location_id"].eq(str(location.location_id))
        ]
        weather = city_weather.loc[city_weather["city_id"].eq(str(location.city_id))]
        result = _generate_location_frame(
            location=location,
            assortment=location_assortment,
            calendar=calendar,
            weather=weather,
            rng=rng,
            source_run_id=source_run_id,
        )
        daily_frames.append(result.daily)
        oracle_frames.append(result.oracle)
    return BatchSimulationResult(
        daily=pd.concat(daily_frames, ignore_index=True)
        if daily_frames
        else pd.DataFrame(columns=MODEL_FACING_COLUMNS),
        oracle=pd.concat(oracle_frames, ignore_index=True)
        if oracle_frames
        else pd.DataFrame(columns=ORACLE_COLUMNS),
    )


def _stable_location_offset(location_id: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(location_id))


def _generate_location_frame(
    *,
    location: DemandLocationRecord,
    assortment: pd.DataFrame,
    calendar: SyntheticCalendar,
    weather: pd.DataFrame,
    rng: np.random.Generator,
    source_run_id: str,
) -> BatchSimulationResult:
    if assortment.empty:
        return BatchSimulationResult(
            pd.DataFrame(columns=MODEL_FACING_COLUMNS),
            pd.DataFrame(columns=ORACLE_COLUMNS),
        )
    matrices = _build_location_matrices(location, assortment, calendar, weather, rng)
    selected = _selected_model_rows(matrices, rng)
    daily = _model_frame(
        location, assortment, calendar, weather, matrices, selected, source_run_id
    )
    oracle = _oracle_frame(location, assortment, calendar, matrices, selected)
    return BatchSimulationResult(daily=daily, oracle=oracle)


def _build_location_matrices(
    location: DemandLocationRecord,
    assortment: pd.DataFrame,
    calendar: SyntheticCalendar,
    weather: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    date_count = len(calendar.dates)
    product_count = len(assortment)
    mu = _latent_mean_matrix(location, assortment, calendar, weather, rng)
    latent = negative_binomial(mu=mu, rng=rng)
    active = active_matrix(assortment, calendar)
    closed = closed_day_mask(str(location.dataset_source), calendar, rng)
    promo = promo_matrix(str(location.dataset_source), product_count, date_count, rng)
    latent = np.where(active & ~closed[None, :], latent, 0)
    latent = np.where(
        zero_inflation_mask(str(location.dataset_source), mu.shape, rng), 0, latent
    )
    available = available_quantity(str(location.dataset_source), mu, latent, rng)
    capacity = capacity_factor(str(location.dataset_source), date_count, rng)
    observed = np.floor(np.minimum(latent * capacity[None, :], available)).astype(float)
    lost = np.maximum(latent - observed, 0.0)
    waste = np.maximum(available - latent, 0.0) * waste_multiplier(
        str(location.dataset_source)
    )
    return {
        "active": active,
        "closed": closed,
        "promo": promo,
        "latent": latent,
        "observed": observed,
        "lost": lost,
        "waste": waste,
        "available": available,
        "capacity": capacity,
        "day_complete": day_complete_mask(date_count, rng),
    }


def _latent_mean_matrix(
    location: DemandLocationRecord,
    assortment: pd.DataFrame,
    calendar: SyntheticCalendar,
    weather: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    source = str(location.dataset_source)
    spec = VERTICAL_SPEC_BY_SOURCE[source]
    base = assortment["base_daily_units"].to_numpy(dtype=float)
    base = scaled_product_base(base, spec.base_units_low, spec.base_units_high, rng)
    dow = dow_lift(source, calendar.day_of_week)
    seasonal = seasonal_lift(source, calendar.month)
    weather_effect = weather_lift(source, weather)
    event_lift = np.where(
        calendar.event_type.astype(str) == "tourism_weekend", 1.08, 1.0
    )
    holiday_effect = np.where(calendar.holiday_flag, holiday_lift(source), 1.0)
    site_noise = rng.lognormal(mean=0.0, sigma=0.06, size=len(calendar.dates))
    return (
        base[:, None]
        * dow[None, :]
        * seasonal[None, :]
        * weather_effect[None, :]
        * event_lift[None, :]
        * holiday_effect[None, :]
        * site_noise[None, :]
    )


def _selected_model_rows(
    matrices: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    observed = matrices["observed"]
    promo = matrices["promo"]
    active = matrices["active"]
    closed = matrices["closed"]
    censored = matrices["lost"] > 0
    sampled_zero = rng.random(observed.shape) < 0.018
    selected = (
        active & ~closed[None, :] & ((observed > 0) | censored | promo | sampled_zero)
    )
    return selected | _closure_sentinel_mask(active, closed)


def _closure_sentinel_mask(active: np.ndarray, closed: np.ndarray) -> np.ndarray:
    sentinel = np.zeros_like(active, dtype=bool)
    if active.shape[0] == 0:
        return sentinel
    sentinel[0, :] = closed
    return sentinel


def _model_frame(
    location: DemandLocationRecord,
    assortment: pd.DataFrame,
    calendar: SyntheticCalendar,
    weather: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    selected: np.ndarray,
    source_run_id: str,
) -> pd.DataFrame:
    product_index, date_index = np.where(selected)
    frame = _base_output_frame(
        location, assortment, calendar, product_index, date_index, source_run_id
    )
    _assign_observed_fields(frame, assortment, matrices, product_index, date_index)
    _assign_calendar_weather_fields(frame, calendar, weather, date_index)
    return frame.loc[:, list(MODEL_FACING_COLUMNS)]


def _base_output_frame(
    location: DemandLocationRecord,
    assortment: pd.DataFrame,
    calendar: SyntheticCalendar,
    product_index: np.ndarray,
    date_index: np.ndarray,
    source_run_id: str,
) -> pd.DataFrame:
    products = assortment.iloc[product_index].reset_index(drop=True)
    source = str(location.dataset_source)
    frame = pd.DataFrame(
        {
            "dataset_source": source,
            "source_partition": "historical_synthetic",
            "source_run_id": source_run_id,
            "dt": calendar.dates[date_index],
            "location_id": str(location.location_id),
            "product_id": products["product_id"].to_numpy(),
            "region_id": str(location.region_id),
            "org_group_id": str(location.org_group_id),
            "category_level_1": products["category_level_1"].to_numpy(),
            "category_level_2": products["category_level_2"].to_numpy(),
            "category_level_3": products["category_level_3"].to_numpy(),
            "silver_run_id": source_run_id,
            "source_name": source,
            "source_policy_id": source,
        }
    )
    frame["series_id"] = frame["location_id"] + "__" + frame["product_id"]
    return frame


def _assign_observed_fields(
    frame: pd.DataFrame,
    assortment: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    product_index: np.ndarray,
    date_index: np.ndarray,
) -> None:
    price = assortment.iloc[product_index]["base_price"].to_numpy(dtype=float)
    observed = matrices["observed"][product_index, date_index]
    lost = matrices["lost"][product_index, date_index]
    discount_rate = np.where(matrices["promo"][product_index, date_index], 0.12, 0.0)
    closed = matrices["closed"][date_index]
    complete = matrices["day_complete"][date_index] & ~closed
    frame["observed_demand_qty"] = observed.round(3)
    frame["target_semantics"] = "observed_sales"
    frame["censor_flag"] = lost > 0
    frame["target_source"] = np.where(
        complete, "observed_sales", "closed_or_missing_observation"
    )
    frame["label_quality_score"] = np.where(complete, 1.0, 0.0)
    frame["usable_for_training_flag"] = complete
    frame["observed_revenue_net"] = (observed * price * (1.0 - discount_rate)).round(2)
    frame["observed_discount_amount"] = (observed * price * discount_rate).round(2)
    frame["promo_flag"] = matrices["promo"][product_index, date_index]
    frame["activity_flag"] = ~closed
    frame["observed_stockout_flag"] = lost > 0
    frame["observed_stockout_available"] = True
    frame["observed_stockout_intensity"] = (
        lost / np.maximum(matrices["latent"][product_index, date_index], 1.0)
    ).round(4)
    frame["day_complete_flag"] = complete


def _assign_calendar_weather_fields(
    frame: pd.DataFrame,
    calendar: SyntheticCalendar,
    weather: pd.DataFrame,
    date_index: np.ndarray,
) -> None:
    frame["holiday_flag"] = calendar.holiday_flag[date_index]
    frame["calendar_weekday_name"] = calendar.frame["calendar_weekday_name"].to_numpy()[
        date_index
    ]
    frame["calendar_day_of_week"] = calendar.frame["calendar_day_of_week"].to_numpy()[
        date_index
    ]
    frame["calendar_month"] = calendar.frame["calendar_month"].to_numpy()[date_index]
    frame["calendar_year"] = calendar.frame["calendar_year"].to_numpy()[date_index]
    frame["calendar_week_key"] = calendar.frame["calendar_week_key"].to_numpy()[
        date_index
    ]
    frame["event_name_1"] = _nullable_array(calendar.event_name[date_index])
    frame["event_type_1"] = _nullable_array(calendar.event_type[date_index])
    frame["event_name_2"] = None
    frame["event_type_2"] = None
    for column in (
        "weather_precipitation",
        "weather_temperature",
        "weather_humidity",
        "weather_wind_level",
    ):
        frame[column] = weather[column].to_numpy(dtype=float)[date_index]


def _nullable_array(values: np.ndarray) -> list[str | None]:
    return [str(value) if str(value) else None for value in values]


def _oracle_frame(
    location: DemandLocationRecord,
    assortment: pd.DataFrame,
    calendar: SyntheticCalendar,
    matrices: dict[str, np.ndarray],
    selected: np.ndarray,
) -> pd.DataFrame:
    product_index, date_index = np.where(selected)
    products = assortment.iloc[product_index].reset_index(drop=True)
    latent = matrices["latent"][product_index, date_index]
    lost = matrices["lost"][product_index, date_index]
    frame = pd.DataFrame(
        {
            "dataset_source": str(location.dataset_source),
            "dt": calendar.dates[date_index],
            "location_id": str(location.location_id),
            "product_id": products["product_id"].to_numpy(),
            "latent_demand_qty_debug": latent.round(3),
            "lost_sales_qty_debug": lost.round(3),
            "waste_qty_debug": matrices["waste"][product_index, date_index].round(3),
            "production_qty_debug": matrices["available"][
                product_index, date_index
            ].round(3),
            "staffing_required_hours_debug": staffing_hours(
                latent, str(location.dataset_source)
            ).round(3),
        }
    )
    return frame.loc[:, list(ORACLE_COLUMNS)]


def batch_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return compact quality statistics for one generated model-facing batch."""

    if frame.empty:
        return {"rows": 0, "series": 0, "censored_rows": 0, "zero_rows": 0}
    return {
        "rows": int(len(frame)),
        "series": int(frame["series_id"].nunique()),
        "censored_rows": int(frame["censor_flag"].astype(bool).sum()),
        "zero_rows": int(frame["observed_demand_qty"].eq(0).sum()),
    }

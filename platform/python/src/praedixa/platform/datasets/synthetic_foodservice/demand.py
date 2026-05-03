from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.shocks import (
    ShockEvent,
    apply_shocks_to_mu,
)
from praedixa.platform.datasets.synthetic_foodservice.simulation_math import (
    active_matrix,
    closed_day_mask,
    day_complete_mask,
    dow_lift,
    holiday_lift,
    negative_binomial,
    promo_matrix,
    scaled_product_base,
    seasonal_lift,
    staffing_hours,
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
    school_zone: str


def generate_location_batch(
    *,
    locations: pd.DataFrame,
    assortments: pd.DataFrame,
    calendar: SyntheticCalendar,
    city_weather: pd.DataFrame,
    seed: int,
    source_run_id: str,
    site_shocks_by_location_id: dict[str, list[ShockEvent]] | None = None,
    allow_data_gaps: bool = False,
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
        site_shocks = (site_shocks_by_location_id or {}).get(
            str(location.location_id), []
        )
        result = _generate_location_frame(
            location=location,
            assortment=location_assortment,
            calendar=calendar,
            weather=weather,
            rng=rng,
            source_run_id=source_run_id,
            site_shocks=site_shocks,
            allow_data_gaps=allow_data_gaps,
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
    site_shocks: list[ShockEvent],
    allow_data_gaps: bool,
) -> BatchSimulationResult:
    if assortment.empty:
        return BatchSimulationResult(
            pd.DataFrame(columns=MODEL_FACING_COLUMNS),
            pd.DataFrame(columns=ORACLE_COLUMNS),
        )
    matrices = _build_location_matrices(
        location,
        assortment,
        calendar,
        weather,
        rng,
        site_shocks=site_shocks,
        allow_data_gaps=allow_data_gaps,
    )
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
    site_shocks: list[ShockEvent] | None = None,
    allow_data_gaps: bool = False,
) -> dict[str, np.ndarray]:
    shocks = site_shocks or []
    date_count = len(calendar.dates)
    product_count = len(assortment)
    active = active_matrix(assortment, calendar)
    closed = closed_day_mask(str(location.dataset_source), calendar, rng)
    closed = _apply_forced_closure_shocks(closed, shocks)
    promo = promo_matrix(str(location.dataset_source), product_count, date_count, rng)
    mu = _latent_mean_matrix(location, assortment, calendar, weather, rng)
    mu, discount_rate = _apply_promo_volume_effects(
        mu, promo, assortment, str(location.dataset_source), rng
    )
    demand_shocks = _shocks_for_demand(shocks)
    if demand_shocks:
        mu = apply_shocks_to_mu(mu, demand_shocks, date_count)
    latent = negative_binomial(mu=mu, rng=rng)
    latent = np.where(active & ~closed[None, :], latent, 0)
    latent = np.where(
        zero_inflation_mask(str(location.dataset_source), mu.shape, rng), 0, latent
    )
    available = latent.copy()
    capacity = np.ones(date_count, dtype=float)
    observed = latent.astype(float)
    lost = np.zeros_like(latent, dtype=float)
    waste = np.zeros_like(latent, dtype=float)
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
        "day_complete": _day_complete_matrix(date_count, rng, shocks, allow_data_gaps),
        "discount_rate": discount_rate,
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
    school_holiday_effect = _school_holiday_lift(location, calendar)
    bridge_effect = np.where(calendar.bridge_day_flag, _bridge_day_lift(source), 1.0)
    price_effect = _price_elasticity_lift(source, assortment)
    site_noise = rng.lognormal(mean=0.0, sigma=0.06, size=len(calendar.dates))
    return (
        base[:, None]
        * price_effect[:, None]
        * dow[None, :]
        * seasonal[None, :]
        * weather_effect[None, :]
        * event_lift[None, :]
        * holiday_effect[None, :]
        * school_holiday_effect[None, :]
        * bridge_effect[None, :]
        * site_noise[None, :]
    )


def _selected_model_rows(
    matrices: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    active = matrices["active"]
    return active


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
    _assign_calendar_weather_fields(frame, location, calendar, weather, date_index)
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
    discount_rate = matrices["discount_rate"][product_index, date_index]
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
    location: DemandLocationRecord,
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
    frame["event_name_1"] = _clean_string_array(calendar.event_name[date_index])
    frame["event_type_1"] = _clean_string_array(calendar.event_type[date_index])
    school_holiday = _school_holiday_mask_for_location(location, calendar)[date_index]
    frame["event_name_2"] = np.where(
        school_holiday,
        "school_holiday_zone_" + str(location.school_zone).lower(),
        "nothing",
    )
    frame["event_type_2"] = np.where(school_holiday, "school_holiday", "nothing")
    for column in (
        "weather_precipitation",
        "weather_temperature",
        "weather_humidity",
        "weather_wind_level",
    ):
        frame[column] = weather[column].to_numpy(dtype=float)[date_index]


def _clean_string_array(values: np.ndarray) -> list[str]:
    return [
        str(value) if str(value) and str(value) != "None" else "nothing"
        for value in values
    ]


def _price_elasticity_lift(source: str, assortment: pd.DataFrame) -> np.ndarray:
    prices = assortment["base_price"].to_numpy(dtype=float)
    medians = assortment.groupby("category_level_1")["base_price"].transform("median")
    reference = np.maximum(medians.to_numpy(dtype=float), 0.01)
    elasticity = _price_elasticity(source)
    return np.clip((prices / reference) ** elasticity, 0.62, 1.42)


def _price_elasticity(source: str) -> float:
    elasticities: dict[str, float] = {
        "synthetic_foodservice_qsr": -0.95,
        "synthetic_foodservice_bakery": -0.65,
        "synthetic_foodservice_restaurant": -0.55,
        "synthetic_foodservice_brasserie": -0.60,
        "synthetic_foodservice_coffee_shop": -0.70,
        "synthetic_foodservice_snack": -0.85,
        "synthetic_foodservice_dark_kitchen": -1.10,
        "synthetic_foodservice_sandwich_shop": -0.80,
        "synthetic_foodservice_seasonal_tourist": -0.50,
    }
    return elasticities.get(source, -0.75)


def _apply_promo_volume_effects(
    mu: np.ndarray,
    promo: np.ndarray,
    assortment: pd.DataFrame,
    source: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    discount_rate = np.where(
        promo,
        rng.uniform(0.08, 0.24, size=promo.shape),
        0.0,
    )
    sensitivity = _promo_sensitivity(source)
    adjusted = mu * (1.0 + sensitivity * discount_rate)
    _apply_category_cannibalization(adjusted, promo, assortment, source)
    return np.clip(adjusted, 0.01, None), discount_rate


def _promo_sensitivity(source: str) -> float:
    sensitivities: dict[str, float] = {
        "synthetic_foodservice_qsr": 2.2,
        "synthetic_foodservice_bakery": 1.4,
        "synthetic_foodservice_restaurant": 1.1,
        "synthetic_foodservice_brasserie": 1.2,
        "synthetic_foodservice_coffee_shop": 1.5,
        "synthetic_foodservice_snack": 1.9,
        "synthetic_foodservice_dark_kitchen": 2.4,
        "synthetic_foodservice_sandwich_shop": 1.8,
        "synthetic_foodservice_seasonal_tourist": 1.3,
    }
    return sensitivities.get(source, 1.5)


def _apply_category_cannibalization(
    adjusted: np.ndarray,
    promo: np.ndarray,
    assortment: pd.DataFrame,
    source: str,
) -> None:
    categories = assortment["category_level_1"].astype(str).to_numpy()
    rate = _cannibalization_rate(source)
    for category in sorted(set(categories.tolist())):
        category_mask = categories == category
        promo_share = promo[category_mask, :].mean(axis=0)
        if not np.any(promo_share > 0):
            continue
        daily_factor = np.clip(1.0 - rate * promo_share, 0.82, 1.0)
        non_promo = category_mask[:, None] & ~promo
        adjusted[non_promo] *= np.broadcast_to(daily_factor, adjusted.shape)[non_promo]


def _cannibalization_rate(source: str) -> float:
    rates: dict[str, float] = {
        "synthetic_foodservice_qsr": 0.16,
        "synthetic_foodservice_bakery": 0.10,
        "synthetic_foodservice_restaurant": 0.08,
        "synthetic_foodservice_brasserie": 0.09,
        "synthetic_foodservice_coffee_shop": 0.11,
        "synthetic_foodservice_snack": 0.14,
        "synthetic_foodservice_dark_kitchen": 0.18,
        "synthetic_foodservice_sandwich_shop": 0.15,
        "synthetic_foodservice_seasonal_tourist": 0.08,
    }
    return rates.get(source, 0.10)


def _school_holiday_lift(
    location: DemandLocationRecord,
    calendar: SyntheticCalendar,
) -> np.ndarray:
    holiday = _school_holiday_mask_for_location(location, calendar)
    source = str(location.dataset_source)
    lift_by_source: dict[str, float] = {
        "synthetic_foodservice_qsr": 1.06,
        "synthetic_foodservice_bakery": 0.96,
        "synthetic_foodservice_restaurant": 1.04,
        "synthetic_foodservice_brasserie": 1.03,
        "synthetic_foodservice_coffee_shop": 0.88,
        "synthetic_foodservice_snack": 1.08,
        "synthetic_foodservice_dark_kitchen": 1.10,
        "synthetic_foodservice_sandwich_shop": 0.86,
        "synthetic_foodservice_seasonal_tourist": 1.18,
    }
    return np.where(holiday, lift_by_source.get(source, 1.0), 1.0)


def _school_holiday_mask_for_location(
    location: DemandLocationRecord,
    calendar: SyntheticCalendar,
) -> np.ndarray:
    zone = str(location.school_zone).upper()
    if zone == "A":
        return calendar.school_holiday_zone_a
    if zone == "B":
        return calendar.school_holiday_zone_b
    return calendar.school_holiday_zone_c


def _bridge_day_lift(source: str) -> float:
    lifts: dict[str, float] = {
        "synthetic_foodservice_qsr": 0.96,
        "synthetic_foodservice_bakery": 0.92,
        "synthetic_foodservice_restaurant": 1.08,
        "synthetic_foodservice_brasserie": 1.10,
        "synthetic_foodservice_coffee_shop": 0.88,
        "synthetic_foodservice_snack": 0.98,
        "synthetic_foodservice_dark_kitchen": 1.06,
        "synthetic_foodservice_sandwich_shop": 0.84,
        "synthetic_foodservice_seasonal_tourist": 1.14,
    }
    return lifts.get(source, 1.0)


def _shocks_for_demand(site_shocks: list[ShockEvent]) -> list[ShockEvent]:
    excluded = {"data", "supply", "ops", "regulatory"}
    return [shock for shock in site_shocks if shock.category not in excluded]


def _apply_forced_closure_shocks(
    closed: np.ndarray,
    site_shocks: list[ShockEvent],
) -> np.ndarray:
    adjusted = closed.copy()
    for shock in site_shocks:
        if shock.category != "regulatory" and not shock.shock_id.startswith(
            "forced_closure"
        ):
            continue
        adjusted[shock.start_day_index : shock.end_day_index] = True
    return adjusted


def _apply_data_shocks(
    day_complete: np.ndarray,
    site_shocks: list[ShockEvent],
) -> np.ndarray:
    adjusted = day_complete.copy()
    for shock in site_shocks:
        if shock.category != "data":
            continue
        adjusted[shock.start_day_index : shock.end_day_index] = False
    return adjusted


def _day_complete_matrix(
    date_count: int,
    rng: np.random.Generator,
    site_shocks: list[ShockEvent],
    allow_data_gaps: bool,
) -> np.ndarray:
    if not allow_data_gaps:
        return np.ones(date_count, dtype=bool)
    return _apply_data_shocks(day_complete_mask(date_count, rng), site_shocks)


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

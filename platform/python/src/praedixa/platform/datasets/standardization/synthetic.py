from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from praedixa.platform.datasets.standardization.schema import align_lazy_frame_to_canonical_schema, build_series_id_expr


@dataclass(frozen=True)
class SyntheticColdStartConfig:
    """Configuration for synthetic cold-start daily demand generation."""

    start_date: str
    days: int
    num_locations: int
    products_per_location: int
    random_seed: int = 7
    promo_probability: float = 0.12
    stockout_probability: float = 0.05


def _calendar_columns(dates: pd.Series) -> dict[str, pd.Series]:
    datetimes = pd.to_datetime(dates, errors="coerce")
    return {
        "calendar_weekday_name": datetimes.dt.day_name(),
        "calendar_day_of_week": datetimes.dt.dayofweek.astype("Int8"),
        "calendar_month": datetimes.dt.month.astype("Int8"),
        "calendar_year": datetimes.dt.year.astype("Int16"),
        "calendar_week_key": datetimes.dt.isocalendar().week.astype("Int32"),
    }


def _synthetic_environment_row(
    *,
    rng: np.random.Generator,
    base_price: float,
    promo_flag: bool,
    stockout_flag: bool,
) -> dict[str, float | bool | None]:
    temperature = float(rng.normal(18.0, 6.0))
    precipitation = max(0.0, float(rng.normal(1.5, 1.0)))
    return {
        "observed_discount_amount": base_price * 0.15 if promo_flag else None,
        "promo_flag": promo_flag,
        "observed_stockout_flag": stockout_flag,
        "observed_stockout_available": True,
        "observed_stockout_intensity": 1.0 if stockout_flag else 0.0,
        "weather_precipitation": precipitation,
        "weather_temperature": temperature,
        "weather_humidity": float(rng.uniform(0.35, 0.85)),
        "weather_wind_level": float(rng.uniform(1.0, 6.0)),
    }


def _synthetic_metadata_row(
    *,
    config: SyntheticColdStartConfig,
    location_id: str,
    product_id: str,
    product_index: int,
    dt: pd.Timestamp,
) -> dict[str, object]:
    return {
        "dataset_source": "synthetic_v1",
        "source_partition": "synthetic_train",
        "source_run_id": f"synthetic_seed_{config.random_seed}",
        "dt": dt.date(),
        "location_id": location_id,
        "product_id": product_id,
        "region_id": None,
        "org_group_id": None,
        "category_level_1": "synthetic_food_service",
        "category_level_2": "synthetic_menu",
        "category_level_3": f"synthetic_family_{product_index + 1}",
        "holiday_flag": False,
        "activity_flag": False,
        "location_open_flag": True,
        "day_complete_flag": True,
        "missing_sales_flag": False,
        "event_name_1": None,
        "event_type_1": None,
        "event_name_2": None,
        "event_type_2": None,
        "anomaly_flag": False,
        "silver_run_id": "synthetic_generator",
    }


def _synthetic_observed_row(
    *,
    rng: np.random.Generator,
    demand_signal: float,
    base_price: float,
) -> dict[str, float]:
    observed_demand_qty = max(0.0, float(rng.normal(demand_signal, max(1.0, demand_signal * 0.12))))
    return {
        "observed_demand_qty": observed_demand_qty,
        "observed_revenue_net": observed_demand_qty * base_price,
        "avg_selling_price": base_price,
    }


def _synthetic_row(
    *,
    config: SyntheticColdStartConfig,
    rng: np.random.Generator,
    location_id: str,
    product_id: str,
    product_index: int,
    dt: pd.Timestamp,
    day_offset: int,
    site_base: float,
    product_base: float,
    base_price: float,
    ramp_days: int,
) -> dict[str, object]:
    promo_flag = bool(rng.random() < config.promo_probability)
    stockout_flag = bool(rng.random() < config.stockout_probability)
    demand_signal = _synthetic_demand_signal(
        dt=dt,
        day_offset=day_offset,
        ramp_days=ramp_days,
        site_base=site_base,
        product_base=product_base,
        promo_flag=promo_flag,
        stockout_flag=stockout_flag,
    )
    row = _synthetic_metadata_row(
        config=config,
        location_id=location_id,
        product_id=product_id,
        product_index=product_index,
        dt=dt,
    )
    row.update(
        _synthetic_environment_row(
            rng=rng,
            base_price=base_price,
            promo_flag=promo_flag,
            stockout_flag=stockout_flag,
        )
    )
    row.update(_synthetic_observed_row(rng=rng, demand_signal=demand_signal, base_price=base_price))
    return row


def _synthetic_demand_signal(
    *,
    dt: pd.Timestamp,
    day_offset: int,
    ramp_days: int,
    site_base: float,
    product_base: float,
    promo_flag: bool,
    stockout_flag: bool,
) -> float:
    weekend_multiplier = 1.2 if dt.dayofweek >= 5 else 1.0
    opening_ramp = min(1.0, (day_offset + 1) / ramp_days)
    demand_signal = site_base * product_base * weekend_multiplier * opening_ramp
    demand_signal *= 1.18 if promo_flag else 1.0
    demand_signal *= 0.75 if stockout_flag else 1.0
    return demand_signal


def _iter_synthetic_rows(
    config: SyntheticColdStartConfig,
    rng: np.random.Generator,
    all_dates: pd.DatetimeIndex,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for location_index in range(config.num_locations):
        location_id = f"synthetic_site_{location_index + 1}"
        site_base = rng.uniform(20.0, 80.0)
        for product_index in range(config.products_per_location):
            product_id = f"synthetic_sku_{product_index + 1}"
            base_price = float(rng.uniform(3.5, 14.0))
            product_base = float(rng.uniform(0.4, 1.4))
            ramp_days = int(rng.integers(7, 22))
            for day_offset, dt in enumerate(all_dates):
                rows.append(
                    _synthetic_row(
                        config=config,
                        rng=rng,
                        location_id=location_id,
                        product_id=product_id,
                        product_index=product_index,
                        dt=dt,
                        day_offset=day_offset,
                        site_base=site_base,
                        product_base=product_base,
                        base_price=base_price,
                        ramp_days=ramp_days,
                    )
                )
    return rows


def generate_synthetic_cold_start_frame(config: SyntheticColdStartConfig) -> pl.DataFrame:
    """Generate a canonical synthetic daily demand dataset for cold-start training."""

    rng = np.random.default_rng(config.random_seed)
    all_dates = pd.date_range(start=pd.Timestamp(config.start_date), periods=config.days, freq="D")
    frame = pd.DataFrame(_iter_synthetic_rows(config, rng, all_dates))

    for column_name, values in _calendar_columns(frame["dt"]).items():
        frame[column_name] = values
    lazy_frame = pl.from_pandas(frame).lazy().with_columns(
        build_series_id_expr("location_id", "product_id").alias("series_id")
    )
    return align_lazy_frame_to_canonical_schema(lazy_frame).collect()


def write_synthetic_cold_start_dataset(
    output_path: str | Path,
    *,
    config: SyntheticColdStartConfig,
) -> Path:
    """Persist a synthetic cold-start dataset to parquet."""

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_cold_start_frame(config).write_parquet(target_path)
    return target_path

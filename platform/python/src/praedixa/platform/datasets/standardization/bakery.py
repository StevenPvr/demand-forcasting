from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import polars as pl

from praedixa.platform.datasets.standardization.schema import align_lazy_frame_to_canonical_schema
from praedixa.platform.runtime.paths import GLOBAL_DATASET_DIR
from praedixa.platform.runtime.paths import SOURCES_DIR


DEFAULT_INPUT_PATH = SOURCES_DIR / "bakery_sales" / "Bakery sales.csv"
DEFAULT_OUTPUT_PATH = GLOBAL_DATASET_DIR / "bakery_daily.parquet"
DEFAULT_LOCATION_ID = "bakery_store_1"
DEFAULT_SOURCE_RUN_ID = "manual_local"
DEFAULT_SILVER_RUN_ID = "manual_local"


def parse_bakery_unit_price(raw_value: str | float | int | None) -> float | None:
    """Parse bakery unit prices stored as French monetary strings."""

    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    normalized = re.sub(r"[^0-9,.-]", "", text).replace(",", ".")
    if not normalized:
        return None
    return float(normalized)


def _normalize_bakery_article(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["product_id"] = normalized["article"].astype(str).str.strip().str.upper()
    return normalized


def _build_bakery_grouped_frame(frame: pd.DataFrame, *, location_id: str) -> pd.DataFrame:
    normalized = _normalize_bakery_article(frame)
    normalized["dt"] = pd.to_datetime(normalized["date"], errors="coerce").dt.date
    normalized["quantity"] = pd.to_numeric(normalized["Quantity"], errors="coerce")
    unit_price = normalized["unit_price"].map(parse_bakery_unit_price)
    normalized["observed_revenue_net"] = normalized["quantity"] * unit_price
    normalized["location_id"] = location_id
    return (
        normalized.groupby(["dt", "location_id", "product_id"], dropna=False)
        .agg(
            observed_demand_qty=("quantity", "sum"),
            observed_revenue_net=("observed_revenue_net", "sum"),
        )
        .reset_index()
    )


def _attach_bakery_canonical_columns(
    grouped: pd.DataFrame,
    *,
    source_run_id: str,
    silver_run_id: str,
) -> pd.DataFrame:
    enriched = grouped.copy()
    enriched["dataset_source"] = "bakery"
    enriched["source_partition"] = "historical"
    enriched["source_run_id"] = source_run_id
    enriched["silver_run_id"] = silver_run_id
    enriched["series_id"] = enriched["location_id"].astype(str) + "__" + enriched["product_id"].astype(str)
    enriched["region_id"] = None
    enriched["org_group_id"] = None
    enriched["category_level_1"] = None
    enriched["category_level_2"] = None
    enriched["category_level_3"] = None
    enriched["observed_discount_amount"] = None
    enriched["promo_flag"] = None
    enriched["holiday_flag"] = None
    enriched["activity_flag"] = None
    enriched["observed_stockout_flag"] = None
    enriched["observed_stockout_available"] = False
    enriched["observed_stockout_intensity"] = None
    enriched["day_complete_flag"] = None
    dt_series = pd.to_datetime(enriched["dt"])
    enriched["calendar_weekday_name"] = dt_series.dt.day_name()
    enriched["calendar_day_of_week"] = dt_series.dt.dayofweek.astype("int8")
    enriched["calendar_month"] = dt_series.dt.month.astype("int8")
    enriched["calendar_year"] = dt_series.dt.year.astype("int16")
    enriched["calendar_week_key"] = dt_series.dt.isocalendar().week.astype("int32")
    enriched["event_name_1"] = None
    enriched["event_type_1"] = None
    enriched["event_name_2"] = None
    enriched["event_type_2"] = None
    enriched["weather_precipitation"] = None
    enriched["weather_temperature"] = None
    enriched["weather_humidity"] = None
    enriched["weather_wind_level"] = None
    return enriched


def build_bakery_standardized_frame(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    location_id: str = DEFAULT_LOCATION_ID,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.DataFrame:
    """Aggregate bakery ticket-line data into the canonical daily demand contract in memory."""

    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Bakery source file does not exist: {source_path}")

    frame = pd.read_csv(source_path)
    grouped = _build_bakery_grouped_frame(frame, location_id=location_id)
    enriched = _attach_bakery_canonical_columns(
        grouped,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
    return align_lazy_frame_to_canonical_schema(pl.from_pandas(enriched).lazy()).collect()


def build_bakery_standardized_dataset(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    location_id: str = DEFAULT_LOCATION_ID,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> Path:
    """Aggregate bakery ticket-line data into the canonical daily demand contract."""

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    canonical = build_bakery_standardized_frame(
        input_path=input_path,
        location_id=location_id,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
    canonical.write_parquet(target_path)
    return target_path

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import polars as pl

from praedixa.platform.datasets.standardization.schema import align_lazy_frame_to_canonical_schema, build_series_id_expr


FIRST_PARTY_DAILY_TEMPLATE_COLUMNS = [
    "dt",
    "location_id",
    "product_id",
    "observed_demand_qty",
    "observed_revenue_net",
    "observed_discount_amount",
    "promo_flag",
    "holiday_flag",
    "activity_flag",
    "observed_stockout_flag",
    "observed_stockout_available",
    "observed_stockout_intensity",
    "day_complete_flag",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "weather_precipitation",
    "weather_temperature",
    "weather_humidity",
    "weather_wind_level",
]

FIRST_PARTY_LOCATION_PROFILE_TEMPLATE_COLUMNS = [
    "location_id",
    "country_code",
    "city_name",
    "drive_through_flag",
    "delivery_flag",
    "pickup_flag",
]

FrameColumnDefault: TypeAlias = pd.Series | str | bool | float | None

FIRST_PARTY_PRODUCT_PROFILE_TEMPLATE_COLUMNS = [
    "product_id",
    "product_family",
    "product_subfamily",
]


def _empty_template_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _template_readme_text() -> str:
    return "\n".join(
        [
            "# Praedixa first-party onboarding bundle",
            "",
            "Files:",
            "- first_party_daily_template.csv: minimum daily demand feed.",
            "- first_party_location_profile_template.csv: static site metadata for cold start.",
            "- first_party_product_profile_template.csv: static product/menu metadata.",
            "",
            "Required daily minimum:",
            "- dt",
            "- location_id",
            "- product_id",
            "- observed_demand_qty",
            "",
            "Once filled, convert the daily feed to parquet and save it as",
            "`var/sources/commercial_datasets/raw/first_party_daily.parquet`.",
        ]
    )


def build_first_party_onboarding_templates(output_dir: str | Path) -> dict[str, Path]:
    """Write the first-party onboarding CSV templates and README."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    daily_template = target_dir / "first_party_daily_template.csv"
    location_profile_template = target_dir / "first_party_location_profile_template.csv"
    product_profile_template = target_dir / "first_party_product_profile_template.csv"
    readme = target_dir / "README.md"

    _empty_template_frame(FIRST_PARTY_DAILY_TEMPLATE_COLUMNS).to_csv(daily_template, index=False)
    _empty_template_frame(FIRST_PARTY_LOCATION_PROFILE_TEMPLATE_COLUMNS).to_csv(
        location_profile_template,
        index=False,
    )
    _empty_template_frame(FIRST_PARTY_PRODUCT_PROFILE_TEMPLATE_COLUMNS).to_csv(
        product_profile_template,
        index=False,
    )
    readme.write_text(_template_readme_text(), encoding="utf-8")

    return {
        "daily_template": daily_template,
        "location_profile_template": location_profile_template,
        "product_profile_template": product_profile_template,
        "readme": readme,
    }


def _normalize_first_party_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["dt"] = pd.to_datetime(normalized["dt"], errors="coerce").dt.date
    normalized["location_id"] = normalized["location_id"].astype(str).str.strip()
    normalized["product_id"] = normalized["product_id"].astype(str).str.strip()
    normalized["observed_demand_qty"] = pd.to_numeric(normalized["observed_demand_qty"], errors="coerce")
    return normalized.loc[
        normalized["dt"].notna()
        & normalized["location_id"].ne("")
        & normalized["product_id"].ne("")
        & normalized["observed_demand_qty"].notna()
    ].copy()


def _calendar_defaults(normalized: pd.DataFrame) -> dict[str, FrameColumnDefault]:
    datetime_index = pd.to_datetime(normalized["dt"], errors="coerce")
    return {
        "calendar_weekday_name": datetime_index.dt.day_name(),
        "calendar_day_of_week": datetime_index.dt.dayofweek.astype("Int8"),
        "calendar_month": datetime_index.dt.month.astype("Int8"),
        "calendar_year": datetime_index.dt.year.astype("Int16"),
        "calendar_week_key": datetime_index.dt.isocalendar().week.astype("Int32"),
    }


def _default_first_party_values(
    *,
    source_partition: str,
    source_run_id: str,
    silver_run_id: str,
    normalized: pd.DataFrame,
) -> dict[str, FrameColumnDefault]:
    defaults: dict[str, FrameColumnDefault] = {
        "dataset_source": "first_party_daily",
        "source_partition": source_partition,
        "source_run_id": source_run_id,
        "silver_run_id": silver_run_id,
        "region_id": None,
        "org_group_id": None,
        "category_level_1": None,
        "category_level_2": None,
        "category_level_3": None,
        "observed_revenue_net": None,
        "observed_discount_amount": None,
        "target_semantics": "observed_sales",
        "target_source": "observed_sales",
        "promo_flag": False,
        "holiday_flag": False,
        "activity_flag": False,
        "observed_stockout_flag": None,
        "observed_stockout_available": False,
        "observed_stockout_intensity": None,
        "day_complete_flag": None,
        "event_name_1": None,
        "event_type_1": None,
        "event_name_2": None,
        "event_type_2": None,
        "weather_precipitation": None,
        "weather_temperature": None,
        "weather_humidity": None,
        "weather_wind_level": None,
    }
    defaults.update(_calendar_defaults(normalized))
    return defaults


def _apply_missing_defaults(
    normalized: pd.DataFrame,
    defaults: dict[str, FrameColumnDefault],
) -> pd.DataFrame:
    for column_name, default_value in defaults.items():
        if column_name not in normalized.columns:
            normalized[column_name] = default_value
    return normalized


def _finalize_label_columns(normalized: pd.DataFrame) -> pd.DataFrame:
    finalized = normalized.copy()
    stockout_flag = finalized.get("observed_stockout_flag")
    if stockout_flag is None:
        stockout_mask = pd.Series(False, index=finalized.index, dtype=bool)
    else:
        stockout_mask = pd.Series(stockout_flag).fillna(False).astype(bool)
    stockout_available = pd.Series(
        finalized.get("observed_stockout_available", False),
        index=finalized.index,
    ).fillna(False).astype(bool)
    finalized["censor_flag"] = stockout_mask
    finalized["label_quality_score"] = np.where(
        stockout_mask,
        0.5,
        np.where(stockout_available, 1.0, 0.75),
    ).astype("float32")
    finalized["usable_for_training_flag"] = (~stockout_mask).astype(bool)
    return finalized


def standardize_first_party_daily_frame(
    frame: pd.DataFrame,
    *,
    source_partition: str,
    source_run_id: str,
    silver_run_id: str,
) -> pl.DataFrame:
    """Map a minimal first-party daily feed to the canonical daily schema."""

    normalized = _normalize_first_party_frame(frame)
    normalized = _apply_missing_defaults(
        normalized,
        _default_first_party_values(
            source_partition=source_partition,
            source_run_id=source_run_id,
            silver_run_id=silver_run_id,
            normalized=normalized,
        ),
    )
    normalized = _finalize_label_columns(normalized)

    lazy_frame = pl.from_pandas(normalized).lazy().with_columns(
        [
            pl.lit("first_party_daily").alias("dataset_source"),
            build_series_id_expr("location_id", "product_id").alias("series_id"),
        ]
    )
    return align_lazy_frame_to_canonical_schema(lazy_frame).collect()

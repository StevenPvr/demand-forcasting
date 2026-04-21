from __future__ import annotations

from typing import Any, cast

import pandas as pd

UCI_ONLINE_RETAIL_ASSUMPTION_SOURCE = "uci_online_retail_single_uk_retailer_hypothesis"
ECOMMERCE_SITE_FORMAT = "ecommerce"
DELIVERY_ONLY_SERVICE_MODEL = "delivery_only"
COUNTER_SERVICE_MODEL = "counter_service"
STORE_PICK_PACK_SERVICE_MODEL = "store_pick_pack"
DATASET_SOURCE_COL = "dataset_source"
LOCATION_ID_COL = "location_id"
COUNTRY_CODE_COL = "country_code"
REGION_CODE_COL = "region_code"
ASSUMPTION_SOURCE_COL = "assumption_source"
SCHOOL_ZONE_COL = "school_zone"
WEATHER_LOCATION_LABEL_COL = "weather_location_label"
SITE_FORMAT_COL = "site_format"
SERVICE_MODEL_COL = "service_model"


def _is_missing_country_code(country_code: object) -> bool:
    if country_code is None:
        return True
    if isinstance(country_code, float):
        return bool(pd.isna(country_code))
    return False

def _fixed_source_metadata(
    *,
    country_code: str | None,
    assumption_source: str,
    site_format: str,
    service_model: str,
) -> dict[str, object]:
    return {
        "country_code": country_code,
        "assumption_source": assumption_source,
        "site_format": site_format,
        "service_model": service_model,
    }

FIXED_SOURCE_METADATA = {
    "freshretail_lt": _fixed_source_metadata(
        country_code="CN",
        assumption_source="freshretail_lt_without_public_geocoding",
        site_format="grocery",
        service_model=STORE_PICK_PACK_SERVICE_MODEL,
    ),
    "uci_online_retail": _fixed_source_metadata(
        country_code="GB",
        assumption_source=UCI_ONLINE_RETAIL_ASSUMPTION_SOURCE,
        site_format=ECOMMERCE_SITE_FORMAT,
        service_model=DELIVERY_ONLY_SERVICE_MODEL,
    ),
    "uci_online_retail_ii": _fixed_source_metadata(
        country_code="GB",
        assumption_source=UCI_ONLINE_RETAIL_ASSUMPTION_SOURCE,
        site_format=ECOMMERCE_SITE_FORMAT,
        service_model=DELIVERY_ONLY_SERVICE_MODEL,
    ),
    "mendeley_pharmacy_id": _fixed_source_metadata(
        country_code="ID",
        assumption_source="mendeley_pharmacy_indonesia_country_hypothesis",
        site_format="pharmacy",
        service_model=COUNTER_SERVICE_MODEL,
    ),
    "mendeley_bangladesh_retail": _fixed_source_metadata(
        country_code="BD",
        assumption_source="mendeley_bangladesh_retail_country_hypothesis",
        site_format="general_trade",
        service_model="field_replenishment",
    ),
    "mendeley_ecommerce": _fixed_source_metadata(
        country_code=None,
        assumption_source="mendeley_ecommerce_country_unknown",
        site_format=ECOMMERCE_SITE_FORMAT,
        service_model=DELIVERY_ONLY_SERVICE_MODEL,
    ),
    "first_party_daily": _fixed_source_metadata(
        country_code=None,
        assumption_source="first_party_location_metadata_pending_client_onboarding",
        site_format="unknown",
        service_model="unknown",
    ),
    "synthetic_v1": _fixed_source_metadata(
        country_code=None,
        assumption_source="synthetic_location_metadata_generated_with_internal_defaults",
        site_format="synthetic_food_service",
        service_model=COUNTER_SERVICE_MODEL,
    ),
}


def _location_profile_fields(
    *,
    site_format: str,
    service_model: str,
    trade_area_type: str,
    drive_through_flag: bool,
    delivery_flag: bool,
    pickup_flag: bool,
    late_night_flag: bool,
    mall_flag: bool,
    transit_hub_flag: bool,
    tourism_flag: bool,
    office_density_bucket: str,
    residential_density_bucket: str,
    competition_intensity_bucket: str,
) -> dict[str, object]:
    return {
        "site_format": site_format,
        "service_model": service_model,
        "drive_through_flag": drive_through_flag,
        "delivery_flag": delivery_flag,
        "pickup_flag": pickup_flag,
        "late_night_flag": late_night_flag,
        "trade_area_type": trade_area_type,
        "mall_flag": mall_flag,
        "transit_hub_flag": transit_hub_flag,
        "tourism_flag": tourism_flag,
        "office_density_bucket": office_density_bucket,
        "residential_density_bucket": residential_density_bucket,
        "competition_intensity_bucket": competition_intensity_bucket,
    }
def _bakery_location_metadata_row(
    location: object,
    *,
    bakery_city_name: str,
    bakery_school_zone: str,
    bakery_latitude: float,
    bakery_longitude: float,
) -> dict[str, object]:
    return {
        DATASET_SOURCE_COL: getattr(location, DATASET_SOURCE_COL),
        LOCATION_ID_COL: getattr(location, LOCATION_ID_COL),
        COUNTRY_CODE_COL: "FR",
        REGION_CODE_COL: None,
        "city_name": bakery_city_name,
        "latitude": bakery_latitude,
        "longitude": bakery_longitude,
        SCHOOL_ZONE_COL: bakery_school_zone,
        WEATHER_LOCATION_LABEL_COL: "manual_bakery_location_proxy",
        ASSUMPTION_SOURCE_COL: "bakery_manual_default_location_hypothesis",
        **_location_profile_fields(
            site_format="bakery",
            service_model=COUNTER_SERVICE_MODEL,
            trade_area_type="urban_high_street",
            drive_through_flag=False,
            delivery_flag=False,
            pickup_flag=True,
            late_night_flag=False,
            mall_flag=False,
            transit_hub_flag=False,
            tourism_flag=False,
            office_density_bucket="medium",
            residential_density_bucket="medium",
            competition_intensity_bucket="high",
        ),
    }


def _freshretail_location_metadata_row(location: object) -> dict[str, object]:
    return {
        DATASET_SOURCE_COL: getattr(location, DATASET_SOURCE_COL),
        LOCATION_ID_COL: getattr(location, LOCATION_ID_COL),
        COUNTRY_CODE_COL: "CN",
        REGION_CODE_COL: getattr(location, REGION_CODE_COL),
        "city_name": None,
        "latitude": None,
        "longitude": None,
        SCHOOL_ZONE_COL: None,
        WEATHER_LOCATION_LABEL_COL: None,
        ASSUMPTION_SOURCE_COL: "freshretail_city_id_without_public_geocoding",
        **_location_profile_fields(
            site_format="grocery",
            service_model=STORE_PICK_PACK_SERVICE_MODEL,
            trade_area_type="urban_neighborhood",
            drive_through_flag=False,
            delivery_flag=True,
            pickup_flag=True,
            late_night_flag=False,
            mall_flag=False,
            transit_hub_flag=False,
            tourism_flag=False,
            office_density_bucket="unknown",
            residential_density_bucket="medium",
            competition_intensity_bucket="medium",
        ),
    }


def _fixed_country_location_metadata_row(
    location: object,
    *,
    country_code: str | None,
    assumption_source: str,
    site_format: str,
    service_model: str,
) -> dict[str, object]:
    return {
        DATASET_SOURCE_COL: getattr(location, DATASET_SOURCE_COL),
        LOCATION_ID_COL: getattr(location, LOCATION_ID_COL),
        COUNTRY_CODE_COL: country_code,
        REGION_CODE_COL: getattr(location, REGION_CODE_COL),
        "city_name": None,
        "latitude": None,
        "longitude": None,
        SCHOOL_ZONE_COL: None,
        WEATHER_LOCATION_LABEL_COL: None,
        ASSUMPTION_SOURCE_COL: assumption_source,
        **_location_profile_fields(
            site_format=site_format,
            service_model=service_model,
            trade_area_type="unknown",
            drive_through_flag=False,
            delivery_flag=service_model in {"delivery_only", "store_pick_pack"},
            pickup_flag=service_model in {"counter_service", "store_pick_pack"},
            late_night_flag=False,
            mall_flag=False,
            transit_hub_flag=False,
            tourism_flag=False,
            office_density_bucket="unknown",
            residential_density_bucket="unknown",
            competition_intensity_bucket="unknown",
        ),
    }


def _location_metadata_row(
    location: object,
    *,
    bakery_city_name: str,
    bakery_school_zone: str,
    bakery_latitude: float,
    bakery_longitude: float,
) -> dict[str, object] | None:
    dataset_source = getattr(location, DATASET_SOURCE_COL)
    if dataset_source == "bakery":
        return _bakery_location_metadata_row(
            location,
            bakery_city_name=bakery_city_name,
            bakery_school_zone=bakery_school_zone,
            bakery_latitude=bakery_latitude,
            bakery_longitude=bakery_longitude,
        )
    if dataset_source == "freshretail":
        return _freshretail_location_metadata_row(location)
    spec = FIXED_SOURCE_METADATA.get(str(dataset_source))
    if spec is None:
        return None
    return _fixed_country_location_metadata_row(
        location,
        country_code=cast(str | None, spec["country_code"]),
        assumption_source=str(spec["assumption_source"]),
        site_format=str(spec["site_format"]),
        service_model=str(spec["service_model"]),
    )


def build_location_metadata_frame(
    silver_locations: pd.DataFrame,
    *,
    bakery_city_name: str,
    bakery_school_zone: str,
    bakery_latitude: float,
    bakery_longitude: float,
) -> pd.DataFrame:
    """Create explicit location metadata and geocoding assumptions for exogenous joins."""

    rows: list[dict[str, object]] = []
    for location in silver_locations.itertuples(index=False):
        row = _location_metadata_row(
            location,
            bakery_city_name=bakery_city_name,
            bakery_school_zone=bakery_school_zone,
            bakery_latitude=bakery_latitude,
            bakery_longitude=bakery_longitude,
        )
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["dataset_source", "location_id"]).reset_index(drop=True)


def compute_country_years(
    silver_locations: pd.DataFrame,
    location_metadata: pd.DataFrame,
) -> dict[str, list[int]]:
    """Return required calendar years per country, including target-day spillover."""

    merged = silver_locations.merge(
        location_metadata[[DATASET_SOURCE_COL, LOCATION_ID_COL, COUNTRY_CODE_COL]],
        on=[DATASET_SOURCE_COL, LOCATION_ID_COL],
        how="left",
    )
    country_years: dict[str, set[int]] = {}
    for row in cast(list[dict[str, Any]], merged.to_dict(orient="records")):
        country_code = row.get(COUNTRY_CODE_COL)
        if _is_missing_country_code(country_code):
            continue
        min_year = int(pd.Timestamp(row["min_dt"]).year)
        max_year = int((pd.Timestamp(row["max_dt"]) + pd.Timedelta(days=1)).year)
        country_years.setdefault(str(country_code), set()).update(range(min_year, max_year + 1))
    return {country: sorted(years) for country, years in country_years.items()}

from __future__ import annotations

from typing import TypedDict

from praedixa.demand_forecast.backends.tft.feature_mapping import (
    TFT_EXPLICIT_ROLE_BY_COLUMN,
)
from praedixa.demand_forecast.backends.tft.feature_mapping_spec import (
    FEATURE_ROLE_ORDER,
    TFTColumnRole,
)


class FeatureContractEntry(TypedDict):
    role: str
    available_at_prediction: bool
    source_system: str


_STATIC_ROLES: frozenset[TFTColumnRole] = frozenset(
    {"group_id", "static_categorical", "static_real"}
)
_KNOWN_ROLES: frozenset[TFTColumnRole] = frozenset(
    {"time_varying_known_categorical", "time_varying_known_real"}
)


def _feature_role(column: str) -> TFTColumnRole:
    role = TFT_EXPLICIT_ROLE_BY_COLUMN.get(column)
    if role is None:
        raise ValueError(f"Explicit TFT mapping is missing feature column `{column}`.")
    return role


def feature_available_at_prediction(role: TFTColumnRole) -> bool:
    return role in _STATIC_ROLES or role in _KNOWN_ROLES


def _source_system_for_column(column: str) -> str:
    if column.startswith("weather_"):
        return "weather"
    if column.startswith("target_") or column.startswith("current_"):
        return "calendar_operational"
    if column.startswith("fr_") or column.startswith("gdp_"):
        return "macro"
    if column.startswith("lending_") or column.startswith("government_"):
        return "macro"
    if column.startswith("event_") or column.startswith("major_"):
        return "events"
    if column.endswith("_holiday_name") or column.endswith("_holiday_flag"):
        return "calendar"
    if column.startswith("data_quality_"):
        return "data_quality"
    if "stockout" in column or "closure" in column or "saturation" in column:
        return "operations"
    if "price" in column or "discount" in column or "promo" in column:
        return "pricing"
    if column in {"dataset_source", "client_id"}:
        return "metadata"
    if column.endswith("_status"):
        return "data_quality"
    return "operations"


def build_feature_contract(feature_cols: list[str]) -> dict[str, FeatureContractEntry]:
    contract: dict[str, FeatureContractEntry] = {}
    for column in feature_cols:
        role = _feature_role(column)
        contract[column] = {
            "role": role,
            "available_at_prediction": feature_available_at_prediction(role),
            "source_system": _source_system_for_column(column),
        }
    return contract


def validate_feature_contract(feature_cols: list[str]) -> dict[str, FeatureContractEntry]:
    contract = build_feature_contract(feature_cols)
    for column, metadata in contract.items():
        role = str(metadata["role"])
        if role in FEATURE_ROLE_ORDER[:4] and not bool(metadata["available_at_prediction"]):
            raise ValueError(
                "TFT feature contract marks a known or static feature as unavailable at prediction: "
                f"{column} ({role})."
            )
    return contract

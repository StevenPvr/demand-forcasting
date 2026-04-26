from __future__ import annotations

from typing import Literal, TypedDict

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
    max_publication_lag_hours: int
    decision_time_scope: str
    leakage_risk_level: str
    decision_profile: str


DecisionProfile = Literal["post_close_d_plus_1", "pre_close_d_plus_1"]
DEFAULT_DECISION_PROFILE: DecisionProfile = "post_close_d_plus_1"

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
    if column.startswith("label_quality_score") or column.startswith("data_quality_"):
        return "data_quality"
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
    if (
        "stockout" in column
        or "closure" in column
        or "saturation" in column
        or "restriction" in column
        or "censor" in column
        or "anomaly" in column
        or "missing_sales" in column
        or "day_complete" in column
    ):
        return "operations"
    if "price" in column or "discount" in column or "promo" in column:
        return "pricing"
    if column in {
        "dataset_source",
        "client_id",
        "source_role",
        "vertical_level_1",
        "vertical_level_2",
        "commerce_modality",
        "operation_type",
        "service_pattern",
        "country_code",
        "region_code",
        "region_known_flag",
        "city_name",
        "product_family",
        "product_subfamily",
        "product_subfamily_known_flag",
        "category_level_1",
        "category_level_1_known_flag",
        "category_level_2",
        "category_level_2_known_flag",
        "category_level_3",
        "category_level_3_known_flag",
        "product_taxonomy_depth",
    }:
        return "metadata"
    if column.endswith("_status"):
        return "data_quality"
    return "operations"


def _decision_time_scope_for_column(column: str) -> str:
    if (
        column.startswith("target_")
        or column.startswith("sin_target_")
        or column.startswith("cos_target_")
    ):
        return "target_calendar_known"
    if "_lag_" in column or column.startswith("lag_"):
        return "historical_observed"
    if (
        column.startswith("rolling_")
        or "_rolling_" in column
        or column.endswith("_rate_7")
        or column.endswith("_rate_28")
    ):
        return "decision_day_history"
    if column.startswith("current_") or column.startswith("observed_"):
        return "decision_day_observed"
    if column.startswith("data_quality_"):
        return "pipeline_metadata"
    return "static_or_planned"


def _max_publication_lag_hours(column: str, *, source_system: str) -> int:
    if source_system == "macro":
        return 744
    if source_system in {"weather", "events"}:
        return 24
    if column.startswith("current_") or column.startswith("observed_"):
        return 0
    return 0


def _leakage_risk_level(
    *,
    role: TFTColumnRole,
    decision_time_scope: str,
    source_system: str,
) -> str:
    if decision_time_scope == "target_observed":
        return "blocked"
    if role in _KNOWN_ROLES and decision_time_scope == "decision_day_observed":
        return "medium"
    if source_system == "weather" and decision_time_scope != "historical_observed":
        return "medium"
    return "low"


def _pre_close_block_reason(metadata: FeatureContractEntry) -> str | None:
    decision_time_scope = str(metadata["decision_time_scope"])
    source_system = str(metadata["source_system"])
    if decision_time_scope in {"decision_day_observed", "decision_day_history"}:
        return decision_time_scope
    if source_system == "weather" and decision_time_scope != "historical_observed":
        return "weather_not_point_in_time"
    return None


def build_feature_contract(
    feature_cols: list[str],
    *,
    decision_profile: DecisionProfile = DEFAULT_DECISION_PROFILE,
) -> dict[str, FeatureContractEntry]:
    contract: dict[str, FeatureContractEntry] = {}
    for column in feature_cols:
        role = _feature_role(column)
        source_system = _source_system_for_column(column)
        decision_time_scope = _decision_time_scope_for_column(column)
        contract[column] = {
            "role": role,
            "available_at_prediction": feature_available_at_prediction(role),
            "source_system": source_system,
            "max_publication_lag_hours": _max_publication_lag_hours(
                column,
                source_system=source_system,
            ),
            "decision_time_scope": decision_time_scope,
            "leakage_risk_level": _leakage_risk_level(
                role=role,
                decision_time_scope=decision_time_scope,
                source_system=source_system,
            ),
            "decision_profile": decision_profile,
        }
    return contract


def validate_feature_contract(
    feature_cols: list[str],
    *,
    decision_profile: DecisionProfile = DEFAULT_DECISION_PROFILE,
) -> dict[str, FeatureContractEntry]:
    contract = build_feature_contract(feature_cols, decision_profile=decision_profile)
    for column, metadata in contract.items():
        role = str(metadata["role"])
        if role in FEATURE_ROLE_ORDER[:4] and not bool(
            metadata["available_at_prediction"]
        ):
            raise ValueError(
                "TFT feature contract marks a known or static feature as unavailable at prediction: "
                f"{column} ({role})."
            )
        if str(metadata["leakage_risk_level"]) == "blocked":
            raise ValueError(
                f"TFT feature contract blocks a target-observed feature: {column}."
            )
        if decision_profile == "pre_close_d_plus_1":
            block_reason = _pre_close_block_reason(metadata)
            if block_reason is not None:
                raise ValueError(
                    "TFT feature contract blocks a feature unavailable for pre-close D+1 decisions: "
                    f"{column} ({block_reason})."
                )
    return contract

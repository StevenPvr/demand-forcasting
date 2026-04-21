from __future__ import annotations

from typing import Iterable, TypedDict

import pandas as pd
from praedixa.demand_forecast.backends.tft.feature_mapping_spec import (
    FEATURE_ROLE_ORDER,
    NON_FEATURE_COLUMN_ROLES,
    TFTColumnRole,
    TFT_GROUP_ID_COLUMNS,
    TFT_STATIC_CATEGORICAL_COLUMNS,
    TFT_STATIC_REAL_COLUMNS,
    TFT_TIME_VARYING_KNOWN_CATEGORICAL_COLUMNS,
    TFT_TIME_VARYING_KNOWN_REAL_COLUMNS,
    TFT_TIME_VARYING_UNKNOWN_CATEGORICAL_COLUMNS,
    TFT_TIME_VARYING_UNKNOWN_REAL_COLUMNS,
)


class TFTLayout(TypedDict):
    static_categoricals: list[str]
    static_reals: list[str]
    time_varying_known_categoricals: list[str]
    time_varying_known_reals: list[str]
    time_varying_unknown_categoricals: list[str]
    time_varying_unknown_reals: list[str]

_CATEGORICAL_FEATURE_ROLES: frozenset[TFTColumnRole] = frozenset(
    {
        "static_categorical",
        "time_varying_known_categorical",
        "time_varying_unknown_categorical",
    }
)


def _ordered_columns(
    source: pd.DataFrame | Iterable[str],
) -> list[str]:
    if isinstance(source, pd.DataFrame):
        return list(source.columns)
    return list(source)


def _build_explicit_role_map() -> dict[str, TFTColumnRole]:
    role_map: dict[str, TFTColumnRole] = {}
    grouped_columns: tuple[tuple[TFTColumnRole, Iterable[str]], ...] = (
        *tuple((role, (column,)) for column, role in NON_FEATURE_COLUMN_ROLES.items()),
        ("group_id", TFT_GROUP_ID_COLUMNS),
        ("static_categorical", TFT_STATIC_CATEGORICAL_COLUMNS),
        ("static_real", TFT_STATIC_REAL_COLUMNS),
        ("time_varying_known_categorical", TFT_TIME_VARYING_KNOWN_CATEGORICAL_COLUMNS),
        ("time_varying_known_real", TFT_TIME_VARYING_KNOWN_REAL_COLUMNS),
        ("time_varying_unknown_categorical", TFT_TIME_VARYING_UNKNOWN_CATEGORICAL_COLUMNS),
        ("time_varying_unknown_real", TFT_TIME_VARYING_UNKNOWN_REAL_COLUMNS),
    )
    duplicates: list[str] = []
    for role, columns in grouped_columns:
        for column in columns:
            if column in role_map:
                duplicates.append(column)
                continue
            role_map[column] = role
    if duplicates:
        raise ValueError(f"Duplicate TFT feature mapping entries: {sorted(set(duplicates))}")
    return role_map


TFT_EXPLICIT_ROLE_BY_COLUMN: dict[str, TFTColumnRole] = _build_explicit_role_map()


def validate_explicit_tft_mapping(columns: pd.DataFrame | Iterable[str]) -> None:
    ordered_columns = _ordered_columns(columns)
    unmapped = sorted(column for column in ordered_columns if column not in TFT_EXPLICIT_ROLE_BY_COLUMN)
    if unmapped:
        raise ValueError(
            "Explicit TFT mapping is missing columns: "
            + ", ".join(unmapped)
        )


def select_explicit_tft_group_id_columns(columns: pd.DataFrame | Iterable[str]) -> list[str]:
    ordered_columns = _ordered_columns(columns)
    validate_explicit_tft_mapping(ordered_columns)
    missing_group_ids = [column for column in TFT_GROUP_ID_COLUMNS if column not in ordered_columns]
    if missing_group_ids:
        raise ValueError(f"Required TFT group_id columns are missing: {missing_group_ids}")
    return [column for column in TFT_GROUP_ID_COLUMNS if column in ordered_columns]


def select_explicit_tft_feature_columns(
    columns: pd.DataFrame | Iterable[str],
    *,
    excluded_cols: Iterable[str] = (),
) -> list[str]:
    ordered_columns = _ordered_columns(columns)
    validate_explicit_tft_mapping(ordered_columns)
    excluded = set(excluded_cols)
    feature_columns: list[str] = []
    for role in FEATURE_ROLE_ORDER:
        role_columns = [
            column
            for column, column_role in TFT_EXPLICIT_ROLE_BY_COLUMN.items()
            if column_role == role
        ]
        for column in role_columns:
            if column in ordered_columns and column not in excluded:
                feature_columns.append(column)
    return feature_columns


def select_explicit_tft_categorical_columns(feature_cols: Iterable[str]) -> list[str]:
    categorical_columns: list[str] = []
    for column in feature_cols:
        role = TFT_EXPLICIT_ROLE_BY_COLUMN.get(column)
        if role is None:
            raise ValueError(f"Explicit TFT mapping is missing feature column `{column}`.")
        if role in _CATEGORICAL_FEATURE_ROLES:
            categorical_columns.append(column)
    return categorical_columns


def _build_empty_tft_layout() -> TFTLayout:
    return {
        "static_categoricals": [],
        "static_reals": [],
        "time_varying_known_categoricals": [],
        "time_varying_known_reals": [],
        "time_varying_unknown_categoricals": [],
        "time_varying_unknown_reals": [],
    }


def resolve_explicit_tft_layout(feature_cols: Iterable[str]) -> TFTLayout:
    layout: TFTLayout = _build_empty_tft_layout()
    for column in feature_cols:
        role = TFT_EXPLICIT_ROLE_BY_COLUMN.get(column)
        if role is None:
            raise ValueError(f"Explicit TFT mapping is missing feature column `{column}`.")
        if role not in FEATURE_ROLE_ORDER:
            raise ValueError(f"Column `{column}` has non-feature TFT role `{role}`.")
        if role == "static_categorical":
            layout["static_categoricals"].append(column)
        elif role == "static_real":
            layout["static_reals"].append(column)
        elif role == "time_varying_known_categorical":
            layout["time_varying_known_categoricals"].append(column)
        elif role == "time_varying_known_real":
            layout["time_varying_known_reals"].append(column)
        elif role == "time_varying_unknown_categorical":
            layout["time_varying_unknown_categoricals"].append(column)
        elif role == "time_varying_unknown_real":
            layout["time_varying_unknown_reals"].append(column)
    return layout

from __future__ import annotations

import pandas as pd

from praedixa.demand_forecast.backends.tft.feature_contract import build_feature_contract
from praedixa.demand_forecast.training.constants import DEFAULT_DATASET_SOURCE_COL, DEFAULT_EXCLUDED_RISKY_FEATURE_COLS
from praedixa.demand_forecast.contracts.targets import TargetContract, build_target_contract_metadata


def drop_constant_feature_columns(
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[str], list[str]]:
    # TFT must keep static metadata even when they are constant inside one train slice.
    _ = frame
    return list(feature_cols), []


def build_feature_audit_payload(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    feature_cols: list[str],
    raw_feature_cols: list[str],
    constant_feature_cols: list[str],
    identifier_feature_cols: list[str],
    folds: list[dict[str, object]],
    target_contract: TargetContract,
) -> dict[str, object]:
    return {
        "raw_feature_count": len(raw_feature_cols),
        "final_feature_count": len(feature_cols),
        "constant_feature_cols": constant_feature_cols,
        "identifier_feature_cols": identifier_feature_cols,
        "excluded_risky_feature_cols": list(DEFAULT_EXCLUDED_RISKY_FEATURE_COLS),
        **build_target_contract_metadata(target_contract),
        "learning_target_summary_train": _target_summary(train_frame, target_contract.learning_target_col),
        "learning_target_summary_tuning": _target_summary(tuning_frame, target_contract.learning_target_col),
        "absolute_target_summary_train": _target_summary(train_frame, target_contract.absolute_target_col),
        "absolute_target_summary_tuning": _target_summary(tuning_frame, target_contract.absolute_target_col),
        "folds": [
            {key: value for key, value in fold.items() if key not in {"train_idx", "valid_idx"}}
            for fold in folds
        ],
        "feature_contract": build_feature_contract(feature_cols),
        "top_missingness": _top_missingness_rows(train_frame, tuning_frame, feature_cols),
    }


def _target_summary(frame: pd.DataFrame, target_col: str) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for dataset_source, dataset_frame in frame.groupby(DEFAULT_DATASET_SOURCE_COL, sort=False):
        target = dataset_frame[target_col]
        summary[str(dataset_source)] = {
            "rows": int(len(dataset_frame)),
            "min": float(target.min()),
            "max": float(target.max()),
            "mean": float(target.mean()),
            "std": float(target.std()),
            "nonpositive_rows": int((target <= 0).fillna(False).sum()),
        }
    return summary


def _top_missingness_rows(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict[str, float | str]]:
    missingness = [
        {
            "feature": column,
            "null_rate_train": float(train_frame[column].isna().mean()),
            "null_rate_tuning": float(tuning_frame[column].isna().mean()),
        }
        for column in feature_cols
    ]
    missingness.sort(key=lambda row: float(row["null_rate_train"]), reverse=True)
    return missingness[:25]

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from praedixa.demand_forecast.evaluation.bakery_metrics import REFERENCE_DATE_COL, REFERENCE_PRODUCT_COL, REFERENCE_TARGET_COL
from praedixa.demand_forecast.evaluation.bakery_metrics import load_reference_split
from praedixa.demand_forecast.evaluation.reference import build_bakery_reference_feature_frame
from praedixa.platform.utils.memory import read_parquet_projected
from praedixa.demand_forecast.training.constants import DEFAULT_DATASET_SOURCE_COL, DEFAULT_DATE_COL
from praedixa.demand_forecast.training.pipeline import load_gold_train_tuning_frames
from praedixa.demand_forecast.contracts.targets import TargetContract, ensure_learning_target_column, resolve_target_contract


def resolve_reference_product_col(frame: pd.DataFrame) -> str:
    for candidate in ("product_id", "product", "series_id", "sku_id"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Unable to resolve a product identifier column for evaluation.")


def resolve_reference_date_col(frame: pd.DataFrame) -> str:
    for candidate in ("target_dt", "dt", "date"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Unable to resolve a date column for evaluation.")


def normalize_reference_split_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[REFERENCE_DATE_COL] = pd.to_datetime(normalized[REFERENCE_DATE_COL])
    normalized[REFERENCE_PRODUCT_COL] = normalized[REFERENCE_PRODUCT_COL].astype(str)
    normalized[REFERENCE_TARGET_COL] = normalized[REFERENCE_TARGET_COL].astype(float)
    return normalized.sort_values([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL]).reset_index(drop=True)


def to_reference_frame(
    frame: pd.DataFrame,
    *,
    target_col: str,
) -> pd.DataFrame:
    date_col = resolve_reference_date_col(frame)
    product_col = resolve_reference_product_col(frame)
    reference = pd.DataFrame(
        {
            REFERENCE_DATE_COL: pd.to_datetime(frame[date_col]),
            REFERENCE_PRODUCT_COL: frame[product_col].astype(str),
            REFERENCE_TARGET_COL: frame[target_col].astype(float),
        }
    )
    return reference.sort_values([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL]).reset_index(drop=True)


def load_gold_bakery_overlap_test_frame(
    *,
    duckdb_path: str | Path,
    reference_full_df: pd.DataFrame,
    reference_test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        gold_base_df = connection.execute(
            """
            select *
            from gold.gold_base_panel_d1
            where dataset_source = 'bakery'
            """
        ).fetchdf()
    finally:
        connection.close()

    overlap_df = build_bakery_reference_feature_frame(reference_full_df, reference_test_df, gold_base_df)
    if overlap_df.empty:
        raise ValueError("No bakery reference rows were materialized for evaluation.")

    scored_reference_test = normalize_reference_split_frame(reference_test_df)
    actual_diff = float(
        abs(
            scored_reference_test[REFERENCE_TARGET_COL].astype(float).to_numpy()
            - overlap_df["target_demand_qty_d_plus_1"].astype(float).to_numpy()
        ).max()
    )
    if actual_diff > 0.0:
        raise ValueError(
            f"Materialized bakery reference target does not match bakery_sales actuals (max_abs_diff={actual_diff})."
        )

    overlap_start_date = pd.Timestamp(scored_reference_test[REFERENCE_DATE_COL].min())
    metadata: dict[str, object] = {
        "reference_test_rows": int(len(reference_test_df)),
        "overlap_test_rows": int(len(scored_reference_test)),
        "missing_reference_test_rows": 0,
        "overlap_start_date": overlap_start_date.strftime("%Y-%m-%d"),
        "overlap_end_date": pd.Timestamp(scored_reference_test[REFERENCE_DATE_COL].max()).strftime("%Y-%m-%d"),
        "product_count": int(scored_reference_test[REFERENCE_PRODUCT_COL].nunique()),
    }
    return overlap_df, scored_reference_test, metadata


def prepare_training_target_frame(
    frame: pd.DataFrame,
    *,
    target_contract: TargetContract,
) -> pd.DataFrame:
    prepared = ensure_learning_target_column(frame.copy(), target_contract)
    return prepared[prepared[target_contract.learning_target_col].notna()].copy()


def prepare_scored_test_frame(
    test_frame: pd.DataFrame,
    scored_reference_test: pd.DataFrame,
    *,
    target_contract: TargetContract,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    prepared = ensure_learning_target_column(test_frame.copy(), target_contract)
    valid_mask = prepared[target_contract.learning_target_col].notna().to_numpy()
    filtered_test = prepared.loc[valid_mask].copy().reset_index(drop=True)
    filtered_reference = scored_reference_test.loc[valid_mask].copy().reset_index(drop=True)
    dropped_rows = int((~valid_mask).sum())
    return filtered_test, filtered_reference, dropped_rows


def load_local_mode_frames(
    *,
    train_selection_input_path: str | Path,
    train_tuning_input_path: str | Path,
    val_input_path: str | Path,
    requested_target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object] | None]:
    train_frame = read_parquet_projected(Path(train_selection_input_path))
    valid_frame = read_parquet_projected(Path(train_tuning_input_path))
    test_frame = read_parquet_projected(Path(val_input_path))
    combined_history = pd.concat([train_frame, valid_frame], ignore_index=True)
    target_contract = resolve_target_contract(combined_history, test_frame, requested_target_col=requested_target_col)
    absolute_target_col = target_contract.absolute_target_col
    history_reference = to_reference_frame(combined_history, target_col=absolute_target_col)
    test_reference = to_reference_frame(test_frame, target_col=absolute_target_col)
    test_frame = test_frame.copy()
    test_frame[REFERENCE_DATE_COL] = test_reference[REFERENCE_DATE_COL].to_numpy()
    test_frame[REFERENCE_PRODUCT_COL] = test_reference[REFERENCE_PRODUCT_COL].to_numpy()
    return train_frame, valid_frame, test_frame, history_reference, test_reference, None


def load_gold_reference_mode_frames(
    *,
    duckdb_path: str | Path,
    gold_table: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    bakery_reference_train_csv: str | Path,
    bakery_reference_val_csv: str | Path,
    bakery_reference_test_csv: str | Path,
    train_frame: pd.DataFrame | None = None,
    valid_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    resolved_train_frame = train_frame
    resolved_valid_frame = valid_frame
    if resolved_train_frame is None or resolved_valid_frame is None:
        resolved_train_frame, resolved_valid_frame, _, _, _ = load_gold_train_tuning_frames(
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            logger=logging.getLogger(__name__),
            date_col=DEFAULT_DATE_COL,
            dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
            train_sample_fraction=train_sample_fraction,
            tuning_sample_fraction=tuning_sample_fraction,
        )
    reference_train = load_reference_split(bakery_reference_train_csv)
    reference_val = load_reference_split(bakery_reference_val_csv)
    reference_test = load_reference_split(bakery_reference_test_csv)
    reference_full = pd.concat([reference_train, reference_val, reference_test], ignore_index=True)
    overlap_test_frame, scored_reference_test, overlap_metadata = load_gold_bakery_overlap_test_frame(
        duckdb_path=duckdb_path,
        reference_full_df=reference_full,
        reference_test_df=reference_test,
    )
    history_reference = pd.concat([reference_train, reference_val], ignore_index=True)
    history_reference = history_reference.sort_values([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL]).reset_index(drop=True)
    return (
        resolved_train_frame,
        resolved_valid_frame,
        overlap_test_frame,
        history_reference,
        scored_reference_test,
        overlap_metadata,
    )

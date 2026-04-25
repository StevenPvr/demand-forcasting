from __future__ import annotations

"""Split chronologique du dataset journalier en train, validation et test."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data_preprocessing.constants import (
    CSV_ENCODING,
    DAILY_FREQUENCY,
    ORIGIN_DATE_COLUMN,
    TARGET_COLUMN,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data_preprocessing.standardization import standardize_splits
from src.data_preprocessing.stationarity import build_stationarity_report
from src.data_preprocessing.temporal_features import build_modeling_features

LOGGER: logging.Logger = logging.getLogger(__name__)


def _validate_split_ratios() -> None:
    """Verifie que les ratios de split couvrent tout le dataset."""

    ratio_sum: float = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")


def _sort_by_time(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Trie le dataset selon ses metadonnees temporelles."""

    return dataset_df.sort_values([ORIGIN_DATE_COLUMN]).reset_index(drop=True)


def _split_boundaries(row_count: int) -> tuple[int, int]:
    """Calcule les indices de coupure 70/15/15."""

    train_end: int = int(row_count * TRAIN_RATIO)
    val_end: int = train_end + int(row_count * VAL_RATIO)
    return train_end, val_end


def split_baguette_dataframe(
    dataset_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cree des splits chronologiques train, validation et test."""

    _validate_split_ratios()
    ordered_df: pd.DataFrame = _sort_by_time(dataset_df)
    train_end, val_end = _split_boundaries(len(ordered_df))
    train_df: pd.DataFrame = ordered_df.iloc[:train_end].reset_index(drop=True)
    val_df: pd.DataFrame = ordered_df.iloc[train_end:val_end].reset_index(drop=True)
    test_df: pd.DataFrame = ordered_df.iloc[val_end:].reset_index(drop=True)
    LOGGER.info(
        "Split %d rows into train=%d, val=%d, test=%d",
        len(ordered_df),
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def _write_split(df: pd.DataFrame, output_csv: Path) -> None:
    """Ecrit un split CSV sur disque."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding=CSV_ENCODING)


def _write_stationarity_report(report: dict[str, Any], output_json: Path) -> None:
    """Ecrit le rapport de stationnarite sur disque."""

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding=CSV_ENCODING)


def _write_preprocessing_bundle(bundle: dict[str, Any], output_path: Path) -> None:
    """Persiste le bundle de preprocessing pour la future inference."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def split_baguette_dataset(
    input_csv: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    stationarity_report_json: Path,
    preprocessing_bundle_path: Path,
) -> dict[str, int]:
    """Charge le dataset, cree les features, puis ecrit les splits et le bundle."""

    start_time: float = time.perf_counter()
    raw_df: pd.DataFrame = pd.read_csv(input_csv)
    featured_df: pd.DataFrame = build_modeling_features(raw_df)
    train_probe_df, _, _ = split_baguette_dataframe(featured_df)
    stationarity_report = build_stationarity_report(
        train_probe_df,
        target_column=TARGET_COLUMN,
    )
    _write_stationarity_report(stationarity_report, stationarity_report_json)
    train_df, val_df, test_df = split_baguette_dataframe(featured_df)
    train_df, val_df, test_df, preprocessing_bundle = standardize_splits(train_df, val_df, test_df)
    preprocessing_bundle["target_column"] = TARGET_COLUMN
    preprocessing_bundle["frequency"] = DAILY_FREQUENCY
    _write_preprocessing_bundle(preprocessing_bundle, preprocessing_bundle_path)
    _write_split(train_df, train_csv)
    _write_split(val_df, val_csv)
    _write_split(test_df, test_csv)
    LOGGER.info("Saved train split to %s", train_csv)
    LOGGER.info("Saved val split to %s", val_csv)
    LOGGER.info("Saved test split to %s", test_csv)
    LOGGER.info("Saved stationarity report to %s", stationarity_report_json)
    LOGGER.info("Saved preprocessing bundle to %s", preprocessing_bundle_path)
    LOGGER.info("Split completed in %.3f seconds", time.perf_counter() - start_time)
    return {
        "raw_row_count": len(raw_df),
        "row_count": len(featured_df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }

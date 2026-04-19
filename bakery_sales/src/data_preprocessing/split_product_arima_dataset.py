from __future__ import annotations

"""Split chronologique par produit pour le pipeline ARIMA journalier."""

import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd

from src.data_preprocessing.constants_arima import (
    CSV_ENCODING,
    DAILY_FREQUENCY,
    DATE_COLUMN,
    PRODUCT_COLUMN,
    TARGET_COLUMN,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data_preprocessing.stationarity import build_stationarity_report

LOGGER: logging.Logger = logging.getLogger(__name__)


def _validate_split_ratios() -> None:
    """Verifie que les ratios couvrent bien tout le dataset."""

    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")


def _sort_product_frame(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Trie le dataset par produit puis date."""

    ordered_df = dataset_df.copy()
    ordered_df[DATE_COLUMN] = pd.to_datetime(ordered_df[DATE_COLUMN], format="%Y-%m-%d")
    return ordered_df.sort_values([PRODUCT_COLUMN, DATE_COLUMN]).reset_index(drop=True)


def _split_boundaries(row_count: int) -> tuple[int, int]:
    """Calcule les indices de coupure par ratio."""

    train_end = int(row_count * TRAIN_RATIO)
    val_end = train_end + int(row_count * VAL_RATIO)
    return train_end, val_end


def _split_single_product_frame(
    product_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decoupe une serie produit en train/val/test chronologiques."""

    train_end, val_end = _split_boundaries(len(product_df))
    train_df = product_df.iloc[:train_end].reset_index(drop=True)
    val_df = product_df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = product_df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def split_product_arima_dataframe(
    dataset_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cree les splits chronologiques independants pour chaque produit."""

    _validate_split_ratios()
    ordered_df = _sort_product_frame(dataset_df)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, product_df in ordered_df.groupby(PRODUCT_COLUMN, sort=True):
        train_df, val_df, test_df = _split_single_product_frame(product_df.reset_index(drop=True))
        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    for split_df in (train_df, val_df, test_df):
        split_df[DATE_COLUMN] = pd.to_datetime(split_df[DATE_COLUMN], format="%Y-%m-%d").dt.strftime(
            "%Y-%m-%d"
        )
    LOGGER.info(
        "Split %d rows across %d products into train=%d, val=%d, test=%d",
        len(ordered_df),
        ordered_df[PRODUCT_COLUMN].nunique(),
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def _write_split(df: pd.DataFrame, output_csv: Path) -> None:
    """Ecrit un split sur disque."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding=CSV_ENCODING)


def _write_json(payload: dict[str, Any], output_json: Path) -> None:
    """Ecrit un JSON sur disque."""

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding=CSV_ENCODING)


def _stationarity_payload(train_df: pd.DataFrame) -> dict[str, Any]:
    """Construit un rapport descriptif de stationnarite par produit."""

    product_reports: dict[str, Any] = {}
    for product_name, product_df in train_df.groupby(PRODUCT_COLUMN, sort=True):
        product_reports[str(product_name)] = build_stationarity_report(
            product_df.reset_index(drop=True),
            target_column=TARGET_COLUMN,
        )
    return {
        "date_column": DATE_COLUMN,
        "product_column": PRODUCT_COLUMN,
        "target_column": TARGET_COLUMN,
        "product_count": len(product_reports),
        "product_reports": product_reports,
    }


def _preprocessing_bundle(train_df: pd.DataFrame) -> dict[str, Any]:
    """Construit le bundle minimum utile a l'inference future."""

    product_names = sorted(str(name) for name in train_df[PRODUCT_COLUMN].unique().tolist())
    return {
        "date_column": DATE_COLUMN,
        "product_column": PRODUCT_COLUMN,
        "target_column": TARGET_COLUMN,
        "frequency": DAILY_FREQUENCY,
        "products": product_names,
    }


def split_product_arima_dataset(
    input_csv: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    stationarity_report_json: Path,
    preprocessing_bundle_path: Path,
) -> dict[str, int]:
    """Charge le dataset long, cree les splits par produit puis ecrit les artefacts."""

    start_time = time.perf_counter()
    raw_df = pd.read_csv(input_csv)
    train_df, val_df, test_df = split_product_arima_dataframe(raw_df)
    stationarity_payload = _stationarity_payload(train_df)
    preprocessing_bundle = _preprocessing_bundle(train_df)
    _write_split(train_df, train_csv)
    _write_split(val_df, val_csv)
    _write_split(test_df, test_csv)
    _write_json(stationarity_payload, stationarity_report_json)
    preprocessing_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessing_bundle, preprocessing_bundle_path)
    LOGGER.info("Saved train split to %s", train_csv)
    LOGGER.info("Saved val split to %s", val_csv)
    LOGGER.info("Saved test split to %s", test_csv)
    LOGGER.info("Saved stationarity report to %s", stationarity_report_json)
    LOGGER.info("Saved preprocessing bundle to %s", preprocessing_bundle_path)
    LOGGER.info("Split completed in %.3f seconds", time.perf_counter() - start_time)
    return {
        "raw_row_count": int(len(raw_df)),
        "row_count": int(len(raw_df)),
        "product_count": int(cast(pd.Series, raw_df[PRODUCT_COLUMN]).nunique()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }

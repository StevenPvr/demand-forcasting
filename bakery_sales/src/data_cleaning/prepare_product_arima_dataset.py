from __future__ import annotations

"""Prepare un dataset journalier long pour un ARIMA distinct par produit."""

import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.data_cleaning.constants_arima import (
    CSV_ENCODING,
    DAILY_FREQUENCY,
    DATE_COLUMN,
    MAX_ZERO_DAY_RATIO,
    MISSING_FLAG_COLUMN,
    PRODUCT_COLUMN,
    TARGET_COLUMN,
)
from src.data_cleaning.prepare_baguette_dataset import (
    aggregate_sales_by_day,
    filter_invalid_articles,
    load_sales_dataset,
    regularize_daily_sales_grid,
    remove_negative_cancellations,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def _clean_sales_frame(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Applique le nettoyage transactionnel partage avant agregations."""

    filtered_df = filter_invalid_articles(sales_df)
    return remove_negative_cancellations(filtered_df)


def _long_product_frame(
    regularized_df: pd.DataFrame,
    missing_mask: pd.Series,
) -> pd.DataFrame:
    """Transforme une matrice date x produit en format long journalier."""

    long_df = regularized_df.reset_index(names=DATE_COLUMN).melt(
        id_vars=[DATE_COLUMN],
        var_name=PRODUCT_COLUMN,
        value_name=TARGET_COLUMN,
    )
    long_df[DATE_COLUMN] = pd.to_datetime(long_df[DATE_COLUMN], format="%Y-%m-%d").dt.strftime(
        "%Y-%m-%d"
    )
    long_df[MISSING_FLAG_COLUMN] = [
        int(missing_mask.loc[pd.Timestamp(date_value)])
        for date_value in cast(pd.Series, long_df[DATE_COLUMN]).tolist()
    ]
    long_df[TARGET_COLUMN] = cast(pd.Series, long_df[TARGET_COLUMN]).astype(float)
    return long_df.sort_values([DATE_COLUMN, PRODUCT_COLUMN]).reset_index(drop=True)


def _filter_sparse_products(
    regularized_df: pd.DataFrame,
    max_zero_day_ratio: float = MAX_ZERO_DAY_RATIO,
) -> tuple[pd.DataFrame, list[str]]:
    """Exclut les produits dont la part de jours a zero depasse le seuil autorise."""

    zero_day_ratios = cast(pd.Series, regularized_df.eq(0.0).mean(axis=0)).astype(float)
    kept_products: list[str] = [
        str(product_name)
        for product_name, zero_ratio in zero_day_ratios.items()
        if float(zero_ratio) <= max_zero_day_ratio
    ]
    excluded_products: list[str] = [
        str(product_name)
        for product_name, zero_ratio in zero_day_ratios.items()
        if float(zero_ratio) > max_zero_day_ratio
    ]
    if not kept_products:
        raise ValueError("No product remaining after zero-day ratio filtering")
    filtered_df = regularized_df.loc[:, kept_products].copy()
    if excluded_products:
        LOGGER.info(
            "Excluded %d products with zero-day ratio above %.3f: %s",
            len(excluded_products),
            max_zero_day_ratio,
            ", ".join(excluded_products),
        )
    return filtered_df, excluded_products


def _metadata_payload(
    regularized_df: pd.DataFrame,
    missing_mask: pd.Series,
    excluded_products: list[str],
) -> dict[str, Any]:
    """Construit le resume JSON du dataset ARIMA multi-produit."""

    product_names = sorted(str(column) for column in regularized_df.columns)
    return {
        "aggregation_level": "daily",
        "frequency": DAILY_FREQUENCY,
        "date_column": DATE_COLUMN,
        "product_column": PRODUCT_COLUMN,
        "target_column": TARGET_COLUMN,
        "missing_flag_column": MISSING_FLAG_COLUMN,
        "product_count": len(product_names),
        "products": product_names,
        "excluded_products_due_to_zero_ratio": excluded_products,
        "max_zero_day_ratio": float(MAX_ZERO_DAY_RATIO),
        "row_count": int(len(regularized_df.index) * len(product_names)),
        "missing_day_count": int(missing_mask.sum()),
    }


def build_daily_product_training_frame(
    sales_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construit le dataset journalier long qui servira aux ARIMA par produit."""

    cleaned_df = _clean_sales_frame(sales_df)
    daily_df = aggregate_sales_by_day(cleaned_df)
    regularized_df, missing_mask = regularize_daily_sales_grid(daily_df)
    regularized_df, excluded_products = _filter_sparse_products(regularized_df)
    training_df = _long_product_frame(regularized_df, missing_mask)
    metadata = _metadata_payload(regularized_df, missing_mask, excluded_products)
    LOGGER.info(
        "Prepared %d daily product rows for %d products",
        len(training_df),
        metadata["product_count"],
    )
    return training_df, metadata


def _write_metadata(output_json: Path, payload: dict[str, Any]) -> None:
    """Ecrit le resume JSON du dataset long."""

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding=CSV_ENCODING)


def prepare_daily_product_arima_dataset(
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
) -> dict[str, int | str]:
    """Prepare et sauvegarde le dataset journalier long utilise par ARIMA."""

    start_time = time.perf_counter()
    sales_df = load_sales_dataset(input_csv)
    training_df, metadata = build_daily_product_training_frame(sales_df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(output_csv, index=False, encoding=CSV_ENCODING)
    _write_metadata(output_json, metadata)
    LOGGER.info("Saved daily product ARIMA dataset to %s", output_csv)
    LOGGER.info("Saved daily product ARIMA summary to %s", output_json)
    LOGGER.info("Preparation completed in %.3f seconds", time.perf_counter() - start_time)
    return {
        "row_count": int(metadata["row_count"]),
        "product_count": int(metadata["product_count"]),
        "output_csv": str(output_csv),
        "output_json": str(output_json),
    }

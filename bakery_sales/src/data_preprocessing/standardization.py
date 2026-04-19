from __future__ import annotations

"""Standardisation causale des exogenes continues pour ElasticNet."""

import logging
from typing import Any, cast

import pandas as pd
from sklearn.preprocessing import StandardScaler

LOGGER: logging.Logger = logging.getLogger(__name__)


def _scaled_feature_columns(dataset_df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes exogenes numeriques non binaires a standardiser."""

    return [
        column
        for column in dataset_df.columns
        if column.startswith("exog_")
        and pd.api.types.is_numeric_dtype(dataset_df[column])
        and not _is_binary_indicator(cast(pd.Series, dataset_df[column]))
    ]


def _is_binary_indicator(value_series: pd.Series) -> bool:
    """Retourne vrai si une serie numerique ne contient que des indicateurs 0/1."""

    non_null_values = value_series.dropna()
    if non_null_values.empty:
        return True
    unique_values = {float(value) for value in non_null_values.to_list()}
    return unique_values.issubset({0.0, 1.0})


def _replace_scaled_features(
    dataset_df: pd.DataFrame,
    scaled_columns: list[str],
    scaled_values: object,
) -> pd.DataFrame:
    """Remplace les colonnes standardisees par leur version float standardisee."""

    if not scaled_columns:
        return dataset_df.copy()
    scaled_frame = pd.DataFrame(scaled_values, columns=scaled_columns, index=dataset_df.index)
    passthrough_columns = [column for column in dataset_df.columns if column not in scaled_columns]
    return pd.concat([dataset_df.loc[:, passthrough_columns].copy(), scaled_frame], axis=1).loc[
        :,
        dataset_df.columns,
    ]


def _preprocessing_bundle(
    train_df: pd.DataFrame,
    scaled_columns: list[str],
    scaler: StandardScaler | None,
) -> dict[str, Any]:
    """Construit le bundle de preprocessing persistant."""

    passthrough_columns = [column for column in train_df.columns if column not in scaled_columns]
    return {
        "sales_scaler": scaler,
        "scaled_sales_columns": scaled_columns,
        "passthrough_columns": passthrough_columns,
    }


def standardize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit sur train uniquement puis transforme train, val et test."""

    scaled_columns = _scaled_feature_columns(train_df)
    if not scaled_columns:
        bundle = _preprocessing_bundle(train_df, scaled_columns, scaler=None)
        return train_df.copy(), val_df.copy(), test_df.copy(), bundle
    scaler = StandardScaler()
    scaler.fit(train_df[scaled_columns])
    scaled_train = _replace_scaled_features(train_df, scaled_columns, scaler.transform(train_df[scaled_columns]))
    scaled_val = _replace_scaled_features(val_df, scaled_columns, scaler.transform(val_df[scaled_columns]))
    scaled_test = _replace_scaled_features(test_df, scaled_columns, scaler.transform(test_df[scaled_columns]))
    bundle = _preprocessing_bundle(train_df, scaled_columns, scaler)
    LOGGER.info("Standardized %d non-binary exogenous columns using train-only statistics", len(scaled_columns))
    return scaled_train, scaled_val, scaled_test, bundle


def apply_preprocessing_bundle(
    dataset_df: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Applique un bundle persiste a un nouveau dataframe de features."""

    sales_columns = list(bundle["scaled_sales_columns"])
    scaler = bundle["sales_scaler"]
    if not sales_columns or scaler is None:
        return dataset_df.copy()
    return _replace_scaled_features(dataset_df, sales_columns, scaler.transform(dataset_df[sales_columns]))

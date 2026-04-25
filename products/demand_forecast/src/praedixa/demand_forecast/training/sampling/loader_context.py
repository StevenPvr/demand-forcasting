from __future__ import annotations

from pathlib import Path

import pandas as pd

from praedixa.demand_forecast.contracts.targets import TargetContract, resolve_target_contract
from praedixa.demand_forecast.training.sampling.common import resolve_sampling_store_col


def parquet_relation_sql(path: Path) -> str:
    return f"select * from parquet_scan('{path.as_posix()}')"


def relation_sql_with_dataset_source(
    *,
    base_relation_sql: str,
    columns: list[str],
    dataset_source_col: str,
) -> tuple[str, list[str]]:
    if dataset_source_col in columns:
        return base_relation_sql, columns
    raise ValueError(
        f"Parquet optimisation inputs must include `{dataset_source_col}`."
    )


def common_schema_previews(
    *,
    train_schema_preview: pd.DataFrame,
    tuning_schema_preview: pd.DataFrame,
    dataset_source_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_columns = [
        column
        for column in train_schema_preview.columns
        if column in set(tuning_schema_preview.columns)
    ]
    train_target_schema_preview = train_schema_preview.loc[:, common_columns].copy()
    tuning_target_schema_preview = tuning_schema_preview.loc[:, common_columns].copy()
    if dataset_source_col not in train_target_schema_preview.columns:
        raise ValueError(f"Training schema is missing `{dataset_source_col}`.")
    if dataset_source_col not in tuning_target_schema_preview.columns:
        raise ValueError(f"Tuning schema is missing `{dataset_source_col}`.")
    return train_target_schema_preview, tuning_target_schema_preview


def target_filter_columns(
    *,
    train_columns: list[str],
    tuning_columns: list[str],
    train_target_schema_preview: pd.DataFrame,
    tuning_target_schema_preview: pd.DataFrame,
    target_col: str,
) -> tuple[TargetContract, str, str]:
    target_contract = resolve_target_contract(
        train_target_schema_preview,
        tuning_target_schema_preview,
        requested_target_col=target_col,
    )
    train_target_filter_col = (
        target_contract.learning_target_col
        if target_contract.learning_target_col in train_columns
        else target_contract.absolute_target_col
    )
    tuning_target_filter_col = (
        target_contract.learning_target_col
        if target_contract.learning_target_col in tuning_columns
        else target_contract.absolute_target_col
    )
    return target_contract, train_target_filter_col, tuning_target_filter_col


def parquet_relations_with_dataset_source(
    *,
    train_base_relation_sql: str,
    tuning_base_relation_sql: str,
    train_columns: list[str],
    tuning_columns: list[str],
    dataset_source_col: str,
) -> tuple[str, list[str], str, list[str]]:
    train_relation_sql, resolved_train_columns = relation_sql_with_dataset_source(
        base_relation_sql=train_base_relation_sql,
        columns=train_columns,
        dataset_source_col=dataset_source_col,
    )
    tuning_relation_sql, resolved_tuning_columns = relation_sql_with_dataset_source(
        base_relation_sql=tuning_base_relation_sql,
        columns=tuning_columns,
        dataset_source_col=dataset_source_col,
    )
    return (
        train_relation_sql,
        resolved_train_columns,
        tuning_relation_sql,
        resolved_tuning_columns,
    )


def parquet_target_filter_context(
    *,
    train_schema_preview: pd.DataFrame,
    tuning_schema_preview: pd.DataFrame,
    dataset_source_col: str,
    target_col: str,
) -> tuple[str, str, str]:
    train_target_schema_preview, tuning_target_schema_preview = common_schema_previews(
        train_schema_preview=train_schema_preview,
        tuning_schema_preview=tuning_schema_preview,
        dataset_source_col=dataset_source_col,
    )
    _target_contract, train_target_filter_col, tuning_target_filter_col = target_filter_columns(
        train_columns=list(train_target_schema_preview.columns),
        tuning_columns=list(tuning_target_schema_preview.columns),
        train_target_schema_preview=train_target_schema_preview,
        tuning_target_schema_preview=tuning_target_schema_preview,
        target_col=target_col,
    )
    return (
        resolve_sampling_store_col(train_target_schema_preview),
        train_target_filter_col,
        tuning_target_filter_col,
    )


__all__ = [
    "common_schema_previews",
    "parquet_relations_with_dataset_source",
    "parquet_relation_sql",
    "parquet_target_filter_context",
    "relation_sql_with_dataset_source",
    "target_filter_columns",
]

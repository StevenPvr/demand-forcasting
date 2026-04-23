from __future__ import annotations

import math
from typing import Any, cast

import duckdb
import pandas as pd

from praedixa.demand_forecast.training.sampling_models import (
    DEFAULT_IDENTIFIER_FEATURE_COLS,
    GoldSplitSamplingSpec,
    RelationSamplingSpec,
)


def _per_stratum_primary_sample_expr(sample_fraction: float) -> str:
    return f"cast(floor(stratum_rows * {sample_fraction:.12f}) as bigint)"


def load_sampling_metadata_from_relation(
    connection: duckdb.DuckDBPyConnection,
    *,
    spec: RelationSamplingSpec,
    available_columns: list[str],
) -> dict[str, object]:
    metadata_projection_cols = _metadata_projection_columns(
        available_columns=available_columns,
        dataset_source_col=spec.dataset_source_col,
        date_col=spec.date_col,
        sample_store_col=spec.sample_store_col,
    )
    sampled_counts = _relation_sampling_counts(
        connection,
        spec=spec,
        metadata_projection_cols=metadata_projection_cols,
    )
    return _relation_sampling_metadata_payload(sampled_counts=sampled_counts, spec=spec)


def _relation_sampling_metadata_payload(
    *,
    sampled_counts: list[tuple[object, object, object, object, object, object, object]],
    spec: RelationSamplingSpec,
) -> dict[str, object]:
    datasets: dict[str, dict[str, int | float]] = {}
    total_original_rows = 0
    total_sampled_rows = 0
    for (
        dataset_source,
        original_rows,
        sampled_rows,
        primary_sampled_rows,
        store_count,
        unique_dates,
        strata_count,
    ) in sampled_counts:
        dataset_original_rows = int(cast(Any, original_rows))
        _ = sampled_rows
        dataset_sampled_rows = _target_dataset_sample_size(
            dataset_original_rows=dataset_original_rows,
            sample_fraction=spec.sample_fraction,
            min_samples_per_dataset=spec.min_samples_per_dataset,
        )
        datasets[str(dataset_source)] = _relation_sampling_dataset_metadata(
            dataset_original_rows=dataset_original_rows,
            dataset_sampled_rows=dataset_sampled_rows,
            primary_sampled_rows=primary_sampled_rows,
            sample_fraction=spec.sample_fraction,
            min_samples_per_dataset=spec.min_samples_per_dataset,
            store_count=store_count,
            unique_dates=unique_dates,
            strata_count=strata_count,
        )
        total_original_rows += dataset_original_rows
        total_sampled_rows += dataset_sampled_rows
    return {
        "sample_fraction": float(spec.sample_fraction),
        "min_samples_per_dataset": int(spec.min_samples_per_dataset),
        "sample_strategy": "date_store_stratified",
        "sample_store_col": spec.sample_store_col,
        "original_rows": total_original_rows,
        "sampled_rows": total_sampled_rows,
        "effective_sample_fraction": _effective_sample_fraction(
            sampled_rows=total_sampled_rows,
            original_rows=total_original_rows,
        ),
        "datasets": datasets,
    }


def _metadata_projection_columns(
    *, available_columns: list[str], dataset_source_col: str, date_col: str, sample_store_col: str
) -> list[str]:
    projection_cols = [dataset_source_col, date_col, sample_store_col]
    for candidate in ("product_id", "series_id"):
        if candidate in available_columns and candidate not in projection_cols:
            projection_cols.append(candidate)
    return projection_cols


def _relation_sampling_dataset_metadata(
    *,
    dataset_original_rows: int,
    dataset_sampled_rows: int,
    primary_sampled_rows: object,
    sample_fraction: float,
    min_samples_per_dataset: int,
    store_count: object,
    unique_dates: object,
    strata_count: object,
) -> dict[str, int | float]:
    primary_rows = int(cast(Any, primary_sampled_rows))
    return {
        "original_rows": dataset_original_rows,
        "sampled_rows": dataset_sampled_rows,
        "primary_sampled_rows": primary_rows,
        "sample_fraction": float(sample_fraction),
        "effective_sample_fraction": _effective_sample_fraction(
            sampled_rows=dataset_sampled_rows,
            original_rows=dataset_original_rows,
        ),
        "min_samples_per_dataset": int(min_samples_per_dataset),
        "store_count": int(cast(Any, store_count)),
        "unique_dates": int(cast(Any, unique_dates)),
        "strata_count": int(cast(Any, strata_count)),
        "top_up_required": int(dataset_sampled_rows > primary_rows),
    }


def _relation_sampling_counts(
    connection: duckdb.DuckDBPyConnection,
    *,
    spec: RelationSamplingSpec,
    metadata_projection_cols: list[str],
) -> list[tuple[object, object, object, object, object, object, object]]:
    select_list = ", ".join(metadata_projection_cols)
    return connection.execute(
        f"""
        with scoped as (
            select {select_list}
            from ({spec.relation_sql}) as relation_source
        ),
        strata as (
            select
                {spec.dataset_source_col} as dataset_source,
                {spec.date_col} as sampled_dt,
                {spec.sample_store_col} as sample_store,
                count(*) as stratum_rows
            from scoped
            group by 1, 2, 3
        ),
        dataset_stats as (
            select
                dataset_source,
                sum(stratum_rows) as original_rows,
                sum({_per_stratum_primary_sample_expr(spec.sample_fraction)}) as primary_sampled_rows,
                count(distinct sample_store) as store_count,
                count(distinct sampled_dt) as unique_dates,
                count(*) as strata_count
            from strata
            group by 1
        )
        select
            dataset_source,
            original_rows,
            least(
                original_rows,
                greatest(primary_sampled_rows, {int(spec.min_samples_per_dataset)})
            ) as sampled_rows,
            primary_sampled_rows,
            store_count,
            unique_dates,
            strata_count
        from dataset_stats
        order by 1
        """
    ).fetchall()


def _target_dataset_sample_size(
    *,
    dataset_original_rows: int,
    sample_fraction: float,
    min_samples_per_dataset: int,
) -> int:
    return int(
        min(
            dataset_original_rows,
            max(
                int(math.ceil(dataset_original_rows * sample_fraction)),
                min_samples_per_dataset,
            ),
        )
    )


def _effective_sample_fraction(*, sampled_rows: int, original_rows: int) -> float:
    if original_rows <= 0:
        return 0.0
    return float(sampled_rows) / float(original_rows)


def resolve_gold_projection_columns(
    schema_preview: pd.DataFrame,
    *,
    date_col: str,
    dataset_source_col: str,
    sample_store_col: str,
) -> list[str]:
    required_columns = {
        date_col,
        dataset_source_col,
        sample_store_col,
        *DEFAULT_IDENTIFIER_FEATURE_COLS,
    }
    for column in schema_preview.columns:
        if _is_numeric_or_bool_projection_column(schema_preview[column]):
            required_columns.add(column)
    return [column for column in schema_preview.columns if column in required_columns]


def _is_numeric_or_bool_projection_column(series: pd.Series) -> bool:
    return bool(pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series))


def load_gold_split_sampling_metadata(
    connection: duckdb.DuckDBPyConnection,
    *,
    spec: GoldSplitSamplingSpec,
) -> dict[str, object]:
    original_counts = connection.execute(
        f"""
        with strata as (
            select
                {spec.dataset_source_col} as dataset_source,
                {spec.date_col} as sampled_dt,
                {spec.sample_store_col} as sample_store,
                count(*) as stratum_rows
            from {spec.gold_table}
            where split_bucket = '{spec.split_bucket}'
            group by 1, 2, 3
        )
        select
            dataset_source,
            sum(stratum_rows) as original_rows,
            sum({_per_stratum_primary_sample_expr(spec.sample_fraction)}) as sampled_rows,
            count(distinct sample_store) as store_count,
            count(distinct sampled_dt) as unique_dates,
            count(*) as strata_count
        from strata
        group by 1
        order by 1
        """
    ).fetchall()
    return _gold_sampling_metadata_payload(original_counts, spec=spec)


def _gold_sampling_metadata_payload(
    original_counts: list[tuple[object, object, object, object, object, object]],
    *,
    spec: GoldSplitSamplingSpec,
) -> dict[str, object]:
    datasets: dict[str, dict[str, int | float]] = {}
    total_original_rows = 0
    total_sampled_rows = 0
    for dataset_source, original_rows, sampled_rows, store_count, unique_dates, strata_count in original_counts:
        dataset_original_rows = int(cast(Any, original_rows))
        dataset_sampled_rows = int(cast(Any, sampled_rows))
        datasets[str(dataset_source)] = {
            "original_rows": dataset_original_rows,
            "sampled_rows": dataset_sampled_rows,
            "sample_fraction": float(spec.sample_fraction),
            "effective_sample_fraction": _effective_sample_fraction(
                sampled_rows=dataset_sampled_rows,
                original_rows=dataset_original_rows,
            ),
            "store_count": int(cast(Any, store_count)),
            "unique_dates": int(cast(Any, unique_dates)),
            "strata_count": int(cast(Any, strata_count)),
        }
        total_original_rows += dataset_original_rows
        total_sampled_rows += dataset_sampled_rows
    return {
        "sample_fraction": float(spec.sample_fraction),
        "sample_strategy": "date_store_stratified",
        "sample_store_col": spec.sample_store_col,
        "original_rows": total_original_rows,
        "sampled_rows": total_sampled_rows,
        "effective_sample_fraction": _effective_sample_fraction(
            sampled_rows=total_sampled_rows,
            original_rows=total_original_rows,
        ),
        "datasets": datasets,
    }


def refresh_sampling_metadata_from_sampled_frame(
    metadata: dict[str, object],
    sampled_frame: pd.DataFrame,
    *,
    dataset_source_col: str,
    sample_store_col: str,
    date_col: str,
) -> dict[str, object]:
    datasets = cast(dict[str, dict[str, object]], metadata["datasets"])
    refreshed = {
        **metadata,
        "datasets": {
            str(dataset_source): dict(dataset_metadata)
            for dataset_source, dataset_metadata in datasets.items()
        },
    }
    refreshed_datasets = cast(dict[str, dict[str, object]], refreshed["datasets"])
    sampled_counts = sampled_frame[dataset_source_col].value_counts(dropna=False).sort_index()
    refreshed["sampled_rows"] = int(len(sampled_frame))
    refreshed["effective_sample_fraction"] = _effective_sample_fraction(
        sampled_rows=int(len(sampled_frame)),
        original_rows=int(cast(Any, metadata["original_rows"])),
    )
    for dataset_source, sampled_rows in sampled_counts.items():
        dataset_key = str(dataset_source)
        refreshed_datasets[dataset_key] = _refreshed_dataset_metadata(
            metadata=metadata,
            sampled_frame=sampled_frame,
            dataset_source_col=dataset_source_col,
            dataset_source=dataset_source,
            sample_store_col=sample_store_col,
            date_col=date_col,
            sampled_rows=int(sampled_rows),
            existing=refreshed_datasets.get(dataset_key),
        )
    return refreshed


def _refreshed_dataset_metadata(
    *,
    metadata: dict[str, object],
    sampled_frame: pd.DataFrame,
    dataset_source_col: str,
    dataset_source: object,
    sample_store_col: str,
    date_col: str,
    sampled_rows: int,
    existing: dict[str, object] | None,
) -> dict[str, object]:
    if existing is None:
        return {
            "original_rows": sampled_rows,
            "sample_fraction": float(cast(Any, metadata["sample_fraction"])),
            "effective_sample_fraction": 1.0,
            "store_count": int(sampled_frame.loc[sampled_frame[dataset_source_col] == dataset_source, sample_store_col].nunique()),
            "unique_dates": int(sampled_frame.loc[sampled_frame[dataset_source_col] == dataset_source, date_col].nunique()),
            "strata_count": int(sampled_frame.loc[sampled_frame[dataset_source_col] == dataset_source].groupby([date_col, sample_store_col], sort=False).ngroups),
            "sampled_rows": sampled_rows,
        }
    updated = dict(existing)
    updated["sampled_rows"] = sampled_rows
    updated["effective_sample_fraction"] = _effective_sample_fraction(
        sampled_rows=sampled_rows,
        original_rows=int(cast(Any, updated["original_rows"])),
    )
    return updated


def relation_sampling_requires_top_up(metadata: dict[str, object]) -> bool:
    datasets = cast(dict[str, dict[str, object]], metadata["datasets"])
    return any(bool(dataset_metadata.get("top_up_required", False)) for dataset_metadata in datasets.values())


__all__ = [
    "load_gold_split_sampling_metadata",
    "load_sampling_metadata_from_relation",
    "relation_sampling_requires_top_up",
    "refresh_sampling_metadata_from_sampled_frame",
    "resolve_gold_projection_columns",
]

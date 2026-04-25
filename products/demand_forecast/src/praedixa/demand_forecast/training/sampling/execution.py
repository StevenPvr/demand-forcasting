from __future__ import annotations

import duckdb
import pandas as pd

from praedixa.demand_forecast.training.sampling import (
    RelationSamplingQuery,
    RelationSamplingSpec,
    build_sampling_query_for_relation,
    load_sampling_metadata_from_relation,
)


def configure_sampling_connection(
    connection: duckdb.DuckDBPyConnection,
    *,
    threads: int,
) -> None:
    connection.execute(f"PRAGMA threads={threads}")
    connection.execute("PRAGMA preserve_insertion_order=false")


def sample_relation_frame(
    *,
    connection: duckdb.DuckDBPyConnection,
    spec: RelationSamplingSpec,
    selected_columns: list[str],
) -> pd.DataFrame:
    return connection.execute(
        build_sampling_query_for_relation(
            query=RelationSamplingQuery(spec=spec, selected_columns=selected_columns)
        )
    ).fetchdf()


def load_full_relation_frame(
    *,
    connection: duckdb.DuckDBPyConnection,
    relation_sql: str,
    selected_columns: list[str],
) -> pd.DataFrame:
    return connection.execute(
        f"""
        select {", ".join(selected_columns)}
        from ({relation_sql}) as relation_source
        """
    ).fetchdf()


def sampling_metadata_for_relation(
    *,
    connection: duckdb.DuckDBPyConnection,
    spec: RelationSamplingSpec,
    available_columns: list[str],
) -> dict[str, object]:
    return load_sampling_metadata_from_relation(
        connection,
        spec=spec,
        available_columns=available_columns,
    )


def full_relation_metadata(
    *,
    frame: pd.DataFrame,
    spec: RelationSamplingSpec,
) -> dict[str, object]:
    from praedixa.demand_forecast.training.sampling import refresh_sampling_metadata_from_sampled_frame

    return refresh_sampling_metadata_from_sampled_frame(
        {
            "sample_fraction": float(spec.sample_fraction),
            "min_samples_per_dataset": int(spec.min_samples_per_dataset),
            "sample_strategy": "full_relation",
            "sample_store_col": spec.sample_store_col,
            "original_rows": int(len(frame)),
            "sampled_rows": int(len(frame)),
            "datasets": {},
        },
        frame,
        dataset_source_col=spec.dataset_source_col,
        sample_store_col=spec.sample_store_col,
        date_col=spec.date_col,
    )


__all__ = [
    "configure_sampling_connection",
    "full_relation_metadata",
    "load_full_relation_frame",
    "sample_relation_frame",
    "sampling_metadata_for_relation",
]

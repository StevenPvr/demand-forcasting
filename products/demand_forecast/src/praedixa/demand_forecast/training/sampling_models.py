from __future__ import annotations

from dataclasses import dataclass

from praedixa.demand_forecast.training.constants import (
    DEFAULT_IDENTIFIER_FEATURE_COLS,
    DEFAULT_MIN_SAMPLES_PER_DATASET,
)


@dataclass(frozen=True)
class RelationSamplingSpec:
    relation_sql: str
    date_col: str
    dataset_source_col: str
    sample_store_col: str
    sample_fraction: float
    min_samples_per_dataset: int = DEFAULT_MIN_SAMPLES_PER_DATASET


@dataclass(frozen=True)
class RelationSamplingQuery:
    spec: RelationSamplingSpec
    selected_columns: list[str]


@dataclass(frozen=True)
class GoldSplitSamplingSpec:
    gold_table: str
    split_bucket: str
    date_col: str
    dataset_source_col: str
    sample_store_col: str
    sample_fraction: float


@dataclass(frozen=True)
class FrameSamplingSpec:
    date_col: str
    dataset_source_col: str
    sample_store_col: str
    sample_fraction: float
    min_samples_per_dataset: int = DEFAULT_MIN_SAMPLES_PER_DATASET


__all__ = [
    "DEFAULT_IDENTIFIER_FEATURE_COLS",
    "DEFAULT_MIN_SAMPLES_PER_DATASET",
    "FrameSamplingSpec",
    "GoldSplitSamplingSpec",
    "RelationSamplingQuery",
    "RelationSamplingSpec",
]

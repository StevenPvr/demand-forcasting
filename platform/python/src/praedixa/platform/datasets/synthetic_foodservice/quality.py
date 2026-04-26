from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.demand import ORACLE_COLUMNS

DEBUG_ONLY_COLUMNS: frozenset[str] = frozenset(
    column for column in ORACLE_COLUMNS if column.endswith("_debug")
)


@dataclass(frozen=True)
class SyntheticGenerationStats:
    """Streaming summary of a one-shot synthetic generation run."""

    row_count: int
    oracle_row_count: int
    series_count: int
    site_count: int
    city_count: int
    min_dt: str | None
    max_dt: str | None
    censored_rows: int
    zero_rows: int
    promo_rows: int
    sources: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GenerationStatsBuilder:
    """Accumulate generation stats without reading the final CSV back."""

    def __init__(self, *, site_count: int, city_count: int) -> None:
        self._site_count = site_count
        self._city_count = city_count
        self._row_count = 0
        self._oracle_row_count = 0
        self._series: set[str] = set()
        self._min_dt: date | None = None
        self._max_dt: date | None = None
        self._censored_rows = 0
        self._zero_rows = 0
        self._promo_rows = 0
        self._sources: dict[str, dict[str, int]] = {}
        self._source_sites: dict[str, set[str]] = {}
        self._source_series: dict[str, set[str]] = {}

    def add_batch(self, *, daily: pd.DataFrame, oracle: pd.DataFrame) -> None:
        if daily.empty:
            return
        self._row_count += len(daily)
        self._oracle_row_count += len(oracle)
        self._series.update(daily["series_id"].astype(str).unique().tolist())
        self._min_dt = _min_date(self._min_dt, pd.to_datetime(daily["dt"]).min().date())
        self._max_dt = _max_date(self._max_dt, pd.to_datetime(daily["dt"]).max().date())
        self._censored_rows += int(daily["censor_flag"].astype(bool).sum())
        self._zero_rows += int(pd.to_numeric(daily["observed_demand_qty"]).eq(0).sum())
        self._promo_rows += int(daily["promo_flag"].astype(bool).sum())
        for source, source_rows in daily.groupby("dataset_source"):
            self._add_source_rows(str(source), source_rows)

    def build(self) -> SyntheticGenerationStats:
        return SyntheticGenerationStats(
            row_count=self._row_count,
            oracle_row_count=self._oracle_row_count,
            series_count=len(self._series),
            site_count=self._site_count,
            city_count=self._city_count,
            min_dt=self._min_dt.isoformat() if self._min_dt is not None else None,
            max_dt=self._max_dt.isoformat() if self._max_dt is not None else None,
            censored_rows=self._censored_rows,
            zero_rows=self._zero_rows,
            promo_rows=self._promo_rows,
            sources=self._sources,
        )

    def _add_source_rows(self, source: str, frame: pd.DataFrame) -> None:
        summary = self._sources.setdefault(
            source,
            {"rows": 0, "sites": 0, "series": 0, "censored_rows": 0},
        )
        sites = self._source_sites.setdefault(source, set())
        series = self._source_series.setdefault(source, set())
        sites.update(frame["location_id"].astype(str).unique().tolist())
        series.update(frame["series_id"].astype(str).unique().tolist())
        summary["rows"] += len(frame)
        summary["sites"] = len(sites)
        summary["series"] = len(series)
        summary["censored_rows"] += int(frame["censor_flag"].astype(bool).sum())


def validate_model_facing_frame(frame: pd.DataFrame) -> None:
    """Fail when oracle/debug fields leak into the model-facing source feed."""

    leaked = sorted(set(frame.columns).intersection(DEBUG_ONLY_COLUMNS))
    if leaked:
        raise ValueError(
            f"Oracle/debug columns leaked into synthetic daily feed: {leaked}"
        )
    duplicate_rows = frame.duplicated(
        ["dataset_source", "dt", "location_id", "product_id"]
    )
    if duplicate_rows.any():
        raise ValueError(
            "Synthetic source violates dataset_source x dt x location_id x product_id uniqueness."
        )
    if pd.to_numeric(frame["observed_demand_qty"], errors="coerce").lt(0).any():
        raise ValueError("Synthetic observed_demand_qty must be non-negative.")


def build_manifest_payload(
    *,
    seed: int,
    start_date: date,
    end_date: date,
    stats: SyntheticGenerationStats,
    location_metadata_path: str,
    hidden_oracle_columns: tuple[str, ...],
) -> dict[str, Any]:
    """Build the JSON manifest written next to the one-shot source CSV."""

    return {
        "generator": "synthetic_foodservice_v1",
        "seed": seed,
        "date_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "stats": stats.to_dict(),
        "model_facing_policy": {
            "daily_source_is_stable_one_shot_csv": True,
            "medallion_generates_no_synthetic_data": True,
            "gold_split_policy": "train_only",
            "allowed_gold_split_buckets": ["train"],
            "forbidden_gold_split_buckets": ["val", "test"],
            "hidden_oracle_columns": list(hidden_oracle_columns),
        },
        "location_metadata_path": location_metadata_path,
        "source_policy_metadata": {
            "commercial_use_allowed": True,
            "ml_training_allowed": True,
            "dataset_kind": "synthetic_structural_foodservice",
        },
    }


def _min_date(left: date | None, right: date) -> date:
    return right if left is None else min(left, right)


def _max_date(left: date | None, right: date) -> date:
    return right if left is None else max(left, right)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from praedixa.platform.datasets.standardization.supplemental_corpus_standardizers import (
    DEFAULT_SILVER_RUN_ID,
    DEFAULT_SOURCE_RUN_ID,
    freshretail_promo_flag_expr,
    standardize_freshretail_lt_lazy_frame,
)
from praedixa.platform.datasets.standardization.schema import (
    align_lazy_frame_to_canonical_schema,
    build_series_id_expr,
)
from praedixa.platform.governance.source_registry import (
    allowed_training_dataset_sources,
)
from praedixa.platform.runtime.paths import GLOBAL_DATASET_DIR
from praedixa.platform.runtime.paths import SOURCES_DIR


DEFAULT_RAW_DIR = SOURCES_DIR / "commercial_datasets" / "raw"
DEFAULT_OUTPUT_PATH = GLOBAL_DATASET_DIR / "supplemental_corpus_daily.parquet"

DEFAULT_FRESHRETAIL_LT_TRAIN_PATH = DEFAULT_RAW_DIR / "freshretail_lt_train.parquet"
DEFAULT_FRESHRETAIL_LT_EVAL_PATH = DEFAULT_RAW_DIR / "freshretail_lt_eval.parquet"
DEFAULT_FIRST_PARTY_DAILY_PATH = DEFAULT_RAW_DIR / "first_party_daily.parquet"
SUPPORTED_SUPPLEMENTAL_CORPUS_SOURCES = frozenset({"freshretail_lt", "first_party_daily"})

__all__ = [
    "SupplementalCorpusCompatibility",
    "build_supplemental_corpus_frame",
    "build_supplemental_corpus_standardized_dataset",
    "freshretail_promo_flag_expr",
]

SUPPLEMENTAL_CORPUS_COMPATIBILITY_ROWS: tuple[
    tuple[str, str, bool, bool, str],
    ...,
] = (
    (
        "freshretail_lt",
        "daily normalized product sales by store x product",
        True,
        True,
        "FreshRetailNet-LT exposes sale_amount at daily store-product grain under CC BY 4.0.",
    ),
    (
        "first_party_daily",
        "canonical daily client-owned demand feed",
        True,
        True,
        "Client onboarding data is already aligned to the canonical daily schema and stays eligible for training.",
    ),
)


@dataclass(frozen=True)
class SupplementalCorpusCompatibility:
    """Compatibility summary for one supplemental corpus dataset."""

    dataset_source: str
    target_semantics: str
    compatible_with_pipeline: bool
    commercial_use_allowed: bool
    reason: str


def _compatibility_from_row(
    row: tuple[str, str, bool, bool, str],
) -> SupplementalCorpusCompatibility:
    dataset_source, target_semantics, compatible_with_pipeline, commercial_use_allowed, reason = row
    return SupplementalCorpusCompatibility(
        dataset_source=dataset_source,
        target_semantics=target_semantics,
        compatible_with_pipeline=compatible_with_pipeline,
        commercial_use_allowed=commercial_use_allowed,
        reason=reason,
    )


def supplemental_corpus_compatibility_matrix() -> list[SupplementalCorpusCompatibility]:
    return [_compatibility_from_row(row) for row in SUPPLEMENTAL_CORPUS_COMPATIBILITY_ROWS]


def _resolve_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _standardize_freshretail_lt_files(
    *,
    train_path: Path | None,
    eval_path: Path | None,
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.DataFrame]:
    frames: list[pl.DataFrame] = []
    if train_path is not None:
        frames.append(
            standardize_freshretail_lt_lazy_frame(
                pl.scan_parquet(train_path),
                source_partition="historical_train",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            ).collect()
        )
    if eval_path is not None:
        frames.append(
            standardize_freshretail_lt_lazy_frame(
                pl.scan_parquet(eval_path),
                source_partition="historical_eval",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            ).collect()
        )
    return frames


def _with_missing_canonical_metadata(
    frame: pl.LazyFrame,
    *,
    available_columns: set[str],
    source_partition: str,
    source_run_id: str,
    silver_run_id: str,
) -> pl.LazyFrame:
    normalized = frame
    if "source_partition" not in available_columns:
        normalized = normalized.with_columns(pl.lit(source_partition).alias("source_partition"))
    if "source_run_id" not in available_columns:
        normalized = normalized.with_columns(pl.lit(source_run_id).alias("source_run_id"))
    if "silver_run_id" not in available_columns:
        normalized = normalized.with_columns(pl.lit(silver_run_id).alias("silver_run_id"))
    return normalized


def _with_optional_series_id(frame: pl.LazyFrame, *, available_columns: set[str]) -> pl.LazyFrame:
    if "series_id" in available_columns:
        return frame
    if {"location_id", "product_id"}.issubset(available_columns):
        return frame.with_columns(build_series_id_expr("location_id", "product_id").alias("series_id"))
    return frame


def _standardize_canonical_daily_parquet(
    *,
    path: Path,
    dataset_source: str,
    source_partition: str,
    source_run_id: str,
    silver_run_id: str,
) -> pl.DataFrame:
    frame = pl.scan_parquet(path)
    available_columns = set(frame.collect_schema().names())
    normalized = frame.with_columns(pl.lit(dataset_source).alias("dataset_source"))
    normalized = _with_missing_canonical_metadata(
        normalized,
        available_columns=available_columns,
        source_partition=source_partition,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
    normalized = _with_optional_series_id(normalized, available_columns=available_columns)
    return align_lazy_frame_to_canonical_schema(normalized).collect()


def _load_freshretail_lt_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.DataFrame]:
    if "freshretail_lt" not in allowed_sources:
        return []
    freshretail_lt_train = _resolve_existing_path(root / DEFAULT_FRESHRETAIL_LT_TRAIN_PATH.name)
    freshretail_lt_eval = _resolve_existing_path(root / DEFAULT_FRESHRETAIL_LT_EVAL_PATH.name)
    return _standardize_freshretail_lt_files(
        train_path=freshretail_lt_train,
        eval_path=freshretail_lt_eval,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


def _load_canonical_parquet_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.DataFrame]:
    parquet_specs = [
        ("first_party_daily", DEFAULT_FIRST_PARTY_DAILY_PATH.name, "client_onboarding"),
    ]
    frames: list[pl.DataFrame] = []
    for dataset_source, file_name, source_partition in parquet_specs:
        if dataset_source not in allowed_sources:
            continue
        path = _resolve_existing_path(root / file_name)
        if path is None:
            continue
        frames.append(
            _standardize_canonical_daily_parquet(
                path=path,
                dataset_source=dataset_source,
                source_partition=source_partition,
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )
    return frames


def _supplemental_corpus_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.DataFrame]:
    frames: list[pl.DataFrame] = []
    frames.extend(
        _load_freshretail_lt_frames(
            root=root,
            allowed_sources=allowed_sources,
            source_run_id=source_run_id,
            silver_run_id=silver_run_id,
        )
    )
    frames.extend(
        _load_canonical_parquet_frames(
            root=root,
            allowed_sources=allowed_sources,
            source_run_id=source_run_id,
            silver_run_id=silver_run_id,
        )
    )
    return frames


def build_supplemental_corpus_frame(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.DataFrame | None:
    """Build the supplemental corpus in memory without persisting an intermediate parquet."""

    root = Path(raw_dir)
    allowed_sources = set(allowed_training_dataset_sources()) & set(SUPPORTED_SUPPLEMENTAL_CORPUS_SOURCES)
    frames = _supplemental_corpus_frames(
        root=root,
        allowed_sources=allowed_sources,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
    if not frames:
        return None
    return (
        pl.concat(frames, how="vertical_relaxed")
        .filter(pl.col("dataset_source").is_in(sorted(allowed_sources)))
        .sort(["dataset_source", "dt", "location_id", "product_id"])
    )


def build_supplemental_corpus_standardized_dataset(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> Path | None:
    combined = build_supplemental_corpus_frame(
        raw_dir=raw_dir,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
    if combined is None:
        return None
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(target_path)
    return target_path

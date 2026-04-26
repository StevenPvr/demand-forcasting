from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import polars as pl

from praedixa.platform.datasets.standardization.supplemental_corpus_standardizers import (
    DEFAULT_SILVER_RUN_ID,
    DEFAULT_SOURCE_RUN_ID,
    freshretail_promo_flag_expr,
    standardize_freshretail_lt_lazy_frame,
    standardize_m5_sales_lazy_frame,
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
DEFAULT_M5_DIR = DEFAULT_RAW_DIR / "m5_forecasting_accuracy_zenodo"
DEFAULT_M5_ZIP_PATH = DEFAULT_RAW_DIR / "m5_forecasting_accuracy_zenodo.zip"

SUPPORTED_SUPPLEMENTAL_CORPUS_SOURCES = frozenset(
    {
        "freshretail_lt",
        "m5_forecasting_accuracy",
    }
)

__all__ = [
    "SupplementalCorpusCompatibility",
    "build_supplemental_corpus_frame",
    "build_supplemental_corpus_standardized_dataset",
    "freshretail_promo_flag_expr",
    "supplemental_corpus_compatibility_matrix",
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
        "m5_forecasting_accuracy",
        "daily retail unit sales by store x product",
        True,
        True,
        "M5 Zenodo exposes daily store-product unit sales under CC BY 4.0.",
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
    (
        dataset_source,
        target_semantics,
        compatible_with_pipeline,
        commercial_use_allowed,
        reason,
    ) = row
    return SupplementalCorpusCompatibility(
        dataset_source=dataset_source,
        target_semantics=target_semantics,
        compatible_with_pipeline=compatible_with_pipeline,
        commercial_use_allowed=commercial_use_allowed,
        reason=reason,
    )


def supplemental_corpus_compatibility_matrix() -> list[SupplementalCorpusCompatibility]:
    return [
        _compatibility_from_row(row) for row in SUPPLEMENTAL_CORPUS_COMPATIBILITY_ROWS
    ]


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
) -> list[pl.LazyFrame]:
    frames: list[pl.LazyFrame] = []
    if train_path is not None:
        frames.append(
            standardize_freshretail_lt_lazy_frame(
                pl.scan_parquet(train_path),
                source_partition="historical_train",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )
    if eval_path is not None:
        frames.append(
            standardize_freshretail_lt_lazy_frame(
                pl.scan_parquet(eval_path),
                source_partition="historical_eval",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )
    return frames


def _load_freshretail_lt_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.LazyFrame]:
    if "freshretail_lt" not in allowed_sources:
        return []
    freshretail_lt_train = _resolve_existing_path(
        root / DEFAULT_FRESHRETAIL_LT_TRAIN_PATH.name
    )
    freshretail_lt_eval = _resolve_existing_path(
        root / DEFAULT_FRESHRETAIL_LT_EVAL_PATH.name
    )
    return _standardize_freshretail_lt_files(
        train_path=freshretail_lt_train,
        eval_path=freshretail_lt_eval,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


def _ensure_m5_files(root: Path) -> Path | None:
    m5_dir = root / DEFAULT_M5_DIR.name
    required_members = ("calendar.csv", "sales_train_evaluation.csv")
    if all((m5_dir / member).exists() for member in required_members):
        return m5_dir
    zip_path = _resolve_existing_path(root / DEFAULT_M5_ZIP_PATH.name)
    if zip_path is None:
        return None
    m5_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        if not set(required_members).issubset(names):
            return None
        for member in required_members:
            archive.extract(member, path=m5_dir)
    return m5_dir


def _load_m5_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.LazyFrame]:
    if "m5_forecasting_accuracy" not in allowed_sources:
        return []
    m5_dir = _ensure_m5_files(root)
    if m5_dir is None:
        return []
    return [
        standardize_m5_sales_lazy_frame(
            pl.scan_csv(
                m5_dir / "sales_train_evaluation.csv", infer_schema_length=1000
            ),
            pl.scan_csv(m5_dir / "calendar.csv", infer_schema_length=1000),
            source_partition="sales_train_evaluation",
            source_run_id=source_run_id,
            silver_run_id=silver_run_id,
        )
    ]


def _supplemental_corpus_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.LazyFrame]:
    frames: list[pl.LazyFrame] = []
    for loader in (
        _load_freshretail_lt_frames,
        _load_m5_frames,
    ):
        frames.extend(
            loader(
                root=root,
                allowed_sources=allowed_sources,
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )
    return frames


def _combined_supplemental_corpus_lazy(
    *,
    raw_dir: str | Path,
    source_run_id: str,
    silver_run_id: str,
) -> pl.LazyFrame | None:
    root = Path(raw_dir)
    allowed_sources = set(allowed_training_dataset_sources()) & set(
        SUPPORTED_SUPPLEMENTAL_CORPUS_SOURCES
    )
    frames = _supplemental_corpus_frames(
        root=root,
        allowed_sources=allowed_sources,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
    if not frames:
        return None
    return pl.concat(frames, how="vertical_relaxed").filter(
        pl.col("dataset_source").is_in(sorted(allowed_sources))
    )


def build_supplemental_corpus_frame(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.DataFrame | None:
    """Build the supplemental corpus in memory without persisting an intermediate parquet."""

    combined = _combined_supplemental_corpus_lazy(
        raw_dir=raw_dir,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
    if combined is None:
        return None
    return combined.collect().sort(
        ["dataset_source", "dt", "location_id", "product_id"]
    )


def build_supplemental_corpus_standardized_dataset(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> Path | None:
    combined = _combined_supplemental_corpus_lazy(
        raw_dir=raw_dir,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
    if combined is None:
        return None
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    combined.sink_parquet(target_path, compression="zstd")
    return target_path

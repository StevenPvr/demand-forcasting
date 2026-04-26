from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import zipfile

import pandas as pd
import polars as pl

from praedixa.platform.datasets.standardization.schema import (
    align_lazy_frame_to_canonical_schema,
    build_series_id_expr,
)
from praedixa.platform.datasets.standardization.supplemental_corpus_standardizers import (
    DEFAULT_SILVER_RUN_ID,
    DEFAULT_SOURCE_RUN_ID,
    freshretail_promo_flag_expr,
    standardize_freshretail_lt_lazy_frame,
    standardize_m5_sales_lazy_frame,
    standardize_perishable_goods_management_lazy_frame,
    standardize_restaurant_sales_report_lazy_frame,
    standardize_uci_online_retail_lazy_frame,
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
DEFAULT_M5_DIR = DEFAULT_RAW_DIR / "m5_forecasting_accuracy_zenodo"
DEFAULT_M5_ZIP_PATH = DEFAULT_RAW_DIR / "m5_forecasting_accuracy_zenodo.zip"
DEFAULT_UCI_ONLINE_RETAIL_II_PATH = DEFAULT_RAW_DIR / "uci_online_retail_ii.zip"
DEFAULT_UCI_ONLINE_RETAIL_PATH = DEFAULT_RAW_DIR / "uci_online_retail.zip"
DEFAULT_RESTAURANT_SALES_REPORT_PATH = (
    DEFAULT_RAW_DIR / "restaurant_sales_report_kaggle.zip"
)
DEFAULT_PERISHABLE_GOODS_PATH = DEFAULT_RAW_DIR / "perishable_goods_management.csv"
DEFAULT_PERISHABLE_GOODS_ZIP_PATH = (
    DEFAULT_RAW_DIR / "perishable_goods_management_kaggle.zip"
)

SUPPORTED_SUPPLEMENTAL_CORPUS_SOURCES = frozenset(
    {
        "freshretail_lt",
        "first_party_daily",
        "m5_forecasting_accuracy",
        "uci_online_retail_ii",
        "uci_online_retail",
        "restaurant_sales_report",
        "perishable_goods_management",
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
        "first_party_daily",
        "canonical daily client-owned demand feed",
        True,
        True,
        "Client onboarding data is already aligned to the canonical daily schema and stays eligible for training.",
    ),
    (
        "m5_forecasting_accuracy",
        "daily retail unit sales by store x product",
        True,
        True,
        "M5 Zenodo exposes daily store-product unit sales under CC BY 4.0.",
    ),
    (
        "uci_online_retail_ii",
        "transaction lines aggregated to daily country-product sales",
        True,
        True,
        "UCI Online Retail II is CC BY 4.0 and maps cleanly to observed daily product sales.",
    ),
    (
        "uci_online_retail",
        "transaction lines aggregated to daily country-product sales",
        True,
        True,
        "UCI Online Retail is CC BY 4.0 and maps cleanly to observed daily product sales.",
    ),
    (
        "restaurant_sales_report",
        "restaurant item transactions aggregated to daily item sales",
        True,
        True,
        "Fast Food Sales Report is Apache 2.0 and exposes date-item-quantity transaction rows.",
    ),
    (
        "perishable_goods_management",
        "synthetic daily perishable-goods demand by store x product",
        True,
        True,
        "The CC0 synthetic perishable-goods corpus exposes daily demand and waste signals at store-product grain.",
    ),
    (
        "maven_cafe_rewards_offers",
        "promotion offer and customer-response events",
        False,
        True,
        "Public Domain but not a daily product-demand target; keep it out until a promotion-event layer exists.",
    ),
    (
        "restaurant_sales_forecasting_zenodo",
        "supplementary restaurant forecasting document",
        False,
        True,
        "CC BY 4.0 but the artifact is a DOCX supplement with model tables, not raw sales rows.",
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
        normalized = normalized.with_columns(
            pl.lit(source_partition).alias("source_partition")
        )
    if "source_run_id" not in available_columns:
        normalized = normalized.with_columns(
            pl.lit(source_run_id).alias("source_run_id")
        )
    if "silver_run_id" not in available_columns:
        normalized = normalized.with_columns(
            pl.lit(silver_run_id).alias("silver_run_id")
        )
    return normalized


def _with_optional_series_id(
    frame: pl.LazyFrame, *, available_columns: set[str]
) -> pl.LazyFrame:
    if "series_id" in available_columns:
        return frame
    if {"location_id", "product_id"}.issubset(available_columns):
        return frame.with_columns(
            build_series_id_expr("location_id", "product_id").alias("series_id")
        )
    return frame


def _standardize_canonical_daily_parquet(
    *,
    path: Path,
    dataset_source: str,
    source_partition: str,
    source_run_id: str,
    silver_run_id: str,
) -> pl.LazyFrame:
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
    normalized = _with_optional_series_id(
        normalized, available_columns=available_columns
    )
    return align_lazy_frame_to_canonical_schema(normalized)


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


def _load_canonical_parquet_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.LazyFrame]:
    parquet_specs = [
        ("first_party_daily", DEFAULT_FIRST_PARTY_DAILY_PATH.name, "client_onboarding"),
    ]
    frames: list[pl.LazyFrame] = []
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


def _normalize_partition_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()
    return normalized or "default"


def _read_excel_sheets_from_zip(
    path: Path, member_name: str
) -> list[tuple[str, pl.DataFrame]]:
    with zipfile.ZipFile(path) as archive:
        workbook_bytes = archive.read(member_name)
    excel = pd.ExcelFile(BytesIO(workbook_bytes))
    frames: list[tuple[str, pl.DataFrame]] = []
    for sheet_name in excel.sheet_names:
        pandas_frame = pd.read_excel(excel, sheet_name=sheet_name, dtype=str)
        frames.append(
            (_normalize_partition_name(sheet_name), pl.from_pandas(pandas_frame))
        )
    return frames


def _read_csv_from_zip(path: Path, member_name: str) -> pl.DataFrame:
    with zipfile.ZipFile(path) as archive:
        return pl.read_csv(BytesIO(archive.read(member_name)), infer_schema_length=1000)


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


def _load_uci_online_retail_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.LazyFrame]:
    specs = [
        (
            "uci_online_retail_ii",
            DEFAULT_UCI_ONLINE_RETAIL_II_PATH.name,
            "online_retail_II.xlsx",
        ),
        (
            "uci_online_retail",
            DEFAULT_UCI_ONLINE_RETAIL_PATH.name,
            "Online Retail.xlsx",
        ),
    ]
    frames: list[pl.LazyFrame] = []
    for dataset_source, zip_name, member_name in specs:
        if dataset_source not in allowed_sources:
            continue
        path = _resolve_existing_path(root / zip_name)
        if path is None:
            continue
        for partition_name, sheet_frame in _read_excel_sheets_from_zip(
            path, member_name
        ):
            frames.append(
                standardize_uci_online_retail_lazy_frame(
                    sheet_frame.lazy(),
                    dataset_source=dataset_source,
                    source_partition=partition_name,
                    source_run_id=source_run_id,
                    silver_run_id=silver_run_id,
                )
            )
    return frames


def _load_restaurant_sales_report_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.LazyFrame]:
    if "restaurant_sales_report" not in allowed_sources:
        return []
    path = _resolve_existing_path(root / DEFAULT_RESTAURANT_SALES_REPORT_PATH.name)
    if path is None:
        return []
    frame = _read_csv_from_zip(path, "Balaji Fast Food Sales.csv")
    return [
        standardize_restaurant_sales_report_lazy_frame(
            frame.lazy(),
            source_partition="balaji_fast_food_sales",
            source_run_id=source_run_id,
            silver_run_id=silver_run_id,
        )
    ]


def _load_perishable_goods_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.LazyFrame]:
    if "perishable_goods_management" not in allowed_sources:
        return []
    csv_path = _resolve_existing_path(root / DEFAULT_PERISHABLE_GOODS_PATH.name)
    if csv_path is not None:
        frame = pl.scan_csv(csv_path, infer_schema_length=1000)
    else:
        zip_path = _resolve_existing_path(root / DEFAULT_PERISHABLE_GOODS_ZIP_PATH.name)
        if zip_path is None:
            return []
        frame = _read_csv_from_zip(zip_path, "perishable_goods_management.csv").lazy()
    return [
        standardize_perishable_goods_management_lazy_frame(
            frame,
            source_partition="perishable_goods_management",
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
        _load_canonical_parquet_frames,
        _load_m5_frames,
        _load_uci_online_retail_frames,
        _load_restaurant_sales_report_frames,
        _load_perishable_goods_frames,
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

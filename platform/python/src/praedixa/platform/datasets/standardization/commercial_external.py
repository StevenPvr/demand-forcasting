from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from zipfile import BadZipFile, ZipFile, is_zipfile

import pandas as pd
import polars as pl

from praedixa.platform.datasets.standardization.commercial_external_sql import (
    standardize_mendeley_pharmacy_sql_text,
)
from praedixa.platform.datasets.standardization.commercial_external_standardizers import (
    DEFAULT_SILVER_RUN_ID,
    DEFAULT_SOURCE_RUN_ID,
    freshretail_promo_flag_expr,
    standardize_freshretail_lt_lazy_frame,
    standardize_mendeley_bangladesh_frame,
    standardize_mendeley_ecommerce_frame,
    standardize_uci_online_retail_frame,
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
DEFAULT_OUTPUT_PATH = GLOBAL_DATASET_DIR / "commercial_external_daily.parquet"

DEFAULT_FRESHRETAIL_LT_TRAIN_PATH = DEFAULT_RAW_DIR / "freshretail_lt_train.parquet"
DEFAULT_FRESHRETAIL_LT_EVAL_PATH = DEFAULT_RAW_DIR / "freshretail_lt_eval.parquet"
DEFAULT_UCI_ONLINE_RETAIL_PATH = DEFAULT_RAW_DIR / "uci_online_retail.xlsx"
DEFAULT_UCI_ONLINE_RETAIL_II_PATH = DEFAULT_RAW_DIR / "uci_online_retail_ii.xlsx"
DEFAULT_MENDELEY_ECOMMERCE_PATH = DEFAULT_RAW_DIR / "mendeley_ecommerce.xlsx"
DEFAULT_MENDELEY_PHARMACY_SQL_PATH = DEFAULT_RAW_DIR / "mendeley_pharmacy.sql"
DEFAULT_MENDELEY_PHARMACY_ZIP_PATH = DEFAULT_RAW_DIR / "mendeley_pharmacy.zip"
DEFAULT_MENDELEY_BANGLADESH_PATH = DEFAULT_RAW_DIR / "mendeley_bangladesh.xlsx"
DEFAULT_FIRST_PARTY_DAILY_PATH = DEFAULT_RAW_DIR / "first_party_daily.parquet"
DEFAULT_SYNTHETIC_DAILY_PATH = DEFAULT_RAW_DIR / "synthetic_v1.parquet"
PANDAS_API: Any = pd

__all__ = [
    "CommercialDatasetCompatibility",
    "build_commercial_external_standardized_dataset",
    "freshretail_promo_flag_expr",
]

COMMERCIAL_COMPATIBILITY_ROWS: tuple[
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
        "uci_online_retail",
        "daily sold quantity by product after filtering returns and non-merchandise charges",
        True,
        True,
        "Transaction lines expose quantity and invoice timestamp; we aggregate to daily product demand.",
    ),
    (
        "uci_online_retail_ii",
        "daily sold quantity by product after filtering returns and non-merchandise charges",
        True,
        True,
        "Transaction lines expose quantity and invoice timestamp across two yearly sheets.",
    ),
    (
        "mendeley_ecommerce",
        "daily sold quantity by SKU aggregated from order lines",
        True,
        True,
        "Order lines expose order_date, prod_sku, and prod_qty under CC BY 4.0.",
    ),
    (
        "mendeley_pharmacy_id",
        "daily sold quantity by medicine code aggregated from pharmacy transactions",
        True,
        True,
        "SQL dump exposes transaction date, medicine code, quantity, and price from a real pharmacy DB.",
    ),
    (
        "mendeley_bangladesh_retail",
        "daily sold quantity for one product",
        True,
        True,
        "The dataset is a single-product daily demand series under CC BY 4.0.",
    ),
    (
        "uci_hierarchical_sales",
        "daily SKU sales by brand hierarchy",
        False,
        False,
        "Excluded for now because the surfaced licensing signals conflict between UCI and the upstream DOI record.",
    ),
)


@dataclass(frozen=True)
class CommercialDatasetCompatibility:
    """Compatibility summary for one commercially usable dataset."""

    dataset_source: str
    target_semantics: str
    compatible_with_pipeline: bool
    commercial_use_allowed: bool
    reason: str


def _compatibility_from_row(
    row: tuple[str, str, bool, bool, str],
) -> CommercialDatasetCompatibility:
    dataset_source, target_semantics, compatible_with_pipeline, commercial_use_allowed, reason = row
    return CommercialDatasetCompatibility(
        dataset_source=dataset_source,
        target_semantics=target_semantics,
        compatible_with_pipeline=compatible_with_pipeline,
        commercial_use_allowed=commercial_use_allowed,
        reason=reason,
    )


def commercial_dataset_compatibility_matrix() -> list[CommercialDatasetCompatibility]:
    return [_compatibility_from_row(row) for row in COMMERCIAL_COMPATIBILITY_ROWS]


def _resolve_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _concat_excel_workbook(workbook: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(workbook.values(), ignore_index=True)


def _embedded_excel_names(archive: ZipFile) -> list[str]:
    return [
        name
        for name in archive.namelist()
        if name.lower().endswith(".xlsx")
    ]


def _read_embedded_excel_frame(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or not is_zipfile(path):
        return None
    try:
        with ZipFile(path) as archive:
            excel_names = _embedded_excel_names(archive)
            if not excel_names:
                return None
            workbook = cast(
                dict[str, pd.DataFrame],
                PANDAS_API.read_excel(
                    BytesIO(archive.read(excel_names[0])),
                    sheet_name=None,
                    engine="openpyxl",
                ),
            )
            return _concat_excel_workbook(workbook)
    except BadZipFile:
        return None


def _read_excel_all_sheets(path: Path) -> pd.DataFrame:
    embedded_frame = _read_embedded_excel_frame(path)
    if embedded_frame is not None:
        return embedded_frame
    workbook = cast(dict[str, pd.DataFrame], PANDAS_API.read_excel(path, sheet_name=None, engine="openpyxl"))
    return _concat_excel_workbook(workbook)


def _read_excel_frame(path: Path) -> pd.DataFrame:
    return cast(pd.DataFrame, PANDAS_API.read_excel(path, engine="openpyxl"))


def _read_mendeley_pharmacy_sql_text(raw_dir: Path) -> str | None:
    sql_path = _resolve_existing_path(raw_dir / DEFAULT_MENDELEY_PHARMACY_SQL_PATH.name)
    if sql_path is not None:
        return sql_path.read_text(encoding="utf-8", errors="ignore")

    zip_path = _resolve_existing_path(raw_dir / DEFAULT_MENDELEY_PHARMACY_ZIP_PATH.name)
    if zip_path is None:
        return None
    with ZipFile(zip_path) as archive:
        for member_name in archive.namelist():
            if member_name.lower().endswith(".sql"):
                return archive.read(member_name).decode("utf-8", errors="ignore")
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


def build_commercial_external_standardized_dataset(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> Path | None:
    root = Path(raw_dir)
    allowed_sources = set(allowed_training_dataset_sources())
    frames = _commercial_external_frames(
        root=root,
        allowed_sources=allowed_sources,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
    if not frames:
        return None

    combined = (
        pl.concat(frames, how="vertical_relaxed")
        .filter(pl.col("dataset_source").is_in(sorted(allowed_sources)))
        .sort(["dataset_source", "dt", "location_id", "product_id"])
    )
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(target_path)
    return target_path


def _commercial_external_frames(
    *,
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.DataFrame]:
    frames: list[pl.DataFrame] = []
    frames.extend(
        _load_freshretail_lt_frames(root, allowed_sources, source_run_id, silver_run_id)
    )
    frames.extend(
        _load_uci_online_retail_frames(
            root, allowed_sources, source_run_id, silver_run_id
        )
    )
    frames.extend(
        _load_mendeley_frames(root, allowed_sources, source_run_id, silver_run_id)
    )
    frames.extend(
        _load_canonical_parquet_frames(
            root, allowed_sources, source_run_id, silver_run_id
        )
    )
    return frames


def _load_freshretail_lt_frames(
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.DataFrame]:
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


def _load_uci_online_retail_frames(
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.DataFrame]:
    frames: list[pl.DataFrame] = []
    uci_specs = [
        (
            "uci_online_retail",
            DEFAULT_UCI_ONLINE_RETAIL_PATH.name,
            "uci_online_retail.zip",
            "uk_online_retail_1",
        ),
        (
            "uci_online_retail_ii",
            DEFAULT_UCI_ONLINE_RETAIL_II_PATH.name,
            "uci_online_retail_ii.zip",
            "uk_online_retail_ii_1",
        ),
    ]
    for dataset_source, default_name, zip_name, location_id in uci_specs:
        if dataset_source not in allowed_sources:
            continue
        path = _resolve_existing_path(root / default_name, root / zip_name)
        if path is None:
            continue
        frames.append(
            standardize_uci_online_retail_frame(
                _read_excel_all_sheets(path),
                dataset_source=dataset_source,
                location_id=location_id,
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )
    return frames


def _load_mendeley_frames(
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.DataFrame]:
    frames: list[pl.DataFrame] = []
    ecommerce_frame = _mendeley_ecommerce_frame(root, allowed_sources, source_run_id, silver_run_id)
    pharmacy_frame = _mendeley_pharmacy_frame(root, allowed_sources, source_run_id, silver_run_id)
    bangladesh_frame = _mendeley_bangladesh_frame(root, allowed_sources, source_run_id, silver_run_id)
    for frame in (ecommerce_frame, pharmacy_frame, bangladesh_frame):
        if frame is not None:
            frames.append(frame)
    return frames


def _mendeley_ecommerce_frame(
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> pl.DataFrame | None:
    if "mendeley_ecommerce" not in allowed_sources:
        return None
    path = _resolve_existing_path(root / DEFAULT_MENDELEY_ECOMMERCE_PATH.name)
    if path is None:
        return None
    return standardize_mendeley_ecommerce_frame(
        _read_excel_frame(path),
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


def _mendeley_pharmacy_frame(
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> pl.DataFrame | None:
    if "mendeley_pharmacy_id" not in allowed_sources:
        return None
    pharmacy_sql_text = _read_mendeley_pharmacy_sql_text(root)
    if pharmacy_sql_text is None:
        return None
    return standardize_mendeley_pharmacy_sql_text(
        pharmacy_sql_text,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


def _mendeley_bangladesh_frame(
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> pl.DataFrame | None:
    if "mendeley_bangladesh_retail" not in allowed_sources:
        return None
    path = _resolve_existing_path(root / DEFAULT_MENDELEY_BANGLADESH_PATH.name)
    if path is None:
        return None
    return standardize_mendeley_bangladesh_frame(
        _read_excel_frame(path),
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


def _load_canonical_parquet_frames(
    root: Path,
    allowed_sources: set[str],
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.DataFrame]:
    parquet_specs = [
        ("first_party_daily", DEFAULT_FIRST_PARTY_DAILY_PATH.name, "client_onboarding"),
        ("synthetic_v1", DEFAULT_SYNTHETIC_DAILY_PATH.name, "synthetic_train"),
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

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import cast

import polars as pl

from praedixa.platform.datasets.standardization.bakery import DEFAULT_INPUT_PATH as DEFAULT_BAKERY_INPUT_PATH
from praedixa.platform.datasets.standardization.bakery import build_bakery_standardized_dataset
from praedixa.platform.datasets.standardization.commercial_external import (
    DEFAULT_RAW_DIR as DEFAULT_COMMERCIAL_RAW_DIR,
)
from praedixa.platform.datasets.standardization.commercial_external import build_commercial_external_standardized_dataset
from praedixa.platform.datasets.standardization.freshretail import build_freshretail_standardized_dataset
from praedixa.platform.datasets.standardization.quality import raise_on_error_issues, validate_canonical_frame
from praedixa.platform.datasets.standardization.schema import CANONICAL_COLUMNS
from praedixa.platform.runtime.paths import GLOBAL_DATASET_DIR


DEFAULT_OUTPUT_DIR = GLOBAL_DATASET_DIR
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalDatasetArtifacts:
    """Paths emitted by the canonical daily dataset pipeline."""

    freshretail_daily: Path | None
    commercial_external_daily: Path | None
    bakery_daily: Path | None
    global_gold: Path
    manifest_path: Path


@dataclass(frozen=True)
class DatasetOutputPaths:
    """Output paths emitted by one global dataset standardization run."""

    freshretail_output: Path
    commercial_external_output: Path
    bakery_output: Path
    combined_output: Path
    manifest_output: Path


def _build_output_paths(target_dir: Path) -> DatasetOutputPaths:
    return DatasetOutputPaths(
        freshretail_output=target_dir / "freshretail_daily.parquet",
        commercial_external_output=target_dir / "commercial_external_daily.parquet",
        bakery_output=target_dir / "bakery_daily.parquet",
        combined_output=target_dir / "global_daily_demand.parquet",
        manifest_output=target_dir / "global_daily_demand_manifest.json",
    )


def _summarize_frame(frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": int(frame.height),
        "series_count": int(frame.select(pl.col("series_id").n_unique()).item()),
        "start_date": frame.select(pl.col("dt").min()).item().isoformat() if frame.height else None,
        "end_date": frame.select(pl.col("dt").max()).item().isoformat() if frame.height else None,
    }


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _collect_source_frames(
    *,
    freshretail_path: Path,
    commercial_external_path: Path | None,
    bakery_path: Path | None,
) -> tuple[list[pl.DataFrame], dict[str, object]]:
    freshretail_frame = pl.read_parquet(freshretail_path)
    frames = [freshretail_frame]
    sources: dict[str, object] = {
        "freshretail": _summarize_frame(freshretail_frame),
    }
    if commercial_external_path is not None:
        commercial_external_frame = pl.read_parquet(commercial_external_path)
        frames.append(commercial_external_frame)
        sources["commercial_external"] = _summarize_frame(commercial_external_frame)
    if bakery_path is not None:
        bakery_frame = pl.read_parquet(bakery_path)
        frames.append(bakery_frame)
        sources["bakery"] = _summarize_frame(bakery_frame)
    return frames, sources


def _validate_frame_or_raise(frame: pl.DataFrame, *, dataset_name: str) -> dict[str, object]:
    report = validate_canonical_frame(frame, dataset_name=dataset_name)
    raise_on_error_issues(report)
    if report.warning_count:
        logger.warning(
            "Canonical dataset quality warnings: dataset=%s warnings=%s",
            dataset_name,
            ", ".join(f"{issue.code}={issue.row_count}" for issue in report.issues if issue.severity == "warning"),
        )
    return report.to_manifest_dict()


def _maybe_build_bakery_dataset(
    *,
    bakery_input_path: str | Path | None,
    bakery_output: Path,
) -> Path | None:
    bakery_source_path = Path(bakery_input_path) if bakery_input_path is not None else None
    if bakery_source_path is None or not bakery_source_path.exists():
        return None
    return build_bakery_standardized_dataset(output_path=bakery_output)


def _append_combined_summary(sources: dict[str, object], combined: pl.DataFrame) -> None:
    sources["combined"] = {
        "rows": int(combined.height),
        "series_count": int(combined.select(pl.col("series_id").n_unique()).item()),
        "start_date": combined.select(pl.col("dt").min()).item().isoformat() if combined.height else None,
        "end_date": combined.select(pl.col("dt").max()).item().isoformat() if combined.height else None,
    }


def _build_source_outputs(
    *,
    outputs: DatasetOutputPaths,
    bakery_input_path: str | Path | None,
    commercial_raw_dir: str | Path,
) -> tuple[Path, Path | None, Path | None]:
    freshretail_path = build_freshretail_standardized_dataset(output_path=outputs.freshretail_output)
    commercial_external_path = build_commercial_external_standardized_dataset(
        output_path=outputs.commercial_external_output,
        raw_dir=commercial_raw_dir,
    )
    bakery_path = _maybe_build_bakery_dataset(
        bakery_input_path=bakery_input_path,
        bakery_output=outputs.bakery_output,
    )
    return freshretail_path, commercial_external_path, bakery_path


def _build_combined_frame(
    *,
    freshretail_path: Path,
    commercial_external_path: Path | None,
    bakery_path: Path | None,
) -> tuple[pl.DataFrame, dict[str, object], dict[str, object]]:
    frames, sources = _collect_source_frames(
        freshretail_path=freshretail_path,
        commercial_external_path=commercial_external_path,
        bakery_path=bakery_path,
    )
    source_quality = {
        source_name: _validate_frame_or_raise(frame, dataset_name=source_name)
        for source_name, frame in zip(sources.keys(), frames, strict=False)
    }
    combined = pl.concat([frame.select(CANONICAL_COLUMNS) for frame in frames], how="vertical_relaxed")
    return combined, sources, cast(dict[str, object], source_quality)


def _build_manifest_payload(
    *,
    sources: dict[str, object],
    source_quality: dict[str, object],
    combined_quality: dict[str, object],
) -> dict[str, object]:
    return {
        "grain": "dt x location_id x product_id",
        "cadence": "daily",
        "target": "observed_demand_qty",
        "sources": sources,
        "known_limitations": [
            "FreshRetail and FreshRetail-LT expose native stockout-like signals while most external commercial datasets and bakery do not.",
            "Some commercial external datasets are single-location or single-product and are useful mainly as regularization support.",
            "The hierarchical UCI sales dataset stays excluded until its commercial licensing is unambiguously resolved.",
            "The local bronze replacement in data/ is not the future production raw/bronze POS ingestion path.",
            "The canonical daily dataset remains sparse and does not densify missing dates in silver.",
        ],
        "data_quality": {
            "sources": source_quality,
            "combined": combined_quality,
        },
    }


def build_global_daily_standardization(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    bakery_input_path: str | Path | None = DEFAULT_BAKERY_INPUT_PATH,
    commercial_raw_dir: str | Path = DEFAULT_COMMERCIAL_RAW_DIR,
) -> GlobalDatasetArtifacts:
    """Build canonical daily standardized datasets from local bronze-like source files."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = _build_output_paths(target_dir)

    freshretail_path, commercial_external_path, bakery_path = _build_source_outputs(
        outputs=outputs,
        bakery_input_path=bakery_input_path,
        commercial_raw_dir=commercial_raw_dir,
    )
    combined, sources, source_quality = _build_combined_frame(
        freshretail_path=freshretail_path,
        commercial_external_path=commercial_external_path,
        bakery_path=bakery_path,
    )
    combined_quality = _validate_frame_or_raise(combined, dataset_name="combined")
    combined.write_parquet(outputs.combined_output)
    _append_combined_summary(sources, combined)
    _write_manifest(
        outputs.manifest_output,
        _build_manifest_payload(
            sources=sources,
            source_quality=source_quality,
            combined_quality=combined_quality,
        ),
    )
    return GlobalDatasetArtifacts(
        freshretail_daily=freshretail_path,
        commercial_external_daily=commercial_external_path,
        bakery_daily=bakery_path,
        global_gold=outputs.combined_output,
        manifest_path=outputs.manifest_output,
    )

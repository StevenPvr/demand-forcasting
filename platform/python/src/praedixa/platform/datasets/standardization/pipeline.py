from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path

import polars as pl

from praedixa.platform.datasets.standardization.bakery import DEFAULT_INPUT_PATH as DEFAULT_BAKERY_INPUT_PATH
from praedixa.platform.datasets.standardization.bakery import build_bakery_standardized_frame
from praedixa.platform.datasets.standardization.freshretail import build_freshretail_standardized_frame
from praedixa.platform.datasets.standardization.quality import raise_on_error_issues, validate_canonical_frame
from praedixa.platform.datasets.standardization.supplemental_corpus import DEFAULT_RAW_DIR as DEFAULT_SUPPLEMENTAL_CORPUS_RAW_DIR
from praedixa.platform.datasets.standardization.supplemental_corpus import build_supplemental_corpus_frame


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalDatasetArtifacts:
    """In-memory quality summary for the canonical dataset assembly path."""

    source_summaries: Mapping[str, object]
    data_quality: Mapping[str, object]


def _summarize_frame(frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": int(frame.height),
        "series_count": int(frame.select(pl.col("series_id").n_unique()).item()),
        "start_date": frame.select(pl.col("dt").min()).item().isoformat() if frame.height else None,
        "end_date": frame.select(pl.col("dt").max()).item().isoformat() if frame.height else None,
    }


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


def _build_source_frames(
    *,
    bakery_input_path: str | Path | None,
    supplemental_corpus_raw_dir: str | Path,
) -> dict[str, pl.DataFrame]:
    frames: dict[str, pl.DataFrame] = {
        "freshretail": build_freshretail_standardized_frame(),
    }
    supplemental_corpus = build_supplemental_corpus_frame(raw_dir=supplemental_corpus_raw_dir)
    if supplemental_corpus is not None:
        frames["supplemental_corpus"] = supplemental_corpus
    bakery_source_path = Path(bakery_input_path) if bakery_input_path is not None else None
    if bakery_source_path is not None and bakery_source_path.exists():
        frames["bakery"] = build_bakery_standardized_frame(input_path=bakery_source_path)
    return frames


def build_global_daily_standardization(
    *,
    bakery_input_path: str | Path | None = DEFAULT_BAKERY_INPUT_PATH,
    supplemental_corpus_raw_dir: str | Path = DEFAULT_SUPPLEMENTAL_CORPUS_RAW_DIR,
) -> GlobalDatasetArtifacts:
    """Validate the canonical dataset assembly path without persisting intermediate parquets."""

    source_frames = _build_source_frames(
        bakery_input_path=bakery_input_path,
        supplemental_corpus_raw_dir=supplemental_corpus_raw_dir,
    )
    source_summaries = {
        source_name: _summarize_frame(frame)
        for source_name, frame in source_frames.items()
    }
    source_quality = {
        source_name: _validate_frame_or_raise(frame, dataset_name=source_name)
        for source_name, frame in source_frames.items()
    }
    combined = pl.concat(list(source_frames.values()), how="vertical_relaxed")
    source_summaries["combined"] = _summarize_frame(combined)
    combined_quality = _validate_frame_or_raise(combined, dataset_name="combined")
    return GlobalDatasetArtifacts(
        source_summaries=source_summaries,
        data_quality={
            "sources": source_quality,
            "combined": combined_quality,
        },
    )

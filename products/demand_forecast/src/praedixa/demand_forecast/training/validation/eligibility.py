from __future__ import annotations

import logging

import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE: float = 0.75
NON_TRAINABLE_TARGET_SOURCES: tuple[str, ...] = (
    "closed_or_missing_observation",
    "dense_calendar_zero_fill",
)
TRAINING_ELIGIBILITY_COLUMNS: tuple[str, ...] = (
    "usable_for_training_flag",
    "censor_flag",
    "label_quality_score",
    "target_semantics",
    "target_source",
)


def training_eligibility_mask(
    frame: pd.DataFrame,
    *,
    min_label_quality_score: float = DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE,
    expected_target_semantics: str | None = None,
) -> pd.Series:
    """Return rows eligible for model fitting, not necessarily for scoring."""

    mask = pd.Series(True, index=frame.index, dtype=bool)
    if "usable_for_training_flag" in frame.columns:
        mask &= frame["usable_for_training_flag"].fillna(False).astype(bool)
    if "label_quality_score" in frame.columns:
        label_quality = pd.to_numeric(frame["label_quality_score"], errors="coerce")
        mask &= label_quality.ge(float(min_label_quality_score)).fillna(False)
    if "target_source" in frame.columns:
        mask &= ~frame["target_source"].astype("string").isin(NON_TRAINABLE_TARGET_SOURCES)
    if expected_target_semantics is not None and "target_semantics" in frame.columns:
        mask &= frame["target_semantics"].astype("string").eq(expected_target_semantics)
    return mask


def filter_training_eligible_rows(
    frame: pd.DataFrame,
    *,
    label: str,
    min_label_quality_score: float = DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE,
    expected_target_semantics: str | None = None,
    logger: logging.Logger | None = None,
    log_level: int = logging.INFO,
) -> pd.DataFrame:
    """Filter rows used for training while preserving scoring/evaluation rows elsewhere."""

    if frame.empty:
        return frame.copy()
    mask = training_eligibility_mask(
        frame,
        min_label_quality_score=min_label_quality_score,
        expected_target_semantics=expected_target_semantics,
    )
    filtered = frame.loc[mask].copy()
    dropped_rows = int(len(frame) - len(filtered))
    if dropped_rows > 0:
        resolved_logger = logger or LOGGER
        resolved_logger.log(
            log_level,
            "Filtered non-trainable rows: label=%s input_rows=%s output_rows=%s dropped_rows=%s min_label_quality_score=%.3f expected_target_semantics=%s",
            label,
            len(frame),
            len(filtered),
            dropped_rows,
            min_label_quality_score,
            expected_target_semantics,
        )
    return filtered

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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
LATENT_TARGET_SEMANTICS = "latent_demand_estimated"


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_tuple(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(_sql_literal(value) for value in values) + ")"


def _qualified(column: str, table_alias: str | None) -> str:
    return column if table_alias is None else f"{table_alias}.{column}"


@dataclass(frozen=True)
class GoldTrainingEligibilityContract:
    """Single train/tuning eligibility contract shared by SQL and Pandas paths."""

    min_label_quality_score: float = DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE
    non_trainable_target_sources: tuple[str, ...] = NON_TRAINABLE_TARGET_SOURCES
    latent_target_semantics: str = LATENT_TARGET_SEMANTICS
    required_sql_columns: tuple[str, ...] = TRAINING_ELIGIBILITY_COLUMNS

    def sql_filter(
        self,
        *,
        available_columns: Iterable[str] | None,
        table_alias: str | None = None,
    ) -> str:
        if available_columns is not None and not set(
            self.required_sql_columns
        ).issubset(set(available_columns)):
            return "true"
        usable = _qualified("usable_for_training_flag", table_alias)
        label_quality = _qualified("label_quality_score", table_alias)
        target_source = _qualified("target_source", table_alias)
        censor_flag = _qualified("censor_flag", table_alias)
        target_semantics = _qualified("target_semantics", table_alias)
        return f"""
(
    coalesce({usable}, false)
    and coalesce({label_quality}, 0.0) >= {self.min_label_quality_score:.6f}
    and coalesce({target_source}, '') not in {_sql_tuple(self.non_trainable_target_sources)}
    and (
        not coalesce({censor_flag}, false)
        or coalesce({target_semantics}, '') = {_sql_literal(self.latent_target_semantics)}
    )
)""".strip()

    def pandas_mask(
        self,
        frame: pd.DataFrame,
        *,
        expected_target_semantics: str | None = None,
    ) -> pd.Series:
        mask = pd.Series(True, index=frame.index, dtype=bool)
        if "usable_for_training_flag" in frame.columns:
            mask &= frame["usable_for_training_flag"].fillna(False).astype(bool)
        if "label_quality_score" in frame.columns:
            label_quality = pd.to_numeric(frame["label_quality_score"], errors="coerce")
            mask &= label_quality.ge(float(self.min_label_quality_score)).fillna(False)
        if "target_source" in frame.columns:
            mask &= ~frame["target_source"].astype("string").isin(
                self.non_trainable_target_sources
            )
        if {"censor_flag", "target_semantics"}.issubset(frame.columns):
            latent_mask = (
                frame["target_semantics"]
                .astype("string")
                .eq(self.latent_target_semantics)
            )
            uncensored_mask = ~frame["censor_flag"].fillna(False).astype(bool)
            mask &= uncensored_mask | latent_mask
        if (
            expected_target_semantics is not None
            and "target_semantics" in frame.columns
        ):
            mask &= (
                frame["target_semantics"].astype("string").eq(expected_target_semantics)
            )
        return mask


GOLD_TRAINING_ELIGIBILITY = GoldTrainingEligibilityContract()


def training_eligibility_mask(
    frame: pd.DataFrame,
    *,
    min_label_quality_score: float = DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE,
    expected_target_semantics: str | None = None,
) -> pd.Series:
    """Return rows eligible for model fitting, not necessarily for scoring."""

    return GoldTrainingEligibilityContract(
        min_label_quality_score=min_label_quality_score
    ).pandas_mask(frame, expected_target_semantics=expected_target_semantics)


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

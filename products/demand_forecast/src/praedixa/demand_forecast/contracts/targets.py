from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_ABSOLUTE_TARGET_COL = "target_demand_qty_d_plus_1"
DEFAULT_VARIATION_TARGET_COL = "target_delta_log_wow_d_plus_1"
DEFAULT_FIELD_BASELINE_COL = "field_baseline_blend_lag_1_lag_7_d_plus_1"
DEFAULT_FIELD_BASELINE_RESIDUAL_TARGET_COL = (
    "target_residual_field_blend_lag_1_lag_7_d_plus_1"
)
DEFAULT_ABSOLUTE_TARGET_CANDIDATES = (
    DEFAULT_ABSOLUTE_TARGET_COL,
    "target",
    "sale_amount",
)
DEFAULT_WOW_ANCHOR_CANDIDATES = (
    "target_lag_7",
    "sale_amount_lag_7",
    "lag_7",
    "seasonal_naive_d7",
)
DEFAULT_WOW_MEAN_BASELINE_CANDIDATES = (
    "target_same_dow_mean_4w",
    "same_dow_mean_4w",
)
DEFAULT_FIELD_BASELINE_CANDIDATES = (
    DEFAULT_FIELD_BASELINE_COL,
    "field_baseline_prediction_d_plus_1",
)


@dataclass(frozen=True)
class TargetContract:
    learning_target_col: str
    absolute_target_col: str
    target_mode: str
    reconstruction_anchor_col: str | None


def _resolve_common_column(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    for candidate in candidates:
        if candidate in train_frame.columns and candidate in tuning_frame.columns:
            return candidate
    return None


def _absolute_target_column_or_raise(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
) -> str:
    absolute_target_col = _resolve_common_column(
        train_frame,
        tuning_frame,
        DEFAULT_ABSOLUTE_TARGET_CANDIDATES,
    )
    if absolute_target_col is None:
        raise ValueError(
            "Unable to resolve a common absolute target column across frames. "
            f"Tried: {', '.join(DEFAULT_ABSOLUTE_TARGET_CANDIDATES)}."
        )
    return absolute_target_col


def _learning_target_column(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    requested_target_col: str,
    absolute_target_col: str,
) -> str:
    if (
        requested_target_col in train_frame.columns
        and requested_target_col in tuning_frame.columns
    ):
        return requested_target_col
    if requested_target_col != absolute_target_col:
        raise ValueError(
            "Requested learning target is unavailable across evaluation frames: "
            f"`{requested_target_col}`."
        )
    return absolute_target_col


def resolve_target_contract(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    *,
    requested_target_col: str = DEFAULT_ABSOLUTE_TARGET_COL,
) -> TargetContract:
    """Resolve the shared learning/scoring target contract across two frames."""

    absolute_target_col = _absolute_target_column_or_raise(train_frame, tuning_frame)
    wow_anchor_col = _resolve_common_column(
        train_frame,
        tuning_frame,
        DEFAULT_WOW_ANCHOR_CANDIDATES,
    )
    field_baseline_col = _resolve_common_column(
        train_frame,
        tuning_frame,
        DEFAULT_FIELD_BASELINE_CANDIDATES,
    )
    learning_target_col = _learning_target_column(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        requested_target_col=requested_target_col,
        absolute_target_col=absolute_target_col,
    )
    if (
        learning_target_col == DEFAULT_VARIATION_TARGET_COL
        and wow_anchor_col is not None
    ):
        return TargetContract(
            learning_target_col=DEFAULT_VARIATION_TARGET_COL,
            absolute_target_col=absolute_target_col,
            target_mode="delta_log_wow",
            reconstruction_anchor_col=wow_anchor_col,
        )
    if learning_target_col == DEFAULT_VARIATION_TARGET_COL:
        raise ValueError("WoW delta target requires a reconstruction anchor column.")
    if (
        learning_target_col == DEFAULT_FIELD_BASELINE_RESIDUAL_TARGET_COL
        and field_baseline_col is not None
    ):
        return TargetContract(
            learning_target_col=DEFAULT_FIELD_BASELINE_RESIDUAL_TARGET_COL,
            absolute_target_col=absolute_target_col,
            target_mode="additive_residual",
            reconstruction_anchor_col=field_baseline_col,
        )
    if learning_target_col == DEFAULT_FIELD_BASELINE_RESIDUAL_TARGET_COL:
        raise ValueError(
            "Baseline residual target requires a field baseline reconstruction column."
        )
    return TargetContract(
        learning_target_col=absolute_target_col,
        absolute_target_col=absolute_target_col,
        target_mode="identity",
        reconstruction_anchor_col=None,
    )


def ensure_learning_target_column(
    frame: pd.DataFrame,
    target_contract: TargetContract,
) -> pd.DataFrame:
    """Return a frame with the resolved learning target column available."""

    if target_contract.learning_target_col in frame.columns:
        return frame
    raise ValueError(
        f"Frame is missing learning target column `{target_contract.learning_target_col}`."
    )


def reconstruct_absolute_predictions(
    predictions: np.ndarray,
    frame: pd.DataFrame,
    target_contract: TargetContract,
) -> np.ndarray:
    """Map model outputs back to absolute demand for business scoring."""

    raw_predictions = np.asarray(predictions, dtype=float)
    if target_contract.target_mode == "delta_log_wow":
        anchor_col = target_contract.reconstruction_anchor_col
        if anchor_col is None or anchor_col not in frame.columns:
            raise ValueError(
                "WoW delta reconstruction requires an anchor column in the scoring frame."
            )
        anchor = frame[anchor_col].to_numpy(dtype=float)
        absolute_predictions = np.expm1(
            raw_predictions + np.log1p(np.clip(anchor, 0.0, None))
        )
    elif target_contract.target_mode == "additive_residual":
        anchor_col = target_contract.reconstruction_anchor_col
        if anchor_col is None or anchor_col not in frame.columns:
            raise ValueError(
                "Baseline residual reconstruction requires an anchor column in the scoring frame."
            )
        anchor = frame[anchor_col].to_numpy(dtype=float)
        absolute_predictions = anchor + raw_predictions
    elif target_contract.target_mode == "log1p":
        absolute_predictions = np.expm1(raw_predictions)
    else:
        absolute_predictions = raw_predictions
    return np.asarray(np.clip(absolute_predictions, 0.0, None), dtype=float)


def build_target_contract_metadata(
    target_contract: TargetContract,
) -> dict[str, str | None]:
    """Serialize the target contract into JSON-friendly metadata fields."""

    return {
        "learning_target_col": target_contract.learning_target_col,
        "absolute_target_col": target_contract.absolute_target_col,
        "target_transform": target_contract.target_mode,
        "reconstruction_anchor_col": target_contract.reconstruction_anchor_col,
    }

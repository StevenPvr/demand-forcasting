from __future__ import annotations

"""Validation temporelle walk-forward et selection du meilleur trial."""

from typing import Any, cast

import optuna
import pandas as pd

from src.data_preprocessing.constants import ORIGIN_DATE_COLUMN


def build_walk_forward_folds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_folds: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Construit des folds walk-forward sur le split validation."""

    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if len(val_df) < n_folds:
        raise ValueError("Validation split must contain at least one row per fold")
    ordered_train = train_df.sort_values(ORIGIN_DATE_COLUMN).reset_index(drop=True)
    ordered_val = val_df.sort_values(ORIGIN_DATE_COLUMN).reset_index(drop=True)
    fold_sizes: list[int] = [len(ordered_val) // n_folds] * n_folds
    for index in range(len(ordered_val) % n_folds):
        fold_sizes[index] += 1
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    start_index = 0
    for fold_size in fold_sizes:
        end_index = start_index + fold_size
        fold_train = pd.concat([ordered_train, ordered_val.iloc[:start_index]], ignore_index=True)
        fold_val = ordered_val.iloc[start_index:end_index].reset_index(drop=True)
        folds.append((fold_train, fold_val))
        start_index = end_index
    return folds


def build_recent_history_walk_forward_folds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_folds: int,
    tail_multiple: int = 2,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Construit des folds walk-forward sur la queue recente de train+validation."""

    if tail_multiple <= 0:
        raise ValueError("tail_multiple must be positive")
    ordered_train = train_df.sort_values(ORIGIN_DATE_COLUMN).reset_index(drop=True)
    ordered_val = val_df.sort_values(ORIGIN_DATE_COLUMN).reset_index(drop=True)
    combined_history = pd.concat([ordered_train, ordered_val], ignore_index=True)
    if len(combined_history) <= 1:
        raise ValueError("Combined history must contain at least two rows")
    target_tail_size = max(1, len(ordered_val) * tail_multiple)
    tail_size = min(target_tail_size, len(combined_history) - 1)
    initial_train_size = len(combined_history) - tail_size
    recent_train = combined_history.iloc[:initial_train_size].reset_index(drop=True)
    recent_val = combined_history.iloc[initial_train_size:].reset_index(drop=True)
    resolved_fold_count = n_folds if n_folds > 0 else len(recent_val)
    return build_walk_forward_folds(
        train_df=recent_train,
        val_df=recent_val,
        n_folds=resolved_fold_count,
    )


def best_trial_by_tiebreakers(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Retourne le meilleur trial selon MAE, puis MASE, puis penalite, puis numero."""

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed_trials:
        raise RuntimeError("No completed optimization trial available")

    def _sort_key(trial: optuna.trial.FrozenTrial) -> tuple[float, float, float, int]:
        trial_value = trial.value
        if trial_value is None:
            raise RuntimeError("Completed trial is missing a value")
        mase_value = trial.user_attrs.get("mean_mase")
        penalty_value = trial.user_attrs.get("complexity_penalty")
        mase = float(mase_value) if mase_value is not None else float("inf")
        penalty = float(penalty_value) if penalty_value is not None else float("inf")
        return float(trial_value), mase, penalty, int(trial.number)

    return cast(list[optuna.trial.FrozenTrial], sorted(completed_trials, key=_sort_key))[0]

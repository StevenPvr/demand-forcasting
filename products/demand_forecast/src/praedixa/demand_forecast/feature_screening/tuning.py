from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import logging
import os
from typing import Any, Callable, cast

import numpy as np
import optuna
import pandas as pd

from praedixa.demand_forecast.feature_screening.constants import (
    DEFAULT_MODEL_PARAMS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET_COL,
    DEFAULT_TUNING_PROGRESS_LOG_EVERY,
    DEFAULT_TUNING_TRIALS,
)
from praedixa.demand_forecast.feature_screening.tft_fold_cache import (
    cached_screening_fold_artifacts,
)
from praedixa.demand_forecast.feature_screening.metrics import compute_wape, compute_wape_improvement_pct
from praedixa.demand_forecast.backends.tft.backend import raise_if_tft_backend_required
from praedixa.demand_forecast.backends.tft.model_utils import DEFAULT_TFT_MODEL_PARAMS, fit_tft_model, predict_with_tft_model


def sample_trial_params(trial_index: int, random_seed: int = DEFAULT_RANDOM_SEED) -> dict[str, object]:
    rng = np.random.default_rng(random_seed + trial_index)
    return {
        "max_epochs": int(rng.integers(10, 41)),
        "learning_rate": float(math.exp(float(rng.uniform(math.log(1e-3), math.log(5e-2))))),
        "hidden_size": int(rng.choice(np.asarray([8, 16, 24, 32, 48, 64], dtype=np.int32))),
        "hidden_continuous_size": int(rng.choice(np.asarray([4, 8, 12, 16, 24, 32], dtype=np.int32))),
        "attention_head_size": int(rng.choice(np.asarray([1, 2, 4], dtype=np.int32))),
        "dropout": float(rng.uniform(0.05, 0.30)),
        "weight_decay": float(math.exp(float(rng.uniform(math.log(1e-6), math.log(1e-2))))),
        "random_state": int(random_seed + trial_index),
    }


def sample_optuna_params(
    trial: optuna.trial.Trial,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    return {
        "max_epochs": trial.suggest_int("max_epochs", 10, 40),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [8, 16, 24, 32, 48, 64]),
        "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [4, 8, 12, 16, 24, 32]),
        "attention_head_size": trial.suggest_categorical("attention_head_size", [1, 2, 4]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "random_state": int(random_seed + trial.number + 1),
    }


def _resolve_fold_workers(
    *,
    folds: list[dict[str, object]],
    resolved_params: dict[str, object],
    num_threads_override: int | None,
) -> int:
    num_threads = int(num_threads_override or cast(Any, resolved_params.get("n_jobs", os.cpu_count() or 1)))
    return max(1, min(len(folds), num_threads))


def _fit_single_screening_fold(
    *,
    frame: pd.DataFrame,
    fold: dict[str, object],
    feature_cols: list[str],
    target_col: str,
    resolved_params: dict[str, object],
) -> float:
    cached_fold = cached_screening_fold_artifacts(
        frame=frame,
        fold=fold,
        feature_cols=feature_cols,
        target_col=target_col,
        model_params=resolved_params,
    )
    model = fit_tft_model(
        cached_fold.train_frame,
        feature_cols,
        target_col=target_col,
        model_params=resolved_params,
        default_params=DEFAULT_TFT_MODEL_PARAMS,
        default_max_iter=300,
        valid_frame=cached_fold.valid_frame,
        dataset_artifacts=cached_fold.dataset_artifacts,
    )
    predictions = predict_with_tft_model(model, cached_fold.valid_frame, feature_cols)
    return compute_wape(cached_fold.valid_frame[target_col], predictions)


def _serial_screening_fold_scores(
    *,
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_col: str,
    resolved_params: dict[str, object],
) -> list[float]:
    return [
        _fit_single_screening_fold(
            frame=frame,
            fold=fold,
            feature_cols=feature_cols,
            target_col=target_col,
            resolved_params=resolved_params,
        )
        for fold in folds
    ]


def _parallel_screening_fold_scores(
    *,
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_col: str,
    resolved_params: dict[str, object],
    fold_workers: int,
) -> list[float]:
    with ThreadPoolExecutor(max_workers=fold_workers) as executor:
        future_to_index = {
            executor.submit(
                _fit_single_screening_fold,
                frame=frame,
                fold=fold,
                feature_cols=feature_cols,
                target_col=target_col,
                resolved_params=resolved_params,
            ): index
            for index, fold in enumerate(folds)
        }
        scores_by_index: dict[int, float] = {}
        for future in as_completed(future_to_index):
            scores_by_index[future_to_index[future]] = float(future.result())
    return [scores_by_index[index] for index in range(len(folds))]


def fit_and_score_tft_model(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    *,
    logger: logging.Logger,
    target_col: str = DEFAULT_TARGET_COL,
    model_params: dict[str, object] | None = None,
    num_threads_override: int | None = None,
) -> list[float]:
    resolved_params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
    fold_workers = _resolve_fold_workers(
        folds=folds,
        resolved_params=resolved_params,
        num_threads_override=num_threads_override,
    )
    if fold_workers == 1:
        return _serial_screening_fold_scores(
            frame=frame,
            folds=folds,
            feature_cols=feature_cols,
            target_col=target_col,
            resolved_params=resolved_params,
        )
    return _parallel_screening_fold_scores(
        frame=frame,
        folds=folds,
        feature_cols=feature_cols,
        target_col=target_col,
        resolved_params=resolved_params,
        fold_workers=fold_workers,
    )


def _objective_improvement(
    *,
    trial: optuna.trial.Trial,
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    logger: logging.Logger,
    target_col: str,
    random_seed: int,
    baseline_wape: float,
    base_model_params: dict[str, object] | None,
    total_threads: int,
) -> tuple[float, float, list[float]]:
    trial_params = {**(base_model_params or {}), **sample_optuna_params(trial=trial, random_seed=random_seed)}
    fold_scores = fit_and_score_tft_model(
        frame=frame,
        folds=folds,
        feature_cols=feature_cols,
        logger=logger,
        target_col=target_col,
        model_params=trial_params,
        num_threads_override=total_threads,
    )
    mean_wape = float(np.mean(fold_scores))
    return mean_wape, compute_wape_improvement_pct(baseline_wape, mean_wape), fold_scores


def _trial_rows(study: optuna.study.Study) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trial in study.trials:
        if trial.value is None:
            raise ValueError(f"Optuna trial {trial.number} completed without an objective value.")
        rows.append(
            {
                "trial": trial.number,
                "mean_wape": float(trial.user_attrs["mean_wape"]),
                "baseline_wape_improvement_pct": float(trial.value),
                "fold_wape_scores": trial.user_attrs["fold_wape_scores"],
                **trial.params,
            }
        )
    return rows


def _configure_optuna_study(random_seed: int) -> optuna.study.Study:
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    return optuna.create_study(direction="maximize", sampler=sampler)


def _should_log_tuning_progress(completed: int, n_trials: int) -> bool:
    if completed == 1 or completed == n_trials:
        return True
    return completed % DEFAULT_TUNING_PROGRESS_LOG_EVERY == 0


def _log_tuning_progress(
    *,
    completed: int,
    n_trials: int,
    study: optuna.study.Study,
    improvement_pct: float,
    logger: logging.Logger,
) -> None:
    if not _should_log_tuning_progress(completed, n_trials):
        return
    logger.info(
        "Tuning progress: trial=%s/%s current_best_improvement_pct=%.6f",
        completed,
        n_trials,
        max((t.value for t in study.trials if t.value is not None), default=improvement_pct),
    )


def _validate_screening_tuning_args(
    *,
    n_trials: int,
    baseline_wape: float | None,
) -> None:
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    if baseline_wape is None:
        raise ValueError("baseline_wape must be provided for non-lag tuning.")


def _best_screening_params(
    *,
    study: optuna.study.Study,
    base_model_params: dict[str, object] | None,
    random_seed: int,
    total_threads: int,
) -> dict[str, object]:
    best_params = {**(base_model_params or {}), **study.best_params}
    best_params["random_state"] = random_seed + study.best_trial.number + 1
    best_params["n_jobs"] = total_threads
    return best_params


def optimize_non_lag_model_params(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    *,
    logger: logging.Logger,
    target_col: str = DEFAULT_TARGET_COL,
    n_trials: int = DEFAULT_TUNING_TRIALS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    baseline_wape: float | None = None,
    base_model_params: dict[str, object] | None = None,
    num_workers: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    del num_workers
    raise_if_tft_backend_required("features_selection_lag.optimize_non_lag_model_params")
    _validate_screening_tuning_args(n_trials=n_trials, baseline_wape=baseline_wape)
    assert baseline_wape is not None
    total_threads = int(cast(Any, (base_model_params or {}).get("n_jobs", os.cpu_count() or 1)))
    study = _configure_optuna_study(random_seed)
    log_tuning_start(logger=logger, n_trials=n_trials, feature_count=len(feature_cols), baseline_wape=baseline_wape)
    run_optuna_search(
        study=study,
        frame=frame,
        folds=folds,
        feature_cols=feature_cols,
        logger=logger,
        target_col=target_col,
        random_seed=random_seed,
        baseline_wape=baseline_wape,
        base_model_params=base_model_params,
        total_threads=total_threads,
        n_trials=n_trials,
    )
    tuning_report = tuning_report_frame(study)
    best_params = _best_screening_params(
        study=study,
        base_model_params=base_model_params,
        random_seed=random_seed,
        total_threads=total_threads,
    )
    logger.info(
        "Non-lag parameter tuning complete: best_trial=%s best_wape=%.6f best_improvement_pct=%.6f",
        study.best_trial.number,
        study.best_trial.user_attrs["mean_wape"],
        study.best_trial.value,
    )
    return best_params, tuning_report


def run_optuna_search(
    *,
    study: optuna.study.Study,
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    logger: logging.Logger,
    target_col: str,
    random_seed: int,
    baseline_wape: float,
    base_model_params: dict[str, object] | None,
    total_threads: int,
    n_trials: int,
) -> None:
    study.optimize(
        build_objective(
            study=study,
            frame=frame,
            folds=folds,
            feature_cols=feature_cols,
            logger=logger,
            target_col=target_col,
            random_seed=random_seed,
            baseline_wape=baseline_wape,
            base_model_params=base_model_params,
            total_threads=total_threads,
            n_trials=n_trials,
        ),
        n_trials=n_trials,
        n_jobs=1,
    )


def log_tuning_start(
    *,
    logger: logging.Logger,
    n_trials: int,
    feature_count: int,
    baseline_wape: float,
) -> None:
    logger.info(
        "Starting non-lag parameter tuning with Optuna: trials=%s feature_count=%s baseline_wape=%.6f",
        n_trials,
        feature_count,
        baseline_wape,
    )


def build_objective(
    *,
    study: optuna.study.Study,
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    logger: logging.Logger,
    target_col: str,
    random_seed: int,
    baseline_wape: float,
    base_model_params: dict[str, object] | None,
    total_threads: int,
    n_trials: int,
) -> Callable[[optuna.trial.Trial], float]:
    trial_counter = {"completed": 0}

    def objective(trial: optuna.trial.Trial) -> float:
        improvement_pct = _objective_for_search(
            trial=trial,
            frame=frame,
            folds=folds,
            feature_cols=feature_cols,
            logger=logger,
            target_col=target_col,
            random_seed=random_seed,
            baseline_wape=baseline_wape,
            base_model_params=base_model_params,
            total_threads=total_threads,
        )
        trial_counter["completed"] += 1
        _log_tuning_progress(
            completed=trial_counter["completed"],
            n_trials=n_trials,
            study=study,
            improvement_pct=improvement_pct,
            logger=logger,
        )
        return improvement_pct

    return objective


def _objective_for_search(
    *,
    trial: optuna.trial.Trial,
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    logger: logging.Logger,
    target_col: str,
    random_seed: int,
    baseline_wape: float,
    base_model_params: dict[str, object] | None,
    total_threads: int,
) -> float:
    mean_wape, improvement_pct, fold_scores = _objective_improvement(
        trial=trial,
        frame=frame,
        folds=folds,
        feature_cols=feature_cols,
        logger=logger,
        target_col=target_col,
        random_seed=random_seed,
        baseline_wape=baseline_wape,
        base_model_params=base_model_params,
        total_threads=total_threads,
    )
    trial.set_user_attr("mean_wape", mean_wape)
    trial.set_user_attr("fold_wape_scores", [float(score) for score in fold_scores])
    return improvement_pct


def tuning_report_frame(study: optuna.study.Study) -> pd.DataFrame:
    return pd.DataFrame(_trial_rows(study)).sort_values(
        by="baseline_wape_improvement_pct",
        ascending=False,
    ).reset_index(drop=True)

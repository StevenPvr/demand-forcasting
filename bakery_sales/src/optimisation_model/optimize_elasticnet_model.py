from __future__ import annotations

"""Optimisation Optuna d'un modele ElasticNet autoregressif."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from src.elasticnet_time_series import (
    batch_forecast,
    elasticnet_params_payload,
    feature_columns as shared_feature_columns,
    fit_elasticnet_model,
    model_complexity_penalty,
    predict_frame,
    save_elasticnet_model,
)
from src.optimisation_model.constants import (
    CSV_ENCODING,
    DEFAULT_ELASTICNET_MAX_ITER,
    DEFAULT_FOLD_JOBS,
    DEFAULT_FOLDS,
    DEFAULT_TRIALS,
    MAX_ALPHA,
    MAX_L1_RATIO,
    MIN_ALPHA,
    MIN_L1_RATIO,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from src.time_series_metrics import mae_score as base_mae_score, mase_score
from src.time_series_validation import (
    best_trial_by_tiebreakers,
    build_recent_history_walk_forward_folds,
    build_walk_forward_folds,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
VARIANCE_GAP_WEIGHT: float = 0.15


class _OptunaTrialLike(Protocol):
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float: ...

    def suggest_categorical(self, name: str, choices: Any) -> Any: ...


def _feature_columns(dataset_df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes de features du dataset journalier."""

    return shared_feature_columns(dataset_df, TARGET_COLUMN)


def _load_split(csv_path: Path) -> pd.DataFrame:
    """Charge et trie un split temporel journalier."""

    dataset_df = pd.read_csv(csv_path)
    return dataset_df.sort_values(["origin_date"]).reset_index(drop=True)


def mae_score_wrapper(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Expose la MAE pour les tests."""

    return base_mae_score(y_true, y_pred)


mae_score = mae_score_wrapper


def suggest_optimization_params(trial: _OptunaTrialLike) -> dict[str, Any]:
    """Definit un espace de recherche ElasticNet pour l'optimisation finale."""

    return elasticnet_params_payload(
        alpha=float(trial.suggest_float("alpha", MIN_ALPHA, MAX_ALPHA, log=True)),
        l1_ratio=float(trial.suggest_float("l1_ratio", MIN_L1_RATIO, MAX_L1_RATIO)),
        fit_intercept=bool(trial.suggest_categorical("fit_intercept", [True, False])),
        max_iter=DEFAULT_ELASTICNET_MAX_ITER,
    )


def optimization_params_from_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Reconstruit le payload ElasticNet final a partir d'un trial."""

    return elasticnet_params_payload(
        alpha=float(trial.params["alpha"]),
        l1_ratio=float(trial.params["l1_ratio"]),
        fit_intercept=bool(trial.params["fit_intercept"]),
        max_iter=DEFAULT_ELASTICNET_MAX_ITER,
    )


def enqueue_optimization_baseline_trials(study: optuna.Study) -> None:
    """Ajoute quelques candidats simples et robustes avant les tirages TPE."""

    study.enqueue_trial({"alpha": 1.0, "l1_ratio": 0.5, "fit_intercept": True})
    study.enqueue_trial({"alpha": 0.1, "l1_ratio": 0.5, "fit_intercept": True})
    study.enqueue_trial({"alpha": 1.0, "l1_ratio": 0.9, "fit_intercept": False})


def _evaluate_fold(
    params: dict[str, Any],
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[float, float, float, float]:
    """Entraine sur un fold puis evalue MAE, MASE et penalite de complexite."""

    fitted_model = fit_elasticnet_model(
        history_df=train_fold,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        params=params,
        random_seed=RANDOM_SEED,
    )
    actual_values = cast(pd.Series, val_fold.loc[:, TARGET_COLUMN])
    predicted_values = predict_frame(fitted_model, val_fold, feature_columns)
    fold_mae = base_mae_score(actual_values, predicted_values)
    fold_mase = mase_score(
        actual_values,
        predicted_values,
        cast(pd.Series, train_fold.loc[:, TARGET_COLUMN]),
        seasonal_period=7,
    )
    actual_std = float(np.std(np.asarray(actual_values, dtype=float), ddof=0))
    predicted_std = float(np.std(np.asarray(predicted_values, dtype=float), ddof=0))
    variance_gap = abs(actual_std - predicted_std)
    return fold_mae, fold_mase, model_complexity_penalty(fitted_model), variance_gap


def _evaluate_folds_in_parallel(
    params: dict[str, Any],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_columns: list[str],
    n_jobs_folds: int,
) -> list[tuple[float, float, float, float]]:
    """Evalue les folds en multiprocessus pour utiliser les coeurs CPU."""

    tasks: list[Any] = [
        delayed(_evaluate_fold)(params, train_fold, val_fold, feature_columns)
        for train_fold, val_fold in folds
    ]
    results = Parallel(n_jobs=n_jobs_folds, prefer="processes")(tasks)
    return cast(list[tuple[float, float, float, float]], results)


def _resolved_fold_jobs(
    n_jobs_folds: int,
    fold_count: int,
    cpu_count: int | None = None,
) -> int:
    """Retourne un nombre de workers borne par les folds disponibles."""

    if fold_count <= 0:
        return 1
    detected_cpu_count = cpu_count if cpu_count is not None else os.cpu_count()
    available_cpus = max(1, detected_cpu_count or 1)
    if n_jobs_folds == -1:
        return min(available_cpus, fold_count)
    return min(max(1, int(n_jobs_folds)), fold_count)


def _optimization_folds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Construit des folds sur la queue recente de train+validation."""

    return build_recent_history_walk_forward_folds(
        train_df=train_df,
        val_df=val_df,
        n_folds=len(val_df),
    )


def _objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_jobs_folds: int,
) -> float:
    """Fonction objectif Optuna basee sur la moyenne des folds walk-forward."""

    params = suggest_optimization_params(trial)
    selected_feature_columns = _feature_columns(train_df)
    folds = _optimization_folds(train_df=train_df, val_df=val_df)
    resolved_jobs = _resolved_fold_jobs(n_jobs_folds=n_jobs_folds, fold_count=len(folds))
    LOGGER.info(
        "Starting trial %d with ElasticNet params=%s, %d linewise folds and fold_workers=%d",
        trial.number,
        params,
        len(folds),
        resolved_jobs,
    )
    fold_results = _evaluate_folds_in_parallel(
        params=params,
        folds=folds,
        feature_columns=selected_feature_columns,
        n_jobs_folds=resolved_jobs,
    )
    maes = [result[0] for result in fold_results]
    mases = [result[1] for result in fold_results]
    penalties = [result[2] for result in fold_results]
    variance_gaps = [result[3] for result in fold_results]
    mean_score = float(np.mean(maes) + (VARIANCE_GAP_WEIGHT * np.mean(variance_gaps)))
    trial.set_user_attr("fold_scores", maes)
    trial.set_user_attr("mean_mase", float(np.mean(mases)))
    trial.set_user_attr("complexity_penalty", float(np.mean(penalties)))
    trial.set_user_attr("variance_gap_penalty", float(np.mean(variance_gaps)))
    LOGGER.info(
        "Finished trial %d with mean MAE %.6f, mean MASE %.6f, mean active-coef penalty %.6f and variance gap %.6f",
        trial.number,
        float(np.mean(maes)),
        float(np.mean(mases)),
        float(np.mean(penalties)),
        float(np.mean(variance_gaps)),
    )
    return mean_score


def _predict_fold(
    params: dict[str, Any],
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    feature_columns: list[str],
    fold_index: int,
) -> pd.DataFrame:
    """Genere les predictions batch d'un fold pour les sauvegarder."""

    predictions_df = batch_forecast(
        history_df=train_fold,
        future_df=val_fold,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        params=params,
        random_seed=RANDOM_SEED,
    )
    predictions_df.insert(0, "fold", fold_index)
    return predictions_df


def run_optimization(
    train_csv: Path,
    val_csv: Path,
    best_params_json: Path,
    trials_csv: Path,
    best_model_pkl: Path,
    best_predictions_csv: Path,
    n_trials: int = DEFAULT_TRIALS,
    n_folds: int = DEFAULT_FOLDS,
    n_jobs_folds: int = DEFAULT_FOLD_JOBS,
) -> dict[str, int | float | str]:
    """Lance Optuna sur train+validation sans jamais toucher au split test."""

    del n_folds
    start_time = time.perf_counter()
    train_df = _load_split(train_csv)
    val_df = _load_split(val_csv)
    optimization_folds = _optimization_folds(train_df=train_df, val_df=val_df)
    selected_feature_columns = _feature_columns(train_df)
    feature_count = len(selected_feature_columns)
    LOGGER.info(
        "Loaded optimization inputs: train=%d rows, val=%d rows, features=%d",
        len(train_df),
        len(val_df),
        feature_count,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    LOGGER.info(
        "Starting ElasticNet optimization with %d trials, %d linewise folds and alpha[%s,%s], l1_ratio[%s,%s]",
        n_trials,
        len(optimization_folds),
        MIN_ALPHA,
        MAX_ALPHA,
        MIN_L1_RATIO,
        MAX_L1_RATIO,
    )
    enqueue_optimization_baseline_trials(study)
    study.optimize(
        lambda trial: _objective(
            trial=trial,
            train_df=train_df,
            val_df=val_df,
            n_jobs_folds=n_jobs_folds,
        ),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )
    best_trial = best_trial_by_tiebreakers(study)
    best_params = optimization_params_from_trial(best_trial)
    trials_df = study.trials_dataframe().loc[lambda df: df["state"] != "WAITING"].reset_index(drop=True)
    trials_csv.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(trials_csv, index=False, encoding=CSV_ENCODING)
    history_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = fit_elasticnet_model(
        history_df=history_df,
        feature_columns=selected_feature_columns,
        target_column=TARGET_COLUMN,
        params=best_params,
        random_seed=RANDOM_SEED,
    )
    save_elasticnet_model(final_model, best_model_pkl)
    reporting_folds = build_walk_forward_folds(
        train_df=train_df,
        val_df=val_df,
        n_folds=len(val_df),
    )
    best_predictions_df = pd.concat(
        [
            _predict_fold(
                params=best_params,
                train_fold=fold_train,
                val_fold=fold_val,
                feature_columns=selected_feature_columns,
                fold_index=fold_index,
            )
            for fold_index, (fold_train, fold_val) in enumerate(reporting_folds, start=1)
        ],
        ignore_index=True,
    )
    best_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    best_predictions_df.to_csv(best_predictions_csv, index=False, encoding=CSV_ENCODING)
    best_fold_scores: list[float] = cast(list[float], best_trial.user_attrs.get("fold_scores", []))
    best_payload: dict[str, Any] = {
        "model_family": "ElasticNet",
        "uses_exogenous_features": True,
        "best_objective_score": float(cast(float, best_trial.value)),
        "best_score_mae": float(cast(float, np.mean(best_fold_scores))) if best_fold_scores else float("inf"),
        "best_score_mase": float(cast(float, best_trial.user_attrs.get("mean_mase", float("inf")))),
        "complexity_penalty": float(
            cast(float, best_trial.user_attrs.get("complexity_penalty", float("inf")))
        ),
        "variance_gap_penalty": float(
            cast(float, best_trial.user_attrs.get("variance_gap_penalty", float("inf")))
        ),
        "trial_count": int(n_trials),
        "fold_count": int(len(optimization_folds)),
        "feature_count": int(feature_count),
        "feature_columns": selected_feature_columns,
        "best_params": best_params,
        "best_trial_number": int(best_trial.number),
        "best_fold_scores": best_fold_scores,
        "best_model_path": str(best_model_pkl),
        "best_predictions_path": str(best_predictions_csv),
        "search_space": {
            "alpha": [MIN_ALPHA, MAX_ALPHA],
            "l1_ratio": [MIN_L1_RATIO, MAX_L1_RATIO],
            "fit_intercept": [True, False],
        },
    }
    best_params_json.parent.mkdir(parents=True, exist_ok=True)
    best_params_json.write_text(json.dumps(best_payload, indent=2), encoding=CSV_ENCODING)
    LOGGER.info(
        "Optimization completed in %.3f seconds with best penalized score %.6f and mean fold MAE %.6f",
        time.perf_counter() - start_time,
        float(cast(float, best_trial.value)),
        float(cast(float, np.mean(best_fold_scores))) if best_fold_scores else float("inf"),
    )
    return {
        "model_family": "ElasticNet",
        "best_score": float(cast(float, np.mean(best_fold_scores))) if best_fold_scores else float("inf"),
        "trial_count": int(n_trials),
        "fold_count": int(len(optimization_folds)),
        "feature_count": int(feature_count),
    }

from __future__ import annotations

"""Optimisation Optuna d'un modele SARIMAX avec regressors exogenes."""

import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.optimisation_model.constants import CSV_ENCODING, DEFAULT_FOLDS, DEFAULT_TRIALS, TARGET_COLUMN
from src.sarimax_time_series import (
    batch_forecast,
    feature_columns as shared_feature_columns,
    fit_sarimax_model,
    model_complexity_penalty,
    sarimax_params_payload,
    save_sarimax_model,
)
from src.time_series_metrics import mae_score, mase_score
from src.time_series_validation import (
    best_trial_by_tiebreakers,
    build_recent_history_walk_forward_folds,
    build_walk_forward_folds,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
VARIANCE_GAP_WEIGHT: float = 0.15


class _NonConvergedFoldError(RuntimeError):
    """Signale qu'un fold SARIMAX n'a pas converge et doit etre prune."""


def _load_split(csv_path: Path) -> pd.DataFrame:
    """Charge et trie un split temporel journalier."""

    dataset_df = pd.read_csv(csv_path)
    return dataset_df.sort_values(["origin_date"]).reset_index(drop=True)


def _feature_columns(dataset_df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes exogenes du dataset journalier."""

    return shared_feature_columns(dataset_df, TARGET_COLUMN)


def suggest_optimization_params(trial: optuna.Trial) -> dict[str, Any]:
    """Definit un espace de recherche SARIMAX avec saisonnalite optionnelle."""

    seasonal_period = int(trial.suggest_categorical("seasonal_period", [0, 7, 14]))
    order = (
        int(trial.suggest_int("p", 0, 2)),
        int(trial.suggest_int("d", 0, 1)),
        int(trial.suggest_int("q", 0, 2)),
    )
    seasonal_order = (
        int(trial.suggest_int("P", 0, 1)) if seasonal_period > 0 else 0,
        int(trial.suggest_int("D", 0, 1)) if seasonal_period > 0 else 0,
        int(trial.suggest_int("Q", 0, 1)) if seasonal_period > 0 else 0,
        seasonal_period,
    )
    trend = cast(str, trial.suggest_categorical("trend", ["n"]))
    return sarimax_params_payload(order=order, seasonal_order=seasonal_order, trend=trend)


def optimization_params_from_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Reconstruit le payload SARIMAX final a partir d'un trial."""

    seasonal_period = int(trial.params["seasonal_period"])
    return sarimax_params_payload(
        order=(int(trial.params["p"]), int(trial.params["d"]), int(trial.params["q"])),
        seasonal_order=(
            int(trial.params["P"]) if seasonal_period > 0 else 0,
            int(trial.params["D"]) if seasonal_period > 0 else 0,
            int(trial.params["Q"]) if seasonal_period > 0 else 0,
            seasonal_period,
        ),
        trend=str(trial.params["trend"]),
    )


def enqueue_optimization_baseline_trials(study: optuna.Study) -> None:
    """Ajoute quelques candidats SARIMAX simples et robustes avant les tirages TPE."""

    study.enqueue_trial(
        {
            "seasonal_period": 0,
            "p": 0,
            "d": 0,
            "q": 0,
            "trend": "n",
        }
    )
    study.enqueue_trial(
        {
            "seasonal_period": 7,
            "p": 1,
            "d": 0,
            "q": 0,
            "P": 1,
            "D": 0,
            "Q": 0,
            "trend": "n",
        }
    )
    study.enqueue_trial(
        {
            "seasonal_period": 0,
            "p": 1,
            "d": 1,
            "q": 1,
            "trend": "n",
        }
    )
    study.enqueue_trial(
        {
            "seasonal_period": 7,
            "p": 1,
            "d": 0,
            "q": 0,
            "P": 0,
            "D": 1,
            "Q": 1,
            "trend": "n",
        }
    )
    study.enqueue_trial(
        {
            "seasonal_period": 14,
            "p": 1,
            "d": 0,
            "q": 0,
            "P": 0,
            "D": 1,
            "Q": 1,
            "trend": "n",
        }
    )


def _minimum_seed_trials() -> int:
    """Retourne le nombre minimal d'essais pour couvrir les candidats de depart."""

    return 2


def _optimization_folds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_folds: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Construit des folds walk-forward bloques pour l'opti SARIMAX."""

    resolved_fold_count = max(1, min(int(n_folds), len(val_df)))
    return build_recent_history_walk_forward_folds(
        train_df=train_df,
        val_df=val_df,
        n_folds=resolved_fold_count,
    )


def _evaluate_fold(
    params: dict[str, Any],
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[float, float, float, float]:
    """Entraine sur un fold puis evalue MAE, MASE, complexite et variance."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted_model = fit_sarimax_model(
            history_df=train_fold,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
            params=params,
        )
    mle_retvals = cast(dict[str, Any], getattr(fitted_model, "mle_retvals", {}))
    if not bool(mle_retvals.get("converged", True)):
        raise _NonConvergedFoldError(f"SARIMAX fold did not converge for params={params}")
    predictions_df = batch_forecast(
        history_df=train_fold,
        future_df=val_fold,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        params=params,
    )
    actual_values = cast(pd.Series, predictions_df["actual"])
    predicted_values = cast(pd.Series, predictions_df["prediction_raw"])
    actual_std = float(np.std(np.asarray(actual_values, dtype=float), ddof=0))
    predicted_std = float(np.std(np.asarray(predicted_values, dtype=float), ddof=0))
    return (
        mae_score(actual_values, predicted_values),
        mase_score(actual_values, predicted_values, cast(pd.Series, train_fold[TARGET_COLUMN]), seasonal_period=7),
        model_complexity_penalty(params, len(feature_columns)) + float(cast(float, fitted_model.aic)),
        abs(actual_std - predicted_std),
    )


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


def _objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_folds: int,
    n_jobs_folds: int,
) -> float:
    """Fonction objectif Optuna basee sur la moyenne des folds walk-forward."""

    params = suggest_optimization_params(trial)
    order_values = cast(list[int], params["order"])
    seasonal_values = cast(list[int], params["seasonal_order"])
    non_seasonal_order_size = int(order_values[0]) + int(order_values[2])
    seasonal_order_size = int(seasonal_values[0]) + int(seasonal_values[2])
    if (non_seasonal_order_size + seasonal_order_size) > 4:
        raise optuna.TrialPruned("SARIMAX order too complex for the current optimization budget")
    folds = _optimization_folds(train_df=train_df, val_df=val_df, n_folds=n_folds)
    selected_feature_columns = _feature_columns(train_df)
    resolved_jobs = _resolved_fold_jobs(n_jobs_folds=n_jobs_folds, fold_count=len(folds))
    LOGGER.info(
        "Starting trial %d with SARIMAX params=%s, %d blocked folds and fold_workers=%d",
        trial.number,
        params,
        len(folds),
        resolved_jobs,
    )
    try:
        fold_results = _evaluate_folds_in_parallel(
            params=params,
            folds=folds,
            feature_columns=selected_feature_columns,
            n_jobs_folds=resolved_jobs,
        )
    except _NonConvergedFoldError as exc:
        LOGGER.info("Pruned trial %d because a fold did not converge: %s", trial.number, exc)
        raise optuna.TrialPruned(str(exc)) from exc
    except Exception as exc:
        LOGGER.warning("Trial %d failed for params=%s with %s", trial.number, params, exc)
        return float("inf")
    maes = [result[0] for result in fold_results]
    mases = [result[1] for result in fold_results]
    penalties = [result[2] for result in fold_results]
    variance_gaps = [result[3] for result in fold_results]
    trial.set_user_attr("fold_scores", maes)
    trial.set_user_attr("mean_mase", float(np.mean(mases)))
    trial.set_user_attr("complexity_penalty", float(np.mean(penalties)))
    trial.set_user_attr("variance_gap_penalty", float(np.mean(variance_gaps)))
    LOGGER.info(
        "Finished trial %d with mean MAE %.6f, mean MASE %.6f, mean complexity penalty %.6f and variance gap %.6f",
        trial.number,
        float(np.mean(maes)),
        float(np.mean(mases)),
        float(np.mean(penalties)),
        float(np.mean(variance_gaps)),
    )
    return float(np.mean(maes) + (VARIANCE_GAP_WEIGHT * np.mean(variance_gaps)))


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
    n_jobs_folds: int = -1,
) -> dict[str, int | float | str]:
    """Lance Optuna SARIMAX sur train+validation sans jamais toucher au split test."""

    start_time = time.perf_counter()
    train_df = _load_split(train_csv)
    val_df = _load_split(val_csv)
    optimization_folds = _optimization_folds(train_df=train_df, val_df=val_df, n_folds=n_folds)
    selected_feature_columns = _feature_columns(train_df)
    LOGGER.info(
        "Loaded SARIMAX optimization inputs: train=%d rows, val=%d rows, features=%d",
        len(train_df),
        len(val_df),
        len(selected_feature_columns),
    )
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
    enqueue_optimization_baseline_trials(study)
    effective_n_trials = max(n_trials, _minimum_seed_trials())
    study.optimize(
        lambda trial: _objective(
            trial=trial,
            train_df=train_df,
            val_df=val_df,
            n_folds=n_folds,
            n_jobs_folds=n_jobs_folds,
        ),
        n_trials=effective_n_trials,
        n_jobs=1,
    )
    best_trial = best_trial_by_tiebreakers(study)
    best_params = optimization_params_from_trial(best_trial)
    trials_df = study.trials_dataframe().loc[lambda df: df["state"] != "WAITING"].reset_index(drop=True)
    trials_csv.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(trials_csv, index=False, encoding=CSV_ENCODING)
    history_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = fit_sarimax_model(
        history_df=history_df,
        feature_columns=selected_feature_columns,
        target_column=TARGET_COLUMN,
        params=best_params,
    )
    save_sarimax_model(final_model, best_model_pkl)
    reporting_folds = build_walk_forward_folds(train_df=train_df, val_df=val_df, n_folds=len(val_df))
    best_predictions_df = pd.concat(
        [
            _predict_fold(best_params, train_fold, val_fold, selected_feature_columns, fold_index)
            for fold_index, (train_fold, val_fold) in enumerate(reporting_folds, start=1)
        ],
        ignore_index=True,
    )
    best_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    best_predictions_df.to_csv(best_predictions_csv, index=False, encoding=CSV_ENCODING)
    best_payload = {
        "model_family": "SARIMAX",
        "uses_exogenous_features": True,
        "feature_columns": selected_feature_columns,
        "feature_count": int(len(selected_feature_columns)),
        "best_params": best_params,
        "best_score_mae": float(best_trial.user_attrs["fold_scores"] and np.mean(best_trial.user_attrs["fold_scores"])),
        "best_score_mase": float(best_trial.user_attrs["mean_mase"]),
        "complexity_penalty": float(best_trial.user_attrs["complexity_penalty"]),
        "variance_gap_penalty": float(best_trial.user_attrs["variance_gap_penalty"]),
        "trial_count": int(len(trials_df)),
        "fold_count": int(len(optimization_folds)),
    }
    best_params_json.parent.mkdir(parents=True, exist_ok=True)
    best_params_json.write_text(json.dumps(best_payload, indent=2), encoding=CSV_ENCODING)
    LOGGER.info(
        "SARIMAX optimization completed in %.3f seconds with best validation MAE %.6f",
        time.perf_counter() - start_time,
        float(best_payload["best_score_mae"]),
    )
    return {
        "model_family": "SARIMAX",
        "trial_count": int(len(trials_df)),
        "fold_count": int(len(optimization_folds)),
        "feature_count": int(len(selected_feature_columns)),
        "best_score_mae": float(best_payload["best_score_mae"]),
    }

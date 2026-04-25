from __future__ import annotations

"""Optimisation Optuna d'un modele ARIMA distinct pour chaque produit."""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from src.arima_time_series import (
    batch_forecast,
    fit_sarima_model,
    model_complexity_penalty,
    sarima_params_payload,
    save_sarima_model,
)
from src.optimisation_model.constants_arima import (
    CSV_ENCODING,
    DATE_COLUMN,
    PRODUCT_COLUMN,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from src.time_series_metrics import mae_score, mase_score
from src.time_series_validation import best_trial_by_tiebreakers

LOGGER: logging.Logger = logging.getLogger(__name__)
VARIANCE_GAP_WEIGHT: float = 0.15
FALLBACK_PARAMS: dict[str, Any] = sarima_params_payload(
    order=(0, 0, 0),
    seasonal_order=(0, 0, 0, 0),
    trend="n",
)


class _NonConvergedFoldError(RuntimeError):
    """Signale qu'un fold ARIMA n'a pas converge et doit etre prune."""


def _load_split(csv_path: Path) -> pd.DataFrame:
    """Charge un split et le trie par produit puis date."""

    dataset_df = pd.read_csv(csv_path)
    dataset_df[DATE_COLUMN] = pd.to_datetime(dataset_df[DATE_COLUMN], format="%Y-%m-%d")
    ordered_df = dataset_df.sort_values([PRODUCT_COLUMN, DATE_COLUMN]).reset_index(drop=True)
    ordered_df[DATE_COLUMN] = ordered_df[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    return ordered_df


def _product_frames(dataset_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Retourne un dictionnaire produit -> sous-serie chronologique."""

    return {
        str(product_name): product_df.reset_index(drop=True)
        for product_name, product_df in dataset_df.groupby(PRODUCT_COLUMN, sort=True)
    }


def suggest_optimization_params(trial: optuna.Trial) -> dict[str, Any]:
    """Definit un espace de recherche ARIMA raisonnable pour des series courtes."""

    seasonal_period = int(trial.suggest_categorical("seasonal_period", [0, 7]))
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
    return sarima_params_payload(order=order, seasonal_order=seasonal_order, trend="n")


def optimization_params_from_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Reconstruit le payload ARIMA final a partir d'un trial."""

    seasonal_period = int(trial.params["seasonal_period"])
    return sarima_params_payload(
        order=(int(trial.params["p"]), int(trial.params["d"]), int(trial.params["q"])),
        seasonal_order=(
            int(trial.params.get("P", 0)) if seasonal_period > 0 else 0,
            int(trial.params.get("D", 0)) if seasonal_period > 0 else 0,
            int(trial.params.get("Q", 0)) if seasonal_period > 0 else 0,
            seasonal_period,
        ),
        trend="n",
    )


def enqueue_optimization_baseline_trials(study: optuna.Study) -> None:
    """Ajoute quelques candidats simples avant les tirages TPE."""

    study.enqueue_trial({"seasonal_period": 0, "p": 0, "d": 0, "q": 0})
    study.enqueue_trial({"seasonal_period": 0, "p": 1, "d": 0, "q": 0})
    study.enqueue_trial({"seasonal_period": 7, "p": 1, "d": 0, "q": 0, "P": 0, "D": 1, "Q": 1})


def _build_walk_forward_folds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_folds: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Construit des folds walk-forward ligne par ligne sur la validation d'un produit."""

    del n_folds
    resolved_folds = max(1, len(val_df))
    ordered_train = train_df.sort_values(DATE_COLUMN).reset_index(drop=True)
    ordered_val = val_df.sort_values(DATE_COLUMN).reset_index(drop=True)
    fold_sizes = [len(ordered_val) // resolved_folds] * resolved_folds
    for index in range(len(ordered_val) % resolved_folds):
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


def _evaluate_fold(
    params: dict[str, Any],
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Entraine sur un fold produit puis evalue les metriques associees."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted_model = fit_sarima_model(history_df=train_fold, target_column=TARGET_COLUMN, params=params)
    mle_retvals = cast(dict[str, Any], getattr(fitted_model, "mle_retvals", {}))
    if not bool(mle_retvals.get("converged", True)):
        raise _NonConvergedFoldError(f"ARIMA fold did not converge for params={params}")
    predictions_df = batch_forecast(
        history_df=train_fold,
        future_df=val_fold,
        date_column=DATE_COLUMN,
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
        model_complexity_penalty(params) + float(cast(float, fitted_model.aic)),
        abs(actual_std - predicted_std),
    )


def _evaluate_folds_in_parallel(
    params: dict[str, Any],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    n_jobs_folds: int,
) -> list[tuple[float, float, float, float]]:
    """Evalue les folds en multiprocessus."""

    tasks = [delayed(_evaluate_fold)(params, train_fold, val_fold) for train_fold, val_fold in folds]
    results = Parallel(n_jobs=n_jobs_folds, prefer="processes")(tasks)
    return cast(list[tuple[float, float, float, float]], results)


def _resolved_fold_jobs(
    n_jobs_folds: int,
    fold_count: int,
    cpu_count: int | None = None,
) -> int:
    """Retourne un nombre de workers borne par le nombre de folds."""

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
    n_jobs_folds: int,
    n_folds: int = 1,
    product_name: str = "unknown",
) -> float:
    """Fonction objectif Optuna pour une serie produit."""

    params = suggest_optimization_params(trial)
    folds = _build_walk_forward_folds(train_df=train_df, val_df=val_df, n_folds=max(1, n_folds))
    resolved_jobs = _resolved_fold_jobs(n_jobs_folds=n_jobs_folds, fold_count=len(folds))
    LOGGER.info(
        "Starting ARIMA trial %d for product=%s with params=%s, folds=%d, fold_workers=%d",
        trial.number,
        product_name,
        params,
        len(folds),
        resolved_jobs,
    )
    try:
        fold_results = _evaluate_folds_in_parallel(params=params, folds=folds, n_jobs_folds=resolved_jobs)
    except _NonConvergedFoldError as exc:
        LOGGER.info("Pruned product=%s trial %d because a fold did not converge: %s", product_name, trial.number, exc)
        raise optuna.TrialPruned(str(exc)) from exc
    except Exception as exc:
        LOGGER.warning("Trial %d failed for product=%s with params=%s and error=%s", trial.number, product_name, params, exc)
        return float("inf")
    maes = [result[0] for result in fold_results]
    mases = [result[1] for result in fold_results]
    penalties = [result[2] for result in fold_results]
    variance_gaps = [result[3] for result in fold_results]
    trial.set_user_attr("fold_scores", maes)
    trial.set_user_attr("mean_mase", float(np.mean(mases)))
    trial.set_user_attr("complexity_penalty", float(np.mean(penalties)))
    trial.set_user_attr("variance_gap_penalty", float(np.mean(variance_gaps)))
    return float(np.mean(maes) + (VARIANCE_GAP_WEIGHT * np.mean(variance_gaps)))


def _study_trials_records(study: optuna.Study, product_name: str) -> list[dict[str, Any]]:
    """Serialise les trials d'un produit pour le CSV de suivi."""

    records: list[dict[str, Any]] = []
    for trial in study.trials:
        record = {
            "product": product_name,
            "trial_number": int(trial.number),
            "state": str(trial.state.name),
            "value": None if trial.value is None else float(trial.value),
        }
        record.update(trial.params)
        records.append(record)
    return records


def _fallback_predictions(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """Construit des predictions de secours avec un ARIMA nul."""

    return batch_forecast(
        history_df=train_df,
        future_df=val_df,
        date_column=DATE_COLUMN,
        target_column=TARGET_COLUMN,
        params=FALLBACK_PARAMS,
    )


def _fallback_product_artifacts(
    product_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, SARIMAXResultsWrapper]:
    """Construit les artefacts de secours pour un produit."""

    predictions_df = _fallback_predictions(train_df, val_df)
    final_model = fit_sarima_model(
        history_df=pd.concat([train_df, val_df], ignore_index=True),
        target_column=TARGET_COLUMN,
        params=FALLBACK_PARAMS,
    )
    payload = {
        "best_params": FALLBACK_PARAMS,
        "best_value": float(
            mae_score(
                cast(pd.Series, predictions_df["actual"]),
                cast(pd.Series, predictions_df["prediction_raw"]),
            )
        ),
        "best_trial_number": -1,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
    }
    records = [
        {
            "product": product_name,
            "trial_number": -1,
            "state": "FALLBACK",
            "value": payload["best_value"],
        }
    ]
    return payload, records, predictions_df, final_model


def _optimize_single_product(
    product_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int,
    n_folds: int,
    n_jobs_folds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, SARIMAXResultsWrapper]:
    """Optimise un ARIMA pour un produit et retourne ses artefacts principaux."""

    if cast(pd.Series, train_df[TARGET_COLUMN]).nunique(dropna=False) <= 1:
        return _fallback_product_artifacts(product_name=product_name, train_df=train_df, val_df=val_df)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    enqueue_optimization_baseline_trials(study)
    study.optimize(
        lambda trial: _objective(
            trial,
            train_df=train_df,
            val_df=val_df,
            n_jobs_folds=n_jobs_folds,
            n_folds=n_folds,
            product_name=product_name,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return _fallback_product_artifacts(product_name=product_name, train_df=train_df, val_df=val_df)
    best_trial = best_trial_by_tiebreakers(study)
    best_params = optimization_params_from_trial(best_trial)
    predictions_df = batch_forecast(
        history_df=train_df,
        future_df=val_df,
        date_column=DATE_COLUMN,
        target_column=TARGET_COLUMN,
        params=best_params,
    )
    final_model = fit_sarima_model(
        history_df=pd.concat([train_df, val_df], ignore_index=True),
        target_column=TARGET_COLUMN,
        params=best_params,
    )
    payload = {
        "best_params": best_params,
        "best_value": float(cast(float, best_trial.value)),
        "best_trial_number": int(best_trial.number),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
    }
    return payload, _study_trials_records(study, product_name), predictions_df, final_model


def _save_best_payload(best_payload: dict[str, Any], best_params_json: Path) -> None:
    """Persiste le JSON de resume de l'optimisation multi-produit."""

    best_params_json.parent.mkdir(parents=True, exist_ok=True)
    best_params_json.write_text(json.dumps(best_payload, indent=2), encoding=CSV_ENCODING)


def run_optimization(
    train_csv: Path,
    val_csv: Path,
    best_params_json: Path,
    trials_csv: Path,
    best_model_pkl: Path,
    best_predictions_csv: Path,
    n_trials: int,
    n_folds: int,
    n_jobs_folds: int,
) -> dict[str, int | float | str]:
    """Optimise un ARIMA distinct pour chaque produit present dans train et val."""

    train_df = _load_split(train_csv)
    val_df = _load_split(val_csv)
    train_products = _product_frames(train_df)
    val_products = _product_frames(val_df)
    shared_products = sorted(set(train_products).intersection(val_products))
    product_models: dict[str, Any] = {}
    trial_records: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    fitted_models: dict[str, SARIMAXResultsWrapper] = {}
    for product_name in shared_products:
        LOGGER.info("Optimizing ARIMA for product=%s", product_name)
        payload, product_trials, predictions_df, final_model = _optimize_single_product(
            product_name=product_name,
            train_df=train_products[product_name],
            val_df=val_products[product_name],
            n_trials=max(1, int(n_trials)),
            n_folds=n_folds,
            n_jobs_folds=n_jobs_folds,
        )
        product_models[product_name] = payload
        trial_records.extend(product_trials)
        prediction_parts.append(predictions_df)
        fitted_models[product_name] = final_model
    best_payload = {
        "model_family": "ARIMA_BY_PRODUCT",
        "uses_exogenous_features": False,
        "feature_columns": [],
        "feature_count": 0,
        "date_column": DATE_COLUMN,
        "product_column": PRODUCT_COLUMN,
        "target_column": TARGET_COLUMN,
        "product_count": len(shared_products),
        "product_models": product_models,
    }
    predictions_df = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    trials_df = pd.DataFrame(trial_records)
    _save_best_payload(best_payload, best_params_json)
    trials_csv.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(trials_csv, index=False, encoding=CSV_ENCODING)
    best_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(best_predictions_csv, index=False, encoding=CSV_ENCODING)
    best_model_pkl.parent.mkdir(parents=True, exist_ok=True)
    if len(shared_products) == 1:
        save_sarima_model(cast(SARIMAXResultsWrapper, fitted_models[shared_products[0]]), best_model_pkl)
    else:
        joblib.dump(fitted_models, best_model_pkl)
    mean_best_score = float(np.mean([cast(float, payload["best_value"]) for payload in product_models.values()])) if product_models else float("inf")
    return {
        "model_family": "ARIMA_BY_PRODUCT",
        "product_count": int(len(shared_products)),
        "feature_count": 0,
        "mean_best_mae": mean_best_score,
    }

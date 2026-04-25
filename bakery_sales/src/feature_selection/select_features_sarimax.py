from __future__ import annotations

"""Selection de features pour le pipeline SARIMAX."""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.feature_selection.constants import (
    CORRELATION_THRESHOLD,
    CSV_ENCODING,
    DEFAULT_FOLDS,
    DEFAULT_OPTUNA_TRIALS,
    DEFAULT_SELECTOR_JOBS,
    SARIMAX_MAX_SELECTED_EXOGENOUS_FEATURES,
    TARGET_COLUMN,
)
from src.sarimax_time_series import (
    batch_forecast,
    fit_sarimax_model,
    sarimax_params_payload,
)
from src.time_series_metrics import mae_score
from src.time_series_validation import build_recent_history_walk_forward_folds

LOGGER: logging.Logger = logging.getLogger(__name__)
IMPORTANCE_LOG_EVERY: int = 10
VARIANCE_GAP_WEIGHT: float = 0.15


class _NonConvergedSelectorError(RuntimeError):
    """Signale qu'un fit du selecteur SARIMAX n'a pas converge."""


def _load_split(csv_path: Path) -> pd.DataFrame:
    """Charge un split temporel journalier en preservant l'ordre des dates."""

    dataset_df: pd.DataFrame = pd.read_csv(csv_path)
    return dataset_df.sort_values("origin_date").reset_index(drop=True)


def _context_feature_columns(dataset_df: pd.DataFrame) -> list[str]:
    """Retourne les exogenes de contexte hors ventes et hors autoregression."""

    return [
        column
        for column in dataset_df.columns
        if column.startswith("exog_calendar_")
        or column.startswith("exog_flag_")
        or column.startswith("exog_holiday_")
        or column.startswith("exog_weather_")
    ]


def _sales_feature_columns(dataset_df: pd.DataFrame) -> list[str]:
    """Retourne les exogenes de ventes candidates a la selection."""

    return [column for column in dataset_df.columns if column.startswith("exog_sales_")]


def _drop_constant_candidate_features(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Retire les features candidates constantes ou quasi constantes."""

    kept_columns: list[str] = []
    dropped_columns: list[str] = []
    for column in candidate_columns:
        if train_df[column].nunique(dropna=False) <= 1:
            dropped_columns.append(column)
        else:
            kept_columns.append(column)
    return kept_columns, dropped_columns


def _drop_correlated_candidate_features(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
    target_column: str,
    correlation_threshold: float,
) -> tuple[list[str], list[str]]:
    """Retire les features fortement correlees en gardant la plus predictive."""

    if len(candidate_columns) <= 1:
        return candidate_columns, []
    corr_to_target = cast(
        pd.Series,
        train_df.loc[:, candidate_columns + [target_column]].corr().abs()[target_column],
    )
    ranked_columns = sorted(
        candidate_columns,
        key=lambda column: (-float(cast(float, corr_to_target.loc[column])), column),
    )
    correlation_matrix = train_df.loc[:, ranked_columns].corr().abs()
    kept_columns: list[str] = []
    dropped_columns: list[str] = []
    for column in ranked_columns:
        if column in dropped_columns:
            continue
        kept_columns.append(column)
        correlated_columns = correlation_matrix.index[
            correlation_matrix.loc[column] > correlation_threshold
        ].tolist()
        for correlated_column in correlated_columns:
            if (
                correlated_column == column
                or correlated_column in kept_columns
                or correlated_column in dropped_columns
            ):
                continue
            dropped_columns.append(correlated_column)
    return kept_columns, dropped_columns


def _fit_selector_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
) -> Any:
    """Ajuste le selecteur SARIMAX sur les colonnes fournies."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted_model = fit_sarimax_model(
            history_df=train_df,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
            params=params,
        )
    mle_retvals = cast(dict[str, Any], getattr(fitted_model, "mle_retvals", {}))
    if not bool(mle_retvals.get("converged", True)):
        raise _NonConvergedSelectorError(f"SARIMAX selector did not converge for params={params}")
    return fitted_model


def _predict_mae(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
) -> tuple[float, float]:
    """Calcule la MAE et l'ecart de variance d'un modele SARIMAX sur le split validation."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        predictions_df = batch_forecast(
            history_df=train_df,
            future_df=val_df,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
            params=params,
        )
    actual_values = cast(pd.Series, predictions_df["actual"])
    predicted_values = cast(pd.Series, predictions_df["prediction_raw"])
    actual_std = float(np.std(np.asarray(actual_values, dtype=float), ddof=0))
    predicted_std = float(np.std(np.asarray(predicted_values, dtype=float), ddof=0))
    return mae_score(actual_values, predicted_values), abs(actual_std - predicted_std)


def _evaluate_fold(
    params: dict[str, Any],
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[float, float]:
    """Entraine sur un fold puis retourne sa MAE et son ecart de variance."""

    _fit_selector_model(train_df=train_fold, feature_columns=feature_columns, params=params)
    return _predict_mae(train_df=train_fold, val_df=val_fold, feature_columns=feature_columns, params=params)


def _evaluate_folds_in_parallel(
    params: dict[str, Any],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_columns: list[str],
    n_jobs: int,
) -> list[tuple[float, float]]:
    """Evalue les folds de tuning via des processus pour utiliser tous les coeurs."""

    tasks = [
        delayed(_evaluate_fold)(params, train_fold, val_fold, feature_columns)
        for train_fold, val_fold in folds
    ]
    results = Parallel(n_jobs=n_jobs, prefer="processes")(tasks)
    return cast(list[tuple[float, float]], list(results))


def _should_log_importance_progress(
    index: int,
    total_features: int,
    every_n: int = IMPORTANCE_LOG_EVERY,
) -> bool:
    """Indique si la progression des importances doit etre logguee."""

    if total_features <= 0:
        return False
    if index == 0 or index == total_features - 1:
        return True
    return (index + 1) % every_n == 0


def _log_importance_progress(
    index: int,
    total_features: int,
    feature_name: str,
    every_n: int = IMPORTANCE_LOG_EVERY,
) -> None:
    """Loggue la progression du calcul des importances SARIMAX."""

    if not _should_log_importance_progress(index=index, total_features=total_features, every_n=every_n):
        return
    LOGGER.info(
        "SARIMAX importance progress %d/%d (%.1f%%): feature=%s",
        index + 1,
        total_features,
        ((index + 1) / total_features) * 100.0,
        feature_name,
    )


def _suggest_selector_params(trial: optuna.Trial) -> dict[str, Any]:
    """Definit l'espace de recherche SARIMAX du selecteur."""

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
    trend = cast(str, trial.suggest_categorical("trend", ["n"]))
    return sarimax_params_payload(order=order, seasonal_order=seasonal_order, trend=trend)


def tune_selector_params(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    selector_feature_columns: list[str],
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
    n_folds: int = DEFAULT_FOLDS,
    n_jobs: int = DEFAULT_SELECTOR_JOBS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Cherche un jeu d'hyperparametres SARIMAX robuste pour la selection."""

    feature_columns = selector_feature_columns if selector_feature_columns else []
    fallback_params = sarimax_params_payload(order=(0, 1, 0), seasonal_order=(0, 0, 0, 0), trend="n")

    def _objective(trial: optuna.Trial) -> float:
        params = _suggest_selector_params(trial)
        folds = build_recent_history_walk_forward_folds(train_df=train_df, val_df=val_df, n_folds=n_folds)
        LOGGER.info("Starting selector trial %d with SARIMAX params: %s", trial.number, params)
        try:
            if n_jobs == 1:
                fold_scores = [
                    _evaluate_fold(params, train_fold, val_fold, feature_columns)
                    for train_fold, val_fold in folds
                ]
            else:
                fold_scores = _evaluate_folds_in_parallel(
                    params=params,
                    folds=folds,
                    feature_columns=feature_columns,
                    n_jobs=n_jobs,
                )
        except _NonConvergedSelectorError as exc:
            LOGGER.info("Pruned SARIMAX selector trial %d because %s", trial.number, exc)
            raise optuna.TrialPruned(str(exc)) from exc
        mean_val_mae = float(np.mean([score[0] for score in fold_scores]))
        variance_gap = float(np.mean([score[1] for score in fold_scores]))
        mean_score = float(mean_val_mae + (VARIANCE_GAP_WEIGHT * variance_gap))
        trial.set_user_attr("fold_scores", [score[0] for score in fold_scores])
        trial.set_user_attr("mean_validation_mae", mean_val_mae)
        trial.set_user_attr("variance_gap_penalty", variance_gap)
        LOGGER.info(
            "Finished selector trial %d with penalized score %.6f, validation MAE %.6f and variance gap %.6f",
            trial.number,
            mean_score,
            mean_val_mae,
            variance_gap,
        )
        return mean_score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=7),
    )
    study.enqueue_trial({"seasonal_period": 0, "p": 0, "d": 1, "q": 0, "trend": "n"})
    study.enqueue_trial({"seasonal_period": 7, "p": 1, "d": 0, "q": 0, "P": 1, "D": 0, "Q": 0, "trend": "n"})
    study.optimize(_objective, n_trials=max(n_trials, 2), n_jobs=1, show_progress_bar=False)
    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    trials_df = study.trials_dataframe().copy()
    if not completed_trials:
        LOGGER.warning(
            "All SARIMAX selector trials were pruned; falling back to robust default params=%s",
            fallback_params,
        )
        return {
            "model_family": "SARIMAX",
            "best_params": fallback_params,
            "best_score": float("inf"),
            "best_trial_number": -1,
            "best_fold_scores": [],
            "variance_gap_penalty": float("inf"),
            "trial_count": int(len(trials_df)),
            "fold_count": int(n_folds),
        }, trials_df

    best_trial = cast(optuna.trial.FrozenTrial, min(completed_trials, key=lambda trial: float(cast(float, trial.value))))
    best_params = sarimax_params_payload(
        order=(int(best_trial.params["p"]), int(best_trial.params["d"]), int(best_trial.params["q"])),
        seasonal_order=(
            int(best_trial.params["P"]) if int(best_trial.params["seasonal_period"]) > 0 else 0,
            int(best_trial.params["D"]) if int(best_trial.params["seasonal_period"]) > 0 else 0,
            int(best_trial.params["Q"]) if int(best_trial.params["seasonal_period"]) > 0 else 0,
            int(best_trial.params["seasonal_period"]),
        ),
        trend=str(best_trial.params["trend"]),
    )
    best_score = cast(float, best_trial.value)
    best_fold_scores = cast(list[float], best_trial.user_attrs.get("fold_scores", []))
    return {
        "model_family": "SARIMAX",
        "best_params": best_params,
        "best_score": float(cast(float, best_trial.user_attrs.get("mean_validation_mae", best_score))),
        "best_trial_number": int(best_trial.number),
        "best_fold_scores": best_fold_scores,
        "variance_gap_penalty": float(cast(float, best_trial.user_attrs.get("variance_gap_penalty", float("inf")))),
        "trial_count": int(len(trials_df)),
        "fold_count": int(n_folds),
    }, trials_df


def _importance_row(
    index: int,
    total_features: int,
    feature_name: str,
    importance: float,
    stage_label: str,
) -> dict[str, float | str | bool]:
    """Construit une ligne d'importance et loggue sa progression."""

    _log_importance_progress(index=index, total_features=total_features, feature_name=feature_name)
    return {
        "feature": feature_name,
        "importance": float(importance),
        "normalized_importance": float(max(importance, 0.0)),
        "selected": bool(importance > 0.0),
        "selection_stage": stage_label,
    }


def _candidate_importance(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    base_feature_columns: list[str],
    candidate_column: str,
    selector_params: dict[str, Any],
    baseline_val_mae: float,
) -> dict[str, float | str | bool]:
    """Calcule le gain de MAE apporte par une feature candidate."""

    feature_columns = base_feature_columns + [candidate_column]
    try:
        _fit_selector_model(train_df=train_df, feature_columns=feature_columns, params=selector_params)
        candidate_val_mae, _ = _predict_mae(
            train_df=train_df,
            val_df=val_df,
            feature_columns=feature_columns,
            params=selector_params,
        )
        improvement = baseline_val_mae - candidate_val_mae
    except _NonConvergedSelectorError:
        improvement = float("-inf")
    return _importance_row(
        index=0,
        total_features=1,
        feature_name=candidate_column,
        importance=improvement,
        stage_label="",
    )


def compute_feature_importances(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    selector_params: dict[str, Any],
    base_feature_columns: list[str],
    candidate_feature_columns: list[str],
    stage_label: str,
    n_jobs: int,
) -> tuple[pd.DataFrame, float]:
    """Construit le tableau d'importances SARIMAX pour les features candidates."""

    LOGGER.info(
        "Computing SARIMAX importances for stage=%s with %d base columns and %d candidate columns",
        stage_label,
        len(base_feature_columns),
        len(candidate_feature_columns),
    )
    baseline_val_mae, _ = _predict_mae(
        train_df=train_df,
        val_df=val_df,
        feature_columns=base_feature_columns,
        params=selector_params,
    )
    if not candidate_feature_columns:
        return pd.DataFrame(
            columns=["feature", "importance", "normalized_importance", "selected", "selection_stage"]
        ), baseline_val_mae
    if n_jobs == 1:
        rows = [
            {
                **_candidate_importance(
                    train_df=train_df,
                    val_df=val_df,
                    base_feature_columns=base_feature_columns,
                    candidate_column=feature_name,
                    selector_params=selector_params,
                    baseline_val_mae=baseline_val_mae,
                ),
                "selection_stage": stage_label,
            }
            for index, feature_name in enumerate(candidate_feature_columns)
            for _ in [_log_importance_progress(index, len(candidate_feature_columns), feature_name)]
        ]
    else:
        tasks = [
            delayed(_candidate_importance)(
                train_df=train_df,
                val_df=val_df,
                base_feature_columns=base_feature_columns,
                candidate_column=feature_name,
                selector_params=selector_params,
                baseline_val_mae=baseline_val_mae,
            )
            for feature_name in candidate_feature_columns
        ]
        raw_rows = cast(list[dict[str, float | str | bool]], Parallel(n_jobs=n_jobs, prefer="processes")(tasks))
        rows = []
        for index, row in enumerate(raw_rows):
            feature_name = cast(str, row["feature"])
            _log_importance_progress(index, len(raw_rows), feature_name)
            rows.append({**row, "selection_stage": stage_label})
    importance_df = pd.DataFrame(rows).sort_values(
        ["selected", "importance", "feature"],
        ascending=[False, False, True],
    )
    if (not importance_df.empty) and (not bool(importance_df["selected"].any())):
        importance_df.loc[:, "selected"] = False
        importance_df.loc[importance_df.index[0], "selected"] = True
    return importance_df.reset_index(drop=True), baseline_val_mae


def _selected_features(importance_df: pd.DataFrame) -> list[str]:
    """Retourne les features retenues par le selecteur."""

    if importance_df.empty:
        return []
    return importance_df.loc[importance_df["selected"], "feature"].tolist()


def _top_features(
    importance_df: pd.DataFrame,
    max_feature_count: int,
) -> list[str]:
    """Retourne au plus max_feature_count features avec gain strictement positif."""

    if importance_df.empty or max_feature_count <= 0:
        return []
    positive_df = importance_df.loc[importance_df["importance"] > 0.0].copy()
    if positive_df.empty:
        return []
    ranked_df = positive_df.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return ranked_df.head(max_feature_count)["feature"].tolist()


def _filtered_split(
    dataset_df: pd.DataFrame,
    selected_columns: list[str],
) -> pd.DataFrame:
    """Conserve uniquement les dates, features selectionnees et target."""

    columns = ["origin_date", "target_date", *selected_columns, TARGET_COLUMN]
    return dataset_df.loc[:, columns].copy()


def _write_dataframe(dataset_df: pd.DataFrame, output_csv: Path) -> None:
    """Ecrit un CSV sur disque."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(output_csv, index=False, encoding=CSV_ENCODING)


def run_feature_selection(
    train_input_csv: Path,
    val_input_csv: Path,
    test_input_csv: Path,
    train_output_csv: Path,
    val_output_csv: Path,
    test_output_csv: Path,
    summary_json: Path,
    importances_csv: Path,
    tuning_trials_csv: Path,
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
    n_folds: int = DEFAULT_FOLDS,
    n_jobs: int = DEFAULT_SELECTOR_JOBS,
    max_selected_exogenous_features: int = SARIMAX_MAX_SELECTED_EXOGENOUS_FEATURES,
) -> dict[str, int | float]:
    """Applique un prefiltrage robuste puis une selection SARIMAX sur les exogenes."""

    start_time = time.perf_counter()
    train_df = _load_split(train_input_csv)
    val_df = _load_split(val_input_csv)
    test_df = _load_split(test_input_csv)
    LOGGER.info(
        "Loaded SARIMAX feature selection inputs: train=%d rows, val=%d rows, test=%d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    context_columns = _context_feature_columns(train_df)
    sales_columns = _sales_feature_columns(train_df)
    context_columns, dropped_constant_context = _drop_constant_candidate_features(
        train_df=train_df,
        candidate_columns=context_columns,
    )
    non_constant_sales, dropped_constant_sales = _drop_constant_candidate_features(train_df, sales_columns)
    candidate_exogenous_columns, dropped_correlated_exogenous = _drop_correlated_candidate_features(
        train_df=train_df,
        candidate_columns=context_columns + non_constant_sales,
        target_column=TARGET_COLUMN,
        correlation_threshold=CORRELATION_THRESHOLD,
    )
    LOGGER.info(
        "Feature prefiltering kept %d/%d non-constant context columns, %d/%d non-constant sales columns, and %d candidate exogenous columns after correlation filtering",
        len(context_columns),
        len(_context_feature_columns(train_df)),
        len(non_constant_sales),
        len(sales_columns),
        len(candidate_exogenous_columns),
    )
    LOGGER.info(
        "Tuning SARIMAX selector dynamics on target only, then scoring all non-lag exogenous candidates one by one; retaining at most top %d exogenous features",
        max_selected_exogenous_features,
    )
    tuning_payload, trials_df = tune_selector_params(
        train_df=train_df,
        val_df=val_df,
        selector_feature_columns=[],
        n_trials=n_trials,
        n_folds=n_folds,
        n_jobs=n_jobs,
    )
    tuning_trials_csv.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(tuning_trials_csv, index=False, encoding=CSV_ENCODING)
    stage_one_importance_df, mandatory_baseline_mae = compute_feature_importances(
        train_df=train_df,
        val_df=val_df,
        selector_params=tuning_payload["best_params"],
        base_feature_columns=[],
        candidate_feature_columns=candidate_exogenous_columns,
        stage_label="exogenous",
        n_jobs=n_jobs,
    )
    selected_feature_columns = _top_features(
        importance_df=stage_one_importance_df,
        max_feature_count=max_selected_exogenous_features,
    )
    selected_context_columns = [column for column in selected_feature_columns if column in context_columns]
    selected_sales_columns = [column for column in selected_feature_columns if column in non_constant_sales]
    selected_lag_columns: list[str] = []
    LOGGER.info(
        "SARIMAX feature selection retained %d context features, %d sales features, %d lag features and %d total features",
        len(selected_context_columns),
        len(selected_sales_columns),
        len(selected_lag_columns),
        len(selected_feature_columns),
    )
    filtered_train = _filtered_split(train_df, selected_feature_columns)
    filtered_val = _filtered_split(val_df, selected_feature_columns)
    filtered_test = _filtered_split(test_df, selected_feature_columns)
    importance_df = stage_one_importance_df.copy()
    _write_dataframe(filtered_train, train_output_csv)
    _write_dataframe(filtered_val, val_output_csv)
    _write_dataframe(filtered_test, test_output_csv)
    _write_dataframe(importance_df, importances_csv)
    payload: dict[str, Any] = {
        "target_model_family": "SARIMAX",
        "selector_model_family": "SARIMAX",
        "selector_best_params": tuning_payload["best_params"],
        "selector_best_score": tuning_payload["best_score"],
        "selector_best_trial_number": tuning_payload["best_trial_number"],
        "selector_best_fold_scores": tuning_payload["best_fold_scores"],
        "selector_trial_count": tuning_payload["trial_count"],
        "selector_fold_count": tuning_payload["fold_count"],
        "selector_variance_gap_penalty": tuning_payload["variance_gap_penalty"],
        "context_feature_columns": context_columns,
        "lag_feature_columns": [],
        "dropped_constant_features": dropped_constant_sales + dropped_constant_context,
        "dropped_constant_context_features": dropped_constant_context,
        "dropped_correlated_features": dropped_correlated_exogenous,
        "candidate_exogenous_feature_columns": candidate_exogenous_columns,
        "selected_sales_feature_columns": selected_sales_columns,
        "selected_context_feature_columns": selected_context_columns,
        "selected_lag_feature_columns": selected_lag_columns,
        "selected_feature_columns": selected_feature_columns,
        "selected_feature_count": len(selected_feature_columns),
        "max_selected_exogenous_features": int(max_selected_exogenous_features),
        "univariate_selector_baseline_val_mae": mandatory_baseline_mae,
        "sales_stage_base_feature_columns": [],
        "stage_two_base_val_mae": None,
        "stage_two_base_feature_columns": [],
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2), encoding=CSV_ENCODING)
    LOGGER.info("Saved SARIMAX feature selection summary to %s", summary_json)
    LOGGER.info(
        "SARIMAX feature selection kept %d mandatory+dynamic features in %.3f seconds",
        len(selected_feature_columns),
        time.perf_counter() - start_time,
    )
    return {
        "selector_trial_count": int(tuning_payload["trial_count"]),
        "selected_feature_count": int(len(selected_feature_columns)),
    }

from __future__ import annotations

"""Selection de features pour le pipeline ElasticNet."""

import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from src.elasticnet_time_series import (
    elasticnet_params_payload,
    fit_elasticnet_model,
    predict_frame,
)
from src.feature_selection.constants import (
    COEFFICIENT_SELECTION_THRESHOLD,
    CORRELATION_THRESHOLD,
    CSV_ENCODING,
    DEFAULT_FOLDS,
    DEFAULT_OPTUNA_TRIALS,
    DEFAULT_SELECTOR_JOBS,
    DEFAULT_SELECTOR_MAX_ITER,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from src.time_series_metrics import mae_score
from src.time_series_validation import build_recent_history_walk_forward_folds

LOGGER: logging.Logger = logging.getLogger(__name__)
IMPORTANCE_LOG_EVERY: int = 10
OVERFIT_GAP_WEIGHT: float = 0.35
VARIANCE_GAP_WEIGHT: float = 0.15


def _load_split(csv_path: Path) -> pd.DataFrame:
    """Charge un split temporel journalier en preservant l'ordre des dates."""

    dataset_df: pd.DataFrame = pd.read_csv(csv_path)
    return dataset_df.sort_values("origin_date").reset_index(drop=True)


def _mandatory_feature_columns(dataset_df: pd.DataFrame) -> list[str]:
    """Retourne le noyau obligatoire de features non supprimables."""

    return [
        column
        for column in dataset_df.columns
        if column.startswith("exog_calendar_")
        or column.startswith("exog_flag_")
        or column.startswith("exog_holiday_")
        or column.startswith("exog_weather_")
    ]


def _lag_feature_columns(dataset_df: pd.DataFrame) -> list[str]:
    """Retourne les lags autoregressifs de la cible traites a l'etape 2."""

    return [column for column in dataset_df.columns if column.startswith("exog_autoreg_")]


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


def _selector_feature_columns(
    mandatory_columns: list[str],
    candidate_sales_columns: list[str],
) -> list[str]:
    """Retourne les colonnes effectivement utilisees par le selecteur."""

    return mandatory_columns + candidate_sales_columns


def _fit_selector_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
    eval_df: pd.DataFrame | None = None,
) -> Any:
    """Ajuste le selecteur ElasticNet sur les colonnes fournies."""

    del eval_df
    return fit_elasticnet_model(
        history_df=train_df,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        params=params,
        random_seed=RANDOM_SEED,
    )


def _predict_mae(
    fitted_model: Any,
    dataset_df: pd.DataFrame,
    feature_columns: list[str],
) -> float:
    """Calcule la MAE d'un modele ElasticNet sur un dataset donne."""

    predicted_values = predict_frame(fitted_model, dataset_df, feature_columns)
    actual_values = cast(pd.Series, dataset_df.loc[:, TARGET_COLUMN])
    return mae_score(actual_values, predicted_values)


def _evaluate_fold(
    params: dict[str, Any],
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[float, float]:
    """Entraine sur un fold puis retourne sa MAE et son ecart de variance."""

    fitted_model = _fit_selector_model(
        train_df=train_fold,
        feature_columns=feature_columns,
        params=params,
        eval_df=val_fold,
    )
    val_mae = _predict_mae(fitted_model, val_fold, feature_columns)
    predicted_values = predict_frame(fitted_model, val_fold, feature_columns)
    actual_values = cast(pd.Series, val_fold.loc[:, TARGET_COLUMN])
    actual_std = float(np.std(np.asarray(actual_values, dtype=float), ddof=0))
    predicted_std = float(np.std(np.asarray(predicted_values, dtype=float), ddof=0))
    variance_gap = abs(actual_std - predicted_std)
    return val_mae, variance_gap


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
    """Loggue la progression du calcul des importances ElasticNet."""

    if not _should_log_importance_progress(index=index, total_features=total_features, every_n=every_n):
        return
    LOGGER.info(
        "ElasticNet importance progress %d/%d (%.1f%%): feature=%s",
        index + 1,
        total_features,
        ((index + 1) / total_features) * 100.0,
        feature_name,
    )


def _suggest_selector_params(trial: optuna.Trial) -> dict[str, Any]:
    """Definit l'espace de recherche ElasticNet du selecteur."""

    return elasticnet_params_payload(
        alpha=float(trial.suggest_float("alpha", 1e-4, 10.0, log=True)),
        l1_ratio=float(trial.suggest_float("l1_ratio", 0.05, 1.0)),
        fit_intercept=bool(trial.suggest_categorical("fit_intercept", [True, False])),
        max_iter=DEFAULT_SELECTOR_MAX_ITER,
    )


def _selector_objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    n_folds: int,
    n_jobs: int,
) -> float:
    """Fonction objectif Optuna basee sur MAE, overfit et ecart de variance."""

    params = _suggest_selector_params(trial)
    folds = build_recent_history_walk_forward_folds(train_df=train_df, val_df=val_df, n_folds=n_folds)
    LOGGER.info("Starting selector trial %d with ElasticNet params: %s", trial.number, params)
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
    train_mae = _selector_train_mae(
        train_df=train_df,
        feature_columns=feature_columns,
        params=params,
    )
    mean_val_mae = float(np.mean([score[0] for score in fold_scores]))
    variance_gap = float(np.mean([score[1] for score in fold_scores]))
    overfit_gap = max(0.0, mean_val_mae - train_mae)
    mean_score = float(mean_val_mae + (OVERFIT_GAP_WEIGHT * overfit_gap) + (VARIANCE_GAP_WEIGHT * variance_gap))
    trial.set_user_attr("fold_scores", [score[0] for score in fold_scores])
    trial.set_user_attr("mean_validation_mae", mean_val_mae)
    trial.set_user_attr("train_mae", train_mae)
    trial.set_user_attr("overfit_gap_penalty", overfit_gap)
    trial.set_user_attr("variance_gap_penalty", variance_gap)
    LOGGER.info(
        "Finished selector trial %d with penalized score %.6f, validation MAE %.6f, overfit gap %.6f and variance gap %.6f",
        trial.number,
        mean_score,
        mean_val_mae,
        overfit_gap,
        variance_gap,
    )
    return mean_score


def _selector_train_mae(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
) -> float:
    """Retourne la MAE train du selecteur pour penaliser l'overfitting."""

    fitted_model = _fit_selector_model(
        train_df=train_df,
        feature_columns=feature_columns,
        params=params,
    )
    return _predict_mae(fitted_model, train_df, feature_columns)


def tune_selector_params(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
    n_folds: int = DEFAULT_FOLDS,
    n_jobs: int = DEFAULT_SELECTOR_JOBS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Cherche un jeu d'hyperparametres ElasticNet robuste pour la selection."""

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda trial: _selector_objective(
            trial=trial,
            train_df=train_df,
            val_df=val_df,
            feature_columns=feature_columns,
            n_folds=n_folds,
            n_jobs=n_jobs,
        ),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )
    best_trial = study.best_trial
    best_params = elasticnet_params_payload(
        alpha=float(best_trial.params["alpha"]),
        l1_ratio=float(best_trial.params["l1_ratio"]),
        fit_intercept=bool(best_trial.params["fit_intercept"]),
        max_iter=DEFAULT_SELECTOR_MAX_ITER,
    )
    trials_df = study.trials_dataframe().copy()
    best_score = cast(float, best_trial.value)
    best_fold_scores = cast(list[float], best_trial.user_attrs.get("fold_scores", []))
    return {
        "model_family": "ElasticNet",
        "best_params": best_params,
        "best_score": float(cast(float, best_trial.user_attrs.get("mean_validation_mae", best_score))),
        "best_trial_number": int(best_trial.number),
        "best_fold_scores": best_fold_scores,
        "train_mae": float(cast(float, best_trial.user_attrs.get("train_mae", float("inf")))),
        "overfit_gap_penalty": float(cast(float, best_trial.user_attrs.get("overfit_gap_penalty", float("inf")))),
        "variance_gap_penalty": float(cast(float, best_trial.user_attrs.get("variance_gap_penalty", float("inf")))),
        "trial_count": int(n_trials),
        "fold_count": int(n_folds),
    }, trials_df


def compute_feature_importances(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    selector_params: dict[str, Any],
    mandatory_columns: list[str],
    candidate_feature_columns: list[str],
    stage_label: str,
) -> tuple[pd.DataFrame, float, float]:
    """Construit le tableau d'importances ElasticNet pour les features candidates."""

    model_columns = _selector_feature_columns(mandatory_columns, candidate_feature_columns)
    LOGGER.info(
        "Computing ElasticNet importances for stage=%s with %d mandatory columns and %d candidate columns",
        stage_label,
        len(mandatory_columns),
        len(candidate_feature_columns),
    )
    fitted_model = _fit_selector_model(
        train_df=train_df,
        feature_columns=model_columns,
        params=selector_params,
    )
    train_mae = _predict_mae(fitted_model, train_df, model_columns)
    val_mae = _predict_mae(fitted_model, val_df, model_columns)
    if not candidate_feature_columns:
        empty_df = pd.DataFrame(
            columns=["feature", "importance", "normalized_importance", "selected", "selection_stage"]
        )
        return empty_df, train_mae, val_mae
    importances = pd.Series(np.abs(np.asarray(fitted_model.coef_, dtype=float)), index=model_columns, dtype=float)
    candidate_importances = importances.loc[candidate_feature_columns]
    total_importance = float(candidate_importances.sum())
    importance_rows = [
        _importance_row(
            index=index,
            total_features=len(candidate_feature_columns),
            feature_name=feature_name,
            importance=float(cast(float, candidate_importances.loc[feature_name])),
            total_importance=total_importance,
            stage_label=stage_label,
        )
        for index, feature_name in enumerate(candidate_feature_columns)
    ]
    importance_df = pd.DataFrame(importance_rows).sort_values(
        ["selected", "importance", "feature"],
        ascending=[False, False, True],
    )
    any_selected = bool(importance_df["selected"].any())
    if (not any_selected) and (not importance_df.empty):
        importance_df.loc[:, "selected"] = False
        importance_df.loc[importance_df.index[0], "selected"] = True
    return importance_df.reset_index(drop=True), train_mae, val_mae


def _importance_row(
    index: int,
    total_features: int,
    feature_name: str,
    importance: float,
    total_importance: float,
    stage_label: str,
) -> dict[str, float | str | bool]:
    """Construit une ligne d'importance et loggue sa progression."""

    _log_importance_progress(index=index, total_features=total_features, feature_name=feature_name)
    normalized_importance = 0.0 if total_importance <= 0.0 else importance / total_importance
    return {
        "feature": feature_name,
        "importance": importance,
        "normalized_importance": float(normalized_importance),
        "selected": bool(importance > COEFFICIENT_SELECTION_THRESHOLD),
        "selection_stage": stage_label,
    }


def _selected_features(importance_df: pd.DataFrame) -> list[str]:
    """Retourne les features retenues par le selecteur."""

    if importance_df.empty:
        return []
    return importance_df.loc[importance_df["selected"], "feature"].tolist()


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
) -> dict[str, int | float]:
    """Applique un prefiltrage robuste puis une selection ElasticNet sur les ventes."""

    start_time = time.perf_counter()
    train_df = _load_split(train_input_csv)
    val_df = _load_split(val_input_csv)
    test_df = _load_split(test_input_csv)
    LOGGER.info(
        "Loaded feature selection inputs: train=%d rows, val=%d rows, test=%d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    mandatory_columns = _mandatory_feature_columns(train_df)
    lag_columns = _lag_feature_columns(train_df)
    sales_columns = _sales_feature_columns(train_df)
    non_constant_sales, dropped_constant_sales = _drop_constant_candidate_features(train_df, sales_columns)
    candidate_sales, dropped_correlated_sales = _drop_correlated_candidate_features(
        train_df=train_df,
        candidate_columns=non_constant_sales,
        target_column=TARGET_COLUMN,
        correlation_threshold=CORRELATION_THRESHOLD,
    )
    LOGGER.info(
        "Feature prefiltering kept %d mandatory columns, %d/%d non-constant sales columns, and %d candidate sales columns after correlation filtering",
        len(mandatory_columns),
        len(non_constant_sales),
        len(sales_columns),
        len(candidate_sales),
    )
    selector_columns = _selector_feature_columns(mandatory_columns, candidate_sales)
    LOGGER.info(
        "Starting ElasticNet Optuna selection with %d trials, %d folds and %d total selector columns",
        n_trials,
        n_folds,
        len(selector_columns),
    )
    tuning_payload, trials_df = tune_selector_params(
        train_df=train_df,
        val_df=val_df,
        feature_columns=selector_columns,
        n_trials=n_trials,
        n_folds=n_folds,
        n_jobs=n_jobs,
    )
    LOGGER.info(
        "Selector tuning finished with best MAE %.6f on trial %d",
        tuning_payload["best_score"],
        tuning_payload["best_trial_number"],
    )
    tuning_trials_csv.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(tuning_trials_csv, index=False, encoding=CSV_ENCODING)
    LOGGER.info("Saved selector Optuna trials to %s", tuning_trials_csv)
    stage_one_importance_df, _, _ = compute_feature_importances(
        train_df=train_df,
        val_df=val_df,
        selector_params=tuning_payload["best_params"],
        mandatory_columns=mandatory_columns,
        candidate_feature_columns=candidate_sales,
        stage_label="non_lag",
    )
    stage_one_survivors = _selected_features(stage_one_importance_df)
    stage_two_raw_candidates = stage_one_survivors + lag_columns
    stage_two_non_constant_candidates, stage_two_dropped_constant_features = _drop_constant_candidate_features(
        train_df=train_df,
        candidate_columns=stage_two_raw_candidates,
    )
    stage_two_candidates, stage_two_dropped_correlated_features = _drop_correlated_candidate_features(
        train_df=train_df,
        candidate_columns=stage_two_non_constant_candidates,
        target_column=TARGET_COLUMN,
        correlation_threshold=CORRELATION_THRESHOLD,
    )
    stage_two_importance_df, selector_train_mae, selector_val_mae = compute_feature_importances(
        train_df=train_df,
        val_df=val_df,
        selector_params=tuning_payload["best_params"],
        mandatory_columns=mandatory_columns,
        candidate_feature_columns=stage_two_candidates,
        stage_label="with_lags",
    )
    importance_df = pd.concat([stage_one_importance_df, stage_two_importance_df], ignore_index=True)
    selected_dynamic_columns = _selected_features(stage_two_importance_df)
    selected_sales_columns = [
        column for column in selected_dynamic_columns if column.startswith("exog_sales_")
    ]
    selected_lag_columns = [
        column for column in selected_dynamic_columns if column.startswith("exog_autoreg_")
    ]
    selected_feature_columns = mandatory_columns + selected_dynamic_columns
    LOGGER.info(
        "Two-stage ElasticNet feature selection retained %d sales features, %d lag features and %d total features",
        len(selected_sales_columns),
        len(selected_lag_columns),
        len(selected_feature_columns),
    )
    filtered_train = _filtered_split(train_df, selected_feature_columns)
    filtered_val = _filtered_split(val_df, selected_feature_columns)
    filtered_test = _filtered_split(test_df, selected_feature_columns)
    _write_dataframe(filtered_train, train_output_csv)
    _write_dataframe(filtered_val, val_output_csv)
    _write_dataframe(filtered_test, test_output_csv)
    _write_dataframe(importance_df, importances_csv)
    LOGGER.info("Saved selected train split to %s", train_output_csv)
    LOGGER.info("Saved selected val split to %s", val_output_csv)
    LOGGER.info("Saved selected test split to %s", test_output_csv)
    LOGGER.info("Saved ElasticNet feature importances to %s", importances_csv)
    payload: dict[str, Any] = {
        "selector_model_family": "ElasticNet",
        "selector_best_params": tuning_payload["best_params"],
        "selector_best_score": tuning_payload["best_score"],
        "selector_best_trial_number": tuning_payload["best_trial_number"],
        "selector_best_fold_scores": tuning_payload["best_fold_scores"],
        "selector_trial_count": tuning_payload["trial_count"],
        "selector_fold_count": tuning_payload["fold_count"],
        "selector_train_mae_from_tuning": tuning_payload["train_mae"],
        "selector_overfit_gap_penalty": tuning_payload["overfit_gap_penalty"],
        "selector_variance_gap_penalty": tuning_payload["variance_gap_penalty"],
        "mandatory_feature_columns": mandatory_columns,
        "lag_feature_columns": lag_columns,
        "dropped_constant_features": dropped_constant_sales,
        "dropped_correlated_features": dropped_correlated_sales,
        "candidate_sales_feature_columns": candidate_sales,
        "stage_one_surviving_sales_feature_columns": stage_one_survivors,
        "stage_two_raw_candidate_feature_columns": stage_two_raw_candidates,
        "stage_two_dropped_constant_features": stage_two_dropped_constant_features,
        "stage_two_dropped_correlated_features": stage_two_dropped_correlated_features,
        "stage_two_candidate_feature_columns": stage_two_candidates,
        "selected_sales_feature_columns": selected_sales_columns,
        "selected_lag_feature_columns": selected_lag_columns,
        "selected_feature_columns": selected_feature_columns,
        "selected_feature_count": len(selected_feature_columns),
        "selector_train_mae": selector_train_mae,
        "selector_val_mae": selector_val_mae,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2), encoding=CSV_ENCODING)
    LOGGER.info("Saved feature selection summary to %s", summary_json)
    LOGGER.info(
        "ElasticNet feature selection kept %d mandatory+dynamic features in %.3f seconds",
        len(selected_feature_columns),
        time.perf_counter() - start_time,
    )
    return {
        "selector_trial_count": int(tuning_payload["trial_count"]),
        "selected_feature_count": int(len(selected_feature_columns)),
        "train_rows": int(len(filtered_train)),
        "val_rows": int(len(filtered_val)),
        "test_rows": int(len(filtered_test)),
    }

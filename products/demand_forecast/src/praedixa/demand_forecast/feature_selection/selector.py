"""ElasticNet selector internals for bundle feature selection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from typing import Any, Mapping, cast

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from praedixa.demand_forecast.feature_selection.artifacts import DATE_COL
from praedixa.demand_forecast.feature_selection.constants import (
    COEFFICIENT_SELECTION_THRESHOLD,
    RANDOM_SEED,
)

LOGGER = logging.getLogger(__name__)


DEFAULT_ELASTIC_NET_PARAMS: dict[str, object] = {
    "alpha": 0.001,
    "l1_ratio": 0.5,
    "max_iter": 10_000,
}


@dataclass(frozen=True)
class ElasticNetSelectorModel:
    """Fitted numeric selector with train-fitted scaling."""

    scaler: StandardScaler
    model: ElasticNet

    def coefficients(self) -> np.ndarray:
        coefficients = cast(Any, self.model).coef_
        return np.abs(np.asarray(coefficients, dtype=float))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        scaled_features = cast(Any, self.scaler).transform(features)
        predictions = cast(Any, self.model).predict(scaled_features)
        return np.asarray(predictions, dtype=float)


@dataclass(frozen=True)
class FoldWindow:
    """Walk-forward fold represented by row positions into the shared train frame."""

    train_positions: np.ndarray
    validation_positions: np.ndarray


def selector_candidate_columns(
    train_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    feature_roles: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    numeric_candidates: list[str] = []
    for column in feature_columns:
        if column not in train_df.columns:
            continue
        role = feature_roles.get(column, "")
        if role.endswith("_real") and pd.api.types.is_numeric_dtype(train_df[column]):
            numeric_candidates.append(column)
    return numeric_candidates, []


def drop_constant_candidate_features(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
) -> tuple[list[str], list[str]]:
    kept_columns: list[str] = []
    dropped_columns: list[str] = []
    for column in candidate_columns:
        if train_df[column].nunique(dropna=False) <= 1:
            dropped_columns.append(column)
        else:
            kept_columns.append(column)
    return kept_columns, dropped_columns


def tune_selector_params(
    train_df: pd.DataFrame,
    *,
    numeric_feature_columns: list[str],
    target_column: str,
    n_trials: int,
    n_folds: int,
    n_fold_workers: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not numeric_feature_columns:
        return {
            **DEFAULT_ELASTIC_NET_PARAMS,
            "best_score": float("nan"),
        }, pd.DataFrame()
    folds = _walk_forward_fold_windows(train_df, n_folds=n_folds)
    max_workers = max(1, min(n_fold_workers, len(folds)))
    LOGGER.info(
        "ElasticNet parameter optimisation folds prepared: folds=%d workers=%d "
        "train_rows=%d feature_count=%d shared_dataset=True",
        len(folds),
        max_workers,
        len(train_df),
        len(numeric_feature_columns),
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda trial: _selector_objective(
            trial,
            train_df=train_df,
            folds=folds,
            numeric_feature_columns=numeric_feature_columns,
            target_column=target_column,
            max_workers=max_workers,
        ),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )
    best_trial = study.best_trial
    best_params: dict[str, object] = {
        "alpha": float(best_trial.params["alpha"]),
        "l1_ratio": float(best_trial.params["l1_ratio"]),
        "max_iter": 10_000,
        "best_score": float(cast(float, best_trial.value)),
        "best_trial_number": int(best_trial.number),
        "fold_scores": cast(list[float], best_trial.user_attrs.get("fold_scores", [])),
        "fold_workers": int(best_trial.user_attrs.get("fold_workers", 1)),
    }
    study_any = cast(Any, study)
    return best_params, cast(pd.DataFrame, study_any.trials_dataframe()).copy()


def importance_frame(
    train_df: pd.DataFrame,
    *,
    numeric_feature_columns: list[str],
    target_column: str,
    selector_params: Mapping[str, object],
) -> tuple[pd.DataFrame, float, None]:
    if not numeric_feature_columns:
        return _empty_importance_frame(), float("nan"), None
    model = _fit_selector_model(
        train_df,
        numeric_feature_columns=numeric_feature_columns,
        target_column=target_column,
        params=selector_params,
    )
    raw_coefficients = _elastic_net_coefficients(model)
    importance_df = _importance_frame_from_values(
        numeric_feature_columns,
        raw_coefficients,
    )
    train_mae = _predict_mae(
        model,
        train_df,
        numeric_feature_columns=numeric_feature_columns,
        target_column=target_column,
    )
    return importance_df, train_mae, None


def selected_numeric_features(importance_df: pd.DataFrame) -> list[str]:
    if importance_df.empty:
        return []
    return [
        str(value) for value in importance_df.loc[importance_df["selected"], "feature"]
    ]


def _empty_importance_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "feature",
            "importance",
            "normalized_importance",
            "selected",
        ]
    )


def _importance_frame_from_values(
    feature_columns: list[str],
    raw_importances: np.ndarray,
) -> pd.DataFrame:
    total_importance = float(raw_importances.sum())
    rows: list[dict[str, object]] = []
    for feature, importance in zip(feature_columns, raw_importances, strict=True):
        importance_value = float(importance)
        normalized = (
            0.0
            if total_importance <= 0.0
            else float(importance_value / total_importance)
        )
        rows.append(
            {
                "feature": feature,
                "importance": importance_value,
                "normalized_importance": normalized,
                "selected": bool(importance_value > COEFFICIENT_SELECTION_THRESHOLD),
            }
        )
    importance_df = pd.DataFrame(rows).sort_values(
        ["selected", "importance", "feature"],
        ascending=[False, False, True],
    )
    if not bool(importance_df["selected"].any()) and not importance_df.empty:
        importance_df.loc[importance_df.index[0], "selected"] = True
    return importance_df.reset_index(drop=True)


def _selector_frame(
    frame: pd.DataFrame,
    *,
    numeric_feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    selected = frame.loc[:, [*numeric_feature_columns, target_column]].copy()
    for column in numeric_feature_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce").fillna(0.0)
    selected[target_column] = pd.to_numeric(
        selected[target_column], errors="coerce"
    ).fillna(0.0)
    return selected


def _selector_xy(
    frame: pd.DataFrame,
    *,
    numeric_feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    selector_df = _selector_frame(
        frame,
        numeric_feature_columns=numeric_feature_columns,
        target_column=target_column,
    )
    return selector_df.loc[:, numeric_feature_columns], selector_df[target_column]


def _fit_selector_model(
    train_df: pd.DataFrame,
    *,
    numeric_feature_columns: list[str],
    target_column: str,
    params: Mapping[str, object],
) -> ElasticNetSelectorModel:
    train_x, train_y = _selector_xy(
        train_df,
        numeric_feature_columns=numeric_feature_columns,
        target_column=target_column,
    )
    scaler = StandardScaler()
    scaled_train_x = cast(Any, scaler).fit_transform(train_x)
    model = _build_elastic_net(params)
    cast(Any, model).fit(scaled_train_x, train_y)
    return ElasticNetSelectorModel(scaler=scaler, model=model)


def _build_elastic_net(params: Mapping[str, object]) -> ElasticNet:
    return ElasticNet(
        alpha=float(cast(float, params["alpha"])),
        l1_ratio=float(cast(float, params["l1_ratio"])),
        max_iter=int(cast(int, params.get("max_iter", 10_000))),
        random_state=RANDOM_SEED,
        selection="cyclic",
    )


def _elastic_net_coefficients(model: ElasticNetSelectorModel) -> np.ndarray:
    return model.coefficients()


def _predict_mae(
    fitted_model: ElasticNetSelectorModel,
    dataset_df: pd.DataFrame,
    *,
    numeric_feature_columns: list[str],
    target_column: str,
) -> float:
    features, actual = _selector_xy(
        dataset_df,
        numeric_feature_columns=numeric_feature_columns,
        target_column=target_column,
    )
    predictions = fitted_model.predict(features)
    actual_values = np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(actual_values - predictions)))


def _walk_forward_fold_windows(
    train_df: pd.DataFrame,
    *,
    n_folds: int,
) -> list[FoldWindow]:
    if train_df.empty:
        empty_positions = np.asarray([], dtype=int)
        return [
            FoldWindow(
                train_positions=empty_positions,
                validation_positions=empty_positions,
            )
        ]
    if DATE_COL not in train_df.columns:
        all_positions = np.arange(len(train_df), dtype=int)
        return [
            FoldWindow(
                train_positions=all_positions,
                validation_positions=all_positions,
            )
        ]
    date_series = pd.Series(train_df[DATE_COL])
    dates = list(date_series.drop_duplicates().sort_values())
    if len(dates) < 2:
        all_positions = np.arange(len(train_df), dtype=int)
        return [
            FoldWindow(
                train_positions=all_positions,
                validation_positions=all_positions,
            )
        ]
    split_count = max(2, min(n_folds + 1, len(dates)))
    chunks = np.array_split(
        np.asarray(dates, dtype=object),
        split_count,
    )
    folds: list[FoldWindow] = []
    history_positions = _positions_for_dates(date_series, chunks[0])
    for chunk in chunks[1:]:
        if len(chunk) == 0:
            continue
        validation_positions = _positions_for_dates(date_series, chunk)
        if len(validation_positions) == 0:
            continue
        folds.append(
            FoldWindow(
                train_positions=history_positions.copy(),
                validation_positions=validation_positions,
            )
        )
        history_positions = np.concatenate(
            [history_positions, validation_positions]
        ).astype(int, copy=False)
    if folds:
        return folds
    all_positions = np.arange(len(train_df), dtype=int)
    return [
        FoldWindow(
            train_positions=all_positions,
            validation_positions=all_positions,
        )
    ]


def _positions_for_dates(date_series: pd.Series, dates: np.ndarray) -> np.ndarray:
    mask = date_series.isin(dates)
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(int, copy=False)


def _suggest_selector_params(trial: optuna.Trial) -> dict[str, object]:
    return {
        "alpha": float(trial.suggest_float("alpha", 1e-5, 1e-1, log=True)),
        "l1_ratio": float(trial.suggest_float("l1_ratio", 0.05, 0.95)),
        "max_iter": 10_000,
    }


def _selector_objective(
    trial: optuna.Trial,
    *,
    train_df: pd.DataFrame,
    folds: list[FoldWindow],
    numeric_feature_columns: list[str],
    target_column: str,
    max_workers: int,
) -> float:
    params = _suggest_selector_params(trial)
    if max_workers == 1:
        fold_scores = [
            _score_selector_fold(
                train_df,
                fold,
                numeric_feature_columns=numeric_feature_columns,
                target_column=target_column,
                params=params,
            )
            for fold in folds
        ]
    else:

        def score_fold(fold: FoldWindow) -> float:
            return _score_selector_fold(
                train_df,
                fold,
                numeric_feature_columns=numeric_feature_columns,
                target_column=target_column,
                params=params,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fold_scores = list(executor.map(score_fold, folds))
    mean_score = float(np.mean(fold_scores))
    trial.set_user_attr("fold_scores", fold_scores)
    trial.set_user_attr("fold_workers", max_workers)
    return mean_score


def _score_selector_fold(
    train_df: pd.DataFrame,
    fold: FoldWindow,
    *,
    numeric_feature_columns: list[str],
    target_column: str,
    params: Mapping[str, object],
) -> float:
    train_fold = train_df.iloc[fold.train_positions]
    validation_fold = train_df.iloc[fold.validation_positions]
    model = _fit_selector_model(
        train_fold,
        numeric_feature_columns=numeric_feature_columns,
        target_column=target_column,
        params=params,
    )
    return _predict_mae(
        model,
        validation_fold,
        numeric_feature_columns=numeric_feature_columns,
        target_column=target_column,
    )

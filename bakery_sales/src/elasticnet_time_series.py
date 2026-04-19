from __future__ import annotations

"""Utilitaires partages pour les modeles ElasticNet temporels."""

import math
from pathlib import Path
from typing import Any, Callable, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet


Z_80: float = 1.2815515655446004
Z_95: float = 1.959963984540054
ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def feature_columns(dataset_df: pd.DataFrame, target_column: str) -> list[str]:
    """Retourne les colonnes de features du dataset journalier."""

    return [
        column
        for column in dataset_df.columns
        if column not in {"origin_date", "target_date", target_column}
    ]


def elasticnet_params_payload(
    alpha: float,
    l1_ratio: float,
    fit_intercept: bool,
    max_iter: int,
) -> dict[str, Any]:
    """Construit un payload de parametres ElasticNet serialisable."""

    return {
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "fit_intercept": bool(fit_intercept),
        "max_iter": int(max_iter),
    }


def build_elasticnet_model(params: dict[str, Any], random_seed: int) -> ElasticNet:
    """Construit un modele ElasticNet pret a etre ajuste."""

    return ElasticNet(
        alpha=float(params["alpha"]),
        l1_ratio=float(params["l1_ratio"]),
        fit_intercept=bool(params["fit_intercept"]),
        max_iter=int(params["max_iter"]),
        random_state=int(random_seed),
        selection="cyclic",
    )


def fit_elasticnet_model(
    history_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    params: dict[str, Any],
    random_seed: int,
) -> ElasticNet:
    """Ajuste un modele ElasticNet sur l'historique disponible."""

    fitted_model = build_elasticnet_model(params=params, random_seed=random_seed)
    fitted_model.fit(history_df.loc[:, feature_columns], history_df[target_column])
    return fitted_model


def save_elasticnet_model(
    fitted_model: ElasticNet,
    output_path: Path,
) -> None:
    """Persiste un modele ElasticNet sur disque."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_model, output_path)


def predict_frame(
    fitted_model: ElasticNet,
    dataset_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """Predictions brutes sur un dataframe entier."""

    predictions_array: np.ndarray[Any, np.dtype[np.float64]] = cast(
        np.ndarray[Any, np.dtype[np.float64]],
        np.asarray(fitted_model.predict(dataset_df.loc[:, feature_columns]), dtype=float),
    )
    predictions_list: list[float] = [float(value) for value in predictions_array.tolist()]
    return pd.Series(data=predictions_list, index=dataset_df.index, dtype=float)


def residual_scale(
    fitted_model: ElasticNet,
    history_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> float:
    """Estime une echelle d'incertitude a partir des residus in-sample."""

    fitted_values = predict_frame(fitted_model, history_df, feature_columns)
    residuals = history_df[target_column].reset_index(drop=True) - fitted_values.reset_index(drop=True)
    residual_values: np.ndarray[Any, np.dtype[np.float64]] = cast(
        np.ndarray[Any, np.dtype[np.float64]],
        np.asarray(residuals, dtype=float),
    )
    scale = float(np.std(residual_values, ddof=0))
    return max(scale, 1e-9)


def prediction_interval_bounds(prediction_raw: float, scale: float) -> tuple[float, float, float, float]:
    """Construit des intervalles symetriques 80/95% autour de la prediction."""

    return (
        prediction_raw - (Z_80 * scale),
        prediction_raw + (Z_80 * scale),
        prediction_raw - (Z_95 * scale),
        prediction_raw + (Z_95 * scale),
    )


def batch_forecast(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    params: dict[str, Any],
    random_seed: int,
) -> pd.DataFrame:
    """Predictions batch sur un horizon futur avec un modele ajuste une seule fois."""

    fitted_model = fit_elasticnet_model(
        history_df=history_df,
        feature_columns=feature_columns,
        target_column=target_column,
        params=params,
        random_seed=random_seed,
    )
    predictions = predict_frame(fitted_model, future_df, feature_columns)
    scale = residual_scale(fitted_model, history_df, feature_columns, target_column)
    future_records: list[dict[str, Any]] = cast(
        list[dict[str, Any]],
        future_df.reset_index(drop=True).to_dict(orient="records"),
    )
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(future_records):
        prediction_raw = float(cast(float, predictions.iloc[row_index]))
        lower_80, upper_80, lower_95, upper_95 = prediction_interval_bounds(prediction_raw, scale)
        rows.append(
            {
                "origin_date": row["origin_date"],
                "target_date": row["target_date"],
                "actual": float(cast(float, row[target_column])),
                "prediction_raw": prediction_raw,
                "prediction_rounded": float(np.rint(prediction_raw)),
                "lower_80": float(lower_80),
                "upper_80": float(upper_80),
                "lower_95": float(lower_95),
                "upper_95": float(upper_95),
                "train_rows_used": int(len(history_df)),
                "used_alpha": float(params["alpha"]),
                "used_l1_ratio": float(params["l1_ratio"]),
            }
        )
    return pd.DataFrame(rows)


def rolling_forecast(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    params: dict[str, Any],
    random_seed: int,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Predictions rolling-origin en re-ajustant ElasticNet avant chaque ligne future."""

    ordered_history = history_df.reset_index(drop=True).copy()
    ordered_future = future_df.reset_index(drop=True).copy()
    future_records: list[dict[str, Any]] = cast(
        list[dict[str, Any]],
        ordered_future.to_dict(orient="records"),
    )
    prediction_rows: list[dict[str, Any]] = []
    total_rows = len(ordered_future)
    for row_index in range(total_rows):
        fitted_model = fit_elasticnet_model(
            history_df=ordered_history,
            feature_columns=feature_columns,
            target_column=target_column,
            params=params,
            random_seed=random_seed,
        )
        scale = residual_scale(fitted_model, ordered_history, feature_columns, target_column)
        next_row = ordered_future.iloc[[row_index]]
        next_row_record = future_records[row_index]
        prediction_raw = float(cast(float, predict_frame(fitted_model, next_row, feature_columns).iloc[0]))
        lower_80, upper_80, lower_95, upper_95 = prediction_interval_bounds(prediction_raw, scale)
        prediction_row: dict[str, Any] = {
            "origin_date": str(next_row_record["origin_date"]),
            "target_date": str(next_row_record["target_date"]),
            "actual": float(cast(float, next_row_record[target_column])),
            "prediction_raw": prediction_raw,
            "prediction_rounded": float(np.rint(prediction_raw)),
            "lower_80": float(lower_80),
            "upper_80": float(upper_80),
            "lower_95": float(lower_95),
            "upper_95": float(upper_95),
            "train_rows_used": int(len(ordered_history)),
            "used_alpha": float(params["alpha"]),
            "used_l1_ratio": float(params["l1_ratio"]),
        }
        prediction_rows.append(prediction_row)
        if progress_callback is not None:
            progress_callback(row_index, total_rows, prediction_row)
        ordered_history = pd.concat([ordered_history, next_row], ignore_index=True)
    return pd.DataFrame(prediction_rows)


def model_complexity_penalty(fitted_model: ElasticNet) -> float:
    """Retourne une penalite simple basee sur le nombre de coefficients actifs."""

    non_zero_coefficients = int(np.count_nonzero(np.asarray(fitted_model.coef_, dtype=float)))
    return float(non_zero_coefficients)

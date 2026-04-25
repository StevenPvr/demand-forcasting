from __future__ import annotations

"""Utilitaires partages pour les modeles SARIMAX temporels."""

from pathlib import Path
from typing import Any, Callable, cast

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def feature_columns(dataset_df: pd.DataFrame, target_column: str) -> list[str]:
    """Retourne les colonnes exogenes du dataset journalier."""

    return [
        column
        for column in dataset_df.columns
        if column not in {"origin_date", "target_date", target_column}
    ]


def sarimax_params_payload(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
) -> dict[str, Any]:
    """Construit un payload SARIMAX serialisable."""

    return {
        "order": [int(value) for value in order],
        "seasonal_order": [int(value) for value in seasonal_order],
        "trend": str(trend),
    }


def build_sarimax_model(
    history_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    params: dict[str, Any],
) -> SARIMAX:
    """Construit un modele SARIMAX pret a etre ajuste."""

    order_values = cast(list[int], params["order"])
    seasonal_values = cast(list[int], params["seasonal_order"])
    exog_df: pd.DataFrame | None = None
    if feature_columns:
        exog_df = history_df.loc[:, feature_columns].astype(float)
    return SARIMAX(
        endog=history_df[target_column].astype(float),
        exog=exog_df,
        order=(int(order_values[0]), int(order_values[1]), int(order_values[2])),
        seasonal_order=(
            int(seasonal_values[0]),
            int(seasonal_values[1]),
            int(seasonal_values[2]),
            int(seasonal_values[3]),
        ),
        trend=str(params["trend"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )


def fit_sarimax_model(
    history_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    params: dict[str, Any],
) -> SARIMAXResultsWrapper:
    """Ajuste un modele SARIMAX sur l'historique disponible."""

    fitted_model = build_sarimax_model(
        history_df=history_df,
        feature_columns=feature_columns,
        target_column=target_column,
        params=params,
    )
    return cast(
        SARIMAXResultsWrapper,
        fitted_model.fit(
            disp=False,
            maxiter=200,
            method="lbfgs",
        ),
    )


def save_sarimax_model(
    fitted_model: SARIMAXResultsWrapper,
    output_path: Path,
) -> None:
    """Persiste un modele SARIMAX sur disque."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_model, output_path)


def _forecast_row(
    fitted_model: SARIMAXResultsWrapper,
    next_row: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    row: dict[str, Any],
    train_rows_used: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Construit la prediction a un pas et ses intervalles."""

    next_exog: pd.DataFrame | None = None
    if feature_columns:
        next_exog = next_row.loc[:, feature_columns].astype(float)
    forecast_80 = fitted_model.get_forecast(steps=1, exog=next_exog)
    conf_80 = forecast_80.conf_int(alpha=0.20)
    conf_95 = fitted_model.get_forecast(steps=1, exog=next_exog).conf_int(alpha=0.05)
    prediction_raw = float(cast(float, forecast_80.predicted_mean.iloc[0]))
    return {
        "origin_date": row["origin_date"],
        "target_date": row["target_date"],
        "actual": float(cast(float, row[target_column])),
        "prediction_raw": prediction_raw,
        "prediction_rounded": float(np.rint(prediction_raw)),
        "lower_80": float(cast(float, conf_80.iloc[0, 0])),
        "upper_80": float(cast(float, conf_80.iloc[0, 1])),
        "lower_95": float(cast(float, conf_95.iloc[0, 0])),
        "upper_95": float(cast(float, conf_95.iloc[0, 1])),
        "train_rows_used": int(train_rows_used),
        "used_order": str(tuple(cast(list[int], params["order"]))),
        "used_seasonal_order": str(tuple(cast(list[int], params["seasonal_order"]))),
        "used_trend": str(params["trend"]),
    }


def batch_forecast(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Predictions batch sur un horizon futur avec un modele ajuste une seule fois."""

    fitted_model = fit_sarimax_model(
        history_df=history_df,
        feature_columns=feature_columns,
        target_column=target_column,
        params=params,
    )
    future_records = cast(list[dict[str, Any]], future_df.reset_index(drop=True).to_dict(orient="records"))
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(future_records):
        next_row = future_df.iloc[[row_index]]
        rows.append(_forecast_row(fitted_model, next_row, feature_columns, target_column, row, len(history_df), params))
    return pd.DataFrame(rows)


def rolling_forecast(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    params: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Predictions rolling-origin en re-ajustant SARIMAX avant chaque ligne future."""

    ordered_history = history_df.reset_index(drop=True).copy()
    ordered_future = future_df.reset_index(drop=True).copy()
    future_records = cast(list[dict[str, Any]], ordered_future.to_dict(orient="records"))
    prediction_rows: list[dict[str, Any]] = []
    total_rows = len(ordered_future)
    for row_index, row in enumerate(future_records):
        fitted_model = fit_sarimax_model(
            history_df=ordered_history,
            feature_columns=feature_columns,
            target_column=target_column,
            params=params,
        )
        next_row = ordered_future.iloc[[row_index]]
        prediction_row = _forecast_row(
            fitted_model,
            next_row,
            feature_columns,
            target_column,
            row,
            len(ordered_history),
            params,
        )
        prediction_rows.append(prediction_row)
        if progress_callback is not None:
            progress_callback(row_index, total_rows, prediction_row)
        ordered_history = pd.concat([ordered_history, next_row], ignore_index=True)
    return pd.DataFrame(prediction_rows)


def model_complexity_penalty(params: dict[str, Any], feature_count: int) -> float:
    """Retourne une penalite simple basee sur la somme des ordres et le nombre de regressors."""

    order_values = cast(list[int], params["order"])
    seasonal_values = cast(list[int], params["seasonal_order"])
    return float(sum(order_values) + sum(seasonal_values[:3]) + max(0, feature_count))

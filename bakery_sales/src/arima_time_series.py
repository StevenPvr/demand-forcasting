from __future__ import annotations

"""Utilitaires partages pour les modeles SARIMA temporels."""

from pathlib import Path
from typing import Any, Callable, cast

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper


ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def sarima_params_payload(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
) -> dict[str, Any]:
    """Construit un payload SARIMA serialisable."""

    return {
        "order": [int(value) for value in order],
        "seasonal_order": [int(value) for value in seasonal_order],
        "trend": str(trend),
    }


def build_sarima_model(
    history_df: pd.DataFrame,
    target_column: str,
    params: dict[str, Any],
) -> SARIMAX:
    """Construit un modele SARIMA pret a etre ajuste."""

    order_values = cast(list[int], params["order"])
    seasonal_values = cast(list[int], params["seasonal_order"])
    return SARIMAX(
        endog=history_df[target_column].astype(float),
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


def fit_sarima_model(
    history_df: pd.DataFrame,
    target_column: str,
    params: dict[str, Any],
) -> SARIMAXResultsWrapper:
    """Ajuste un modele SARIMA sur l'historique disponible."""

    fitted_model = build_sarima_model(
        history_df=history_df,
        target_column=target_column,
        params=params,
    )
    return cast(
        SARIMAXResultsWrapper,
        fitted_model.fit(disp=False),
    )


def save_sarima_model(
    fitted_model: SARIMAXResultsWrapper,
    output_path: Path,
) -> None:
    """Persiste un modele SARIMA sur disque."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_model, output_path)


def _prediction_row(
    row: dict[str, Any],
    train_rows_used: int,
    params: dict[str, Any],
    prediction_raw: float,
    lower_80: float,
    upper_80: float,
    lower_95: float,
    upper_95: float,
    origin_date: str | None,
    target_column: str,
    date_column: str,
) -> dict[str, Any]:
    """Construit la prediction a un pas et ses intervalles."""

    prediction_row = {
        "origin_date": origin_date,
        "target_date": str(row[date_column]),
        "actual": float(cast(float, row[target_column])),
        "prediction_raw": prediction_raw,
        "prediction_rounded": float(np.rint(prediction_raw)),
        "lower_80": lower_80,
        "upper_80": upper_80,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "train_rows_used": int(train_rows_used),
        "used_order": str(tuple(cast(list[int], params["order"]))),
        "used_seasonal_order": str(tuple(cast(list[int], params["seasonal_order"]))),
        "used_trend": str(params["trend"]),
    }
    if "product" in row:
        prediction_row["product"] = str(row["product"])
    return prediction_row


def batch_forecast(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    date_column: str,
    target_column: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Predictions batch sur un horizon futur avec un modele ajuste une seule fois."""

    fitted_model = fit_sarima_model(history_df=history_df, target_column=target_column, params=params)
    rows: list[dict[str, Any]] = []
    future_records = cast(list[dict[str, Any]], future_df.reset_index(drop=True).to_dict(orient="records"))
    forecast_80 = fitted_model.get_forecast(steps=len(future_records))
    conf_80 = forecast_80.conf_int(alpha=0.20).reset_index(drop=True)
    conf_95 = fitted_model.get_forecast(steps=len(future_records)).conf_int(alpha=0.05).reset_index(drop=True)
    predicted_mean = forecast_80.predicted_mean.reset_index(drop=True)
    origin_date = None
    if len(history_df) > 0 and date_column in history_df.columns:
        origin_date = str(history_df.iloc[-1][date_column])
    for row_index, row in enumerate(future_records):
        rows.append(
            _prediction_row(
                row=row,
                train_rows_used=len(history_df),
                params=params,
                prediction_raw=float(cast(float, predicted_mean.iloc[row_index])),
                lower_80=float(cast(float, conf_80.iloc[row_index, 0])),
                upper_80=float(cast(float, conf_80.iloc[row_index, 1])),
                lower_95=float(cast(float, conf_95.iloc[row_index, 0])),
                upper_95=float(cast(float, conf_95.iloc[row_index, 1])),
                origin_date=origin_date,
                target_column=target_column,
                date_column=date_column,
            )
        )
    return pd.DataFrame(rows)


def rolling_forecast(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    date_column: str,
    target_column: str,
    params: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Predictions rolling-origin en re-ajustant SARIMA avant chaque ligne future."""

    ordered_history = history_df.reset_index(drop=True).copy()
    ordered_future = future_df.reset_index(drop=True).copy()
    future_records = cast(list[dict[str, Any]], ordered_future.to_dict(orient="records"))
    prediction_rows: list[dict[str, Any]] = []
    total_rows = len(ordered_future)
    for row_index, row in enumerate(future_records):
        fitted_model = fit_sarima_model(
            history_df=ordered_history,
            target_column=target_column,
            params=params,
        )
        forecast_80 = fitted_model.get_forecast(steps=1)
        conf_80 = forecast_80.conf_int(alpha=0.20)
        conf_95 = fitted_model.get_forecast(steps=1).conf_int(alpha=0.05)
        origin_date = None
        if len(ordered_history) > 0 and date_column in ordered_history.columns:
            origin_date = str(ordered_history.iloc[-1][date_column])
        prediction_row = _prediction_row(
            row=row,
            train_rows_used=len(ordered_history),
            params=params,
            prediction_raw=float(cast(float, forecast_80.predicted_mean.iloc[0])),
            lower_80=float(cast(float, conf_80.iloc[0, 0])),
            upper_80=float(cast(float, conf_80.iloc[0, 1])),
            lower_95=float(cast(float, conf_95.iloc[0, 0])),
            upper_95=float(cast(float, conf_95.iloc[0, 1])),
            origin_date=origin_date,
            target_column=target_column,
            date_column=date_column,
        )
        prediction_rows.append(prediction_row)
        if progress_callback is not None:
            progress_callback(row_index, total_rows, prediction_row)
        ordered_history = pd.concat([ordered_history, ordered_future.iloc[[row_index]]], ignore_index=True)
    return pd.DataFrame(prediction_rows)


def model_complexity_penalty(params: dict[str, Any]) -> float:
    """Retourne une penalite simple basee sur la somme des ordres."""

    order_values = cast(list[int], params["order"])
    seasonal_values = cast(list[int], params["seasonal_order"])
    return float(sum(order_values) + sum(seasonal_values[:3]))

from __future__ import annotations

"""Genere un panel de baselines statistiques causales pour tous les produits."""

import json
import sys
from pathlib import Path
from typing import Any, Callable, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.constants_arima import CSV_ENCODING, DATE_COLUMN, FIELD_BASELINE_NAME, PRODUCT_COLUMN, TARGET_COLUMN
from src.evaluation.paths_arima import STATISTICAL_BASELINES_JSON, TEST_CSV, TRAIN_CSV, VAL_CSV
from src.time_series_metrics import mae_score, mase_score, rmse_score, smape

HistoryForecaster = Callable[[list[float], pd.Timestamp], float]


def _array_mean(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Moyenne type-safe pour les reducers."""

    return float(np.mean(values))


def _array_median(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Mediane type-safe pour les reducers."""

    return float(np.median(values))


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    """Convertit une valeur quelconque en Timestamp valide."""

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("date cannot be NaT")
    return cast(pd.Timestamp, timestamp)


def _load_split(csv_path: Path) -> pd.DataFrame:
    """Charge un split temporel en preservant l'ordre chronologique par produit."""

    dataset_df = pd.read_csv(csv_path)
    dataset_df[DATE_COLUMN] = pd.to_datetime(dataset_df[DATE_COLUMN], format="%Y-%m-%d")
    ordered_df = dataset_df.sort_values([PRODUCT_COLUMN, DATE_COLUMN]).reset_index(drop=True)
    return ordered_df


def _require_input_files(csv_paths: list[Path]) -> None:
    """Verifie que les splits requis existent avant de lancer les baselines."""

    missing_paths = [path for path in csv_paths if not path.exists()]
    if not missing_paths:
        return
    missing_list = ", ".join(str(path) for path in missing_paths)
    raise RuntimeError(
        "Missing baseline input split(s): "
        f"{missing_list}. Run the preprocessing pipeline first to generate product train/val/test splits."
    )


def _safe_last(history: list[float]) -> float:
    """Retourne la derniere valeur disponible."""

    return float(history[-1])


def _lag_forecaster(lag: int) -> HistoryForecaster:
    """Construit une baseline de lag causal."""

    def forecast(history: list[float], _: pd.Timestamp) -> float:
        resolved_lag = max(1, lag)
        if len(history) < resolved_lag:
            return _safe_last(history)
        return float(history[-resolved_lag])

    return forecast


def _rolling_mean_forecaster(window: int) -> HistoryForecaster:
    """Construit une baseline de moyenne mobile causale."""

    def forecast(history: list[float], _: pd.Timestamp) -> float:
        resolved_window = max(1, min(window, len(history)))
        return float(np.mean(history[-resolved_window:]))

    return forecast


def _rolling_median_forecaster(window: int) -> HistoryForecaster:
    """Construit une baseline de mediane mobile causale."""

    def forecast(history: list[float], _: pd.Timestamp) -> float:
        resolved_window = max(1, min(window, len(history)))
        return float(np.median(history[-resolved_window:]))

    return forecast


def _ewm_forecaster(span: int) -> HistoryForecaster:
    """Construit une baseline de moyenne exponentielle causale."""

    def forecast(history: list[float], _: pd.Timestamp) -> float:
        history_series = pd.Series(history, dtype=float)
        return float(history_series.ewm(span=max(1, span), adjust=False).mean().iloc[-1])

    return forecast


def _expanding_mean_forecaster(history: list[float], _: pd.Timestamp) -> float:
    """Retourne la moyenne cumulative."""

    return float(np.mean(history))


def _expanding_median_forecaster(history: list[float], _: pd.Timestamp) -> float:
    """Retourne la mediane cumulative."""

    return float(np.median(history))


def _same_weekday_history(history_df: pd.DataFrame, forecast_date: pd.Timestamp) -> list[float]:
    """Retourne l'historique des memes jours de semaine."""

    date_series = cast(pd.Series, history_df[DATE_COLUMN])
    mask = date_series.dt.dayofweek == forecast_date.dayofweek
    values = cast(pd.Series, history_df.loc[mask, TARGET_COLUMN]).astype(float).tolist()
    return values


def _trimmed_mean_forecaster(window: int, trim_ratio: float = 0.2) -> HistoryForecaster:
    """Construit une baseline moyenne tronquee sur une fenetre recente."""

    def forecast(history: list[float], _: pd.Timestamp) -> float:
        resolved_window = max(1, min(window, len(history)))
        values = np.sort(np.asarray(history[-resolved_window:], dtype=float))
        trim_count = int(np.floor(len(values) * trim_ratio))
        if (len(values) - (2 * trim_count)) <= 0:
            return float(np.mean(values))
        trimmed = values[trim_count : len(values) - trim_count]
        return float(np.mean(trimmed))

    return forecast


def _blend_forecaster(weights: dict[str, float]) -> Callable[[dict[str, float]], float]:
    """Construit un blend lineaire de sous-baselines."""

    def forecast(predictions: dict[str, float]) -> float:
        return float(sum(predictions[name] * weight for name, weight in weights.items()))

    return forecast


def _year_ago_date_candidates(forecast_date: pd.Timestamp) -> list[pd.Timestamp]:
    """Retourne quelques dates candidates autour du meme jour l'annee precedente."""

    return [
        cast(pd.Timestamp, forecast_date - pd.Timedelta(days=365)),
        cast(pd.Timestamp, forecast_date - pd.Timedelta(days=364)),
        cast(pd.Timestamp, forecast_date - pd.Timedelta(days=366)),
    ]


def _same_day_last_year_value(
    history_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
) -> float | None:
    """Recupere la cible du meme jour l'annee precedente si disponible."""

    indexed_history = history_df.set_index(DATE_COLUMN)
    for candidate_date in _year_ago_date_candidates(forecast_date):
        if candidate_date in indexed_history.index:
            return float(cast(float, indexed_history.at[candidate_date, TARGET_COLUMN]))
    return None


def _predict_baseline(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictor: HistoryForecaster | Callable[[dict[str, float]], float],
    precomputed_names: list[str] | None = None,
    prediction_cache: dict[str, list[float]] | None = None,
) -> pd.Series:
    """Genere les predictions causales d'une baseline sur le split test."""

    observed_history_df = history_df.copy()
    observed_values = cast(pd.Series, observed_history_df[TARGET_COLUMN]).astype(float).tolist()
    predictions: list[float] = []
    future_records = cast(list[dict[str, Any]], test_df.to_dict(orient="records"))
    for row in future_records:
        forecast_date = _coerce_timestamp(row[DATE_COLUMN])
        if precomputed_names is None:
            direct_predictor = cast(HistoryForecaster, predictor)
            predicted_value = float(direct_predictor(observed_values, forecast_date))
        else:
            if prediction_cache is None:
                raise RuntimeError("prediction_cache is required for blended baselines")
            component_predictions = {
                name: prediction_cache[name][len(predictions)]
                for name in precomputed_names
            }
            blended_predictor = cast(Callable[[dict[str, float]], float], predictor)
            predicted_value = float(blended_predictor(component_predictions))
        predictions.append(predicted_value)
        appended_row = pd.DataFrame([row]).assign(
            **{DATE_COLUMN: lambda df: pd.to_datetime(df[DATE_COLUMN])}
        )
        observed_history_df = pd.concat([observed_history_df, appended_row], ignore_index=True)
        observed_values.append(float(cast(float, row[TARGET_COLUMN])))
    return pd.Series(predictions, dtype=float)


def _same_weekday_predictor(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reducer: Callable[[np.ndarray[Any, np.dtype[np.float64]]], float],
    window: int | None,
) -> pd.Series:
    """Genere une baseline causale basee sur les memes jours de semaine."""

    observed_history_df = history_df.copy()
    predictions: list[float] = []
    future_records = cast(list[dict[str, Any]], test_df.to_dict(orient="records"))
    for row in future_records:
        forecast_date = _coerce_timestamp(row[DATE_COLUMN])
        weekday_history = _same_weekday_history(observed_history_df, forecast_date)
        if not weekday_history:
            predictions.append(float(cast(pd.Series, observed_history_df[TARGET_COLUMN]).astype(float).iloc[-1]))
        else:
            if window is not None:
                weekday_history = weekday_history[-window:]
            predictions.append(float(reducer(np.asarray(weekday_history, dtype=float))))
        appended_row = pd.DataFrame([row]).assign(
            **{DATE_COLUMN: lambda df: pd.to_datetime(df[DATE_COLUMN])}
        )
        observed_history_df = pd.concat([observed_history_df, appended_row], ignore_index=True)
    return pd.Series(predictions, dtype=float)


def _same_day_last_year_predictions(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.Series:
    """Baseline: meme jour de l'annee precedente."""

    observed_history_df = history_df.copy()
    predictions: list[float] = []
    future_records = cast(list[dict[str, Any]], test_df.to_dict(orient="records"))
    for row in future_records:
        forecast_date = _coerce_timestamp(row[DATE_COLUMN])
        year_ago_value = _same_day_last_year_value(observed_history_df, forecast_date)
        if year_ago_value is None:
            year_ago_value = float(cast(pd.Series, observed_history_df[TARGET_COLUMN]).astype(float).iloc[-1])
        predictions.append(year_ago_value)
        appended_row = pd.DataFrame([row]).assign(
            **{DATE_COLUMN: lambda df: pd.to_datetime(df[DATE_COLUMN])}
        )
        observed_history_df = pd.concat([observed_history_df, appended_row], ignore_index=True)
    return pd.Series(predictions, dtype=float)


def _baseline_metrics(
    name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    insample_series: pd.Series,
) -> dict[str, float | str]:
    """Calcule les metriques d'une baseline."""

    errors = y_true.astype(float).reset_index(drop=True) - y_pred.astype(float).reset_index(drop=True)
    return {
        "name": name,
        "mae": mae_score(y_true, y_pred),
        "rmse": rmse_score(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase_score(y_true, y_pred, insample_series, seasonal_period=7),
        "bias_mean_error": float(errors.mean()),
        "mean_prediction": float(y_pred.mean()),
        "std_prediction": float(np.std(np.asarray(y_pred, dtype=float), ddof=1)),
    }


def _prediction_rows_payload(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> list[dict[str, float | str]]:
    """Construit un payload ligne a ligne pour une baseline."""

    target_dates = cast(pd.Series, test_df[DATE_COLUMN]).dt.strftime("%Y-%m-%d")
    payload_df = pd.DataFrame(
        {
            "product": cast(pd.Series, test_df[PRODUCT_COLUMN]).astype(str),
            "origin_date": (cast(pd.Series, test_df[DATE_COLUMN]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
            "target_date": target_dates,
            "actual": y_true.astype(float),
            "prediction": y_pred.astype(float),
        }
    )
    payload_df["absolute_error"] = (payload_df["actual"] - payload_df["prediction"]).abs()
    return cast(list[dict[str, float | str]], payload_df.to_dict(orient="records"))


def _baseline_definitions() -> tuple[
    dict[str, HistoryForecaster],
    dict[str, tuple[list[str], Callable[[dict[str, float]], float]]],
]:
    """Retourne le panel de baselines directes et blendées."""

    direct_baselines: dict[str, HistoryForecaster] = {
        "naive_lag_1": _lag_forecaster(1),
        "seasonal_naive_lag_7": _lag_forecaster(7),
        "lag_14": _lag_forecaster(14),
        "rolling_mean_3": _rolling_mean_forecaster(3),
        "rolling_mean_7": _rolling_mean_forecaster(7),
        "rolling_mean_14": _rolling_mean_forecaster(14),
        "rolling_median_7": _rolling_median_forecaster(7),
        "rolling_median_14": _rolling_median_forecaster(14),
        "ewm_span_3": _ewm_forecaster(3),
        "ewm_span_7": _ewm_forecaster(7),
        "ewm_span_14": _ewm_forecaster(14),
        "expanding_mean": _expanding_mean_forecaster,
        "expanding_median": _expanding_median_forecaster,
        "trimmed_mean_7": _trimmed_mean_forecaster(7),
    }
    blended_baselines: dict[str, tuple[list[str], Callable[[dict[str, float]], float]]] = {
        "blend_lag_1_lag_7_50_50": (
            ["naive_lag_1", "seasonal_naive_lag_7"],
            _blend_forecaster({"naive_lag_1": 0.5, "seasonal_naive_lag_7": 0.5}),
        ),
        "blend_roll_mean_7_ewm_7_50_50": (
            ["rolling_mean_7", "ewm_span_7"],
            _blend_forecaster({"rolling_mean_7": 0.5, "ewm_span_7": 0.5}),
        ),
    }
    return direct_baselines, blended_baselines


def _same_weekday_baselines(history_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Construit les baselines dépendantes du calendrier hebdomadaire."""

    return {
        "same_weekday_mean_4": _same_weekday_predictor(history_df, test_df, _array_mean, 4),
        "same_weekday_median_4": _same_weekday_predictor(history_df, test_df, _array_median, 4),
        "same_weekday_mean_expanding": _same_weekday_predictor(history_df, test_df, _array_mean, None),
        "same_weekday_median_expanding": _same_weekday_predictor(history_df, test_df, _array_median, None),
        "same_day_last_year": _same_day_last_year_predictions(history_df, test_df),
    }


def _ranked_rows(metrics_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """Trie les baselines par MAE puis RMSE et annote leur rang."""

    ranked_rows = sorted(metrics_rows, key=lambda row: (float(row["mae"]), float(row["rmse"]), str(row["name"])))
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank_mae"] = rank
    return ranked_rows


def _product_payload(
    product_name: str,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, pd.Series], pd.Series, pd.Series]:
    """Calcule tout le panel de baselines pour un produit donne."""

    y_true = cast(pd.Series, test_df[TARGET_COLUMN].astype(float).reset_index(drop=True))
    insample_series = cast(pd.Series, history_df[TARGET_COLUMN].astype(float).reset_index(drop=True))
    direct_baselines, blended_baselines = _baseline_definitions()
    prediction_cache: dict[str, list[float]] = {}
    prediction_series_by_name: dict[str, pd.Series] = {}
    metrics_rows: list[dict[str, float | str]] = []

    for name, predictor in direct_baselines.items():
        predictions = _predict_baseline(history_df, test_df, predictor)
        prediction_cache[name] = predictions.tolist()
        prediction_series_by_name[name] = predictions
        metrics_rows.append(_baseline_metrics(name, y_true, predictions, insample_series))

    for name, predictions in _same_weekday_baselines(history_df, test_df).items():
        prediction_cache[name] = predictions.tolist()
        prediction_series_by_name[name] = predictions
        metrics_rows.append(_baseline_metrics(name, y_true, predictions, insample_series))

    for name, (component_names, predictor) in blended_baselines.items():
        predictions = _predict_baseline(
            history_df=history_df,
            test_df=test_df,
            predictor=predictor,
            precomputed_names=component_names,
            prediction_cache=prediction_cache,
        )
        prediction_series_by_name[name] = predictions
        metrics_rows.append(_baseline_metrics(name, y_true, predictions, insample_series))

    ranked_rows = _ranked_rows(metrics_rows)
    best_baseline_name = str(ranked_rows[0]["name"])
    best_baseline_predictions = prediction_series_by_name[best_baseline_name]
    best_payload = {
        **ranked_rows[0],
        "product": product_name,
        "prediction_rows": _prediction_rows_payload(
            test_df=test_df,
            y_true=y_true,
            y_pred=best_baseline_predictions,
        ),
    }
    field_baseline_predictions = prediction_series_by_name[FIELD_BASELINE_NAME]
    field_metrics_row = next(row for row in ranked_rows if str(row["name"]) == FIELD_BASELINE_NAME)
    field_payload = {
        **field_metrics_row,
        "product": product_name,
        "prediction_rows": _prediction_rows_payload(
            test_df=test_df,
            y_true=y_true,
            y_pred=field_baseline_predictions,
        ),
    }
    return best_payload, field_payload, prediction_series_by_name, y_true, insample_series


def run_statistical_baselines(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    output_json: Path,
) -> dict[str, Any]:
    """Calcule un panel de baselines causales sur le split test pour tous les produits."""

    _require_input_files([train_csv, val_csv, test_csv])
    train_df = _load_split(train_csv)
    val_df = _load_split(val_csv)
    test_df = _load_split(test_csv)
    history_df = pd.concat([train_df, val_df], ignore_index=True)
    history_products = {
        str(product_name): product_df.reset_index(drop=True)
        for product_name, product_df in history_df.groupby(PRODUCT_COLUMN, sort=True)
    }
    test_products = {
        str(product_name): product_df.reset_index(drop=True)
        for product_name, product_df in test_df.groupby(PRODUCT_COLUMN, sort=True)
    }

    aggregate_predictions: dict[str, list[pd.DataFrame]] = {}
    aggregate_truth: dict[str, list[pd.Series]] = {}
    aggregate_insample: dict[str, list[pd.Series]] = {}
    per_product_best_baselines: dict[str, Any] = {}
    per_product_field_baselines: dict[str, Any] = {}

    for product_name, product_test_df in test_products.items():
        product_best_payload, product_field_payload, prediction_series_by_name, y_true, insample_series = _product_payload(
            product_name=product_name,
            history_df=history_products[product_name],
            test_df=product_test_df,
        )
        per_product_best_baselines[product_name] = product_best_payload
        per_product_field_baselines[product_name] = product_field_payload
        for baseline_name, y_pred in prediction_series_by_name.items():
            aggregate_predictions.setdefault(baseline_name, []).append(
                pd.DataFrame(
                    {
                        PRODUCT_COLUMN: [product_name] * len(y_pred),
                        "y_true": y_true.astype(float).to_list(),
                        "y_pred": y_pred.astype(float).to_list(),
                    }
                )
            )
            aggregate_truth.setdefault(baseline_name, []).append(y_true)
            aggregate_insample.setdefault(baseline_name, []).append(insample_series)

    aggregate_metrics_rows: list[dict[str, float | str]] = []
    aggregate_predictions_by_name: dict[str, pd.Series] = {}
    for baseline_name, prediction_frames in aggregate_predictions.items():
        merged_predictions_df = pd.concat(prediction_frames, ignore_index=True)
        y_true = cast(pd.Series, merged_predictions_df["y_true"]).astype(float)
        y_pred = cast(pd.Series, merged_predictions_df["y_pred"]).astype(float)
        insample_values = pd.concat(
            [series.reset_index(drop=True) for series in aggregate_insample[baseline_name]],
            ignore_index=True,
        )
        aggregate_metrics_rows.append(
            _baseline_metrics(
                baseline_name,
                y_true=y_true,
                y_pred=y_pred,
                insample_series=cast(pd.Series, insample_values.astype(float)),
            )
        )
        aggregate_predictions_by_name[baseline_name] = y_pred

    ranked_rows = _ranked_rows(aggregate_metrics_rows)
    best_baseline_name = str(ranked_rows[0]["name"])
    best_prediction_frames = pd.concat(aggregate_predictions[best_baseline_name], ignore_index=True)
    best_baseline_payload = {
        **ranked_rows[0],
        "prediction_rows": _prediction_rows_payload(
            test_df=test_df.reset_index(drop=True),
            y_true=cast(pd.Series, best_prediction_frames["y_true"]).astype(float).reset_index(drop=True),
            y_pred=cast(pd.Series, best_prediction_frames["y_pred"]).astype(float).reset_index(drop=True),
        ),
    }

    payload: dict[str, Any] = {
        "date_column": DATE_COLUMN,
        "product_column": PRODUCT_COLUMN,
        "target_column": TARGET_COLUMN,
        "history_rows": int(len(history_df)),
        "test_rows": int(len(test_df)),
        "product_count": int(len(test_products)),
        "history_range": {
            "start": str(cast(pd.Timestamp, cast(pd.Series, history_df[DATE_COLUMN]).min()).date()),
            "end": str(cast(pd.Timestamp, cast(pd.Series, history_df[DATE_COLUMN]).max()).date()),
        },
        "test_range": {
            "start": str(cast(pd.Timestamp, cast(pd.Series, test_df[DATE_COLUMN]).min()).date()),
            "end": str(cast(pd.Timestamp, cast(pd.Series, test_df[DATE_COLUMN]).max()).date()),
        },
        "baseline_count": int(len(ranked_rows)),
        "field_baseline_name": FIELD_BASELINE_NAME,
        "best_baseline": best_baseline_payload,
        "baselines_ranked_by_mae": ranked_rows,
        "per_product_best_baselines": per_product_best_baselines,
        "per_product_field_baselines": per_product_field_baselines,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding=CSV_ENCODING)
    return payload


def main() -> None:
    """Execute la generation des baselines statistiques sur les splits ARIMA actifs."""

    run_statistical_baselines(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        output_json=STATISTICAL_BASELINES_JSON,
    )


if __name__ == "__main__":
    main()

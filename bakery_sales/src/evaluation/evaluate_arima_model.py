from __future__ import annotations

"""Evaluation du meilleur modele ARIMA par produit sur le split test."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd

_CACHE_DIR: Path = Path(tempfile.gettempdir()) / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir())))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from src.arima_time_series import fit_sarima_model, rolling_forecast, save_sarima_model
from src.evaluation.constants_arima import CSV_ENCODING, DATE_COLUMN, DEFAULT_EVALUATION_JOBS, PRODUCT_COLUMN, TARGET_COLUMN
from src.time_series_metrics import interval_coverage, lag_baseline_predictions, mae_score, mase_score, rmse_score, smape

LOGGER: logging.Logger = logging.getLogger(__name__)


def _configure_plot_style() -> None:
    """Applique un style sobre pour les figures d'evaluation."""

    sns.set_theme(style="ticks", context="paper", palette="colorblind")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _load_split(csv_path: Path) -> pd.DataFrame:
    """Charge un split et le trie par produit puis date."""

    dataset_df = pd.read_csv(csv_path)
    dataset_df[DATE_COLUMN] = pd.to_datetime(dataset_df[DATE_COLUMN], format="%Y-%m-%d")
    ordered_df = dataset_df.sort_values([PRODUCT_COLUMN, DATE_COLUMN]).reset_index(drop=True)
    ordered_df[DATE_COLUMN] = ordered_df[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    return ordered_df


def _load_best_payload(best_params_json: Path) -> dict[str, Any]:
    """Charge le JSON des meilleurs hyperparametres par produit."""

    return json.loads(best_params_json.read_text(encoding=CSV_ENCODING))


def _product_frames(dataset_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Retourne un dictionnaire produit -> sous-serie chronologique."""

    return {
        str(product_name): product_df.reset_index(drop=True)
        for product_name, product_df in dataset_df.groupby(PRODUCT_COLUMN, sort=True)
    }


def _log_progress(product_name: str, index: int, total_rows: int, prediction_row: dict[str, Any]) -> None:
    """Loggue la progression de l'evaluation produit par produit."""

    if total_rows <= 0:
        return
    if index not in {0, total_rows - 1} and (index + 1) % 50 != 0:
        return
    LOGGER.info(
        "ARIMA evaluation progress product=%s %d/%d (%.1f%%): actual=%.3f predicted=%.3f",
        product_name,
        index + 1,
        total_rows,
        ((index + 1) / total_rows) * 100.0,
        prediction_row["actual"],
        prediction_row["prediction_raw"],
    )


def rolling_origin_predictions(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_payload: dict[str, Any],
    n_jobs: int = DEFAULT_EVALUATION_JOBS,
) -> pd.DataFrame:
    """Predictions rolling-origin en reentrainant un ARIMA distinct par produit."""

    del n_jobs
    history_products = _product_frames(history_df)
    test_products = _product_frames(test_df)
    prediction_parts: list[pd.DataFrame] = []
    for product_name, product_test_df in test_products.items():
        if product_name not in best_payload["product_models"]:
            continue
        product_params = best_payload["product_models"][product_name]["best_params"]
        product_predictions = rolling_forecast(
            history_df=history_products[product_name],
            future_df=product_test_df,
            date_column=DATE_COLUMN,
            target_column=TARGET_COLUMN,
            params=product_params,
            progress_callback=lambda index, total, row, product_name=product_name: _log_progress(
                product_name, index, total, row
            ),
        )
        prediction_parts.append(product_predictions)
    return pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()


def _metrics_payload(history_df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict[str, float | int]:
    """Construit les metriques d'evaluation pour une seule serie produit."""

    y_true = cast(pd.Series, predictions_df["actual"]).astype(float)
    y_pred = cast(pd.Series, predictions_df["prediction_raw"]).astype(float)
    history_values = cast(pd.Series, history_df[TARGET_COLUMN]).astype(float)
    lag_1_baseline = lag_baseline_predictions(history_values, y_true, lag=1)
    seasonal_baseline = lag_baseline_predictions(history_values, y_true, lag=7)
    return {
        "test_rows": int(len(predictions_df)),
        "mae": mae_score(y_true, y_pred),
        "rmse": rmse_score(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase_score(y_true, y_pred, history_values, seasonal_period=7),
        "naive_lag_1_mae": mae_score(y_true, lag_1_baseline),
        "seasonal_naive_lag_7_mae": mae_score(y_true, seasonal_baseline),
        "coverage_80": interval_coverage(
            y_true,
            cast(pd.Series, predictions_df["lower_80"]),
            cast(pd.Series, predictions_df["upper_80"]),
        ),
        "coverage_95": interval_coverage(
            y_true,
            cast(pd.Series, predictions_df["lower_95"]),
            cast(pd.Series, predictions_df["upper_95"]),
        ),
    }


def _diagnostics_payload(predictions_df: pd.DataFrame) -> dict[str, float | int]:
    """Construit des diagnostics residuels elementaires."""

    residuals = cast(pd.Series, predictions_df["actual"]).astype(float) - cast(
        pd.Series, predictions_df["prediction_raw"]
    ).astype(float)
    resolved_lag = max(min(len(residuals) - 1, 10), 1)
    ljung_box = acorr_ljungbox(residuals, lags=[resolved_lag], return_df=True)
    return {
        "row_count": int(len(residuals)),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std(ddof=0)),
        "ljung_box_stat": float(ljung_box["lb_stat"].iloc[0]),
        "ljung_box_pvalue": float(ljung_box["lb_pvalue"].iloc[0]),
    }


def _per_product_metrics(
    history_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calcule les metriques et diagnostics pour chaque produit."""

    history_products = _product_frames(history_df)
    metrics_payload: dict[str, Any] = {}
    diagnostics_payload: dict[str, Any] = {}
    for product_name, product_predictions_df in predictions_df.groupby(PRODUCT_COLUMN, sort=True):
        metrics_payload[str(product_name)] = _metrics_payload(
            history_products[str(product_name)],
            product_predictions_df.reset_index(drop=True),
        )
        diagnostics_payload[str(product_name)] = _diagnostics_payload(
            product_predictions_df.reset_index(drop=True)
        )
    return metrics_payload, diagnostics_payload


def _overall_metrics(per_product_metrics: dict[str, Any], predictions_df: pd.DataFrame) -> dict[str, float | int]:
    """Agrege les metriques globales a partir des predictions concatenees."""

    actual_series = cast(pd.Series, predictions_df["actual"])
    predicted_series = cast(pd.Series, predictions_df["prediction_raw"])
    lower_80 = cast(pd.Series, predictions_df["lower_80"])
    upper_80 = cast(pd.Series, predictions_df["upper_80"])
    lower_95 = cast(pd.Series, predictions_df["lower_95"])
    upper_95 = cast(pd.Series, predictions_df["upper_95"])
    return {
        "test_rows": int(len(predictions_df)),
        "mae": mae_score(actual_series, predicted_series),
        "rmse": rmse_score(actual_series, predicted_series),
        "smape": smape(actual_series, predicted_series),
        "coverage_80": interval_coverage(actual_series, lower_80, upper_80),
        "coverage_95": interval_coverage(actual_series, lower_95, upper_95),
        "mean_product_mae": float(np.mean([payload["mae"] for payload in per_product_metrics.values()])),
    }


def _plot_actual_vs_predicted(predictions_df: pd.DataFrame, output_png: Path) -> None:
    """Trace les ventes reelles et predites pour chaque produit teste."""

    products = sorted(str(product) for product in predictions_df[PRODUCT_COLUMN].unique().tolist())
    fig, axes = plt.subplots(len(products), 1, figsize=(12, max(4, 3 * len(products))), squeeze=False)
    for axis, product_name in zip(axes.flat, products, strict=False):
        product_df = predictions_df.loc[predictions_df[PRODUCT_COLUMN] == product_name].reset_index(drop=True)
        axis.plot(product_df["target_date"], product_df["actual"], label="Actual", linewidth=1.4, color="#0072B2")
        axis.plot(product_df["target_date"], product_df["prediction_raw"], label="Predicted", linewidth=1.1, color="#D55E00")
        axis.set_title(f"Test set: actual vs predicted for {product_name}")
        axis.tick_params(axis="x", rotation=30)
        axis.legend(frameon=False)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals(predictions_df: pd.DataFrame, output_png: Path) -> None:
    """Trace les residus concatènes sur le split test."""

    residuals = predictions_df["actual"] - predictions_df["prediction_raw"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(residuals)), residuals, color="#0072B2", linewidth=1.2)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_title("Test residuals over concatenated product forecasts")
    ax.set_xlabel("Forecast row index")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals_qq(predictions_df: pd.DataFrame, output_png: Path) -> None:
    """Trace le QQ plot des residus concatènes."""

    residuals = predictions_df["actual"] - predictions_df["prediction_raw"]
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Residual QQ plot")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals_acf_pacf(predictions_df: pd.DataFrame, output_png: Path) -> None:
    """Trace les ACF/PACF des residus concatènes."""

    residuals = predictions_df["actual"] - predictions_df["prediction_raw"]
    acf_lags = max(min(len(residuals) - 1, 10), 1)
    pacf_lags = min(10, max((len(residuals) // 2) - 1, 0))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(residuals, lags=acf_lags, ax=axes[0], zero=False)
    if pacf_lags >= 1:
        plot_pacf(residuals, lags=pacf_lags, ax=axes[1], zero=False, method="ywm")
    else:
        axes[1].text(0.5, 0.5, "Not enough residuals for PACF", ha="center", va="center")
        axes[1].set_axis_off()
    axes[0].set_title("Residual ACF")
    axes[1].set_title("Residual PACF")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def _save_fitted_models(
    history_df: pd.DataFrame,
    best_payload: dict[str, Any],
    final_model_pkl: Path,
) -> None:
    """Ajuste et sauvegarde le modele final de chaque produit sur train+val."""

    history_products = _product_frames(history_df)
    product_models: dict[str, SARIMAXResultsWrapper] = {}
    for product_name, product_payload in best_payload["product_models"].items():
        product_models[str(product_name)] = fit_sarima_model(
            history_df=history_products[str(product_name)],
            target_column=TARGET_COLUMN,
            params=product_payload["best_params"],
        )
    final_model_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(product_models, final_model_pkl)


def run_evaluation(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    best_params_json: Path,
    metrics_json: Path,
    predictions_csv: Path,
    predictions_plot_png: Path,
    diagnostics_json: Path,
    model_card_json: Path,
    final_model_pkl: Path,
    residuals_plot_png: Path,
    residuals_qq_png: Path,
    residuals_acf_pacf_png: Path,
    statistical_baselines_json: Path | None = None,
    n_jobs: int = DEFAULT_EVALUATION_JOBS,
) -> dict[str, int | float | str]:
    """Entraine sur train+val puis evalue l'ARIMA de chaque produit sur le test."""

    del statistical_baselines_json
    start_time = time.perf_counter()
    _configure_plot_style()
    train_df = _load_split(train_csv)
    val_df = _load_split(val_csv)
    test_df = _load_split(test_csv)
    history_df = pd.concat([train_df, val_df], ignore_index=True)
    best_payload = _load_best_payload(best_params_json)
    predictions_df = rolling_origin_predictions(history_df=history_df, test_df=test_df, best_payload=best_payload, n_jobs=n_jobs)
    per_product_metrics, per_product_diagnostics = _per_product_metrics(history_df, predictions_df)
    overall_metrics = _overall_metrics(per_product_metrics, predictions_df)
    overall_diagnostics = _diagnostics_payload(predictions_df)
    metrics_payload = {
        "model_family": "ARIMA_BY_PRODUCT",
        "product_count": int(len(per_product_metrics)),
        "overall_metrics": overall_metrics,
        "per_product_metrics": per_product_metrics,
    }
    diagnostics_payload = {
        "product_count": int(len(per_product_diagnostics)),
        "overall_diagnostics": overall_diagnostics,
        "per_product_diagnostics": per_product_diagnostics,
    }
    _save_fitted_models(history_df, best_payload, final_model_pkl)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding=CSV_ENCODING)
    diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_json.write_text(json.dumps(diagnostics_payload, indent=2), encoding=CSV_ENCODING)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_csv, index=False, encoding=CSV_ENCODING)
    _plot_actual_vs_predicted(predictions_df, predictions_plot_png)
    _plot_residuals(predictions_df, residuals_plot_png)
    _plot_residuals_qq(predictions_df, residuals_qq_png)
    _plot_residuals_acf_pacf(predictions_df, residuals_acf_pacf_png)
    model_card_payload = {
        **best_payload,
        "uses_exogenous_features": bool(best_payload.get("uses_exogenous_features", False)),
        "feature_columns": cast(list[str], best_payload.get("feature_columns", [])),
        "feature_count": int(best_payload.get("feature_count", 0)),
        "metrics": metrics_payload,
        "diagnostics": diagnostics_payload,
        "final_model_path": str(final_model_pkl),
        "predictions_path": str(predictions_csv),
    }
    model_card_json.parent.mkdir(parents=True, exist_ok=True)
    model_card_json.write_text(json.dumps(model_card_payload, indent=2), encoding=CSV_ENCODING)
    LOGGER.info(
        "ARIMA by product evaluation completed in %.3f seconds with global MAE=%.6f RMSE=%.6f sMAPE=%.6f",
        time.perf_counter() - start_time,
        float(overall_metrics["mae"]),
        float(overall_metrics["rmse"]),
        float(overall_metrics["smape"]),
    )
    return {
        "model_family": "ARIMA_BY_PRODUCT",
        "test_rows": int(len(predictions_df)),
        "product_count": int(len(per_product_metrics)),
        "feature_count": 0,
        "mae": float(overall_metrics["mae"]),
        "rmse": float(overall_metrics["rmse"]),
        "smape": float(overall_metrics["smape"]),
    }

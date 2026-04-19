from __future__ import annotations

"""Evaluation du meilleur modele ElasticNet sur le split test."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, cast

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

from src.elasticnet_time_series import (
    feature_columns as shared_feature_columns,
    fit_elasticnet_model,
    rolling_forecast,
    save_elasticnet_model,
)
from src.evaluation.constants import (
    CSV_ENCODING,
    DEFAULT_EVALUATION_JOBS,
    TARGET_COLUMN,
    TEST_BAGUETTE_UNIT_COST_EUR,
)
from src.evaluation.paths import STATISTICAL_BASELINES_JSON
from src.time_series_metrics import (
    interval_coverage,
    lag_baseline_predictions,
    mae_score,
    mase_score,
    rmse_score,
    smape,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
PROGRESS_LOG_EVERY: int = 250
RANDOM_SEED: int = 7


def _configure_plot_style() -> None:
    """Applique un style sobre et lisible pour les figures d'evaluation."""

    sns.set_theme(style="ticks", context="paper", palette="colorblind")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _feature_columns(dataset_df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes de features du dataset journalier."""

    return shared_feature_columns(dataset_df, TARGET_COLUMN)


def _load_split(csv_path: Path) -> pd.DataFrame:
    """Charge un split temporel journalier en preservant l'ordre chronologique."""

    dataset_df = pd.read_csv(csv_path)
    return dataset_df.sort_values(["origin_date"]).reset_index(drop=True)


def _load_best_payload(best_params_json: Path) -> dict[str, Any]:
    """Charge le JSON des meilleurs hyperparametres."""

    return json.loads(best_params_json.read_text(encoding=CSV_ENCODING))


def _should_log_progress(index: int, total_rows: int, every_n: int = PROGRESS_LOG_EVERY) -> bool:
    """Indique si la progression doit etre logguee a cette iteration."""

    if total_rows <= 0:
        return False
    if index == 0 or index == total_rows - 1:
        return True
    return (index + 1) % every_n == 0


def _log_progress(index: int, total_rows: int, prediction_row: dict[str, Any]) -> None:
    """Loggue la progression de l'evaluation rolling-origin."""

    if not _should_log_progress(index=index, total_rows=total_rows):
        return
    LOGGER.info(
        "ElasticNet evaluation progress %d/%d (%.1f%%): actual=%.3f predicted=%.3f train_rows=%d",
        index + 1,
        total_rows,
        ((index + 1) / total_rows) * 100.0,
        prediction_row["actual"],
        prediction_row["prediction_raw"],
        prediction_row["train_rows_used"],
    )


def rolling_origin_predictions(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_payload: dict[str, Any],
    n_jobs: int = DEFAULT_EVALUATION_JOBS,
) -> pd.DataFrame:
    """Predictions rolling-origin en reentrainant ElasticNet avant chaque ligne test."""

    del n_jobs
    selected_feature_columns = _feature_columns(history_df)
    return rolling_forecast(
        history_df=history_df,
        future_df=test_df,
        feature_columns=selected_feature_columns,
        target_column=TARGET_COLUMN,
        params=best_payload["best_params"],
        random_seed=RANDOM_SEED,
        progress_callback=_log_progress,
    )


def _metrics_payload(
    history_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> dict[str, float | int]:
    """Construit les metriques d'evaluation sur predictions brutes."""

    y_true = cast(pd.Series, predictions_df.loc[:, "actual"])
    y_pred = cast(pd.Series, predictions_df.loc[:, "prediction_raw"])
    history_values = cast(pd.Series, history_df.loc[:, TARGET_COLUMN])
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
            cast(pd.Series, predictions_df.loc[:, "lower_80"]),
            cast(pd.Series, predictions_df.loc[:, "upper_80"]),
        ),
        "coverage_95": interval_coverage(
            y_true,
            cast(pd.Series, predictions_df.loc[:, "lower_95"]),
            cast(pd.Series, predictions_df.loc[:, "upper_95"]),
        ),
    }


def _diagnostics_payload(predictions_df: pd.DataFrame) -> dict[str, float | int]:
    """Construit les diagnostics residuels de base."""

    residuals = cast(pd.Series, predictions_df.loc[:, "actual"]) - cast(
        pd.Series, predictions_df.loc[:, "prediction_raw"]
    )
    resolved_lag = max(min(len(residuals) - 1, 10), 1)
    ljung_box = acorr_ljungbox(residuals, lags=[resolved_lag], return_df=True)
    return {
        "row_count": int(len(residuals)),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std(ddof=0)),
        "ljung_box_stat": float(ljung_box["lb_stat"].iloc[0]),
        "ljung_box_pvalue": float(ljung_box["lb_pvalue"].iloc[0]),
    }


def _load_statistical_baselines_payload(
    statistical_baselines_json: Path | None,
) -> dict[str, Any] | None:
    """Charge le JSON de baselines si disponible."""

    if statistical_baselines_json is None:
        return None
    if not statistical_baselines_json.exists():
        LOGGER.warning(
            "Statistical baselines file not found at %s; baseline savings will be skipped",
            statistical_baselines_json,
        )
        return None
    return cast(
        dict[str, Any],
        json.loads(statistical_baselines_json.read_text(encoding=CSV_ENCODING)),
    )


def _baseline_comparison_payload(
    predictions_df: pd.DataFrame,
    statistical_baselines_payload: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Enrichit les predictions avec la meilleure baseline et calcule les gains associes."""

    if statistical_baselines_payload is None:
        return predictions_df, None
    best_baseline = cast(dict[str, Any] | None, statistical_baselines_payload.get("best_baseline"))
    if not best_baseline:
        LOGGER.warning("Statistical baselines payload has no best_baseline section; skipping comparison")
        return predictions_df, None
    prediction_rows = cast(list[dict[str, Any]], best_baseline.get("prediction_rows", []))
    if len(prediction_rows) != len(predictions_df):
        LOGGER.warning(
            "Best baseline row count mismatch (%d vs %d); skipping comparison",
            len(prediction_rows),
            len(predictions_df),
        )
        return predictions_df, None
    baseline_df = pd.DataFrame(prediction_rows)
    baseline_origin_dates = baseline_df["origin_date"].astype(str).tolist()
    prediction_origin_dates = predictions_df["origin_date"].astype(str).tolist()
    baseline_target_dates = baseline_df["target_date"].astype(str).tolist()
    prediction_target_dates = predictions_df["target_date"].astype(str).tolist()
    if (baseline_origin_dates != prediction_origin_dates) or (baseline_target_dates != prediction_target_dates):
        LOGGER.warning("Best baseline dates do not align with evaluation predictions; skipping comparison")
        return predictions_df, None
    enriched_df = predictions_df.copy()
    actual_series = cast(pd.Series, enriched_df["actual"]).astype(float).reset_index(drop=True)
    model_prediction_series = cast(pd.Series, enriched_df["prediction_raw"]).astype(float).reset_index(drop=True)
    model_abs_error = (actual_series - model_prediction_series).abs()
    baseline_prediction = baseline_df["prediction"].astype(float).reset_index(drop=True)
    baseline_abs_error = baseline_df["absolute_error"].astype(float).reset_index(drop=True)
    model_mae_value = mae_score(actual_series, model_prediction_series)
    best_baseline_mae = float(best_baseline["mae"])
    rounded_model_mae = int(round(model_mae_value))
    rounded_best_baseline_mae = int(round(best_baseline_mae))
    rounded_mae_gap_baguettes = rounded_best_baseline_mae - rounded_model_mae
    test_day_count = int(len(predictions_df))
    estimated_savings_eur = float(
        rounded_mae_gap_baguettes * TEST_BAGUETTE_UNIT_COST_EUR * test_day_count
    )
    enriched_df["best_statistical_baseline_name"] = str(best_baseline["name"])
    enriched_df["best_statistical_baseline_prediction"] = baseline_prediction
    enriched_df["best_statistical_baseline_abs_error"] = baseline_abs_error
    enriched_df["model_abs_error"] = model_abs_error
    enriched_df["absolute_error_saved_vs_best_baseline"] = baseline_abs_error - model_abs_error
    savings_series = cast(pd.Series, enriched_df["absolute_error_saved_vs_best_baseline"]).astype(float)
    enriched_df["relative_error_reduction_vs_best_baseline"] = np.where(
        baseline_abs_error > 0.0,
        enriched_df["absolute_error_saved_vs_best_baseline"] / baseline_abs_error,
        0.0,
    )
    savings_payload: dict[str, Any] = {
        "best_statistical_baseline_name": str(best_baseline["name"]),
        "best_statistical_baseline_mae": best_baseline_mae,
        "model_mae": float(model_mae_value),
        "absolute_mae_saved_vs_best_baseline": float(best_baseline_mae - model_mae_value),
        "rounded_best_statistical_baseline_mae_baguettes": rounded_best_baseline_mae,
        "rounded_model_mae_baguettes": rounded_model_mae,
        "rounded_mae_gap_baguettes_vs_best_baseline": rounded_mae_gap_baguettes,
        "baguette_unit_cost_eur": float(TEST_BAGUETTE_UNIT_COST_EUR),
        "test_day_count": test_day_count,
        "estimated_savings_eur_vs_best_baseline": estimated_savings_eur,
        "total_absolute_error_saved_vs_best_baseline": float(savings_series.sum()),
        "rows_better_than_best_baseline": int((savings_series > 0.0).sum()),
        "rows_equal_to_best_baseline": int((savings_series == 0.0).sum()),
        "rows_worse_than_best_baseline": int((savings_series < 0.0).sum()),
        "relative_mae_improvement_vs_best_baseline": float(
            0.0 if best_baseline_mae <= 0.0 else ((best_baseline_mae - model_mae_value) / best_baseline_mae)
        ),
    }
    return enriched_df, savings_payload


def _plot_actual_vs_predicted(predictions_df: pd.DataFrame, predictions_plot_png: Path) -> None:
    """Trace la courbe des ventes reelles et predites sur le split test."""

    fig, ax = plt.subplots(figsize=(12, 4))
    x_axis = np.arange(len(predictions_df))
    ax.plot(x_axis, predictions_df["actual"], label="Actual sales", linewidth=1.5, color="#0072B2")
    ax.plot(x_axis, predictions_df["prediction_raw"], label="Predicted sales", linewidth=1.2, color="#D55E00")
    ax.set_title("Test set: actual vs predicted daily baguette demand")
    ax.set_xlabel("Test observation index")
    ax.set_ylabel("Demand on next day")
    ax.legend(frameon=False)
    fig.tight_layout()
    predictions_plot_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(predictions_plot_png, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals(predictions_df: pd.DataFrame, residuals_plot_png: Path) -> None:
    """Trace les residus dans le temps."""

    residuals = cast(pd.Series, predictions_df.loc[:, "actual"]) - cast(
        pd.Series, predictions_df.loc[:, "prediction_raw"]
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(residuals)), residuals, color="#0072B2", linewidth=1.2)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_title("Test residuals over time")
    ax.set_xlabel("Test observation index")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    residuals_plot_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(residuals_plot_png, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals_qq(predictions_df: pd.DataFrame, residuals_qq_png: Path) -> None:
    """Trace le QQ plot des residus."""

    residuals = cast(pd.Series, predictions_df.loc[:, "actual"]) - cast(
        pd.Series, predictions_df.loc[:, "prediction_raw"]
    )
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Residual QQ plot")
    fig.tight_layout()
    residuals_qq_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(residuals_qq_png, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals_acf_pacf(predictions_df: pd.DataFrame, residuals_acf_pacf_png: Path) -> None:
    """Trace les ACF/PACF des residus."""

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
    residuals_acf_pacf_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(residuals_acf_pacf_png, bbox_inches="tight")
    plt.close(fig)


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
    statistical_baselines_json: Path | None = STATISTICAL_BASELINES_JSON,
    n_jobs: int = DEFAULT_EVALUATION_JOBS,
) -> dict[str, int | float | str]:
    """Entraine sur train+val puis evalue en rolling-origin sur le test."""

    start_time = time.perf_counter()
    _configure_plot_style()
    train_df = _load_split(train_csv)
    val_df = _load_split(val_csv)
    test_df = _load_split(test_csv)
    history_df = pd.concat([train_df, val_df], ignore_index=True)
    best_payload = _load_best_payload(best_params_json)
    LOGGER.info(
        "Loaded evaluation inputs: train=%d rows, val=%d rows, test=%d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    predictions_df = rolling_origin_predictions(history_df=history_df, test_df=test_df, best_payload=best_payload, n_jobs=n_jobs)
    metrics_payload = _metrics_payload(history_df, predictions_df)
    diagnostics_payload = _diagnostics_payload(predictions_df)
    baselines_payload = _load_statistical_baselines_payload(statistical_baselines_json)
    predictions_df, baseline_savings_payload = _baseline_comparison_payload(predictions_df, baselines_payload)
    if baseline_savings_payload is not None:
        metrics_payload.update(baseline_savings_payload)
    business_impact_payload: dict[str, Any] | None = None
    if baseline_savings_payload is not None:
        business_impact_payload = {
            "estimated_savings_eur_vs_best_baseline": baseline_savings_payload[
                "estimated_savings_eur_vs_best_baseline"
            ],
            "rounded_mae_gap_baguettes_vs_best_baseline": baseline_savings_payload[
                "rounded_mae_gap_baguettes_vs_best_baseline"
            ],
            "rounded_model_mae_baguettes": baseline_savings_payload["rounded_model_mae_baguettes"],
            "rounded_best_statistical_baseline_mae_baguettes": baseline_savings_payload[
                "rounded_best_statistical_baseline_mae_baguettes"
            ],
            "baguette_unit_cost_eur": baseline_savings_payload["baguette_unit_cost_eur"],
            "test_day_count": baseline_savings_payload["test_day_count"],
        }
    selected_feature_columns = _feature_columns(history_df)
    final_model = fit_elasticnet_model(
        history_df=history_df,
        feature_columns=selected_feature_columns,
        target_column=TARGET_COLUMN,
        params=best_payload["best_params"],
        random_seed=RANDOM_SEED,
    )
    save_elasticnet_model(final_model, final_model_pkl)
    metrics_output_payload: dict[str, Any] = (
        {"business_impact": business_impact_payload, **metrics_payload}
        if business_impact_payload is not None
        else dict(metrics_payload)
    )
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics_output_payload, indent=2), encoding=CSV_ENCODING)
    diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_json.write_text(json.dumps(diagnostics_payload, indent=2), encoding=CSV_ENCODING)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_csv, index=False, encoding=CSV_ENCODING)
    _plot_actual_vs_predicted(predictions_df, predictions_plot_png)
    _plot_residuals(predictions_df, residuals_plot_png)
    _plot_residuals_qq(predictions_df, residuals_qq_png)
    _plot_residuals_acf_pacf(predictions_df, residuals_acf_pacf_png)
    model_card_payload: dict[str, Any] = {}
    if business_impact_payload is not None:
        model_card_payload["business_impact"] = business_impact_payload
    model_card_payload.update(best_payload)
    model_card_payload["metrics"] = metrics_output_payload
    model_card_payload["diagnostics"] = diagnostics_payload
    model_card_payload["final_model_path"] = str(final_model_pkl)
    model_card_payload["predictions_path"] = str(predictions_csv)
    if baselines_payload is not None and statistical_baselines_json is not None:
        model_card_payload["statistical_baselines_path"] = str(statistical_baselines_json)
    if baseline_savings_payload is not None:
        model_card_payload["best_statistical_baseline"] = {
            "name": baseline_savings_payload["best_statistical_baseline_name"],
            "mae": baseline_savings_payload["best_statistical_baseline_mae"],
        }
        model_card_payload["baseline_savings"] = baseline_savings_payload
    model_card_json.parent.mkdir(parents=True, exist_ok=True)
    model_card_json.write_text(json.dumps(model_card_payload, indent=2), encoding=CSV_ENCODING)
    LOGGER.info(
        "Evaluation completed in %.3f seconds with MAE=%.6f RMSE=%.6f sMAPE=%.6f",
        time.perf_counter() - start_time,
        metrics_payload["mae"],
        metrics_payload["rmse"],
        metrics_payload["smape"],
    )
    if baseline_savings_payload is not None:
        LOGGER.info(
            "Business impact vs best baseline: rounded model MAE=%d, rounded baseline MAE=%d, gap=%d baguettes/day, unit_cost=%.2f EUR, test_days=%d, estimated_savings=%.2f EUR",
            baseline_savings_payload["rounded_model_mae_baguettes"],
            baseline_savings_payload["rounded_best_statistical_baseline_mae_baguettes"],
            baseline_savings_payload["rounded_mae_gap_baguettes_vs_best_baseline"],
            baseline_savings_payload["baguette_unit_cost_eur"],
            baseline_savings_payload["test_day_count"],
            baseline_savings_payload["estimated_savings_eur_vs_best_baseline"],
        )
    return {
        "model_family": str(best_payload.get("model_family", "ElasticNet")),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(selected_feature_columns)),
        "mae": float(metrics_payload["mae"]),
        "rmse": float(metrics_payload["rmse"]),
        "smape": float(metrics_payload["smape"]),
        "absolute_mae_saved_vs_best_baseline": float(
            metrics_payload.get("absolute_mae_saved_vs_best_baseline", 0.0)
        ),
        "estimated_savings_eur_vs_best_baseline": float(
            metrics_payload.get("estimated_savings_eur_vs_best_baseline", 0.0)
        ),
    }

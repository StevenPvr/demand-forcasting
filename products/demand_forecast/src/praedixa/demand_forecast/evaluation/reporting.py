from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from praedixa.platform.runtime.paths import CACHE_DIR

def _resolve_project_root() -> Path:
    current_file = Path(__file__).resolve()
    try:
        return next(
            parent for parent in current_file.parents if (parent / "AGENTS.md").exists()
        )
    except StopIteration as exc:  # pragma: no cover - repository invariant
        raise RuntimeError(
            f"Unable to resolve the Praedixa project root from {current_file}"
        ) from exc


PROJECT_ROOT = _resolve_project_root()
CACHE_ROOT = CACHE_DIR
MPL_CACHE_DIR = CACHE_ROOT / "matplotlib"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtri  # noqa: E402


matplotlib.use("Agg")
PLOT_API: Any = plt
PREDICTION_QUANTILE_COLUMN_PATTERN = re.compile(r"prediction_p([0-9_]+)")


def json_dump(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_best_params(params_path: str | Path) -> dict[str, object]:
    return dict(json.loads(Path(params_path).read_text(encoding="utf-8")))


def _prediction_quantile_columns(predictions_df: pd.DataFrame) -> list[str]:
    resolved: list[tuple[float, str]] = []
    for column in predictions_df.columns:
        match = PREDICTION_QUANTILE_COLUMN_PATTERN.fullmatch(str(column))
        if match is None:
            continue
        quantile_level = float(match.group(1).replace("_", ".")) / 100.0
        resolved.append((quantile_level, column))
    return [column for _, column in sorted(resolved, key=lambda item: item[0])]


def _interval_contract(
    predictions_df: pd.DataFrame,
    *,
    label: str,
    overall_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    lower_col, upper_col = _interval_columns(label)
    lower_available = _has_interval_bound(predictions_df, lower_col)
    upper_available = _has_interval_bound(predictions_df, upper_col)
    return {
        "lower": lower_col if lower_available else None,
        "upper": upper_col if upper_available else None,
        "coverage": _interval_metric(overall_metrics, f"coverage_{label}"),
        "width": _interval_metric(overall_metrics, f"interval_width_{label}"),
    }


def _interval_columns(label: str) -> tuple[str, str]:
    return f"lower_{label}", f"upper_{label}"


def _probabilistic_overall_metrics(overall_metrics: dict[str, Any] | None) -> dict[str, Any]:
    if overall_metrics is None:
        return {}
    return {key: value for key, value in overall_metrics.items() if _is_probabilistic_metric_key(key)}


def _has_interval_bound(predictions_df: pd.DataFrame, column: str) -> bool:
    return column in predictions_df.columns and not predictions_df[column].isna().all()


def _interval_metric(overall_metrics: dict[str, Any] | None, key: str) -> Any:
    return None if overall_metrics is None else overall_metrics.get(key)


def _is_probabilistic_metric_key(key: str) -> bool:
    return (
        key.startswith("pinball_loss_")
        or key == "mean_pinball_loss"
        or key.startswith("coverage_")
        or key.startswith("interval_width_")
    )


def build_probabilistic_summary(
    predictions_df: pd.DataFrame,
    *,
    overall_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quantile_columns = _prediction_quantile_columns(predictions_df)
    return {
        "point_forecast_column": "prediction_raw" if "prediction_raw" in predictions_df.columns else None,
        "median_forecast_column": "prediction_p50" if "prediction_p50" in predictions_df.columns else None,
        "quantile_columns": quantile_columns,
        "intervals": {
            "80": _interval_contract(predictions_df, label="80", overall_metrics=overall_metrics),
            "95": _interval_contract(predictions_df, label="95", overall_metrics=overall_metrics),
        },
        "overall_metrics": _probabilistic_overall_metrics(overall_metrics),
    }


def _quantile_calibration(predictions_df: pd.DataFrame) -> dict[str, float]:
    actual = predictions_df["actual"].astype(float).to_numpy()
    calibration: dict[str, float] = {}
    for column in _prediction_quantile_columns(predictions_df):
        match = PREDICTION_QUANTILE_COLUMN_PATTERN.fullmatch(column)
        if match is None:
            continue
        quantile = float(match.group(1).replace("_", ".")) / 100.0
        predicted = predictions_df[column].astype(float).to_numpy()
        calibration[column] = float(np.mean(actual <= predicted) - quantile)
    return calibration


def _per_product_residuals(predictions_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    residuals: dict[str, dict[str, float]] = {}
    for product_name, product_predictions in predictions_df.groupby("product", sort=True):
        product_residuals = product_predictions["actual"].astype(float) - product_predictions["prediction_raw"].astype(float)
        residuals[str(product_name)] = {
            "mean": float(product_residuals.mean()),
            "std": float(product_residuals.std(ddof=0)),
            "rows": int(len(product_predictions)),
        }
    return residuals


def build_canonical_predictions_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    canonical = predictions_df.copy()
    for column in canonical.columns:
        if PREDICTION_QUANTILE_COLUMN_PATTERN.fullmatch(str(column)):
            canonical = canonical.drop(columns=[column])
    for column in ("lower_80", "upper_80", "lower_95", "upper_95"):
        if column in canonical.columns:
            canonical[column] = np.nan
    return canonical


def build_diagnostics_payload(
    predictions_df: pd.DataFrame,
    *,
    evaluation_mode: str,
    feature_count: int,
    best_iteration: int,
    overlap_metadata: dict[str, object] | None,
    daily_refit: bool,
    overall_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    residuals = predictions_df["actual"].astype(float) - predictions_df[
        "prediction_raw"
    ].astype(float)
    payload: dict[str, Any] = {
        "evaluation_mode": evaluation_mode,
        "row_count": int(len(predictions_df)),
        "feature_count": int(feature_count),
        "best_iteration": int(best_iteration),
        "daily_refit": bool(daily_refit),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std(ddof=0)),
        "residual_min": float(residuals.min()),
        "residual_max": float(residuals.max()),
        "per_product_residuals": _per_product_residuals(predictions_df),
        "quantile_calibration_error": _quantile_calibration(predictions_df),
        "probabilistic_summary": build_probabilistic_summary(
            predictions_df,
            overall_metrics=overall_metrics,
        ),
    }
    if overlap_metadata is not None:
        payload["reference_overlap"] = overlap_metadata
    return payload


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, output_path: Path) -> None:
    products = sorted(str(product) for product in predictions_df["product"].unique().tolist())
    fig, axes = PLOT_API.subplots(
        len(products),
        1,
        figsize=(12, max(4, 3 * len(products))),
        squeeze=False,
    )
    for axis, product_name in zip(axes.flat, products, strict=False):
        product_df = predictions_df.loc[predictions_df["product"].astype(str) == product_name].reset_index(drop=True)
        x_axis = product_df["target_date"] if "target_date" in product_df.columns else np.arange(len(product_df))
        axis.plot(x_axis, product_df["actual"], label="Actual", linewidth=1.4, color="#0072B2")
        axis.plot(x_axis, product_df["prediction_raw"], label="Predicted", linewidth=1.1, color="#D55E00")
        axis.set_title(f"Test set: actual vs predicted for {product_name}")
        axis.tick_params(axis="x", rotation=30)
        axis.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(predictions_df: pd.DataFrame, output_path: Path) -> None:
    residuals = predictions_df["actual"].astype(float) - predictions_df[
        "prediction_raw"
    ].astype(float)
    fig, ax = PLOT_API.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(residuals)), residuals, color="#0072B2", linewidth=1.2)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_title("Test residuals over concatenated product forecasts")
    ax.set_xlabel("Forecast row index")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_qq(predictions_df: pd.DataFrame, output_path: Path) -> None:
    residuals = _residual_values(predictions_df)
    fig = PLOT_API.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    theoretical_quantiles, ordered_residuals = _normal_qq_points(residuals)
    ax.scatter(theoretical_quantiles, ordered_residuals, s=14, color="#0072B2", alpha=0.75)
    _draw_qq_reference_line(ax, theoretical_quantiles, ordered_residuals)
    ax.set_title("Residual QQ plot")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered residuals")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_acf_pacf(predictions_df: pd.DataFrame, output_path: Path) -> None:
    residuals = _residual_values(predictions_df)
    acf_lags = max(min(len(residuals) - 1, 10), 1)
    pacf_lags = min(10, max((len(residuals) // 2) - 1, 0))
    fig, axes = PLOT_API.subplots(1, 2, figsize=(12, 4))
    _plot_correlation_bars(
        axes[0],
        _acf_values(residuals, max_lag=acf_lags),
        title="Residual ACF",
        sample_size=len(residuals),
    )
    if pacf_lags >= 1:
        _plot_correlation_bars(
            axes[1],
            _pacf_values(residuals, max_lag=pacf_lags),
            title="Residual PACF",
            sample_size=len(residuals),
        )
    else:
        axes[1].text(0.5, 0.5, "Not enough residuals for PACF", ha="center", va="center")
        axes[1].set_axis_off()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _residual_values(predictions_df: pd.DataFrame) -> np.ndarray:
    residuals = predictions_df["actual"].astype(float).to_numpy() - predictions_df["prediction_raw"].astype(float).to_numpy()
    return residuals[np.isfinite(residuals)]


def _normal_qq_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered_values = np.sort(values.astype(float))
    if len(ordered_values) == 0:
        return np.asarray([], dtype=float), ordered_values
    plotting_positions = (np.arange(1, len(ordered_values) + 1, dtype=float) - 0.5) / float(len(ordered_values))
    return ndtri(plotting_positions), ordered_values


def _draw_qq_reference_line(
    axis: Any,
    theoretical_quantiles: np.ndarray,
    ordered_values: np.ndarray,
) -> None:
    if len(theoretical_quantiles) < 2 or len(ordered_values) < 2:
        return
    slope, intercept = np.polyfit(theoretical_quantiles, ordered_values, deg=1)
    axis.plot(
        theoretical_quantiles,
        (float(slope) * theoretical_quantiles) + float(intercept),
        color="#D55E00",
        linewidth=1.2,
    )


def _acf_values(values: np.ndarray, *, max_lag: int) -> np.ndarray:
    centered = values - float(np.mean(values))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return np.zeros(max_lag, dtype=float)
    return np.asarray(
        [
            float(np.dot(centered[:-lag], centered[lag:]) / denominator)
            for lag in range(1, max_lag + 1)
        ],
        dtype=float,
    )


def _pacf_values(values: np.ndarray, *, max_lag: int) -> np.ndarray:
    centered = values - float(np.mean(values))
    pacf: list[float] = []
    for lag in range(1, max_lag + 1):
        y_values = centered[lag:]
        lagged_columns = [centered[lag - offset : -offset] for offset in range(1, lag + 1)]
        if len(y_values) == 0 or not lagged_columns:
            pacf.append(float("nan"))
            continue
        design = np.column_stack(lagged_columns)
        coefficients = np.linalg.lstsq(design, y_values, rcond=None)[0]
        pacf.append(float(coefficients[-1]))
    return np.asarray(pacf, dtype=float)


def _plot_correlation_bars(
    axis: Any,
    values: np.ndarray,
    *,
    title: str,
    sample_size: int,
) -> None:
    lags = np.arange(1, len(values) + 1)
    axis.bar(lags, values, color="#0072B2", width=0.6)
    axis.axhline(0.0, color="#444444", linewidth=1)
    confidence = 1.96 / np.sqrt(max(sample_size, 1))
    axis.axhline(confidence, color="#999999", linestyle="--", linewidth=1)
    axis.axhline(-confidence, color="#999999", linestyle="--", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Lag")
    axis.set_ylabel("Correlation")

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

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
CACHE_ROOT = PROJECT_ROOT / "var" / "cache"
MPL_CACHE_DIR = CACHE_ROOT / "matplotlib"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


matplotlib.use("Agg")
PLOT_API: Any = plt


def json_dump(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_best_params(params_path: str | Path) -> dict[str, object]:
    return dict(json.loads(Path(params_path).read_text(encoding="utf-8")))


def _prediction_quantile_columns(predictions_df: pd.DataFrame) -> list[str]:
    resolved: list[tuple[float, str]] = []
    for column in predictions_df.columns:
        match = re.fullmatch(r"prediction_p([0-9_]+)", column)
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


def build_canonical_predictions_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    canonical = predictions_df.copy()
    for column in canonical.columns:
        if re.fullmatch(r"prediction_p([0-9_]+)", column):
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
        "probabilistic_summary": build_probabilistic_summary(
            predictions_df,
            overall_metrics=overall_metrics,
        ),
    }
    if overlap_metadata is not None:
        payload["reference_overlap"] = overlap_metadata
    return payload


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = PLOT_API.subplots(figsize=(12, 4))
    x_axis = np.arange(len(predictions_df))
    ax.plot(
        x_axis,
        predictions_df["actual"],
        label="Actual sales",
        linewidth=1.5,
        color="#0072B2",
    )
    ax.plot(
        x_axis,
        predictions_df["prediction_raw"],
        label="Predicted sales",
        linewidth=1.2,
        color="#D55E00",
    )
    ax.set_title("Test set: actual vs predicted daily demand")
    ax.set_xlabel("Test observation index")
    ax.set_ylabel("Demand on next day")
    ax.legend(frameon=False)
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
    ax.set_title("Test residuals over time")
    ax.set_xlabel("Test observation index")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

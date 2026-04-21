from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.evaluation.bakery_baselines import DEFAULT_UNIT_COST_EUR
from praedixa.demand_forecast.evaluation.bakery_metrics import mae_score


@dataclass(frozen=True)
class BaselineSavingsSummary:
    best_baseline_mae: float
    model_mae_value: float
    rounded_best_baseline_mae: int
    rounded_model_mae: int
    rounded_mae_gap_units: int
    test_day_count: int
    estimated_savings_eur: float


def _best_baseline_prediction_frame(
    predictions_df: pd.DataFrame,
    statistical_baselines_payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    if statistical_baselines_payload is None:
        return None
    best_baseline = cast(dict[str, Any] | None, statistical_baselines_payload.get("best_baseline"))
    if not best_baseline:
        return None
    prediction_rows = cast(list[dict[str, Any]], best_baseline.get("prediction_rows", []))
    if len(prediction_rows) != len(predictions_df):
        return None
    baseline_df = pd.DataFrame(prediction_rows)
    return best_baseline, baseline_df


def _baseline_predictions_align(
    predictions_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> bool:
    return not (
        baseline_df["origin_date"].astype(str).tolist() != predictions_df["origin_date"].astype(str).tolist()
        or baseline_df["target_date"].astype(str).tolist() != predictions_df["target_date"].astype(str).tolist()
        or baseline_df["product"].astype(str).tolist() != predictions_df["product"].astype(str).tolist()
    )


def enrich_predictions_with_best_baseline(
    predictions_df: pd.DataFrame,
    statistical_baselines_payload: dict[str, Any] | None,
    *,
    field_baseline_name: str,
    unit_cost_eur: float = DEFAULT_UNIT_COST_EUR,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    resolved_bundle = _resolved_baseline_bundle(predictions_df, statistical_baselines_payload)
    if resolved_bundle is None:
        return predictions_df, None
    best_baseline, baseline_df, per_product_best_baselines = resolved_bundle
    if not _baseline_predictions_align(predictions_df, baseline_df):
        return predictions_df, None
    enriched_df = predictions_df.copy()
    actual_series, model_prediction_series, model_abs_error = _model_error_columns(enriched_df)
    baseline_prediction, baseline_abs_error = _baseline_error_columns(baseline_df)
    savings_summary = _baseline_savings_summary(
        predictions_df=predictions_df,
        best_baseline=best_baseline,
        actual_series=actual_series,
        model_prediction_series=model_prediction_series,
        unit_cost_eur=unit_cost_eur,
    )
    enriched_df = _attach_baseline_columns(
        enriched_df=enriched_df,
        best_baseline=best_baseline,
        baseline_prediction=baseline_prediction,
        baseline_abs_error=baseline_abs_error,
        per_product_best_baselines=per_product_best_baselines,
        best_baseline_mae=savings_summary.best_baseline_mae,
        model_abs_error=model_abs_error,
    )
    savings_series = enriched_df["absolute_error_saved_vs_best_baseline"].astype(float)
    enriched_df["relative_error_reduction_vs_best_baseline"] = _relative_error_reduction(
        saved_error=enriched_df["absolute_error_saved_vs_best_baseline"].astype(float),
        baseline_abs_error=baseline_abs_error,
    )
    return enriched_df, _savings_payload(
        best_baseline=best_baseline,
        field_baseline_name=field_baseline_name,
        unit_cost_eur=unit_cost_eur,
        savings_summary=savings_summary,
        savings_series=savings_series,
    )


def _resolved_baseline_bundle(
    predictions_df: pd.DataFrame,
    statistical_baselines_payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]] | None:
    baseline_bundle = _best_baseline_prediction_frame(predictions_df, statistical_baselines_payload)
    if baseline_bundle is None or statistical_baselines_payload is None:
        return None
    best_baseline, baseline_df = baseline_bundle
    per_product_best_baselines = cast(
        dict[str, Any],
        statistical_baselines_payload.get("per_product_best_baselines", {}),
    )
    return best_baseline, baseline_df, per_product_best_baselines


def _savings_payload(
    *,
    best_baseline: dict[str, Any],
    field_baseline_name: str,
    unit_cost_eur: float,
    savings_summary: BaselineSavingsSummary,
    savings_series: pd.Series,
) -> dict[str, Any]:
    return _baseline_savings_payload(
        best_baseline=best_baseline,
        field_baseline_name=field_baseline_name,
        best_baseline_mae=savings_summary.best_baseline_mae,
        model_mae_value=savings_summary.model_mae_value,
        rounded_best_baseline_mae=savings_summary.rounded_best_baseline_mae,
        rounded_model_mae=savings_summary.rounded_model_mae,
        rounded_mae_gap_units=savings_summary.rounded_mae_gap_units,
        unit_cost_eur=unit_cost_eur,
        test_day_count=savings_summary.test_day_count,
        estimated_savings_eur=savings_summary.estimated_savings_eur,
        savings_series=savings_series,
    )


def _model_error_columns(
    enriched_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    actual_series = enriched_df["actual"].astype(float).reset_index(drop=True)
    model_prediction_series = enriched_df["prediction_raw"].astype(float).reset_index(drop=True)
    model_abs_error = (actual_series - model_prediction_series).abs()
    return actual_series, model_prediction_series, model_abs_error


def _baseline_error_columns(baseline_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    baseline_prediction = baseline_df["prediction"].astype(float).reset_index(drop=True)
    baseline_abs_error = baseline_df["absolute_error"].astype(float).reset_index(drop=True)
    return baseline_prediction, baseline_abs_error


def _baseline_savings_summary(
    *,
    predictions_df: pd.DataFrame,
    best_baseline: dict[str, Any],
    actual_series: pd.Series,
    model_prediction_series: pd.Series,
    unit_cost_eur: float,
) -> BaselineSavingsSummary:
    model_mae_value = mae_score(actual_series, model_prediction_series)
    best_baseline_mae = float(best_baseline["mae"])
    rounded_model_mae = int(round(model_mae_value))
    rounded_best_baseline_mae = int(round(best_baseline_mae))
    rounded_mae_gap_units = rounded_best_baseline_mae - rounded_model_mae
    test_day_count = int(len(predictions_df))
    estimated_savings_eur = float(rounded_mae_gap_units * float(unit_cost_eur) * test_day_count)
    return BaselineSavingsSummary(
        best_baseline_mae=best_baseline_mae,
        model_mae_value=model_mae_value,
        rounded_best_baseline_mae=rounded_best_baseline_mae,
        rounded_model_mae=rounded_model_mae,
        rounded_mae_gap_units=rounded_mae_gap_units,
        test_day_count=test_day_count,
        estimated_savings_eur=estimated_savings_eur,
    )


def _attach_baseline_columns(
    *,
    enriched_df: pd.DataFrame,
    best_baseline: dict[str, Any],
    baseline_prediction: pd.Series,
    baseline_abs_error: pd.Series,
    per_product_best_baselines: dict[str, Any],
    best_baseline_mae: float,
    model_abs_error: pd.Series,
) -> pd.DataFrame:
    enriched_df["best_statistical_baseline_name"] = str(best_baseline["name"])
    enriched_df["best_statistical_baseline_prediction"] = baseline_prediction
    enriched_df["best_statistical_baseline_abs_error"] = baseline_abs_error
    enriched_df["best_statistical_baseline_product_mae"] = _best_baseline_product_maes(
        products=enriched_df["product"].astype(str),
        per_product_best_baselines=per_product_best_baselines,
        best_baseline_mae=best_baseline_mae,
    )
    enriched_df["model_abs_error"] = model_abs_error
    enriched_df["absolute_error_saved_vs_best_baseline"] = baseline_abs_error - model_abs_error
    return enriched_df


def _relative_error_reduction(
    *,
    saved_error: pd.Series,
    baseline_abs_error: pd.Series,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.where(
        baseline_abs_error > 0.0,
        saved_error / baseline_abs_error,
        0.0,
    )


def _best_baseline_product_maes(
    *,
    products: pd.Series,
    per_product_best_baselines: dict[str, Any],
    best_baseline_mae: float,
) -> list[float]:
    return [
        float(
            cast(dict[str, Any], per_product_best_baselines.get(str(product_name), {})).get("mae", best_baseline_mae)
        )
        for product_name in products
    ]


def _baseline_savings_payload(
    *,
    best_baseline: dict[str, Any],
    field_baseline_name: str,
    best_baseline_mae: float,
    model_mae_value: float,
    rounded_best_baseline_mae: int,
    rounded_model_mae: int,
    rounded_mae_gap_units: int,
    unit_cost_eur: float,
    test_day_count: int,
    estimated_savings_eur: float,
    savings_series: pd.Series,
) -> dict[str, Any]:
    return {
        "best_statistical_baseline_name": str(best_baseline["name"]),
        "field_baseline_name": field_baseline_name,
        "best_statistical_baseline_mae": best_baseline_mae,
        "model_mae": float(model_mae_value),
        "absolute_mae_saved_vs_best_baseline": float(best_baseline_mae - model_mae_value),
        "rounded_best_statistical_baseline_mae_baguettes": rounded_best_baseline_mae,
        "rounded_model_mae_baguettes": rounded_model_mae,
        "rounded_mae_gap_baguettes_vs_best_baseline": rounded_mae_gap_units,
        "baguette_unit_cost_eur": float(unit_cost_eur),
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

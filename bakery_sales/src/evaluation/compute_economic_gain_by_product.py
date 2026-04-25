from __future__ import annotations

"""Calcule un gain economique par produit a partir des predictions ARIMA et des baselines."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning.paths_arima import RAW_SALES_CSV
from src.data_cleaning.prepare_baguette_dataset import (
    filter_invalid_articles,
    load_sales_dataset,
    remove_negative_cancellations,
)
from src.evaluation.constants_arima import DATE_COLUMN, DEFAULT_PRODUCTION_COST_RATIO, FIELD_BASELINE_NAME, PRODUCT_COLUMN
from src.evaluation.paths_arima import ECONOMIC_GAIN_JSON, METRICS_JSON, PREDICTIONS_CSV, STATISTICAL_BASELINES_JSON, TEST_CSV

LOGGER: logging.Logger = logging.getLogger(__name__)
DEFAULT_UNIT_SALE_PRICE_EUR: float = 1.0


def _load_json(input_path: Path) -> dict[str, Any]:
    """Charge un payload JSON depuis le disque."""

    return cast(dict[str, Any], json.loads(input_path.read_text(encoding="utf-8")))


def _load_predictions(csv_path: Path) -> pd.DataFrame:
    """Charge les predictions ARIMA detaillees et les trie par produit/date cible."""

    predictions_df = pd.read_csv(csv_path)
    predictions_df["target_date"] = pd.to_datetime(predictions_df["target_date"], format="%Y-%m-%d")
    ordered_df = predictions_df.sort_values([PRODUCT_COLUMN, "target_date"]).reset_index(drop=True)
    ordered_df["target_date"] = ordered_df["target_date"].dt.strftime("%Y-%m-%d")
    return ordered_df


def _test_window_unit_prices(
    test_csv: Path,
    raw_sales_csv: Path | None,
) -> dict[str, float]:
    """Calcule un prix de vente unitaire moyen pondere par produit sur les dates du split test."""

    if raw_sales_csv is None or not raw_sales_csv.exists():
        LOGGER.info(
            "Raw sales CSV missing; using default unit sale price %.2f EUR for all products",
            DEFAULT_UNIT_SALE_PRICE_EUR,
        )
        return {}
    test_df = pd.read_csv(test_csv)
    test_dates = set(cast(pd.Series, test_df[DATE_COLUMN]).astype(str).tolist())
    test_products = set(cast(pd.Series, test_df[PRODUCT_COLUMN]).astype(str).tolist())
    sales_df = load_sales_dataset(raw_sales_csv)
    sales_df = filter_invalid_articles(sales_df)
    sales_df = remove_negative_cancellations(sales_df)
    filtered_sales_df = sales_df.loc[
        cast(pd.Series, sales_df["date"]).astype(str).isin(test_dates)
        & cast(pd.Series, sales_df["article"]).astype(str).isin(test_products)
    ].copy()
    if len(filtered_sales_df) == 0:
        LOGGER.info(
            "No raw sales rows matched the test window; using default unit sale price %.2f EUR for all products",
            DEFAULT_UNIT_SALE_PRICE_EUR,
        )
        return {}
    filtered_sales_df["line_revenue"] = (
        cast(pd.Series, filtered_sales_df["unit_price"]).astype(float)
        * cast(pd.Series, filtered_sales_df["Quantity"]).astype(float)
    )
    grouped_df = filtered_sales_df.groupby("article", as_index=False).agg(
        total_quantity=("Quantity", "sum"),
        total_revenue=("line_revenue", "sum"),
    )
    grouped_df = grouped_df.loc[cast(pd.Series, grouped_df["total_quantity"]).astype(float) > 0.0].copy()
    grouped_df["unit_sale_price_eur"] = (
        cast(pd.Series, grouped_df["total_revenue"]).astype(float)
        / cast(pd.Series, grouped_df["total_quantity"]).astype(float)
    )
    unit_prices = {
        str(row["article"]): float(row["unit_sale_price_eur"])
        for row in grouped_df.to_dict(orient="records")
    }
    LOGGER.info("Computed test-window unit prices for %d products", len(unit_prices))
    return unit_prices


def _rounded_units(values: pd.Series) -> pd.Series:
    """Arrondit des predictions en unites entieres non negatives."""

    rounded_values = values.astype(float).round().clip(lower=0.0)
    return cast(pd.Series, rounded_values.astype(int))


def _baseline_predictions_frame(baselines_payload: dict[str, Any]) -> pd.DataFrame:
    """Construit un DataFrame consolide des baselines terrain par produit."""

    rows: list[dict[str, Any]] = []
    per_product_field_baselines = cast(
        dict[str, Any],
        baselines_payload.get("per_product_field_baselines", baselines_payload.get("per_product_best_baselines", {})),
    )
    for product_name, baseline_payload in per_product_field_baselines.items():
        prediction_rows = cast(list[dict[str, Any]], baseline_payload.get("prediction_rows", []))
        for row in prediction_rows:
            rows.append(
                {
                    PRODUCT_COLUMN: str(product_name),
                    "target_date": str(row["target_date"]),
                    "baseline_name": str(baseline_payload["name"]),
                    "baseline_prediction_raw": float(row["prediction"]),
                    "baseline_actual": float(row["actual"]),
                    "best_baseline_mae": float(baseline_payload["mae"]),
                }
            )
    return pd.DataFrame(rows)


def _comparison_frame(
    predictions_csv: Path,
    baselines_payload: dict[str, Any],
) -> pd.DataFrame:
    """Assemble les predictions ARIMA et baseline ligne a ligne."""

    model_predictions_df = _load_predictions(predictions_csv).loc[
        :,
        [PRODUCT_COLUMN, "target_date", "actual", "prediction_raw", "prediction_rounded"],
    ].copy()
    model_predictions_df = model_predictions_df.rename(
        columns={
            "actual": "model_actual",
            "prediction_raw": "model_prediction_raw",
            "prediction_rounded": "model_prediction_rounded",
        }
    )
    baseline_predictions_df = _baseline_predictions_frame(baselines_payload)
    merged_df = model_predictions_df.merge(
        baseline_predictions_df,
        on=[PRODUCT_COLUMN, "target_date"],
        how="inner",
    )
    if len(merged_df) == 0:
        return merged_df
    merged_df["actual_units"] = _rounded_units(cast(pd.Series, merged_df["model_actual"]))
    merged_df["model_prediction_units"] = _rounded_units(cast(pd.Series, merged_df["model_prediction_rounded"]))
    merged_df["baseline_prediction_units"] = _rounded_units(cast(pd.Series, merged_df["baseline_prediction_raw"]))
    return merged_df


def _apply_loss_model(
    comparison_df: pd.DataFrame,
    unit_sale_prices: dict[str, float],
    production_cost_ratio: float,
) -> pd.DataFrame:
    """Calcule les pertes modeles pour ARIMA et baseline, ligne par ligne."""

    priced_df = comparison_df.copy()
    priced_df["unit_sale_price_eur"] = [
        float(unit_sale_prices.get(str(product_name), DEFAULT_UNIT_SALE_PRICE_EUR))
        for product_name in cast(pd.Series, priced_df[PRODUCT_COLUMN]).astype(str)
    ]
    priced_df["unit_production_cost_eur"] = (
        cast(pd.Series, priced_df["unit_sale_price_eur"]).astype(float) * float(production_cost_ratio)
    )
    for prefix in ("model", "baseline"):
        predicted_units = cast(pd.Series, priced_df[f"{prefix}_prediction_units"]).astype(int)
        actual_units = cast(pd.Series, priced_df["actual_units"]).astype(int)
        priced_df[f"{prefix}_overproduction_units"] = (predicted_units - actual_units).clip(lower=0)
        priced_df[f"{prefix}_underproduction_units"] = (actual_units - predicted_units).clip(lower=0)
        priced_df[f"{prefix}_overproduction_loss_eur"] = (
            cast(pd.Series, priced_df[f"{prefix}_overproduction_units"]).astype(float)
            * cast(pd.Series, priced_df["unit_production_cost_eur"]).astype(float)
        )
        priced_df[f"{prefix}_underproduction_loss_eur"] = (
            cast(pd.Series, priced_df[f"{prefix}_underproduction_units"]).astype(float)
            * cast(pd.Series, priced_df["unit_sale_price_eur"]).astype(float)
        )
        priced_df[f"{prefix}_total_loss_eur"] = (
            cast(pd.Series, priced_df[f"{prefix}_overproduction_loss_eur"]).astype(float)
            + cast(pd.Series, priced_df[f"{prefix}_underproduction_loss_eur"]).astype(float)
        )
    return priced_df


def _product_gain_payload(
    product_df: pd.DataFrame,
    arima_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Construit le gain economique realise pour un produit unique."""

    best_baseline_name = str(cast(pd.Series, product_df["baseline_name"]).iloc[0])
    unit_sale_price = float(cast(pd.Series, product_df["unit_sale_price_eur"]).iloc[0])
    unit_production_cost = float(cast(pd.Series, product_df["unit_production_cost_eur"]).iloc[0])
    model_mae = float(arima_metrics["mae"])
    baseline_mae = float(cast(pd.Series, product_df["best_baseline_mae"]).iloc[0])
    model_total_loss = float(cast(pd.Series, product_df["model_total_loss_eur"]).sum())
    baseline_total_loss = float(cast(pd.Series, product_df["baseline_total_loss_eur"]).sum())
    estimated_savings_eur = float(baseline_total_loss - model_total_loss)
    return {
        "best_baseline_name": best_baseline_name,
        "model_mae": model_mae,
        "best_baseline_mae": baseline_mae,
        "absolute_mae_saved_vs_best_baseline": float(baseline_mae - model_mae),
        "relative_mae_improvement_vs_best_baseline": float(
            0.0 if baseline_mae <= 0.0 else ((baseline_mae - model_mae) / baseline_mae)
        ),
        "test_day_count": int(arima_metrics["test_rows"]),
        "unit_sale_price_eur": unit_sale_price,
        "unit_production_cost_eur": unit_production_cost,
        "model_overproduction_units": float(cast(pd.Series, product_df["model_overproduction_units"]).sum()),
        "model_underproduction_units": float(cast(pd.Series, product_df["model_underproduction_units"]).sum()),
        "baseline_overproduction_units": float(cast(pd.Series, product_df["baseline_overproduction_units"]).sum()),
        "baseline_underproduction_units": float(cast(pd.Series, product_df["baseline_underproduction_units"]).sum()),
        "model_total_loss_eur": model_total_loss,
        "baseline_total_loss_eur": baseline_total_loss,
        "estimated_realistic_savings_eur_vs_best_baseline": estimated_savings_eur,
        "estimated_savings_eur_vs_best_baseline": estimated_savings_eur,
    }


def compute_economic_gain_report(
    arima_metrics_json: Path,
    baselines_json: Path,
    output_json: Path,
    predictions_csv: Path,
    test_csv: Path,
    raw_sales_csv: Path | None = RAW_SALES_CSV,
    production_cost_ratio: float = DEFAULT_PRODUCTION_COST_RATIO,
) -> dict[str, Any]:
    """Calcule un rapport de gain economique asymetrique global et par produit."""

    arima_payload = _load_json(arima_metrics_json)
    baselines_payload = _load_json(baselines_json)
    LOGGER.info("Loaded ARIMA metrics from %s", arima_metrics_json)
    LOGGER.info("Loaded baseline metrics from %s", baselines_json)
    LOGGER.info(
        "Using production cost ratio %.2f of unit sale price for overproduction losses",
        production_cost_ratio,
    )
    unit_sale_prices = _test_window_unit_prices(test_csv=test_csv, raw_sales_csv=raw_sales_csv)
    comparison_df = _comparison_frame(predictions_csv=predictions_csv, baselines_payload=baselines_payload)
    if len(comparison_df) == 0:
        raise RuntimeError("No overlapping ARIMA and baseline prediction rows were found")
    priced_comparison_df = _apply_loss_model(
        comparison_df=comparison_df,
        unit_sale_prices=unit_sale_prices,
        production_cost_ratio=production_cost_ratio,
    )
    per_product_metrics = cast(dict[str, Any], arima_payload.get("per_product_metrics", {}))
    per_product_field_baselines = cast(
        dict[str, Any],
        baselines_payload.get("per_product_field_baselines", baselines_payload.get("per_product_best_baselines", {})),
    )
    resolved_field_baseline_name = str(baselines_payload.get("field_baseline_name", FIELD_BASELINE_NAME))
    LOGGER.info(
        "Comparing %d ARIMA product metrics against %d per-product field baselines (%s)",
        len(per_product_metrics),
        len(per_product_field_baselines),
        resolved_field_baseline_name,
    )

    per_product_gain: dict[str, Any] = {}
    for product_name, metrics_payload in per_product_metrics.items():
        baseline_payload = cast(dict[str, Any] | None, per_product_field_baselines.get(product_name))
        if baseline_payload is None:
            LOGGER.warning("Skipping product %s because no field baseline was found", product_name)
            continue
        product_comparison_df = priced_comparison_df.loc[
            cast(pd.Series, priced_comparison_df[PRODUCT_COLUMN]).astype(str) == str(product_name)
        ].copy()
        if len(product_comparison_df) == 0:
            LOGGER.warning("Skipping product %s because no aligned prediction rows were found", product_name)
            continue
        product_gain = _product_gain_payload(
            product_df=product_comparison_df,
            arima_metrics=cast(dict[str, Any], metrics_payload),
        )
        per_product_gain[str(product_name)] = product_gain
        LOGGER.info(
            "Product %s: ARIMA loss %.4f EUR vs baseline %s loss %.4f EUR; savings %.4f EUR; sale price %.4f EUR; production cost %.4f EUR",
            product_name,
            float(product_gain["model_total_loss_eur"]),
            str(product_gain["best_baseline_name"]),
            float(product_gain["baseline_total_loss_eur"]),
            float(product_gain["estimated_savings_eur_vs_best_baseline"]),
            float(product_gain["unit_sale_price_eur"]),
            float(product_gain["unit_production_cost_eur"]),
        )

    total_estimated_savings = float(
        sum(
            cast(float, payload["estimated_savings_eur_vs_best_baseline"])
            for payload in per_product_gain.values()
        )
    )
    total_model_loss = float(sum(cast(float, payload["model_total_loss_eur"]) for payload in per_product_gain.values()))
    total_baseline_loss = float(
        sum(cast(float, payload["baseline_total_loss_eur"]) for payload in per_product_gain.values())
    )
    total_absolute_mae_saved = float(
        sum(cast(float, payload["absolute_mae_saved_vs_best_baseline"]) for payload in per_product_gain.values())
    )

    report_payload: dict[str, Any] = {
        "model_family": str(arima_payload.get("model_family", "ARIMA_BY_PRODUCT")),
        "product_count": int(len(per_product_gain)),
        "default_unit_sale_price_eur": float(DEFAULT_UNIT_SALE_PRICE_EUR),
        "production_cost_ratio": float(production_cost_ratio),
        "field_baseline_name": resolved_field_baseline_name,
        "per_product_gain": per_product_gain,
        "total_model_loss_eur": total_model_loss,
        "total_baseline_loss_eur": total_baseline_loss,
        "total_absolute_mae_saved_vs_best_baselines": total_absolute_mae_saved,
        "total_estimated_savings_eur_vs_best_baselines": total_estimated_savings,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    LOGGER.info(
        "Computed economic gain against field baseline for %d products; total savings %.2f EUR",
        len(per_product_gain),
        total_estimated_savings,
    )
    LOGGER.info("Saved economic gain report to %s", output_json)
    return report_payload


def main() -> None:
    """Execute le calcul de gain economique sur les artefacts ARIMA actifs."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    compute_economic_gain_report(
        arima_metrics_json=METRICS_JSON,
        baselines_json=STATISTICAL_BASELINES_JSON,
        output_json=ECONOMIC_GAIN_JSON,
        predictions_csv=PREDICTIONS_CSV,
        test_csv=TEST_CSV,
        raw_sales_csv=RAW_SALES_CSV,
    )


if __name__ == "__main__":
    main()

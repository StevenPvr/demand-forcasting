from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from praedixa.demand_forecast.evaluation.bakery_baselines import FIELD_BASELINE_NAME
from praedixa.demand_forecast.evaluation.bakery_metrics import REFERENCE_PRODUCT_COL


DEFAULT_UNIT_SALE_PRICE_EUR = 1.0
DEFAULT_PRODUCTION_COST_RATIO = 0.35
DEFAULT_RAW_SALES_CSV: Path | None = None
INVALID_ARTICLES: tuple[str, ...] = (
    "COUPON",
    "DECOUVERTE",
    "ARTICLE ANNULER",
    "MERCI DE VOTRE VISITE",
)


def build_simple_economic_gain_payload(
    predictions_df: pd.DataFrame,
    metrics_payload: dict[str, Any],
    *,
    duckdb_path: str | Path | None = None,
    raw_sales_csv: str | Path | None = DEFAULT_RAW_SALES_CSV,
    production_cost_ratio: float = DEFAULT_PRODUCTION_COST_RATIO,
) -> dict[str, Any]:
    per_product_metrics = dict(metrics_payload["per_product_metrics"])
    comparison_df = _economic_comparison_frame(predictions_df)
    if len(comparison_df) == 0:
        return _empty_economic_gain_payload(production_cost_ratio)

    unit_sale_prices = _test_window_unit_prices(
        predictions_df=predictions_df,
        duckdb_path=duckdb_path,
        raw_sales_csv=raw_sales_csv,
    )
    priced_comparison_df = _apply_loss_model(
        comparison_df=comparison_df,
        unit_sale_prices=unit_sale_prices,
        production_cost_ratio=production_cost_ratio,
    )

    per_product_gain = _per_product_gain_payloads(
        priced_comparison_df=priced_comparison_df,
        per_product_metrics=per_product_metrics,
    )
    return {
        "model_family": "FOUNDATION_TFT",
        "product_count": int(len(per_product_gain)),
        "default_unit_sale_price_eur": float(DEFAULT_UNIT_SALE_PRICE_EUR),
        "production_cost_ratio": float(production_cost_ratio),
        "field_baseline_name": FIELD_BASELINE_NAME,
        "per_product_gain": per_product_gain,
        **_economic_gain_totals(per_product_gain),
    }


def _parse_unit_price_series(price_series: pd.Series) -> pd.Series:
    normalized_series = (
        price_series.astype("string")
        .fillna("0")
        .str.replace("€", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    numeric_series = pd.to_numeric(normalized_series, errors="coerce")
    return numeric_series.fillna(0.0).astype(float)


def _load_sales_dataset(raw_sales_csv: str | Path) -> pd.DataFrame:
    sales_df = pd.read_csv(raw_sales_csv)
    keep_columns = _sales_keep_columns(sales_df.columns)
    sales_df = sales_df.loc[:, keep_columns].copy()
    sales_df["date"] = pd.to_datetime(sales_df["date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    sales_df["article"] = sales_df["article"].astype("string").str.strip()
    sales_df["Quantity"] = sales_df["Quantity"].astype(float)
    sales_df["unit_price"] = _parse_unit_price_series(sales_df["unit_price"])
    return sales_df.loc[:, ["date", "article", "Quantity", "unit_price"]]


def _cancel_negative_sales_for_article(article_df: pd.DataFrame) -> pd.DataFrame:
    working_df = article_df.copy()
    positive_indices: list[int] = []
    for row_index in working_df.index:
        quantity = float(cast(float, working_df.at[row_index, "Quantity"]))
        if quantity > 0.0:
            positive_indices.append(row_index)
        else:
            _cancel_negative_quantity_row(
                working_df=working_df,
                row_index=row_index,
                quantity=quantity,
                positive_indices=positive_indices,
            )
    return working_df


def _empty_economic_gain_payload(production_cost_ratio: float) -> dict[str, Any]:
    return {
        "model_family": "FOUNDATION_TFT",
        "product_count": 0,
        "default_unit_sale_price_eur": float(DEFAULT_UNIT_SALE_PRICE_EUR),
        "production_cost_ratio": float(production_cost_ratio),
        "field_baseline_name": FIELD_BASELINE_NAME,
        "per_product_gain": {},
        "total_model_loss_eur": 0.0,
        "total_baseline_loss_eur": 0.0,
        "total_absolute_mae_saved_vs_best_baselines": 0.0,
        "total_estimated_savings_eur_vs_best_baselines": 0.0,
    }


def _economic_gain_totals(per_product_gain: dict[str, Any]) -> dict[str, float]:
    return {
        "total_model_loss_eur": float(
            sum(cast(float, payload["model_total_loss_eur"]) for payload in per_product_gain.values())
        ),
        "total_baseline_loss_eur": float(
            sum(cast(float, payload["baseline_total_loss_eur"]) for payload in per_product_gain.values())
        ),
        "total_absolute_mae_saved_vs_best_baselines": float(
            sum(cast(float, payload["absolute_mae_saved_vs_best_baseline"]) for payload in per_product_gain.values())
        ),
        "total_estimated_savings_eur_vs_best_baselines": float(
            sum(cast(float, payload["estimated_savings_eur_vs_best_baseline"]) for payload in per_product_gain.values())
        ),
    }


def _per_product_gain_payloads(
    *,
    priced_comparison_df: pd.DataFrame,
    per_product_metrics: dict[str, Any],
) -> dict[str, Any]:
    per_product_gain: dict[str, Any] = {}
    for product_name, metrics in per_product_metrics.items():
        product_comparison_df = priced_comparison_df.loc[
            priced_comparison_df[REFERENCE_PRODUCT_COL].astype(str) == str(product_name)
        ].copy()
        if len(product_comparison_df) == 0:
            continue
        per_product_gain[str(product_name)] = _product_gain_payload(
            product_df=product_comparison_df,
            product_metrics=cast(dict[str, Any], metrics),
        )
    return per_product_gain


def _normalized_sales_columns(columns: pd.Index) -> list[str]:
    return [
        str(column)
        for column in columns
        if str(column).strip() and not str(column).lower().startswith("unnamed:")
    ]


def _drop_leading_index_column(columns: list[str]) -> list[str]:
    if columns and columns[0].lower() == "index":
        return columns[1:]
    return columns


def _sales_keep_columns(columns: pd.Index) -> list[str]:
    return _drop_leading_index_column(_normalized_sales_columns(columns))


def _cancel_negative_quantity_row(
    *,
    working_df: pd.DataFrame,
    row_index: int,
    quantity: float,
    positive_indices: list[int],
) -> None:
    remaining_to_cancel = abs(quantity)
    while remaining_to_cancel > 0.0 and positive_indices:
        previous_index = positive_indices[-1]
        previous_quantity = float(cast(float, working_df.at[previous_index, "Quantity"]))
        cancelled_quantity = min(previous_quantity, remaining_to_cancel)
        working_df.at[previous_index, "Quantity"] = previous_quantity - cancelled_quantity
        remaining_to_cancel -= cancelled_quantity
        if float(cast(float, working_df.at[previous_index, "Quantity"])) <= 0.0:
            positive_indices.pop()
    working_df.at[row_index, "Quantity"] = 0.0


def _remove_negative_cancellations(sales_df: pd.DataFrame) -> pd.DataFrame:
    preserved_columns = [str(column) for column in sales_df.columns]
    ordered_df = sales_df.copy()
    ordered_df["row_order"] = range(len(ordered_df))
    ordered_df = ordered_df.sort_values(["article", "date", "row_order"])
    cleaned_articles = [
        _cancel_negative_sales_for_article(article_df)
        for _, article_df in ordered_df.groupby("article", sort=False)
    ]
    cleaned_df = pd.concat(cleaned_articles, ignore_index=True)
    cleaned_df = cleaned_df.loc[cleaned_df["Quantity"] > 0].copy()
    cleaned_df = cleaned_df.sort_values(["date", "row_order"]).reset_index(drop=True)
    return cleaned_df.loc[:, preserved_columns]


def _test_window_unit_prices(
    *,
    predictions_df: pd.DataFrame,
    duckdb_path: str | Path | None,
    raw_sales_csv: str | Path | None,
) -> dict[str, float]:
    from_gold = _test_window_unit_prices_from_gold(
        predictions_df=predictions_df,
        duckdb_path=duckdb_path,
    )
    if from_gold:
        return from_gold
    if raw_sales_csv is None or not Path(raw_sales_csv).exists():
        return {}

    test_dates = set(predictions_df["target_date"].astype(str).tolist())
    test_products = set(predictions_df["product"].astype(str).tolist())
    sales_df = _load_sales_dataset(raw_sales_csv)
    sales_df = sales_df.loc[~sales_df["article"].isin(INVALID_ARTICLES)].copy()
    sales_df = _remove_negative_cancellations(sales_df)
    test_date_values = sorted(test_dates)
    test_product_values = sorted(test_products)
    filtered_sales_df = sales_df.loc[
        sales_df["date"].astype(str).isin(test_date_values)
        & sales_df["article"].astype(str).isin(test_product_values)
    ].copy()
    if len(filtered_sales_df) == 0:
        return {}

    filtered_sales_df["line_revenue"] = (
        filtered_sales_df["unit_price"].astype(float)
        * filtered_sales_df["Quantity"].astype(float)
    )
    grouped_df = filtered_sales_df.groupby("article", as_index=False).agg(
        total_quantity=("Quantity", "sum"),
        total_revenue=("line_revenue", "sum"),
    )
    grouped_df = grouped_df.loc[grouped_df["total_quantity"].astype(float) > 0.0].copy()
    grouped_df["unit_sale_price_eur"] = (
        grouped_df["total_revenue"].astype(float)
        / grouped_df["total_quantity"].astype(float)
    )
    return {
        str(row["article"]): float(row["unit_sale_price_eur"])
        for row in grouped_df.to_dict(orient="records")
    }


def _test_window_unit_prices_from_gold(
    *,
    predictions_df: pd.DataFrame,
    duckdb_path: str | Path | None,
) -> dict[str, float]:
    if duckdb_path is None or not Path(duckdb_path).exists():
        return {}
    prediction_targets = pd.DataFrame(
        {
            "dt": pd.to_datetime(predictions_df["target_date"], format="%Y-%m-%d", errors="raise"),
            "product_id": predictions_df["product"].astype(str),
        }
    ).drop_duplicates(ignore_index=True)
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        connection.register("prediction_targets", prediction_targets)
        grouped_df = connection.execute(
            """
            select
                bakery.product_id as article,
                sum(coalesce(bakery.observed_revenue_net, 0.0)) as total_revenue,
                sum(coalesce(bakery.current_day_demand_qty, 0.0)) as total_quantity
            from gold.gold_base_panel_d1 as bakery
            inner join prediction_targets as target
                on bakery.dt = target.dt
               and bakery.product_id = target.product_id
            where bakery.dataset_source = 'bakery'
            group by 1
            """
        ).fetchdf()
    finally:
        connection.close()
    grouped_df = grouped_df.loc[grouped_df["total_quantity"].astype(float) > 0.0].copy()
    if len(grouped_df) == 0:
        return {}
    grouped_df["unit_sale_price_eur"] = (
        grouped_df["total_revenue"].astype(float)
        / grouped_df["total_quantity"].astype(float)
    )
    return {
        str(row["article"]): float(row["unit_sale_price_eur"])
        for row in grouped_df.to_dict(orient="records")
    }


def _rounded_units(values: pd.Series) -> pd.Series:
    rounded_values = values.astype(float).round().clip(lower=0.0)
    return rounded_values.astype(int)


def _economic_comparison_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "product",
        "target_date",
        "actual",
        "prediction_raw",
        "prediction_rounded",
        "best_statistical_baseline_name",
        "best_statistical_baseline_prediction",
        "best_statistical_baseline_abs_error",
    }
    if not required_columns.issubset(predictions_df.columns):
        return pd.DataFrame()

    comparison_df = predictions_df.loc[
        :,
        [
            "product",
            "target_date",
            "actual",
            "prediction_raw",
            "prediction_rounded",
            "best_statistical_baseline_name",
            "best_statistical_baseline_prediction",
            "best_statistical_baseline_abs_error",
            "best_statistical_baseline_product_mae",
        ],
    ].copy()
    comparison_df = comparison_df.rename(
        columns={
            "actual": "model_actual",
            "prediction_raw": "model_prediction_raw",
            "prediction_rounded": "model_prediction_rounded",
            "best_statistical_baseline_name": "baseline_name",
            "best_statistical_baseline_prediction": "baseline_prediction_raw",
            "best_statistical_baseline_product_mae": "best_baseline_mae",
            "best_statistical_baseline_abs_error": "baseline_absolute_error",
        }
    )
    comparison_df["actual_units"] = _rounded_units(comparison_df["model_actual"])
    comparison_df["model_prediction_units"] = _rounded_units(comparison_df["model_prediction_rounded"])
    comparison_df["baseline_prediction_units"] = _rounded_units(comparison_df["baseline_prediction_raw"])
    return comparison_df


def _apply_loss_model(
    *,
    comparison_df: pd.DataFrame,
    unit_sale_prices: dict[str, float],
    production_cost_ratio: float,
) -> pd.DataFrame:
    priced_df = comparison_df.copy()
    priced_df["unit_sale_price_eur"] = [
        float(unit_sale_prices.get(str(product_name), DEFAULT_UNIT_SALE_PRICE_EUR))
        for product_name in priced_df[REFERENCE_PRODUCT_COL].astype(str)
    ]
    priced_df["unit_production_cost_eur"] = (
        priced_df["unit_sale_price_eur"].astype(float) * float(production_cost_ratio)
    )
    for prefix in ("model", "baseline"):
        predicted_units = priced_df[f"{prefix}_prediction_units"].astype(int)
        actual_units = priced_df["actual_units"].astype(int)
        priced_df[f"{prefix}_overproduction_units"] = (predicted_units - actual_units).clip(lower=0)
        priced_df[f"{prefix}_underproduction_units"] = (actual_units - predicted_units).clip(lower=0)
        priced_df[f"{prefix}_overproduction_loss_eur"] = (
            priced_df[f"{prefix}_overproduction_units"].astype(float)
            * priced_df["unit_production_cost_eur"].astype(float)
        )
        priced_df[f"{prefix}_underproduction_loss_eur"] = (
            priced_df[f"{prefix}_underproduction_units"].astype(float)
            * priced_df["unit_sale_price_eur"].astype(float)
        )
        priced_df[f"{prefix}_total_loss_eur"] = (
            priced_df[f"{prefix}_overproduction_loss_eur"].astype(float)
            + priced_df[f"{prefix}_underproduction_loss_eur"].astype(float)
        )
    return priced_df


def _product_gain_payload(
    *,
    product_df: pd.DataFrame,
    product_metrics: dict[str, Any],
) -> dict[str, Any]:
    best_baseline_name = str(product_df["baseline_name"].iloc[0])
    unit_sale_price = float(product_df["unit_sale_price_eur"].astype(float).iloc[0])
    unit_production_cost = float(product_df["unit_production_cost_eur"].astype(float).iloc[0])
    model_mae = float(product_metrics["mae"])
    baseline_mae = float(product_df["best_baseline_mae"].astype(float).iloc[0])
    model_total_loss = float(product_df["model_total_loss_eur"].astype(float).sum())
    baseline_total_loss = float(product_df["baseline_total_loss_eur"].astype(float).sum())
    estimated_savings_eur = float(baseline_total_loss - model_total_loss)
    return {
        "best_baseline_name": best_baseline_name,
        "model_mae": model_mae,
        "best_baseline_mae": baseline_mae,
        "absolute_mae_saved_vs_best_baseline": float(baseline_mae - model_mae),
        "relative_mae_improvement_vs_best_baseline": float(
            0.0 if baseline_mae <= 0.0 else ((baseline_mae - model_mae) / baseline_mae)
        ),
        "test_day_count": int(product_metrics["test_rows"]),
        "unit_sale_price_eur": unit_sale_price,
        "unit_production_cost_eur": unit_production_cost,
        "model_overproduction_units": float(product_df["model_overproduction_units"].astype(float).sum()),
        "model_underproduction_units": float(product_df["model_underproduction_units"].astype(float).sum()),
        "baseline_overproduction_units": float(product_df["baseline_overproduction_units"].astype(float).sum()),
        "baseline_underproduction_units": float(product_df["baseline_underproduction_units"].astype(float).sum()),
        "model_total_loss_eur": model_total_loss,
        "baseline_total_loss_eur": baseline_total_loss,
        "estimated_realistic_savings_eur_vs_best_baseline": estimated_savings_eur,
        "estimated_savings_eur_vs_best_baseline": estimated_savings_eur,
    }

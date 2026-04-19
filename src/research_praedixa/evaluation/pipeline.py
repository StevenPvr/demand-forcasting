from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import duckdb
import matplotlib
import numpy as np
import pandas as pd

from research_praedixa.evaluation.bakery_style import (
    DEFAULT_UNIT_COST_EUR,
    REFERENCE_DATE_COL,
    REFERENCE_PRODUCT_COL,
    REFERENCE_TARGET_COL,
    build_predictions_frame,
    build_simple_economic_gain_payload,
    build_statistical_baselines_payload,
    compute_metrics_payload,
    enrich_predictions_with_best_baseline,
    load_reference_split,
)
from research_praedixa.memory_utils import read_parquet_projected
from research_praedixa.optimisation.pipeline import (
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_DATE_COL,
    DEFAULT_DUCKDB_PATH,
    DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
    DEFAULT_GOLD_TABLE,
    DEFAULT_IDENTIFIER_FEATURE_COLS,
    DEFAULT_TRAIN_SAMPLE_FRACTION,
    DEFAULT_TUNING_SAMPLE_FRACTION,
    load_gold_train_tuning_frames,
)
from research_praedixa.target_utils import (
    DEFAULT_VARIATION_TARGET_COL,
    TargetContract,
    build_target_contract_metadata,
    ensure_learning_target_column,
    reconstruct_absolute_predictions,
    resolve_target_contract,
)
from research_praedixa.xgboost_utils import (
    DEFAULT_XGBOOST_MODEL_PARAMS,
    fit_xgboost_booster,
    predict_with_xgboost_booster,
    select_numeric_feature_columns,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TRAIN_SELECTION_INPUT_PATH: Path | None = None
DEFAULT_TRAIN_TUNING_INPUT_PATH: Path | None = None
DEFAULT_VAL_INPUT_PATH: Path | None = None
DEFAULT_BEST_PARAMS_PATH = Path("data/optimisation/best_optuna_params.json")
DEFAULT_OUTPUT_DIR = Path("data/evaluation")
DEFAULT_REQUESTED_TARGET_COL = DEFAULT_VARIATION_TARGET_COL
DEFAULT_BAKERY_REFERENCE_TRAIN_CSV = Path("bakery_sales/data/data_preprocessing/daily_product_arima_train_2021_2022.csv")
DEFAULT_BAKERY_REFERENCE_VAL_CSV = Path("bakery_sales/data/data_preprocessing/daily_product_arima_val_2021_2022.csv")
DEFAULT_BAKERY_REFERENCE_TEST_CSV = Path("bakery_sales/data/data_preprocessing/daily_product_arima_test_2021_2022.csv")
DEFAULT_MODEL_FAMILY = "FOUNDATION_XGBOOST"
DEFAULT_EVALUATION_EXCLUDED_HELPER_FEATURE_COLS = (
    "rn_sample",
    "stratum_row_count",
)


logger = logging.getLogger(__name__)


def _json_dump(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_best_params(params_path: str | Path) -> dict[str, object]:
    return json.loads(Path(params_path).read_text(encoding="utf-8"))


def build_daily_walk_forward_folds(
    frame: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
) -> list[dict[str, object]]:
    ordered = frame.copy().reset_index(drop=True)
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    unique_dates = pd.Index(ordered[date_col].drop_duplicates().sort_values())
    if len(unique_dates) < 1:
        raise ValueError("Need at least one evaluation date to build daily folds.")
    folds: list[dict[str, object]] = []
    for fold_idx, eval_date in enumerate(unique_dates, start=1):
        history_mask = ordered[date_col] < eval_date
        valid_mask = ordered[date_col] == eval_date
        folds.append(
            {
                "fold": fold_idx,
                "eval_date": eval_date.strftime("%Y-%m-%d"),
                "history_idx": ordered.index[history_mask].to_numpy(),
                "valid_idx": ordered.index[valid_mask].to_numpy(),
                "history_rows": int(history_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
            }
        )
    return folds


def _evaluate_daily_refit_predictions(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    folds = build_daily_walk_forward_folds(test_frame, date_col=REFERENCE_DATE_COL)
    daily_prediction_parts: list[pd.DataFrame] = []
    daily_reports: list[dict[str, object]] = []
    best_iterations: list[int] = []
    base_train_rows = int(len(train_frame))
    base_valid_rows = int(len(valid_frame))

    for fold in folds:
        eval_date = str(fold["eval_date"])
        bakery_history_frame = test_frame.iloc[fold["history_idx"]].copy()
        refit_train_frame = pd.concat([train_frame, bakery_history_frame], ignore_index=True)
        known_history_rows = int(base_train_rows + base_valid_rows + len(bakery_history_frame))
        model, best_iteration = _fit_evaluation_model(
            train_frame=refit_train_frame,
            valid_frame=valid_frame,
            feature_cols=feature_cols,
            target_contract=target_contract,
            model_params=model_params,
        )
        best_iterations.append(best_iteration)
        fold_test_frame = test_frame.iloc[fold["valid_idx"]].copy()
        fold_reference = fold_test_frame.loc[:, [REFERENCE_DATE_COL, REFERENCE_PRODUCT_COL]].copy()
        absolute_predictions = _predict_absolute(
            model,
            fold_test_frame,
            feature_cols=feature_cols,
            target_contract=target_contract,
        )
        fold_predictions = build_predictions_frame(
            fold_reference,
            actual=fold_test_frame[target_contract.absolute_target_col],
            prediction_raw=absolute_predictions,
            train_rows_used=known_history_rows,
        )
        daily_prediction_parts.append(fold_predictions)
        absolute_target = fold_test_frame[target_contract.absolute_target_col].astype(float)
        prediction_series = fold_predictions["prediction_raw"].astype(float)
        daily_reports.append(
            {
                "fold": int(fold["fold"]),
                "eval_date": eval_date,
                "history_rows": known_history_rows,
                "bakery_history_rows": int(len(bakery_history_frame)),
                "valid_rows": int(len(fold_predictions)),
                "best_iteration": int(best_iteration),
                "mae": float(np.mean(np.abs(absolute_target.to_numpy() - prediction_series.to_numpy()))),
                "rmse": float(
                    np.sqrt(np.mean(np.square(absolute_target.to_numpy() - prediction_series.to_numpy())))
                ),
            }
        )
        logger.info(
            "Daily evaluation refit complete: fold=%s/%s eval_date=%s history_rows=%s valid_rows=%s best_iteration=%s",
            fold["fold"],
            len(folds),
            eval_date,
            known_history_rows,
            len(fold_predictions),
            best_iteration,
        )

    predictions_df = pd.concat(daily_prediction_parts, ignore_index=True)
    predictions_df = predictions_df.sort_values(["product", "target_date"]).reset_index(drop=True)
    daily_report_df = pd.DataFrame(daily_reports).sort_values("fold").reset_index(drop=True)
    resolved_best_iteration = int(round(float(np.mean(best_iterations)))) if best_iterations else 2000
    return predictions_df, daily_report_df, resolved_best_iteration


def _compute_equal_dataset_row_weights(
    frame: pd.DataFrame,
    *,
    dataset_source_col: str,
) -> np.ndarray | None:
    if dataset_source_col not in frame.columns:
        return None
    dataset_counts = frame[dataset_source_col].value_counts(dropna=False)
    dataset_count = max(1, len(dataset_counts))
    relative_weights = frame[dataset_source_col].map(
        lambda dataset: 1.0 / (dataset_count * float(dataset_counts.loc[dataset]))
    ).to_numpy(dtype=float)
    return relative_weights * float(len(frame))


def _drop_constant_feature_columns(
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[str], list[str]]:
    constant_feature_cols = [
        column
        for column in feature_cols
        if frame[column].nunique(dropna=False) <= 1
    ]
    return [column for column in feature_cols if column not in set(constant_feature_cols)], constant_feature_cols


def _resolve_reference_product_col(frame: pd.DataFrame) -> str:
    for candidate in ("product_id", "product", "series_id", "sku_id"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Unable to resolve a product identifier column for evaluation.")


def _resolve_reference_date_col(frame: pd.DataFrame) -> str:
    for candidate in ("target_dt", "dt", "date"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Unable to resolve a date column for evaluation.")


def _normalize_reference_split_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[REFERENCE_DATE_COL] = pd.to_datetime(normalized[REFERENCE_DATE_COL])
    normalized[REFERENCE_PRODUCT_COL] = normalized[REFERENCE_PRODUCT_COL].astype(str)
    normalized[REFERENCE_TARGET_COL] = normalized[REFERENCE_TARGET_COL].astype(float)
    return normalized.sort_values([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL]).reset_index(drop=True)


def _build_bakery_reference_feature_frame(
    reference_full_df: pd.DataFrame,
    reference_test_df: pd.DataFrame,
    gold_base_panel_df: pd.DataFrame,
) -> pd.DataFrame:
    reference_full = _normalize_reference_split_frame(reference_full_df)
    reference_test = _normalize_reference_split_frame(reference_test_df)
    gold_base = gold_base_panel_df.copy()
    gold_base["dt"] = pd.to_datetime(gold_base["dt"])
    gold_base["target_dt"] = pd.to_datetime(gold_base["target_dt"])
    gold_base["product_id"] = gold_base["product_id"].astype(str)

    base = reference_full.rename(
        columns={
            REFERENCE_DATE_COL: "dt",
            REFERENCE_PRODUCT_COL: "product_id",
            REFERENCE_TARGET_COL: "current_day_demand_qty",
        }
    )
    base = base.loc[:, ["dt", "product_id", "current_day_demand_qty"]].copy()

    enrichment_cols = [
        column
        for column in gold_base.columns
        if column not in {"dt", "target_dt", "product_id", "current_day_demand_qty"}
    ]
    merged = base.merge(
        gold_base.loc[:, ["dt", "product_id", *enrichment_cols]],
        how="left",
        on=["dt", "product_id"],
    )

    date_template = gold_base.sort_values(["dt", "product_id"]).groupby("dt").first(numeric_only=False)
    product_template = gold_base.sort_values(["dt"]).groupby("product_id").last(numeric_only=False)
    price_template = gold_base.groupby("product_id")["avg_selling_price"].median(numeric_only=True)

    date_fill_cols = [
        "source_partition",
        "source_run_id",
        "location_open_flag",
        "day_complete_flag",
        "holiday_flag",
        "activity_flag",
        "holiday_name",
        "school_holiday_flag",
        "bridge_day_flag",
        "pre_holiday_flag",
        "post_holiday_flag",
        "snap_flag",
        "weather_precipitation",
        "weather_temperature",
        "weather_temperature_min",
        "weather_temperature_max",
        "weather_humidity",
        "weather_wind_level",
        "inflation_cpi_latest",
        "food_cpi_latest",
        "policy_rate_latest",
        "gdp_growth_latest",
        "gdp_quarterly_level_latest",
        "gdp_current_usd_latest",
        "unemployment_rate_latest",
        "consumer_confidence_latest",
        "retail_sales_index_latest",
        "lending_interest_rate_latest",
        "government_debt_pct_gdp_latest",
        "client_id",
        "vertical_level_1",
        "vertical_level_2",
        "country_code",
        "region_code",
        "city_name",
        "freshretail_rescaled_flag",
        "target_scale_assumption",
        "gold_run_id",
    ]
    product_fill_cols = [
        "series_id",
        "location_id",
        "region_id",
        "org_group_id",
        "category_level_1",
        "category_level_2",
        "category_level_3",
        "location_open_flag",
        "product_active_flag",
        "client_id",
        "vertical_level_1",
        "vertical_level_2",
        "country_code",
        "region_code",
        "city_name",
        "freshretail_rescaled_flag",
        "target_scale_assumption",
        "gold_run_id",
    ]
    for column in date_fill_cols:
        if column in merged.columns and column in date_template.columns:
            merged[column] = merged[column].fillna(merged["dt"].map(date_template[column]))
    for column in product_fill_cols:
        if column in merged.columns and column in product_template.columns:
            merged[column] = merged[column].fillna(merged["product_id"].map(product_template[column]))

    merged["dataset_source"] = "bakery"
    merged["location_id"] = merged["location_id"].fillna("bakery_store_1")
    merged["product_active_flag"] = merged.get("product_active_flag", True)
    merged["product_active_flag"] = merged["product_active_flag"].fillna(True)
    merged["location_open_flag"] = merged["location_open_flag"].fillna(True)
    merged["day_complete_flag"] = merged["day_complete_flag"].fillna(True)
    merged["missing_sales_flag"] = merged.get("is_missing_day", 0)
    merged["missing_sales_flag"] = merged["missing_sales_flag"].fillna(0).astype(bool)
    merged["is_observed_row"] = ~merged["missing_sales_flag"]
    merged["observed_stockout_flag"] = merged.get("observed_stockout_flag", False)
    merged["observed_stockout_flag"] = merged["observed_stockout_flag"].fillna(False)
    merged["observed_stockout_available"] = merged.get("observed_stockout_available", False)
    merged["observed_stockout_available"] = merged["observed_stockout_available"].fillna(False)
    merged["anomaly_flag"] = merged.get("anomaly_flag", False)
    merged["anomaly_flag"] = merged["anomaly_flag"].fillna(False)
    merged["true_zero_demand_flag"] = merged["current_day_demand_qty"].fillna(0.0).eq(0.0)
    merged["location_closed_flag"] = merged.get("location_closed_flag", False)
    merged["location_closed_flag"] = merged["location_closed_flag"].fillna(False)
    merged["avg_selling_price"] = merged["avg_selling_price"].fillna(merged["product_id"].map(price_template))
    merged["observed_discount_amount"] = merged.get("observed_discount_amount", 0.0)
    merged["observed_discount_amount"] = merged["observed_discount_amount"].fillna(0.0)
    merged["promo_flag"] = merged.get("promo_flag", False)
    merged["promo_flag"] = merged["promo_flag"].fillna(False)
    merged["observed_revenue_net"] = merged.get("observed_revenue_net", np.nan)
    revenue_fill = merged["current_day_demand_qty"].fillna(0.0) * merged["avg_selling_price"].fillna(0.0)
    merged["observed_revenue_net"] = merged["observed_revenue_net"].fillna(revenue_fill)
    merged["price_available"] = merged["avg_selling_price"].notna()
    merged["weather_available"] = (
        merged["weather_temperature"].notna()
        | merged["weather_precipitation"].notna()
        | merged["weather_humidity"].notna()
        | merged["weather_wind_level"].notna()
    )
    merged["macro_available"] = (
        merged["inflation_cpi_latest"].notna()
        | merged["food_cpi_latest"].notna()
        | merged["policy_rate_latest"].notna()
        | merged["gdp_growth_latest"].notna()
        | merged["gdp_quarterly_level_latest"].notna()
        | merged["gdp_current_usd_latest"].notna()
        | merged["unemployment_rate_latest"].notna()
        | merged["consumer_confidence_latest"].notna()
        | merged["retail_sales_index_latest"].notna()
        | merged["lending_interest_rate_latest"].notna()
        | merged["government_debt_pct_gdp_latest"].notna()
    )
    merged["target_dt"] = merged["dt"] + pd.Timedelta(days=1)
    merged = merged.sort_values(["product_id", "dt"]).reset_index(drop=True)

    product_group = merged.groupby("product_id", sort=False)
    quantity_series = cast(pd.Series, merged["current_day_demand_qty"]).astype(float)
    merged["target_demand_qty_d_plus_1"] = product_group["current_day_demand_qty"].shift(-1)
    merged["lag_1"] = product_group["current_day_demand_qty"].shift(1)
    merged["lag_7"] = product_group["current_day_demand_qty"].shift(7)
    merged["lag_14"] = product_group["current_day_demand_qty"].shift(14)
    merged["lag_21_same_dow"] = product_group["current_day_demand_qty"].shift(21)
    merged["lag_28"] = product_group["current_day_demand_qty"].shift(28)
    merged["target_lag_7"] = product_group["current_day_demand_qty"].shift(6)
    merged["target_lag_14"] = product_group["current_day_demand_qty"].shift(13)
    merged["target_lag_21"] = product_group["current_day_demand_qty"].shift(20)
    merged["target_lag_28"] = product_group["current_day_demand_qty"].shift(27)
    merged["rolling_mean_7"] = (
        product_group["current_day_demand_qty"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["rolling_mean_14"] = (
        product_group["current_day_demand_qty"].rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["rolling_mean_28"] = (
        product_group["current_day_demand_qty"].rolling(window=28, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["rolling_std_28"] = (
        product_group["current_day_demand_qty"].rolling(window=28, min_periods=1).std().reset_index(level=0, drop=True)
    )
    merged["naive_last_value"] = quantity_series
    merged["avg_selling_price_lag_1"] = product_group["avg_selling_price"].shift(1)
    merged["avg_selling_price_lag_7"] = product_group["avg_selling_price"].shift(7)
    merged["avg_selling_price_lag_28"] = product_group["avg_selling_price"].shift(28)
    merged["avg_selling_price_rolling_mean_7"] = (
        product_group["avg_selling_price"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["avg_selling_price_rolling_mean_28"] = (
        product_group["avg_selling_price"].rolling(window=28, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["observed_discount_amount_lag_1"] = product_group["observed_discount_amount"].shift(1)
    merged["observed_discount_amount_lag_7"] = product_group["observed_discount_amount"].shift(7)
    merged["observed_discount_amount_lag_28"] = product_group["observed_discount_amount"].shift(28)
    merged["observed_discount_amount_rolling_mean_7"] = (
        product_group["observed_discount_amount"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["observed_discount_amount_rolling_mean_28"] = (
        product_group["observed_discount_amount"].rolling(window=28, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["promo_flag_lag_1"] = product_group["promo_flag"].shift(1)
    merged["promo_flag_lag_7"] = product_group["promo_flag"].shift(7)
    merged["promo_flag_lag_28"] = product_group["promo_flag"].shift(28)
    merged["promo_rate_7"] = (
        product_group["promo_flag"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["promo_rate_28"] = (
        product_group["promo_flag"].rolling(window=28, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["activity_flag_lag_1"] = product_group["activity_flag"].shift(1)
    merged["activity_flag_lag_7"] = product_group["activity_flag"].shift(7)
    merged["activity_flag_lag_28"] = product_group["activity_flag"].shift(28)
    merged["activity_rate_7"] = (
        product_group["activity_flag"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["activity_rate_28"] = (
        product_group["activity_flag"].rolling(window=28, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["observed_stockout_flag_lag_1"] = product_group["observed_stockout_flag"].shift(1)
    merged["observed_stockout_flag_lag_7"] = product_group["observed_stockout_flag"].shift(7)
    merged["observed_stockout_flag_lag_28"] = product_group["observed_stockout_flag"].shift(28)
    merged["observed_stockout_rate_7"] = (
        product_group["observed_stockout_flag"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["observed_stockout_rate_28"] = (
        product_group["observed_stockout_flag"].rolling(window=28, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["weather_temperature_lag_0"] = merged["weather_temperature"]
    merged["weather_temperature_lag_1"] = product_group["weather_temperature"].shift(1)
    merged["weather_temperature_lag_7"] = product_group["weather_temperature"].shift(7)
    merged["weather_temperature_rolling_mean_7"] = (
        product_group["weather_temperature"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["weather_temperature_rolling_mean_28"] = (
        product_group["weather_temperature"].rolling(window=28, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["weather_temperature_min_lag_1"] = product_group["weather_temperature_min"].shift(1)
    merged["weather_temperature_min_lag_7"] = product_group["weather_temperature_min"].shift(7)
    merged["weather_temperature_min_rolling_mean_7"] = (
        product_group["weather_temperature_min"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["weather_temperature_max_lag_1"] = product_group["weather_temperature_max"].shift(1)
    merged["weather_temperature_max_lag_7"] = product_group["weather_temperature_max"].shift(7)
    merged["weather_temperature_max_rolling_mean_7"] = (
        product_group["weather_temperature_max"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["weather_precipitation_lag_0"] = merged["weather_precipitation"]
    merged["weather_precipitation_lag_1"] = product_group["weather_precipitation"].shift(1)
    merged["weather_precipitation_lag_7"] = product_group["weather_precipitation"].shift(7)
    merged["weather_precipitation_rolling_mean_7"] = (
        product_group["weather_precipitation"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["weather_precipitation_rolling_mean_28"] = (
        product_group["weather_precipitation"].rolling(window=28, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["weather_humidity_lag_0"] = merged["weather_humidity"]
    merged["weather_humidity_lag_1"] = product_group["weather_humidity"].shift(1)
    merged["weather_humidity_lag_7"] = product_group["weather_humidity"].shift(7)
    merged["weather_humidity_rolling_mean_7"] = (
        product_group["weather_humidity"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["weather_wind_level_lag_0"] = merged["weather_wind_level"]
    merged["weather_wind_level_lag_1"] = product_group["weather_wind_level"].shift(1)
    merged["weather_wind_level_lag_7"] = product_group["weather_wind_level"].shift(7)
    merged["weather_wind_level_rolling_mean_7"] = (
        product_group["weather_wind_level"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    merged["inflation_cpi_latest_lag_28"] = product_group["inflation_cpi_latest"].shift(28)
    merged["inflation_cpi_latest_delta_28"] = (
        merged["inflation_cpi_latest"] - merged["inflation_cpi_latest_lag_28"]
    )
    merged["food_cpi_latest_lag_28"] = product_group["food_cpi_latest"].shift(28)
    merged["food_cpi_latest_delta_28"] = merged["food_cpi_latest"] - merged["food_cpi_latest_lag_28"]
    merged["policy_rate_latest_lag_28"] = product_group["policy_rate_latest"].shift(28)
    merged["policy_rate_latest_delta_28"] = (
        merged["policy_rate_latest"] - merged["policy_rate_latest_lag_28"]
    )
    merged["gdp_growth_latest_lag_28"] = product_group["gdp_growth_latest"].shift(28)
    merged["gdp_growth_latest_delta_28"] = merged["gdp_growth_latest"] - merged["gdp_growth_latest_lag_28"]
    merged["gdp_quarterly_level_latest_lag_28"] = product_group["gdp_quarterly_level_latest"].shift(28)
    merged["gdp_quarterly_level_latest_delta_28"] = (
        merged["gdp_quarterly_level_latest"] - merged["gdp_quarterly_level_latest_lag_28"]
    )
    merged["gdp_current_usd_latest_lag_28"] = product_group["gdp_current_usd_latest"].shift(28)
    merged["gdp_current_usd_latest_delta_28"] = (
        merged["gdp_current_usd_latest"] - merged["gdp_current_usd_latest_lag_28"]
    )
    merged["unemployment_rate_latest_lag_28"] = product_group["unemployment_rate_latest"].shift(28)
    merged["unemployment_rate_latest_delta_28"] = (
        merged["unemployment_rate_latest"] - merged["unemployment_rate_latest_lag_28"]
    )
    merged["consumer_confidence_latest_lag_28"] = product_group["consumer_confidence_latest"].shift(28)
    merged["consumer_confidence_latest_delta_28"] = (
        merged["consumer_confidence_latest"] - merged["consumer_confidence_latest_lag_28"]
    )
    merged["retail_sales_index_latest_lag_28"] = product_group["retail_sales_index_latest"].shift(28)
    merged["retail_sales_index_latest_delta_28"] = (
        merged["retail_sales_index_latest"] - merged["retail_sales_index_latest_lag_28"]
    )
    merged["lending_interest_rate_latest_lag_28"] = product_group["lending_interest_rate_latest"].shift(28)
    merged["lending_interest_rate_latest_delta_28"] = (
        merged["lending_interest_rate_latest"] - merged["lending_interest_rate_latest_lag_28"]
    )
    merged["government_debt_pct_gdp_latest_lag_28"] = product_group["government_debt_pct_gdp_latest"].shift(28)
    merged["government_debt_pct_gdp_latest_delta_28"] = (
        merged["government_debt_pct_gdp_latest"] - merged["government_debt_pct_gdp_latest_lag_28"]
    )
    merged["target_holiday_flag"] = product_group["holiday_flag"].shift(-1)
    merged["target_holiday_name"] = product_group["holiday_name"].shift(-1)
    merged["target_school_holiday_flag"] = product_group["school_holiday_flag"].shift(-1)
    merged["target_bridge_day_flag"] = product_group["bridge_day_flag"].shift(-1)
    merged["target_pre_holiday_flag"] = product_group["pre_holiday_flag"].shift(-1)
    merged["target_post_holiday_flag"] = product_group["post_holiday_flag"].shift(-1)

    lag_cols = ["lag_7", "lag_14", "lag_21_same_dow", "lag_28"]
    target_lag_cols = ["target_lag_7", "target_lag_14", "target_lag_21", "target_lag_28"]
    merged["same_dow_mean_4w"] = merged[lag_cols].mean(axis=1, skipna=True)
    merged["target_same_dow_mean_4w"] = merged[target_lag_cols].mean(axis=1, skipna=True)
    merged["seasonal_naive_d7"] = merged["lag_7"]
    merged["target_seasonal_naive_d7"] = merged["target_lag_7"]
    merged["moving_average_7"] = merged["rolling_mean_7"]
    merged["moving_average_28"] = merged["rolling_mean_28"]
    valid_delta_mask = (
        merged["target_demand_qty_d_plus_1"].notna()
        & merged["target_lag_7"].notna()
        & (merged["target_demand_qty_d_plus_1"] >= 0.0)
        & (merged["target_lag_7"] >= 0.0)
    )
    merged["target_delta_log_wow_d_plus_1"] = np.nan
    merged.loc[valid_delta_mask, "target_delta_log_wow_d_plus_1"] = (
        np.log1p(merged.loc[valid_delta_mask, "target_demand_qty_d_plus_1"].astype(float))
        - np.log1p(merged.loc[valid_delta_mask, "target_lag_7"].astype(float))
    )
    merged["target_day_of_week"] = merged["target_dt"].dt.dayofweek
    merged["target_day_of_month"] = merged["target_dt"].dt.day
    merged["target_week_of_year"] = merged["target_dt"].dt.isocalendar().week.astype(int)
    merged["target_month"] = merged["target_dt"].dt.month
    merged["target_quarter"] = merged["target_dt"].dt.quarter
    merged["target_year"] = merged["target_dt"].dt.year
    merged["target_weekend_flag"] = merged["target_day_of_week"].isin([5, 6])
    merged["sin_target_day_of_week"] = np.sin(2 * np.pi * merged["target_day_of_week"].astype(float) / 7.0)
    merged["cos_target_day_of_week"] = np.cos(2 * np.pi * merged["target_day_of_week"].astype(float) / 7.0)
    merged["sin_target_week_of_year"] = np.sin(2 * np.pi * merged["target_week_of_year"].astype(float) / 53.0)
    merged["cos_target_week_of_year"] = np.cos(2 * np.pi * merged["target_week_of_year"].astype(float) / 53.0)
    merged["sin_target_month"] = np.sin(2 * np.pi * merged["target_month"].astype(float) / 12.0)
    merged["cos_target_month"] = np.cos(2 * np.pi * merged["target_month"].astype(float) / 12.0)

    target_rows = merged.merge(
        reference_test.loc[:, [REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL, REFERENCE_TARGET_COL]],
        how="right",
        left_on=["product_id", "target_dt"],
        right_on=[REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL],
    )
    target_rows["product_id"] = target_rows["product_id"].fillna(target_rows[REFERENCE_PRODUCT_COL]).astype(str)
    target_rows["target_dt"] = pd.to_datetime(target_rows[REFERENCE_DATE_COL])
    target_rows["target_demand_qty_d_plus_1"] = target_rows[REFERENCE_TARGET_COL].astype(float)
    target_rows["target_delta_log_wow_d_plus_1"] = np.where(
        target_rows["target_lag_7"].notna() & (target_rows["target_lag_7"].astype(float) >= 0.0),
        np.log1p(target_rows["target_demand_qty_d_plus_1"].astype(float))
        - np.log1p(target_rows["target_lag_7"].astype(float)),
        np.nan,
    )
    target_rows[REFERENCE_PRODUCT_COL] = target_rows[REFERENCE_PRODUCT_COL].astype(str)
    target_rows[REFERENCE_DATE_COL] = pd.to_datetime(target_rows[REFERENCE_DATE_COL])
    return target_rows.sort_values([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL]).reset_index(drop=True)


def _to_reference_frame(
    frame: pd.DataFrame,
    *,
    target_col: str,
) -> pd.DataFrame:
    date_col = _resolve_reference_date_col(frame)
    product_col = _resolve_reference_product_col(frame)
    reference = pd.DataFrame(
        {
            REFERENCE_DATE_COL: pd.to_datetime(frame[date_col]),
            REFERENCE_PRODUCT_COL: frame[product_col].astype(str),
            REFERENCE_TARGET_COL: frame[target_col].astype(float),
        }
    )
    return reference.sort_values([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL]).reset_index(drop=True)


def _load_gold_bakery_overlap_test_frame(
    *,
    duckdb_path: str | Path,
    gold_table: str,
    reference_full_df: pd.DataFrame,
    reference_test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        gold_base_df = connection.execute(
            f"""
            select *
            from gold.gold_base_panel_d1
            where dataset_source = 'bakery'
            """
        ).fetchdf()
    finally:
        connection.close()

    overlap_df = _build_bakery_reference_feature_frame(reference_full_df, reference_test_df, gold_base_df)
    if overlap_df.empty:
        raise ValueError("No bakery reference rows were materialized for evaluation.")

    scored_reference_test = _normalize_reference_split_frame(reference_test_df)
    actual_diff = float(
        np.abs(
            scored_reference_test[REFERENCE_TARGET_COL].astype(float).to_numpy()
            - overlap_df["target_demand_qty_d_plus_1"].astype(float).to_numpy()
        ).max()
    )
    if actual_diff > 0.0:
        raise ValueError(
            f"Materialized bakery reference target does not match bakery_sales actuals (max_abs_diff={actual_diff})."
        )

    overlap_start_date = cast(pd.Timestamp, scored_reference_test[REFERENCE_DATE_COL].min())
    metadata = {
        "reference_test_rows": int(len(reference_test_df)),
        "overlap_test_rows": int(len(scored_reference_test)),
        "missing_reference_test_rows": 0,
        "overlap_start_date": overlap_start_date.strftime("%Y-%m-%d"),
        "overlap_end_date": cast(pd.Timestamp, scored_reference_test[REFERENCE_DATE_COL].max()).strftime("%Y-%m-%d"),
        "product_count": int(scored_reference_test[REFERENCE_PRODUCT_COL].nunique()),
    }
    return overlap_df, scored_reference_test, metadata


def _prepare_training_target_frame(
    frame: pd.DataFrame,
    *,
    target_contract: TargetContract,
) -> pd.DataFrame:
    prepared = ensure_learning_target_column(frame.copy(), target_contract)
    learning_target_col = target_contract.learning_target_col
    return prepared[prepared[learning_target_col].notna()].copy()


def _prepare_scored_test_frame(
    test_frame: pd.DataFrame,
    scored_reference_test: pd.DataFrame,
    *,
    target_contract: TargetContract,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    prepared = ensure_learning_target_column(test_frame.copy(), target_contract)
    valid_mask = prepared[target_contract.learning_target_col].notna().to_numpy()
    filtered_test = prepared.loc[valid_mask].copy().reset_index(drop=True)
    filtered_reference = scored_reference_test.loc[valid_mask].copy().reset_index(drop=True)
    dropped_rows = int((~valid_mask).sum())
    return filtered_test, filtered_reference, dropped_rows


def _select_feature_columns(
    train_frame: pd.DataFrame,
    *,
    learning_target_col: str,
    absolute_target_col: str,
) -> tuple[list[str], list[str], list[str]]:
    numeric_feature_cols = select_numeric_feature_columns(
        train_frame,
        excluded_cols={
            DEFAULT_DATE_COL,
            learning_target_col,
            absolute_target_col,
            *DEFAULT_EVALUATION_EXCLUDED_HELPER_FEATURE_COLS,
            *DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
        },
    )
    identifier_feature_cols = [
        column
        for column in DEFAULT_IDENTIFIER_FEATURE_COLS
        if column in train_frame.columns and column not in numeric_feature_cols
    ]
    raw_feature_cols = [*numeric_feature_cols, *identifier_feature_cols]
    filtered_feature_cols, constant_feature_cols = _drop_constant_feature_columns(train_frame, raw_feature_cols)
    return filtered_feature_cols, constant_feature_cols, identifier_feature_cols


def _fit_evaluation_model(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
) -> tuple[Any, int]:
    learning_target_col = target_contract.learning_target_col
    train_weights = _compute_equal_dataset_row_weights(train_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL)
    valid_weights = _compute_equal_dataset_row_weights(valid_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL)
    train_for_model = train_frame.copy()
    valid_for_model = valid_frame.copy()
    if target_contract.target_mode == "log1p":
        train_for_model[learning_target_col] = np.log1p(train_for_model[learning_target_col].astype(float))
        valid_for_model[learning_target_col] = np.log1p(valid_for_model[learning_target_col].astype(float))
    model = fit_xgboost_booster(
        train_for_model,
        feature_cols,
        target_col=learning_target_col,
        model_params=model_params,
        default_params=DEFAULT_XGBOOST_MODEL_PARAMS,
        default_n_estimators=2000,
        valid_frame=valid_for_model,
        early_stopping_rounds=50,
        train_weights=train_weights,
        valid_weights=valid_weights,
    )
    best_iteration = getattr(model, "best_iteration", None)
    resolved_rounds = int(best_iteration + 1) if best_iteration is not None and best_iteration >= 0 else 2000
    return model, resolved_rounds


def _predict_absolute(
    model: Any,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_contract: TargetContract,
) -> np.ndarray:
    predictions = predict_with_xgboost_booster(model, frame, feature_cols)
    return reconstruct_absolute_predictions(predictions, frame, target_contract)


def _fit_final_model(
    *,
    fit_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
    num_boost_round: int,
) -> Any:
    learning_target_col = target_contract.learning_target_col
    fit_weights = _compute_equal_dataset_row_weights(fit_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL)
    fit_for_model = fit_frame.copy()
    if target_contract.target_mode == "log1p":
        fit_for_model[learning_target_col] = np.log1p(fit_for_model[learning_target_col].astype(float))
    return fit_xgboost_booster(
        fit_for_model,
        feature_cols,
        target_col=learning_target_col,
        model_params=model_params,
        default_params=DEFAULT_XGBOOST_MODEL_PARAMS,
        default_n_estimators=num_boost_round,
        train_weights=fit_weights,
    )


def _build_diagnostics_payload(
    predictions_df: pd.DataFrame,
    *,
    evaluation_mode: str,
    feature_count: int,
    best_iteration: int,
    overlap_metadata: dict[str, object] | None,
    daily_refit: bool,
) -> dict[str, Any]:
    residuals = cast(pd.Series, predictions_df["actual"]).astype(float) - cast(
        pd.Series, predictions_df["prediction_raw"]
    ).astype(float)
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
    }
    if overlap_metadata is not None:
        payload["reference_overlap"] = overlap_metadata
    return payload


def _plot_actual_vs_predicted(predictions_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    x_axis = np.arange(len(predictions_df))
    ax.plot(x_axis, predictions_df["actual"], label="Actual sales", linewidth=1.5, color="#0072B2")
    ax.plot(x_axis, predictions_df["prediction_raw"], label="Predicted sales", linewidth=1.2, color="#D55E00")
    ax.set_title("Test set: actual vs predicted daily demand")
    ax.set_xlabel("Test observation index")
    ax.set_ylabel("Demand on next day")
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals(predictions_df: pd.DataFrame, output_path: Path) -> None:
    residuals = cast(pd.Series, predictions_df["actual"]).astype(float) - cast(
        pd.Series, predictions_df["prediction_raw"]
    ).astype(float)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(residuals)), residuals, color="#0072B2", linewidth=1.2)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_title("Test residuals over time")
    ax.set_xlabel("Test observation index")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_local_mode_frames(
    *,
    train_selection_input_path: str | Path,
    train_tuning_input_path: str | Path,
    val_input_path: str | Path,
    requested_target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object] | None]:
    train_frame = read_parquet_projected(Path(train_selection_input_path))
    valid_frame = read_parquet_projected(Path(train_tuning_input_path))
    test_frame = read_parquet_projected(Path(val_input_path))
    combined_history = pd.concat([train_frame, valid_frame], ignore_index=True)
    target_contract = resolve_target_contract(combined_history, test_frame, requested_target_col=requested_target_col)
    absolute_target_col = target_contract.absolute_target_col
    history_reference = _to_reference_frame(combined_history, target_col=absolute_target_col)
    test_reference = _to_reference_frame(test_frame, target_col=absolute_target_col)
    test_frame = test_frame.copy()
    test_frame[REFERENCE_DATE_COL] = test_reference[REFERENCE_DATE_COL].to_numpy()
    test_frame[REFERENCE_PRODUCT_COL] = test_reference[REFERENCE_PRODUCT_COL].to_numpy()
    return train_frame, valid_frame, test_frame, history_reference, test_reference, None


def _load_gold_reference_mode_frames(
    *,
    duckdb_path: str | Path,
    gold_table: str,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
    bakery_reference_train_csv: str | Path,
    bakery_reference_val_csv: str | Path,
    bakery_reference_test_csv: str | Path,
    train_frame: pd.DataFrame | None = None,
    valid_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    resolved_train_frame = train_frame
    resolved_valid_frame = valid_frame
    if resolved_train_frame is None or resolved_valid_frame is None:
        resolved_train_frame, resolved_valid_frame, _, _, _ = load_gold_train_tuning_frames(
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            date_col=DEFAULT_DATE_COL,
            dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
            train_sample_fraction=train_sample_fraction,
            tuning_sample_fraction=tuning_sample_fraction,
        )
    reference_train = load_reference_split(bakery_reference_train_csv)
    reference_val = load_reference_split(bakery_reference_val_csv)
    reference_test = load_reference_split(bakery_reference_test_csv)
    reference_full = pd.concat([reference_train, reference_val, reference_test], ignore_index=True)
    overlap_test_frame, scored_reference_test, overlap_metadata = _load_gold_bakery_overlap_test_frame(
        duckdb_path=duckdb_path,
        gold_table=gold_table,
        reference_full_df=reference_full,
        reference_test_df=reference_test,
    )
    history_reference = pd.concat([reference_train, reference_val], ignore_index=True)
    history_reference = history_reference.sort_values(
        [REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL]
    ).reset_index(drop=True)
    return (
        resolved_train_frame,
        resolved_valid_frame,
        overlap_test_frame,
        history_reference,
        scored_reference_test,
        overlap_metadata,
    )


def build_evaluation_outputs(
    train_selection_input_path: str | Path | None = DEFAULT_TRAIN_SELECTION_INPUT_PATH,
    train_tuning_input_path: str | Path | None = DEFAULT_TRAIN_TUNING_INPUT_PATH,
    val_input_path: str | Path | None = DEFAULT_VAL_INPUT_PATH,
    best_params_path: str | Path = DEFAULT_BEST_PARAMS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    target_col: str = DEFAULT_REQUESTED_TARGET_COL,
    duckdb_path: str | Path = DEFAULT_DUCKDB_PATH,
    gold_table: str = DEFAULT_GOLD_TABLE,
    train_sample_fraction: float = DEFAULT_TRAIN_SAMPLE_FRACTION,
    tuning_sample_fraction: float = DEFAULT_TUNING_SAMPLE_FRACTION,
    bakery_reference_train_csv: str | Path = DEFAULT_BAKERY_REFERENCE_TRAIN_CSV,
    bakery_reference_val_csv: str | Path = DEFAULT_BAKERY_REFERENCE_VAL_CSV,
    bakery_reference_test_csv: str | Path = DEFAULT_BAKERY_REFERENCE_TEST_CSV,
) -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    best_params = load_best_params(best_params_path)

    explicit_local_mode = (
        train_selection_input_path is not None
        and train_tuning_input_path is not None
        and val_input_path is not None
    )
    if explicit_local_mode:
        logger.info(
            "Running evaluation in local parquet mode: train=%s valid=%s test=%s",
            train_selection_input_path,
            train_tuning_input_path,
            val_input_path,
        )
        (
            train_frame,
            valid_frame,
            test_frame,
            history_reference,
            scored_reference_test,
            overlap_metadata,
        ) = _load_local_mode_frames(
            train_selection_input_path=train_selection_input_path,
            train_tuning_input_path=train_tuning_input_path,
            val_input_path=val_input_path,
            requested_target_col=target_col,
        )
        evaluation_mode = "local_parquet"
    else:
        logger.info(
            (
                "Running evaluation in bakery reference mode: duckdb_path=%s gold_table=%s "
                "train_sample_fraction=%.4f tuning_sample_fraction=%.4f"
            ),
            duckdb_path,
            gold_table,
            train_sample_fraction,
            tuning_sample_fraction,
        )
        (
            train_frame,
            valid_frame,
            test_frame,
            history_reference,
            scored_reference_test,
            overlap_metadata,
        ) = _load_gold_reference_mode_frames(
            duckdb_path=duckdb_path,
            gold_table=gold_table,
            train_sample_fraction=train_sample_fraction,
            tuning_sample_fraction=tuning_sample_fraction,
            bakery_reference_train_csv=bakery_reference_train_csv,
            bakery_reference_val_csv=bakery_reference_val_csv,
            bakery_reference_test_csv=bakery_reference_test_csv,
        )
        evaluation_mode = "bakery_reference_overlap"

    combined_pretest_frame = pd.concat([train_frame, valid_frame], ignore_index=True)
    target_contract = resolve_target_contract(
        combined_pretest_frame,
        test_frame,
        requested_target_col=target_col,
    )
    absolute_target_col = target_contract.absolute_target_col
    train_frame = _prepare_training_target_frame(train_frame, target_contract=target_contract)
    valid_frame = _prepare_training_target_frame(valid_frame, target_contract=target_contract)
    test_frame, scored_reference_test, dropped_test_rows = _prepare_scored_test_frame(
        test_frame,
        scored_reference_test,
        target_contract=target_contract,
    )
    combined_pretest_frame = pd.concat([train_frame, valid_frame], ignore_index=True)

    feature_cols, constant_feature_cols, identifier_feature_cols = _select_feature_columns(
        combined_pretest_frame,
        learning_target_col=target_contract.learning_target_col,
        absolute_target_col=absolute_target_col,
    )
    missing_in_test_feature_cols = [column for column in feature_cols if column not in test_frame.columns]
    if missing_in_test_feature_cols:
        logger.info(
            "Dropping evaluation features unavailable at inference time: %s",
            missing_in_test_feature_cols,
        )
        feature_cols = [column for column in feature_cols if column in test_frame.columns]
    logger.info(
        (
            "Loaded evaluation datasets: mode=%s train_rows=%s valid_rows=%s test_rows=%s "
            "feature_count=%s learning_target_col=%s absolute_target_col=%s"
        ),
        evaluation_mode,
        len(train_frame),
        len(valid_frame),
        len(test_frame),
        len(feature_cols),
        target_contract.learning_target_col,
        absolute_target_col,
    )
    logger.info("Target contract for evaluation: %s", build_target_contract_metadata(target_contract))
    if identifier_feature_cols:
        logger.info("Identifier features included in evaluation: %s", identifier_feature_cols)
    if constant_feature_cols:
        logger.info("Constant features dropped before evaluation: %s", constant_feature_cols)
    if overlap_metadata is not None:
        overlap_metadata = {
            **overlap_metadata,
            "scorable_test_rows": int(len(test_frame)),
            "dropped_test_rows_without_learning_target": int(dropped_test_rows),
        }
        logger.info("Bakery reference overlap: %s", overlap_metadata)

    predictions_df, daily_report_df, best_iteration = _evaluate_daily_refit_predictions(
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        model_params=best_params,
    )
    baselines_payload = build_statistical_baselines_payload(history_reference, scored_reference_test)
    predictions_df, baseline_savings_payload = enrich_predictions_with_best_baseline(
        predictions_df,
        baselines_payload,
        unit_cost_eur=DEFAULT_UNIT_COST_EUR,
    )
    metrics_payload = {
        "model_family": DEFAULT_MODEL_FAMILY,
        **compute_metrics_payload(history_reference, predictions_df),
    }
    if baseline_savings_payload is not None:
        metrics_payload["business_impact"] = baseline_savings_payload
    diagnostics_payload = _build_diagnostics_payload(
        predictions_df,
        evaluation_mode=evaluation_mode,
        feature_count=len(feature_cols),
        best_iteration=best_iteration,
        overlap_metadata=overlap_metadata,
        daily_refit=True,
    )
    economic_gain_payload = build_simple_economic_gain_payload(
        predictions_df,
        metrics_payload,
    )

    final_refit_frame = pd.concat([train_frame, valid_frame, test_frame], ignore_index=True)
    final_model = _fit_final_model(
        fit_frame=final_refit_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        model_params=best_params,
        num_boost_round=best_iteration,
    )

    metrics_output_path = target_dir / "foundation_xgboost_test_metrics.json"
    baselines_output_path = target_dir / "foundation_xgboost_statistical_baselines.json"
    predictions_output_path = target_dir / "foundation_xgboost_test_predictions.csv"
    diagnostics_output_path = target_dir / "foundation_xgboost_test_diagnostics.json"
    model_card_output_path = target_dir / "foundation_xgboost_model_card.json"
    final_model_output_path = target_dir / "foundation_xgboost_final_model.json"
    metadata_output_path = target_dir / "foundation_xgboost_evaluation_metadata.json"
    economic_gain_output_path = target_dir / "foundation_xgboost_economic_gain_vs_best_baseline.json"
    daily_report_output_path = target_dir / "foundation_xgboost_daily_refit_metrics.csv"
    predictions_plot_path = target_dir / "foundation_xgboost_actual_vs_predicted.png"
    residuals_plot_path = target_dir / "foundation_xgboost_residuals.png"

    _json_dump(metrics_output_path, metrics_payload)
    _json_dump(baselines_output_path, baselines_payload)
    predictions_df.to_csv(predictions_output_path, index=False)
    _json_dump(diagnostics_output_path, diagnostics_payload)
    _json_dump(economic_gain_output_path, economic_gain_payload)
    daily_report_df.to_csv(daily_report_output_path, index=False)
    final_model.save_model(str(final_model_output_path))
    _json_dump(
        metadata_output_path,
        {
            "evaluation_mode": evaluation_mode,
            "feature_cols": feature_cols,
            "constant_feature_cols": constant_feature_cols,
            "identifier_feature_cols": identifier_feature_cols,
            "excluded_risky_feature_cols": list(DEFAULT_EXCLUDED_RISKY_FEATURE_COLS),
            "excluded_helper_feature_cols": list(DEFAULT_EVALUATION_EXCLUDED_HELPER_FEATURE_COLS),
            "missing_in_test_feature_cols": missing_in_test_feature_cols,
            "best_iteration": int(best_iteration),
            "daily_refit": True,
            "train_rows": int(len(train_frame)),
            "valid_rows": int(len(valid_frame)),
            "test_rows": int(len(test_frame)),
            "history_reference_rows": int(len(history_reference)),
            "reference_test_rows": int(len(scored_reference_test)),
            "reference_train_csv": str(bakery_reference_train_csv),
            "reference_val_csv": str(bakery_reference_val_csv),
            "reference_test_csv": str(bakery_reference_test_csv),
            "duckdb_path": str(duckdb_path),
            "gold_table": gold_table,
            "train_sample_fraction": float(train_sample_fraction),
            "tuning_sample_fraction": float(tuning_sample_fraction),
            **build_target_contract_metadata(target_contract),
            "reference_overlap": overlap_metadata,
        },
    )
    _plot_actual_vs_predicted(predictions_df, predictions_plot_path)
    _plot_residuals(predictions_df, residuals_plot_path)

    model_card_payload: dict[str, Any] = {
        "model_family": DEFAULT_MODEL_FAMILY,
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "metrics": metrics_payload,
        "diagnostics": diagnostics_payload,
        "business_impact": baseline_savings_payload,
        "economic_gain": economic_gain_payload,
        "target_contract": build_target_contract_metadata(target_contract),
        "final_model_path": str(final_model_output_path),
        "predictions_path": str(predictions_output_path),
        "statistical_baselines_path": str(baselines_output_path),
        "evaluation_metadata_path": str(metadata_output_path),
        "daily_refit_metrics_path": str(daily_report_output_path),
    }
    _json_dump(model_card_output_path, model_card_payload)

    logger.info(
        (
            "Evaluation complete: mode=%s test_rows=%s mae=%.6f rmse=%.6f smape=%.6f "
            "best_baseline=%s estimated_savings=%.2f"
        ),
        evaluation_mode,
        metrics_payload["overall_metrics"]["test_rows"],
        metrics_payload["overall_metrics"]["mae"],
        metrics_payload["overall_metrics"]["rmse"],
        metrics_payload["overall_metrics"]["smape"],
        baseline_savings_payload["best_statistical_baseline_name"] if baseline_savings_payload is not None else None,
        economic_gain_payload["total_estimated_savings_eur_vs_best_baselines"],
    )
    for product_name, product_gain in economic_gain_payload["per_product_gain"].items():
        logger.info(
            (
                "Economic gain by product: product=%s best_baseline=%s model_mae=%.6f "
                "baseline_mae=%.6f absolute_mae_saved=%.6f estimated_savings_eur=%.2f"
            ),
            product_name,
            product_gain["best_baseline_name"],
            product_gain["model_mae"],
            product_gain["best_baseline_mae"],
            product_gain["absolute_mae_saved_vs_best_baseline"],
            product_gain["estimated_savings_eur_vs_best_baseline"],
        )
    return {
        "metrics_json": metrics_output_path,
        "statistical_baselines_json": baselines_output_path,
        "predictions_csv": predictions_output_path,
        "diagnostics_json": diagnostics_output_path,
        "model_card_json": model_card_output_path,
        "final_model": final_model_output_path,
        "evaluation_metadata": metadata_output_path,
        "economic_gain_json": economic_gain_output_path,
        "daily_refit_metrics_csv": daily_report_output_path,
        "actual_vs_predicted_plot": predictions_plot_path,
        "residuals_plot": residuals_plot_path,
    }

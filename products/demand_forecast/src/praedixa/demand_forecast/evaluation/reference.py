from __future__ import annotations

import numpy as np
import pandas as pd

from praedixa.demand_forecast.evaluation.bakery_style import REFERENCE_DATE_COL
from praedixa.demand_forecast.evaluation.bakery_style import REFERENCE_PRODUCT_COL
from praedixa.demand_forecast.evaluation.bakery_style import REFERENCE_TARGET_COL


def _normalize_reference_split_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[REFERENCE_DATE_COL] = pd.to_datetime(normalized[REFERENCE_DATE_COL])
    normalized[REFERENCE_PRODUCT_COL] = normalized[REFERENCE_PRODUCT_COL].astype(str)
    normalized[REFERENCE_TARGET_COL] = normalized[REFERENCE_TARGET_COL].astype(float)
    return normalized.sort_values([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL]).reset_index(drop=True)


def _build_bakery_overlap_feature_block(merged: pd.DataFrame) -> pd.DataFrame:
    product_group = merged.groupby("product_id", sort=False)
    quantity_series = merged["current_day_demand_qty"].astype(float)
    target_dt = pd.to_datetime(merged["target_dt"])

    gdp_growth_latest_lag_28 = product_group["gdp_growth_latest"].shift(28)
    gdp_current_usd_latest_lag_28 = product_group["gdp_current_usd_latest"].shift(28)
    lending_interest_rate_latest_lag_28 = product_group["lending_interest_rate_latest"].shift(28)
    government_debt_pct_gdp_latest_lag_28 = product_group["government_debt_pct_gdp_latest"].shift(28)

    feature_map: dict[str, object] = {
        "target_demand_qty_d_plus_1": product_group["current_day_demand_qty"].shift(-1),
        "lag_1": product_group["current_day_demand_qty"].shift(1),
        "lag_7": product_group["current_day_demand_qty"].shift(7),
        "lag_14": product_group["current_day_demand_qty"].shift(14),
        "lag_21": product_group["current_day_demand_qty"].shift(21),
        "lag_28": product_group["current_day_demand_qty"].shift(28),
        # `target_lag_7` is anchored on the forecast date (target_dt = dt + 1),
        # so the matching observed quantity sits six rows back on the current-day axis.
        "target_lag_7": product_group["current_day_demand_qty"].shift(6),
        "avg_selling_price_lag_1": product_group["avg_selling_price"].shift(1),
        "promo_flag_lag_1": product_group["promo_flag"].shift(1),
        "promo_rate_7": merged["promo_flag"].astype(float).groupby(merged["product_id"], sort=False).transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        ),
        "activity_rate_7": merged["activity_flag"].astype(float).groupby(merged["product_id"], sort=False).transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        ),
        "weather_temperature_lag_0": merged["weather_temperature"],
        "weather_temperature_lag_1": product_group["weather_temperature"].shift(1),
        "weather_precipitation_lag_0": merged["weather_precipitation"],
        "weather_humidity_lag_0": merged["weather_humidity"],
        "weather_wind_level_lag_0": merged["weather_wind_level"],
        "target_same_dow_mean_4w": quantity_series.groupby(
            [merged["product_id"], target_dt.dt.dayofweek], sort=False
        ).transform(lambda s: s.shift(7).rolling(4, min_periods=1).mean()),
        "gdp_growth_latest_delta_28": merged["gdp_growth_latest"] - gdp_growth_latest_lag_28,
        "gdp_current_usd_latest_delta_28": merged["gdp_current_usd_latest"] - gdp_current_usd_latest_lag_28,
        "lending_interest_rate_latest_delta_28": merged["lending_interest_rate_latest"] - lending_interest_rate_latest_lag_28,
        "government_debt_pct_gdp_latest_delta_28": merged["government_debt_pct_gdp_latest"] - government_debt_pct_gdp_latest_lag_28,
    }
    return pd.DataFrame(feature_map, index=merged.index)


def _reference_base_frame(reference_full_df: pd.DataFrame) -> pd.DataFrame:
    reference_full = _normalize_reference_split_frame(reference_full_df)
    base = reference_full.rename(
        columns={
            REFERENCE_DATE_COL: "dt",
            REFERENCE_PRODUCT_COL: "product_id",
            REFERENCE_TARGET_COL: "current_day_demand_qty",
        }
    )
    return base.loc[:, ["dt", "product_id", "current_day_demand_qty"]].copy()


def _normalized_gold_base_frame(gold_base_panel_df: pd.DataFrame) -> pd.DataFrame:
    gold_base = gold_base_panel_df.copy()
    gold_base["dt"] = pd.to_datetime(gold_base["dt"])
    gold_base["target_dt"] = pd.to_datetime(gold_base["target_dt"])
    gold_base["product_id"] = gold_base["product_id"].astype(str)
    return gold_base


def _reference_enrichment_columns(gold_base: pd.DataFrame) -> list[str]:
    return [
        column
        for column in gold_base.columns
        if column not in {"dt", "target_dt", "product_id", "current_day_demand_qty"}
    ]


def _merged_reference_frame(base: pd.DataFrame, gold_base: pd.DataFrame) -> pd.DataFrame:
    enrichment_cols = _reference_enrichment_columns(gold_base)
    return base.merge(
        gold_base.loc[:, ["dt", "product_id", *enrichment_cols]],
        how="left",
        on=["dt", "product_id"],
    )


def _reference_templates(gold_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    date_template = gold_base.sort_values(["dt", "product_id"]).groupby("dt").first(numeric_only=False)
    product_template = gold_base.sort_values(["dt"]).groupby("product_id").last(numeric_only=False)
    price_template = gold_base.groupby("product_id")["avg_selling_price"].median(numeric_only=True)
    return date_template, product_template, price_template


def _fill_template_columns(
    merged: pd.DataFrame,
    *,
    template: pd.DataFrame,
    fill_columns: list[str],
    key_column: str,
) -> None:
    for column in fill_columns:
        if column in merged.columns and column in template.columns:
            merged[column] = merged[column].fillna(merged[key_column].map(template[column]))


def _fill_reference_templates(
    merged: pd.DataFrame,
    *,
    date_template: pd.DataFrame,
    product_template: pd.DataFrame,
) -> None:
    _fill_template_columns(
        merged,
        template=date_template,
        fill_columns=_DATE_TEMPLATE_FILL_COLUMNS,
        key_column="dt",
    )
    _fill_template_columns(
        merged,
        template=product_template,
        fill_columns=_PRODUCT_TEMPLATE_FILL_COLUMNS,
        key_column="product_id",
    )


def _macro_available_flag(merged: pd.DataFrame) -> pd.Series:
    return (
        merged["gdp_growth_latest"].notna()
        | merged["gdp_current_usd_latest"].notna()
        | merged["lending_interest_rate_latest"].notna()
        | merged["government_debt_pct_gdp_latest"].notna()
    )


def _apply_reference_defaults(merged: pd.DataFrame, *, price_template: pd.Series) -> pd.DataFrame:
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
    merged["macro_available"] = _macro_available_flag(merged)
    merged["target_dt"] = merged["dt"] + pd.Timedelta(days=1)
    merged = merged.sort_values(["product_id", "dt"]).reset_index(drop=True)
    return pd.concat([merged, _build_bakery_overlap_feature_block(merged)], axis=1)


def _target_rows_from_reference(merged: pd.DataFrame, reference_test: pd.DataFrame) -> pd.DataFrame:
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


def build_bakery_reference_feature_frame(
    reference_full_df: pd.DataFrame,
    reference_test_df: pd.DataFrame,
    gold_base_panel_df: pd.DataFrame,
) -> pd.DataFrame:
    reference_test = _normalize_reference_split_frame(reference_test_df)
    gold_base = _normalized_gold_base_frame(gold_base_panel_df)
    merged = _merged_reference_frame(_reference_base_frame(reference_full_df), gold_base)
    date_template, product_template, price_template = _reference_templates(gold_base)
    _fill_reference_templates(merged, date_template=date_template, product_template=product_template)
    merged = _apply_reference_defaults(merged, price_template=price_template)
    return _target_rows_from_reference(merged, reference_test)


_DATE_TEMPLATE_FILL_COLUMNS = [
    "source_partition",
    "source_run_id",
    "location_open_flag",
    "day_complete_flag",
    "holiday_flag",
    "activity_flag",
    "school_holiday_flag",
    "bridge_day_flag",
    "pre_holiday_flag",
    "post_holiday_flag",
    "weather_precipitation",
    "weather_temperature",
    "weather_humidity",
    "weather_wind_level",
    "gdp_growth_latest",
    "gdp_current_usd_latest",
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

_PRODUCT_TEMPLATE_FILL_COLUMNS = [
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

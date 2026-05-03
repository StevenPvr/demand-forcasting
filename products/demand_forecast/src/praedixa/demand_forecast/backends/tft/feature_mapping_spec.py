from __future__ import annotations

from typing import Literal


TFTColumnRole = Literal[
    "group_id",
    "date",
    "target",
    "exclude",
    "static_categorical",
    "static_real",
    "time_varying_known_categorical",
    "time_varying_known_real",
    "time_varying_unknown_categorical",
    "time_varying_unknown_real",
]

FEATURE_ROLE_ORDER: tuple[TFTColumnRole, ...] = (
    "static_categorical",
    "static_real",
    "time_varying_known_categorical",
    "time_varying_known_real",
    "time_varying_unknown_categorical",
    "time_varying_unknown_real",
)

FEATURE_ROLE_SUFFIX_BY_ROLE: dict[TFTColumnRole, str] = {
    "static_categorical": "_static_cat",
    "static_real": "_static_real",
    "time_varying_known_categorical": "_known_cat",
    "time_varying_known_real": "_known_real",
    "time_varying_unknown_categorical": "_unknown_cat",
    "time_varying_unknown_real": "_unknown_real",
}


def feature_role_suffix(role: TFTColumnRole) -> str:
    suffix = FEATURE_ROLE_SUFFIX_BY_ROLE.get(role)
    if suffix is None:
        raise ValueError(f"TFT role `{role}` does not have a feature suffix.")
    return suffix


def suffixed_feature_name(column: str, role: TFTColumnRole) -> str:
    return f"{column}{feature_role_suffix(role)}"


def strip_feature_role_suffix(column: str) -> str:
    for suffix in sorted(FEATURE_ROLE_SUFFIX_BY_ROLE.values(), key=len, reverse=True):
        if column.endswith(suffix):
            return column.removesuffix(suffix)
    return column


def _suffix_columns(columns: tuple[str, ...], role: TFTColumnRole) -> tuple[str, ...]:
    return tuple(suffixed_feature_name(column, role) for column in columns)


TFT_LAG_HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 14, 28)

TFT_TARGET_LAG_HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28)

TFT_BOOLEAN_LAG_PREFIXES: tuple[str, ...] = (
    "promo_flag",
    "activity_flag",
    "observed_stockout_flag",
    "censor_flag",
    "day_complete_flag",
)

TFT_REAL_LAG_PREFIXES: tuple[str, ...] = (
    "label_quality_score",
    "observed_discount_amount",
    "observed_revenue_net",
    "weather_temperature",
    "weather_temperature_min",
    "weather_temperature_max",
    "weather_precipitation",
    "weather_humidity",
    "weather_wind_level",
)

TFT_BOOLEAN_LAG_COLUMNS: tuple[str, ...] = tuple(
    f"{prefix}_lag_{horizon}"
    for prefix in TFT_BOOLEAN_LAG_PREFIXES
    for horizon in TFT_LAG_HORIZONS
)

TFT_REAL_LAG_COLUMNS: tuple[str, ...] = (
    *(f"lag_{horizon}" for horizon in TFT_LAG_HORIZONS),
    *(f"target_lag_{horizon}" for horizon in TFT_TARGET_LAG_HORIZONS),
    *(
        f"{prefix}_lag_{horizon}"
        for prefix in TFT_REAL_LAG_PREFIXES
        for horizon in TFT_LAG_HORIZONS
    ),
)

EXCLUDED_FROM_TFT_LAG_COLUMNS: tuple[str, ...] = (
    "missing_sales_flag_lag_1",
    "missing_sales_flag_lag_7",
    "missing_sales_flag_lag_28",
    "anomaly_flag_lag_1",
    "anomaly_flag_lag_7",
    "anomaly_flag_lag_28",
    "location_closed_flag_lag_1",
    "location_closed_flag_lag_7",
    "location_closed_flag_lag_28",
    "kitchen_saturation_flag_lag_1",
    "kitchen_saturation_flag_lag_7",
    "kitchen_saturation_flag_lag_28",
    "assortment_restriction_flag_lag_1",
    "assortment_restriction_flag_lag_7",
    "assortment_restriction_flag_lag_28",
    "weather_temperature_lag_0",
    "weather_temperature_min_lag_0",
    "weather_temperature_max_lag_0",
    "weather_precipitation_lag_0",
    "weather_humidity_lag_0",
    "weather_wind_level_lag_0",
    "weather_humidity",
    "weather_wind_level",
    "lending_interest_rate_latest_lag_28",
    "avg_selling_price_lag_1",
    "avg_selling_price_lag_7",
    "avg_selling_price_lag_28",
    "closure_minutes_lag_1",
    "closure_minutes_lag_7",
    "closure_minutes_lag_28",
    "naive_last_value",
    "seasonal_naive_d7",
    "target_seasonal_naive_d7",
    "moving_average_7",
    "moving_average_28",
)

REMOVED_MODEL_INPUT_COLUMNS: tuple[str, ...] = (
    "series_id",
    "site_format",
    "service_model",
    "late_night_flag",
    "trade_area_type",
    "office_density_bucket",
    "residential_density_bucket",
    "competition_intensity_bucket",
    "menu_role",
    "price_band",
    "location_closed_flag",
    "location_open_flag",
    "product_active_flag",
    "missing_sales_flag",
    "anomaly_flag",
    "kitchen_saturation_flag",
    "assortment_restriction_flag",
    "target_scale_assumption",
    "price_available",
    "current_holiday_name",
    "current_school_holiday_flag",
    "current_school_holiday_name",
    "current_holiday_name_local",
    "school_holiday_flag_local",
    "current_school_holiday_name_local",
    "payday_flag",
    "target_holiday_name",
    "target_school_holiday_flag",
    "target_school_holiday_name",
    "closure_minutes",
    "avg_selling_price",
    "observed_revenue_net_rolling_mean_7",
    "observed_revenue_net_rolling_mean_28",
    "missing_sales_rate_7",
    "missing_sales_rate_28",
    "anomaly_rate_7",
    "anomaly_rate_28",
    "location_closed_rate_7",
    "location_closed_rate_28",
    "closure_minutes_rolling_mean_7",
    "closure_minutes_rolling_mean_28",
    "kitchen_saturation_rate_7",
    "kitchen_saturation_rate_28",
    "assortment_restriction_rate_7",
    "assortment_restriction_rate_28",
)

TFT_STATIC_CATEGORICAL_BASE_COLUMNS: tuple[str, ...] = (
    "dataset_source",
    "source_role",
    "is_synthetic_source",
    "vertical_level_1",
    "vertical_level_2",
    "commerce_modality",
    "operation_type",
    "service_pattern",
    "country_code",
    "region_code",
    "region_known_flag",
    "city_name",
    "drive_through_flag",
    "delivery_flag",
    "pickup_flag",
    "product_family",
    "product_subfamily",
    "product_subfamily_known_flag",
    "location_id",
    "product_id",
    "category_level_1",
    "category_level_1_known_flag",
    "category_level_2",
    "category_level_2_known_flag",
    "category_level_3",
    "category_level_3_known_flag",
)

TFT_STATIC_REAL_BASE_COLUMNS: tuple[str, ...] = ("product_taxonomy_depth",)

TFT_TIME_VARYING_KNOWN_CATEGORICAL_BASE_COLUMNS: tuple[str, ...] = (
    "cold_start_bucket",
    "is_observed_row",
    "true_zero_demand_flag",
    "day_complete_flag",
    "observed_stockout_flag",
    "observed_stockout_available",
    "promo_flag",
    "activity_flag",
    "target_day_of_week",
    "target_day_of_month",
    "target_week_of_year",
    "target_month",
    "target_quarter",
    "target_year",
    "current_holiday_flag",
    "month_start_flag",
    "month_end_flag",
    "target_weekend_flag",
    "target_holiday_flag",
)

TFT_TIME_VARYING_KNOWN_REAL_BASE_COLUMNS: tuple[str, ...] = (
    "history_available_days",
    "data_quality_metadata_unavailable_count",
    "data_quality_metadata_missing_count",
    "data_quality_pricing_promo_unavailable_count",
    "data_quality_pricing_promo_missing_count",
    "data_quality_weather_unavailable_count",
    "data_quality_weather_missing_count",
    "data_quality_macro_unavailable_count",
    "data_quality_macro_missing_count",
    "data_quality_operations_unavailable_count",
    "data_quality_operations_missing_count",
    "data_quality_history_unavailable_count",
    "data_quality_history_missing_count",
    "field_baseline_blend_lag_1_lag_7_d_plus_1",
    "current_day_demand_qty",
    "observed_discount_amount",
    "sin_target_day_of_week",
    "cos_target_day_of_week",
    "sin_target_week_of_year",
    "cos_target_week_of_year",
    "sin_target_month",
    "cos_target_month",
    "weather_temperature_rolling_mean_7",
    "weather_temperature_rolling_mean_28",
    "weather_precipitation_rolling_mean_7",
    "weather_precipitation_rolling_mean_28",
    "weather_humidity_rolling_mean_7",
    "weather_wind_level_rolling_mean_7",
    "lending_interest_rate_latest",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_28",
    "observed_discount_amount_rolling_mean_7",
    "observed_discount_amount_rolling_mean_28",
    "promo_rate_7",
    "promo_rate_28",
    "activity_rate_7",
    "activity_rate_28",
    "observed_stockout_rate_7",
    "observed_stockout_rate_28",
    "censor_rate_7",
    "censor_rate_28",
    "label_quality_score_rolling_mean_7",
    "label_quality_score_rolling_mean_28",
    "day_complete_rate_7",
    "day_complete_rate_28",
    "same_dow_mean_4w",
    "target_same_dow_mean_4w",
)

TFT_TIME_VARYING_UNKNOWN_CATEGORICAL_BASE_COLUMNS: tuple[str, ...] = (
    TFT_BOOLEAN_LAG_COLUMNS
)
TFT_TIME_VARYING_UNKNOWN_REAL_BASE_COLUMNS: tuple[str, ...] = TFT_REAL_LAG_COLUMNS

BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    *TFT_STATIC_CATEGORICAL_BASE_COLUMNS,
    *TFT_STATIC_REAL_BASE_COLUMNS,
    *TFT_TIME_VARYING_KNOWN_CATEGORICAL_BASE_COLUMNS,
    *TFT_TIME_VARYING_KNOWN_REAL_BASE_COLUMNS,
    *TFT_TIME_VARYING_UNKNOWN_CATEGORICAL_BASE_COLUMNS,
    *TFT_TIME_VARYING_UNKNOWN_REAL_BASE_COLUMNS,
)

BASE_FEATURE_ROLE_BY_COLUMN: dict[str, TFTColumnRole] = {
    **{column: "static_categorical" for column in TFT_STATIC_CATEGORICAL_BASE_COLUMNS},
    **{column: "static_real" for column in TFT_STATIC_REAL_BASE_COLUMNS},
    **{
        column: "time_varying_known_categorical"
        for column in TFT_TIME_VARYING_KNOWN_CATEGORICAL_BASE_COLUMNS
    },
    **{
        column: "time_varying_known_real"
        for column in TFT_TIME_VARYING_KNOWN_REAL_BASE_COLUMNS
    },
    **{
        column: "time_varying_unknown_categorical"
        for column in TFT_TIME_VARYING_UNKNOWN_CATEGORICAL_BASE_COLUMNS
    },
    **{
        column: "time_varying_unknown_real"
        for column in TFT_TIME_VARYING_UNKNOWN_REAL_BASE_COLUMNS
    },
}

NON_FEATURE_COLUMN_ROLES: dict[str, TFTColumnRole] = {
    **{column: "exclude" for column in EXCLUDED_FROM_TFT_LAG_COLUMNS},
    **{column: "exclude" for column in REMOVED_MODEL_INPUT_COLUMNS},
    "__tft_group_id": "exclude",
    "__tft_split": "exclude",
    "__tft_time_idx": "exclude",
    "__tft_sample_weight": "exclude",
    "__tft_prediction_row_id": "exclude",
    "dt": "date",
    "target": "target",
    "target_dt": "exclude",
    "forecast_horizon_days": "exclude",
    "split_bucket": "exclude",
    "source_partition": "exclude",
    "source_run_id": "exclude",
    "target_demand_qty_d_plus_1": "target",
    "target_residual_field_blend_lag_1_lag_7_d_plus_1": "target",
    "target_delta_log_wow_d_plus_1": "exclude",
    "target_semantics": "exclude",
    "censor_flag": "exclude",
    "target_source": "exclude",
    "label_quality_score": "exclude",
    "usable_for_training_flag": "exclude",
    "target_true_zero_demand_flag": "exclude",
    "observed_revenue_net": "exclude",
    "decision_timestamp": "exclude",
    "feature_availability_profile": "exclude",
    "rn_sample": "exclude",
    "source_legal_basis": "exclude",
    "source_license_type": "exclude",
    "source_review_status": "exclude",
    "source_legal_status_snapshot": "exclude",
    "stratum_row_count": "exclude",
    "sample_weight_source": "exclude",
    "sample_weight_business": "exclude",
    "is_primary_eval_dataset": "exclude",
    "gold_run_id": "exclude",
}

TFT_GROUP_ID_COLUMNS: tuple[str, ...] = ("client_id",)

TFT_STATIC_CATEGORICAL_COLUMNS: tuple[str, ...] = _suffix_columns(
    TFT_STATIC_CATEGORICAL_BASE_COLUMNS,
    "static_categorical",
)
TFT_STATIC_REAL_COLUMNS: tuple[str, ...] = _suffix_columns(
    TFT_STATIC_REAL_BASE_COLUMNS,
    "static_real",
)
TFT_TIME_VARYING_KNOWN_CATEGORICAL_COLUMNS: tuple[str, ...] = _suffix_columns(
    TFT_TIME_VARYING_KNOWN_CATEGORICAL_BASE_COLUMNS,
    "time_varying_known_categorical",
)
TFT_TIME_VARYING_KNOWN_REAL_COLUMNS: tuple[str, ...] = _suffix_columns(
    TFT_TIME_VARYING_KNOWN_REAL_BASE_COLUMNS,
    "time_varying_known_real",
)
TFT_TIME_VARYING_UNKNOWN_CATEGORICAL_COLUMNS: tuple[str, ...] = _suffix_columns(
    TFT_TIME_VARYING_UNKNOWN_CATEGORICAL_BASE_COLUMNS,
    "time_varying_unknown_categorical",
)
TFT_TIME_VARYING_UNKNOWN_REAL_COLUMNS: tuple[str, ...] = _suffix_columns(
    TFT_TIME_VARYING_UNKNOWN_REAL_BASE_COLUMNS,
    "time_varying_unknown_real",
)

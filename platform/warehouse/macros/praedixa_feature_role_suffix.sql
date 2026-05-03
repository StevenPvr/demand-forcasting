{% macro praedixa_feature_role_suffix(column_name) %}
    {%- set static_categorical_columns = [
        "dataset_source",
        "source_role",
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
    ] -%}
    {%- set static_real_columns = ["product_taxonomy_depth"] -%}
    {%- set known_categorical_columns = [
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
    ] -%}
    {%- set known_real_columns = [
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
    ] -%}
    {%- set boolean_lag_prefixes = [
        "promo_flag",
        "activity_flag",
        "observed_stockout_flag",
        "censor_flag",
        "day_complete_flag",
    ] -%}
    {%- set real_lag_prefixes = [
        "label_quality_score",
        "observed_discount_amount",
        "observed_revenue_net",
        "weather_temperature",
        "weather_temperature_min",
        "weather_temperature_max",
        "weather_precipitation",
        "weather_humidity",
        "weather_wind_level",
    ] -%}
    {%- if column_name.endswith("_lag_0") -%}
        {{- "" -}}
    {%- elif column_name in static_categorical_columns -%}
        {{- "_static_cat" -}}
    {%- elif column_name in static_real_columns -%}
        {{- "_static_real" -}}
    {%- elif column_name in known_categorical_columns -%}
        {{- "_known_cat" -}}
    {%- elif column_name in known_real_columns -%}
        {{- "_known_real" -}}
    {%- elif column_name.startswith("lag_") or column_name.startswith("target_lag_") -%}
        {{- "_unknown_real" -}}
    {%- elif "_lag_" in column_name and column_name.rsplit("_lag_", 1)[0] in boolean_lag_prefixes -%}
        {{- "_unknown_cat" -}}
    {%- elif "_lag_" in column_name and column_name.rsplit("_lag_", 1)[0] in real_lag_prefixes -%}
        {{- "_unknown_real" -}}
    {%- else -%}
        {{- "" -}}
    {%- endif -%}
{% endmacro %}

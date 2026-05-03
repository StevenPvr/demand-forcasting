{{ config(tags=["silver", "synthetic_foodservice"], materialized="table") }}

select
    cast(coalesce(dataset_source, source_policy_id) as varchar) as dataset_source,
    cast(source_partition as varchar) as source_partition,
    cast(source_run_id as varchar) as source_run_id,
    cast(series_id as varchar) as series_id,
    cast(dt as date) as dt,
    cast(location_id as varchar) as location_id,
    cast(product_id as varchar) as product_id,
    cast(region_id as varchar) as region_id,
    cast(org_group_id as varchar) as org_group_id,
    cast(category_level_1 as varchar) as category_level_1,
    cast(category_level_2 as varchar) as category_level_2,
    cast(category_level_3 as varchar) as category_level_3,
    try_cast(observed_demand_qty as double) as observed_demand_qty,
    observed_demand_qty is null as observed_demand_missing_flag,
    observed_demand_qty is not null
        and try_cast(observed_demand_qty as double) is null
        as observed_demand_parse_error_flag,
    case
        when lower(trim(cast(target_semantics as varchar))) in ('observed_sales', 'latent_demand_estimated')
        then lower(trim(cast(target_semantics as varchar)))
        else null
    end as target_semantics,
    cast(target_semantics as varchar) as target_semantics_raw,
    target_semantics is not null
        and lower(trim(cast(target_semantics as varchar))) not in ('observed_sales', 'latent_demand_estimated')
        as target_semantics_parse_error_flag,
    try_cast(censor_flag as boolean) as censor_flag,
    cast(target_source as varchar) as target_source,
    case
        when coalesce(try_cast(day_complete_flag as boolean), true) = false then 0.0
        when coalesce(cast(target_source as varchar), 'observed_sales') in ('closed_or_missing_observation', 'dense_calendar_zero_fill') then 0.0
        else 1.0
    end as label_quality_score,
    label_quality_score is null as label_quality_missing_flag,
    label_quality_score is not null
        and try_cast(label_quality_score as double) is null
        as label_quality_parse_error_flag,
    case
        when coalesce(try_cast(day_complete_flag as boolean), true) = false then false
        when coalesce(cast(target_source as varchar), 'observed_sales') in ('closed_or_missing_observation', 'dense_calendar_zero_fill') then false
        else true
    end as usable_for_training_flag,
    usable_for_training_flag is null as usable_for_training_missing_flag,
    censor_flag is null as censor_flag_missing_flag,
    try_cast(observed_revenue_net as double) as observed_revenue_net,
    try_cast(observed_discount_amount as double) as observed_discount_amount,
    try_cast(promo_flag as boolean) as promo_flag,
    try_cast(holiday_flag as boolean) as holiday_flag,
    try_cast(activity_flag as boolean) as activity_flag,
    try_cast(observed_stockout_flag as boolean) as observed_stockout_flag,
    try_cast(observed_stockout_available as boolean) as observed_stockout_available,
    observed_stockout_available is null as observed_stockout_available_missing_flag,
    try_cast(observed_stockout_intensity as double) as observed_stockout_intensity,
    try_cast(day_complete_flag as boolean) as day_complete_flag,
    cast(calendar_weekday_name as varchar) as calendar_weekday_name,
    try_cast(calendar_day_of_week as integer) as calendar_day_of_week,
    try_cast(calendar_month as integer) as calendar_month,
    try_cast(calendar_year as integer) as calendar_year,
    try_cast(calendar_week_key as integer) as calendar_week_key,
    cast(event_name_1 as varchar) as event_name_1,
    cast(event_type_1 as varchar) as event_type_1,
    cast(event_name_2 as varchar) as event_name_2,
    cast(event_type_2 as varchar) as event_type_2,
    try_cast(weather_precipitation as double) as weather_precipitation,
    try_cast(weather_temperature as double) as weather_temperature,
    try_cast(weather_humidity as double) as weather_humidity,
    try_cast(weather_wind_level as double) as weather_wind_level,
    cast(silver_run_id as varchar) as silver_run_id,
    cast(source_name as varchar) as source_name,
    cast(source_policy_id as varchar) as source_policy_id,
    cast(source_file_path as varchar) as source_file_path,
    cast(loaded_at as timestamp) as source_loaded_at,
    'synthetic' as data_origin
from {{ ref("stg_synthetic_foodservice_daily") }}

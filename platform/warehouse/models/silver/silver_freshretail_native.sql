{{ config(tags=["silver", "freshretail"], unique_key=["dataset_source", "source_partition", "dt", "location_id", "product_id"]) }}

select
    'freshretail' as dataset_source,
    source_partition,
    'warehouse_run' as source_run_id,
    concat(cast(store_id as varchar), '__', cast(product_id as varchar)) as series_id,
    cast(dt as date) as dt,
    cast(store_id as varchar) as location_id,
    cast(product_id as varchar) as product_id,
    cast(city_id as varchar) as region_id,
    cast(management_group_id as varchar) as org_group_id,
    cast(first_category_id as varchar) as category_level_1,
    cast(second_category_id as varchar) as category_level_2,
    cast(third_category_id as varchar) as category_level_3,
    try_cast(sale_amount as double) as observed_demand_qty,
    'observed_sales' as target_semantics,
    coalesce(try_cast(is_censored as boolean), false) as censor_flag,
    'observed_sales' as target_source,
    case when coalesce(try_cast(is_censored as boolean), false) then 0.5 else 1.0 end as label_quality_score,
    not coalesce(try_cast(is_censored as boolean), false) as usable_for_training_flag,
    cast(null as double) as observed_revenue_net,
    try_cast(discount as double) as observed_discount_amount,
    case
        when try_cast(discount as double) is null then null
        when try_cast(discount as double) <= 0 then false
        when try_cast(discount as double) = 1 then false
        when try_cast(discount as double) < 1 then true
        else true
    end as promo_flag,
    try_cast(holiday_flag as boolean) as holiday_flag,
    try_cast(activity_flag as boolean) as activity_flag,
    try_cast(is_censored as boolean) as observed_stockout_flag,
    true as observed_stockout_available,
    try_cast(stock_hour6_22_cnt as double) as observed_stockout_intensity,
    true as day_complete_flag,
    strftime(dt, '%A') as calendar_weekday_name,
    cast(strftime(dt, '%u') as integer) - 1 as calendar_day_of_week,
    cast(strftime(dt, '%m') as integer) as calendar_month,
    cast(strftime(dt, '%Y') as integer) as calendar_year,
    cast(strftime(dt, '%V') as integer) as calendar_week_key,
    cast(null as varchar) as event_name_1,
    cast(null as varchar) as event_type_1,
    cast(null as varchar) as event_name_2,
    cast(null as varchar) as event_type_2,
    try_cast(precpt as double) as weather_precipitation,
    try_cast(avg_temperature as double) as weather_temperature,
    try_cast(avg_humidity as double) as weather_humidity,
    try_cast(avg_wind_level as double) as weather_wind_level,
    '{{ env_var("PRAEDIXA_SILVER_RUN_ID", "manual") }}' as silver_run_id
from {{ ref("stg_freshretail_daily") }}

{{ config(tags=["silver", "freshretail"], unique_key=["dataset_source", "source_partition", "dt", "location_id", "product_id"]) }}

select
    'freshretail' as dataset_source,
    source_partition,
    'warehouse_run' as source_run_id,
    concat(cast(store_id as varchar), '__', cast(product_id as varchar)) as series_id,
    dt,
    cast(store_id as varchar) as location_id,
    cast(product_id as varchar) as product_id,
    cast(city_id as varchar) as region_id,
    cast(management_group_id as varchar) as org_group_id,
    cast(first_category_id as varchar) as category_level_1,
    cast(second_category_id as varchar) as category_level_2,
    cast(third_category_id as varchar) as category_level_3,
    sale_amount as observed_demand_qty,
    cast(null as double) as observed_revenue_net,
    discount as observed_discount_amount,
    cast(null as double) as avg_selling_price,
    case
        when discount is null then null
        when discount <= 0 then false
        when discount = 1 then false
        when discount < 1 then true
        else true
    end as promo_flag,
    holiday_flag,
    activity_flag,
    is_censored as observed_stockout_flag,
    true as observed_stockout_available,
    cast(stock_hour6_22_cnt as double) as observed_stockout_intensity,
    true as location_open_flag,
    true as day_complete_flag,
    false as missing_sales_flag,
    strftime(dt, '%A') as calendar_weekday_name,
    cast(strftime(dt, '%u') as integer) - 1 as calendar_day_of_week,
    cast(strftime(dt, '%m') as integer) as calendar_month,
    cast(strftime(dt, '%Y') as integer) as calendar_year,
    cast(strftime(dt, '%V') as integer) as calendar_week_key,
    cast(null as varchar) as event_name_1,
    cast(null as varchar) as event_type_1,
    cast(null as varchar) as event_name_2,
    cast(null as varchar) as event_type_2,
    cast(null as boolean) as snap_flag,
    precpt as weather_precipitation,
    avg_temperature as weather_temperature,
    avg_humidity as weather_humidity,
    avg_wind_level as weather_wind_level,
    false as anomaly_flag,
    '{{ env_var("PRAEDIXA_SILVER_RUN_ID", "manual") }}' as silver_run_id
from {{ ref("stg_freshretail_daily") }}

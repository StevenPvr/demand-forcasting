{{ config(tags=["staging", "freshretail"]) }}

select
    source_partition,
    city_id,
    store_id,
    management_group_id,
    first_category_id,
    second_category_id,
    third_category_id,
    product_id,
    cast(dt as date) as dt,
    cast(sale_amount as double) as sale_amount_raw,
    cast(sale_amount as double) as sale_amount,
    cast(stock_hour6_22_cnt as smallint) as stock_hour6_22_cnt,
    cast(discount as double) as discount,
    cast(holiday_flag as boolean) as holiday_flag,
    cast(activity_flag as boolean) as activity_flag,
    cast(precpt as double) as precpt,
    cast(avg_temperature as double) as avg_temperature,
    cast(avg_humidity as double) as avg_humidity,
    cast(avg_wind_level as double) as avg_wind_level,
    cast(is_censored as boolean) as is_censored,
    nullif(cast(source_name as varchar), '') as source_name,
    nullif(cast(source_policy_id as varchar), '') as source_policy_id,
    source_file_path,
    loaded_at
from {{ source("bronze", "bronze_freshretail_daily") }}

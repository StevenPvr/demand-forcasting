{{ config(tags=["silver", "capacity"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

select
    dataset_source,
    location_id,
    dt,
    case
        when bool_and(coalesce(location_open_flag, true)) then 0.0
        else 1440.0
    end as closure_minutes,
    false as channel_disabled_flag,
    bool_or(coalesce(observed_stockout_flag, false) and coalesce(observed_stockout_available, false)) as kitchen_saturation_flag,
    bool_or(coalesce(missing_sales_flag, false)) as assortment_restriction_flag,
    false as order_cutoff_flag
from {{ ref("silver_daily_product_demand") }}
group by
    dataset_source,
    location_id,
    dt

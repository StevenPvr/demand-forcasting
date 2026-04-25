{{ config(tags=["silver", "capacity"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

select
    dataset_source,
    location_id,
    dt,
    false as channel_disabled_flag,
    false as order_cutoff_flag
from {{ ref("silver_daily_product_demand") }}
group by
    dataset_source,
    location_id,
    dt

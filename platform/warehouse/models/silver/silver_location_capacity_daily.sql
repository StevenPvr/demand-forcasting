{{ config(tags=["silver", "capacity"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

select
    dataset_source,
    location_id,
    dt,
    cast(null as boolean) as channel_disabled_flag,
    cast(null as boolean) as order_cutoff_flag,
    false as capacity_available_flag,
    'default_assumption' as capacity_source,
    true as assumption_flag
from {{ ref("silver_location_date_spine") }}

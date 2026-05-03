{{ config(tags=["silver", "profiles"], materialized="table", unique_key=["dataset_source", "location_id"]) }}

select
    dataset_source,
    location_id,
    timezone,
    drive_through_flag,
    delivery_flag,
    pickup_flag
from {{ ref("silver_open_location_metadata") }}

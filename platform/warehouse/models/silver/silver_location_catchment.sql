{{ config(tags=["silver", "catchment"], materialized="table", unique_key=["dataset_source", "location_id"]) }}

select
    dataset_source,
    location_id
from {{ ref("stg_open_location_catchment") }}

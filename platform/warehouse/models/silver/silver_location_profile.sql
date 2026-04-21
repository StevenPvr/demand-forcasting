{{ config(tags=["silver", "profiles"], materialized="table", unique_key=["dataset_source", "location_id"]) }}

select
    dataset_source,
    location_id,
    site_format,
    service_model,
    drive_through_flag,
    delivery_flag,
    pickup_flag,
    late_night_flag,
    trade_area_type,
    mall_flag,
    transit_hub_flag,
    tourism_flag,
    office_density_bucket,
    residential_density_bucket,
    competition_intensity_bucket
from {{ ref("silver_open_location_metadata") }}

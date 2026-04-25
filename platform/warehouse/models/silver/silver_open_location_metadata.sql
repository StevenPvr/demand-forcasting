{{ config(tags=["silver", "exogenous"], materialized="table", unique_key=["dataset_source", "location_id"]) }}

select
    dataset_source,
    location_id,
    max(country_code) as country_code,
    max(region_code) as region_code,
    max(city_name) as city_name,
    max(latitude) as latitude,
    max(longitude) as longitude,
    max(school_zone) as school_zone,
    max(weather_location_label) as weather_location_label,
    max(assumption_source) as assumption_source,
    bool_or(coalesce(drive_through_flag, false)) as drive_through_flag,
    bool_or(coalesce(delivery_flag, false)) as delivery_flag,
    bool_or(coalesce(pickup_flag, false)) as pickup_flag,
    bool_or(coalesce(mall_flag, false)) as mall_flag,
    bool_or(coalesce(transit_hub_flag, false)) as transit_hub_flag,
    bool_or(coalesce(tourism_flag, false)) as tourism_flag
from {{ ref("stg_open_location_metadata") }}
group by
    dataset_source,
    location_id

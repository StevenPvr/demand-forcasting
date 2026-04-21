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
    max(assumption_source) as assumption_source
    ,
    max(site_format) as site_format,
    max(service_model) as service_model,
    bool_or(coalesce(drive_through_flag, false)) as drive_through_flag,
    bool_or(coalesce(delivery_flag, false)) as delivery_flag,
    bool_or(coalesce(pickup_flag, false)) as pickup_flag,
    bool_or(coalesce(late_night_flag, false)) as late_night_flag,
    max(trade_area_type) as trade_area_type,
    bool_or(coalesce(mall_flag, false)) as mall_flag,
    bool_or(coalesce(transit_hub_flag, false)) as transit_hub_flag,
    bool_or(coalesce(tourism_flag, false)) as tourism_flag,
    max(office_density_bucket) as office_density_bucket,
    max(residential_density_bucket) as residential_density_bucket,
    max(competition_intensity_bucket) as competition_intensity_bucket
from {{ ref("stg_open_location_metadata") }}
group by
    dataset_source,
    location_id

{{ config(tags=["silver", "exogenous"], unique_key=["dataset_source", "location_id"]) }}

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
from {{ ref("stg_open_location_metadata") }}
group by
    dataset_source,
    location_id

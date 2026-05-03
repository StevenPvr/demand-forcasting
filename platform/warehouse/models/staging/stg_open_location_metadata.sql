{{ config(tags=["silver", "exogenous"]) }}

select
    cast(dataset_source as varchar) as dataset_source,
    cast(location_id as varchar) as location_id,
    cast(country_code as varchar) as country_code,
    cast(region_code as varchar) as region_code,
    nullif(cast(city_name as varchar), '') as city_name,
    cast(latitude as double) as latitude,
    cast(longitude as double) as longitude,
    nullif(cast(school_zone as varchar), '') as school_zone,
    nullif(cast(weather_location_label as varchar), '') as weather_location_label,
    nullif(cast(assumption_source as varchar), '') as assumption_source,
    cast(drive_through_flag as boolean) as drive_through_flag,
    cast(delivery_flag as boolean) as delivery_flag,
    cast(pickup_flag as boolean) as pickup_flag,
    cast(mall_flag as boolean) as mall_flag,
    cast(transit_hub_flag as boolean) as transit_hub_flag,
    cast(tourism_flag as boolean) as tourism_flag,
    nullif(cast(source_name as varchar), '') as source_name,
    nullif(cast(source_policy_id as varchar), '') as source_policy_id,
    source_file_path,
    loaded_at
from {{ source('bronze', 'bronze_open_location_metadata') }}

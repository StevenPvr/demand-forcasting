{{ config(tags=["silver", "exogenous"], materialized="table", unique_key=["dataset_source", "location_id"]) }}

with source_rows as (
    select stg.*
    from {{ ref("stg_open_location_metadata") }} as stg
    inner join {{ ref("silver_allowed_provider_sources") }} as allowed
      on coalesce(stg.source_policy_id, stg.source_name) = allowed.source_id
       or stg.source_name = allowed.source_id
)
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
    bool_or(coalesce(tourism_flag, false)) as tourism_flag,
    max(source_name) as source_name,
    max(coalesce(source_policy_id, source_name)) as source_policy_id
from source_rows
group by
    dataset_source,
    location_id

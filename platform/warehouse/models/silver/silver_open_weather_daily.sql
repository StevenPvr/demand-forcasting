{{ config(tags=["silver", "exogenous"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

with source_rows as (
    select
        stg.*,
        {{ praedixa_canonical_source_id("stg.source_policy_id", "stg.source_name") }} as canonical_source_id
    from {{ ref("stg_open_weather_daily") }} as stg
    inner join {{ ref("silver_allowed_provider_sources") }} as allowed
      on {{ praedixa_canonical_source_id("stg.source_policy_id", "stg.source_name") }} = allowed.source_id
)
select
    dataset_source,
    location_id,
    dt,
    max(latitude) as latitude,
    max(longitude) as longitude,
    avg(weather_temperature_mean) as weather_temperature_mean,
    avg(weather_temperature_min) as weather_temperature_min,
    avg(weather_temperature_max) as weather_temperature_max,
    avg(weather_precipitation_sum) as weather_precipitation_sum,
    avg(weather_relative_humidity_mean) as weather_relative_humidity_mean,
    avg(weather_wind_speed_mean) as weather_wind_speed_mean,
    max(source_name) as source_name,
    max(canonical_source_id) as source_policy_id,
    string_agg(canonical_source_id, ' | ' order by canonical_source_id) as source_policy_ids,
    count(distinct canonical_source_id) as provider_count,
    true as weather_available_flag
from source_rows
group by
    dataset_source,
    location_id,
    dt

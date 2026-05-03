{{ config(tags=["silver", "exogenous", "weather"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

select
    dataset_source,
    location_id,
    dt,
    latitude,
    longitude,
    weather_temperature_mean,
    weather_temperature_min,
    weather_temperature_max,
    weather_precipitation_sum,
    weather_relative_humidity_mean,
    weather_wind_speed_mean,
    source_name,
    source_policy_id,
    source_policy_ids,
    provider_count,
    weather_available_flag,
    'actual' as weather_observation_type
from {{ ref('silver_open_weather_daily') }}

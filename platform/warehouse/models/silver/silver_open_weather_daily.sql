{{ config(tags=["silver", "exogenous"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

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
    max(source_name) as source_name
from {{ ref("stg_open_weather_daily") }}
group by
    dataset_source,
    location_id,
    dt

{{ config(tags=["silver", "exogenous"]) }}

select
    cast(dataset_source as varchar) as dataset_source,
    cast(location_id as varchar) as location_id,
    cast(dt as date) as dt,
    cast(latitude as double) as latitude,
    cast(longitude as double) as longitude,
    cast(weather_temperature_mean as double) as weather_temperature_mean,
    cast(weather_temperature_min as double) as weather_temperature_min,
    cast(weather_temperature_max as double) as weather_temperature_max,
    cast(weather_precipitation_sum as double) as weather_precipitation_sum,
    cast(weather_relative_humidity_mean as double) as weather_relative_humidity_mean,
    cast(weather_wind_speed_mean as double) as weather_wind_speed_mean,
    nullif(cast(source_name as varchar), '') as source_name,
    source_file_path,
    loaded_at
from {{ source("bronze", "bronze_open_weather_daily") }}

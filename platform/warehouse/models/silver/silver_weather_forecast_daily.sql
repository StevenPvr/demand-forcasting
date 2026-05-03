{{ config(tags=["silver", "exogenous", "weather"], materialized="table", unique_key=["dataset_source", "location_id", "dt", "forecast_generated_at"]) }}

select
    cast(null as varchar) as dataset_source,
    cast(null as varchar) as location_id,
    cast(null as date) as dt,
    cast(null as timestamp) as forecast_generated_at,
    cast(null as double) as latitude,
    cast(null as double) as longitude,
    cast(null as double) as weather_temperature_mean,
    cast(null as double) as weather_temperature_min,
    cast(null as double) as weather_temperature_max,
    cast(null as double) as weather_precipitation_sum,
    cast(null as double) as weather_relative_humidity_mean,
    cast(null as double) as weather_wind_speed_mean,
    cast(null as varchar) as source_name,
    cast(null as varchar) as source_policy_id,
    cast(null as varchar) as source_policy_ids,
    cast(null as integer) as provider_count,
    false as weather_available_flag,
    'forecast' as weather_observation_type
where false

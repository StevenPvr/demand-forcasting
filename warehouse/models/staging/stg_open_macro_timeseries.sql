{{ config(tags=["silver", "exogenous"]) }}

select
    cast(country_code as varchar) as country_code,
    cast(metric_name as varchar) as metric_name,
    cast(source_series_id as varchar) as source_series_id,
    cast(source_frequency as varchar) as source_frequency,
    cast(metric_units as varchar) as metric_units,
    cast(observation_date as date) as observation_date,
    cast(period_end as date) as period_end,
    cast(effective_from as date) as effective_from,
    cast(metric_value as double) as metric_value,
    nullif(cast(source_name as varchar), '') as source_name,
    source_file_path,
    loaded_at
from {{ source("bronze", "bronze_open_macro_timeseries") }}

{{ config(tags=["silver", "exogenous"]) }}

select
    cast(country_code as varchar) as country_code,
    cast(indicator_code as varchar) as indicator_code,
    cast(metric_name as varchar) as metric_name,
    cast(frequency_code as varchar) as frequency_code,
    cast(observation_period as varchar) as observation_period,
    cast(effective_from as date) as effective_from,
    cast(metric_value as double) as metric_value,
    nullif(cast(source_name as varchar), '') as source_name,
    source_file_path,
    loaded_at
from {{ source("bronze", "bronze_open_macro_timeseries") }}

{{ config(tags=["silver", "exogenous"]) }}

select
    cast(country_code as varchar) as country_code,
    cast(indicator_code as varchar) as indicator_code,
    cast(metric_name as varchar) as metric_name,
    cast(observation_year as integer) as observation_year,
    cast(effective_from as date) as effective_from,
    cast(effective_from as date) as available_from,
    true as available_from_assumption_flag,
    cast(metric_value as double) as metric_value,
    nullif(cast(source_name as varchar), '') as source_name,
    nullif(cast(source_policy_id as varchar), '') as source_policy_id,
    source_file_path,
    loaded_at
from {{ source('bronze', 'bronze_open_macro_annual') }}

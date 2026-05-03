{{ config(tags=["silver", "exogenous"]) }}

select
    cast(country_code as varchar) as country_code,
    cast(dt as date) as dt,
    nullif(cast(holiday_name as varchar), '') as holiday_name,
    nullif(cast(holiday_local_name as varchar), '') as holiday_local_name,
    cast(global_flag as boolean) as global_flag,
    nullif(cast(counties_json as varchar), '') as counties_json,
    nullif(cast(holiday_types_json as varchar), '') as holiday_types_json,
    nullif(cast(source_name as varchar), '') as source_name,
    nullif(cast(source_policy_id as varchar), '') as source_policy_id,
    source_file_path,
    loaded_at
from {{ source('bronze', 'bronze_open_public_holidays') }}

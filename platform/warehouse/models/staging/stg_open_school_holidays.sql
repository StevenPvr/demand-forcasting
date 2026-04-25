{{ config(tags=["silver", "exogenous"]) }}

select
    cast(dataset_source as varchar) as dataset_source,
    cast(location_id as varchar) as location_id,
    cast(dt as date) as dt,
    nullif(cast(school_holiday_name as varchar), '') as school_holiday_name,
    nullif(cast(school_zone as varchar), '') as school_zone,
    nullif(cast(source_name as varchar), '') as source_name,
    nullif(cast(source_policy_id as varchar), '') as source_policy_id,
    source_file_path,
    loaded_at
from {{ source("bronze", "bronze_open_school_holidays") }}

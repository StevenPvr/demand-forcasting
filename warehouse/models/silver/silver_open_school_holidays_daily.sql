{{ config(tags=["silver", "exogenous"], unique_key=["dataset_source", "location_id", "dt"]) }}

select
    dataset_source,
    location_id,
    dt,
    true as school_holiday_flag,
    string_agg(school_holiday_name, ' | ' order by school_holiday_name) as school_holiday_name,
    max(school_zone) as school_zone,
    max(source_name) as source_name
from {{ ref("stg_open_school_holidays") }}
group by
    dataset_source,
    location_id,
    dt

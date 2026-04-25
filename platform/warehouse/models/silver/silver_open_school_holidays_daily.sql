{{ config(tags=["silver", "exogenous"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

with source_rows as (
    select stg.*
    from {{ ref("stg_open_school_holidays") }} as stg
    inner join {{ ref("silver_allowed_provider_sources") }} as allowed
      on coalesce(stg.source_policy_id, stg.source_name) = allowed.source_id
       or stg.source_name = allowed.source_id
)
select
    dataset_source,
    location_id,
    dt,
    true as school_holiday_flag,
    string_agg(school_holiday_name, ' | ' order by school_holiday_name) as school_holiday_name,
    max(school_zone) as school_zone,
    max(source_name) as source_name,
    max(coalesce(source_policy_id, source_name)) as source_policy_id
from source_rows
group by
    dataset_source,
    location_id,
    dt

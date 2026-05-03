{{ config(tags=["silver", "exogenous"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

with source_rows as (
    select
        stg.*,
        {{ praedixa_canonical_source_id('stg.source_policy_id', 'stg.source_name') }} as canonical_source_id
    from {{ ref('stg_open_school_holidays') }} as stg
    inner join {{ ref('silver_allowed_provider_sources') }} as allowed
      on {{ praedixa_canonical_source_id('stg.source_policy_id', 'stg.source_name') }} = allowed.source_id
),
positive_holidays as (
    select
        dataset_source,
        location_id,
        dt,
        true as school_holiday_flag,
        string_agg(school_holiday_name, ' | ' order by school_holiday_name) as school_holiday_name,
        max(school_zone) as school_zone,
        max(source_name) as source_name,
        max(canonical_source_id) as source_policy_id,
        true as school_holiday_available_flag
    from source_rows
    group by
        dataset_source,
        location_id,
        dt
)
select
    spine.dataset_source,
    spine.location_id,
    spine.dt,
    coalesce(positive.school_holiday_flag, false) as school_holiday_flag,
    positive.school_holiday_name,
    coalesce(positive.school_zone, spine.school_zone) as school_zone,
    positive.source_name,
    positive.source_policy_id,
    positive.school_holiday_available_flag is not null as school_holiday_available_flag
from {{ ref('silver_location_date_spine') }} as spine
left join positive_holidays as positive
  on spine.dataset_source = positive.dataset_source
 and spine.location_id = positive.location_id
 and spine.dt = positive.dt

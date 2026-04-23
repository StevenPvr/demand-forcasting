{{ config(tags=["silver", "calendar"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

with demand_dates as (
    select distinct
        demand.dataset_source,
        demand.location_id,
        demand.dt,
        metadata.country_code
    from {{ ref("silver_daily_product_demand") }} as demand
    left join {{ ref("silver_open_location_metadata") }} as metadata
      on demand.dataset_source = metadata.dataset_source
     and demand.location_id = metadata.location_id
),
holiday_calendar as (
    select
        country_code,
        dt,
        holiday_flag,
        holiday_name,
        pre_holiday_flag,
        post_holiday_flag,
        bridge_day_flag
    from {{ ref("silver_open_public_holiday_calendar_daily") }}
)
select
    demand_dates.dataset_source,
    demand_dates.location_id,
    demand_dates.dt,
    coalesce(school.school_holiday_flag, false) as school_holiday_flag_local,
    school.school_holiday_name,
    holiday_calendar.holiday_name,
    cast(strftime(demand_dates.dt, '%d') as integer) in (1, 2, 3) as payday_flag,
    cast(strftime(demand_dates.dt, '%d') as integer) <= 3 as month_start_flag,
    cast(strftime(demand_dates.dt + interval 1 day, '%m') as integer) != cast(strftime(demand_dates.dt, '%m') as integer) as month_end_flag,
    case when holiday_calendar.holiday_name is null then 0 else 1 end as event_count_local,
    case when holiday_calendar.holiday_name is null then 0.0 else 1.0 end as event_intensity_score
from demand_dates
left join holiday_calendar
  on demand_dates.country_code = holiday_calendar.country_code
 and demand_dates.dt = holiday_calendar.dt
left join {{ ref("silver_open_school_holidays_daily") }} as school
  on demand_dates.dataset_source = school.dataset_source
 and demand_dates.location_id = school.location_id
 and demand_dates.dt = school.dt

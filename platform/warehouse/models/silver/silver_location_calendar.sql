{{ config(tags=["silver", "calendar"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

with location_dates as (
    select
        dataset_source,
        location_id,
        dt,
        country_code
    from {{ ref('silver_location_date_spine') }}
),
country_calendar as (
    select
        country_code,
        dt,
        holiday_flag,
        holiday_name,
        pre_holiday_flag,
        post_holiday_flag,
        bridge_day_flag
    from {{ ref('silver_open_public_holiday_calendar_daily') }}
),
school_calendar as (
    select
        dataset_source,
        location_id,
        dt,
        school_holiday_flag,
        school_holiday_available_flag
    from {{ ref('silver_open_school_holidays_daily') }}
)
select
    dates.dataset_source,
    dates.location_id,
    dates.dt,
    cast(strftime(dates.dt, '%u') as integer) - 1 as day_of_week,
    cast(strftime(dates.dt, '%V') as integer) as week_of_year,
    cast(strftime(dates.dt, '%m') as integer) as month,
    cast(ceil(cast(strftime(dates.dt, '%m') as integer) / 3.0) as integer) as quarter,
    cast(strftime(dates.dt, '%Y') as integer) as year,
    cast(strftime(dates.dt, '%u') as integer) in (6, 7) as weekend_flag,
    cast(strftime(dates.dt, '%d') as integer) <= 3 as month_start_flag,
    cast(strftime(dates.dt + interval 1 day, '%m') as integer) != cast(strftime(dates.dt, '%m') as integer) as month_end_flag,
    coalesce(country_calendar.holiday_flag, false) as public_holiday_flag,
    coalesce(country_calendar.pre_holiday_flag, false) as pre_public_holiday_flag,
    coalesce(country_calendar.post_holiday_flag, false) as post_public_holiday_flag,
    coalesce(country_calendar.bridge_day_flag, false) as bridge_day_flag,
    coalesce(school_calendar.school_holiday_flag, false) as school_holiday_flag,
    coalesce(school_calendar.school_holiday_available_flag, false) as school_holiday_available_flag,
    case when country_calendar.holiday_name is null then 0 else 1 end as event_count_local,
    case when country_calendar.holiday_name is null then 0.0 else 1.0 end as event_intensity_score
from location_dates as dates
left join country_calendar
  on dates.country_code = country_calendar.country_code
 and dates.dt = country_calendar.dt
left join school_calendar
  on dates.dataset_source = school_calendar.dataset_source
 and dates.location_id = school_calendar.location_id
 and dates.dt = school_calendar.dt

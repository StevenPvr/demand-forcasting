{{ config(tags=["silver", "exogenous"], materialized="table", unique_key=["country_code", "dt"]) }}

with demand_dates as (
    select distinct
        country_code,
        dt
    from {{ ref("silver_location_date_spine") }}
    where country_code is not null
),
holiday_rows as (
    select
        country_code,
        dt,
        true as holiday_flag,
        string_agg(holiday_name, ' | ' order by holiday_name) as holiday_name,
        bool_or(global_flag) as holiday_global_flag
    from {{ ref("stg_open_public_holidays") }} as holidays
    inner join {{ ref("silver_allowed_provider_sources") }} as allowed
      on {{ praedixa_canonical_source_id("holidays.source_policy_id", "holidays.source_name") }} = allowed.source_id
    group by
        country_code,
        dt
),
calendar as (
    select
        dates.country_code,
        cast(dates.dt as date) as dt,
        coalesce(holiday_rows.holiday_flag, false) as holiday_flag,
        holiday_rows.holiday_name,
        coalesce(holiday_rows.holiday_global_flag, false) as holiday_global_flag
    from demand_dates as dates
    left join holiday_rows
      on dates.country_code = holiday_rows.country_code
     and cast(dates.dt as date) = holiday_rows.dt
),
enriched as (
    select
        *,
        lag(holiday_flag, 1, false) over country_window as previous_day_holiday_flag,
        lead(holiday_flag, 1, false) over country_window as next_day_holiday_flag,
        cast(strftime(dt, '%u') as integer) - 1 as calendar_day_of_week
    from calendar
    window
        country_window as (
            partition by country_code
            order by dt
        )
)
select
    country_code,
    dt,
    holiday_flag,
    holiday_name,
    holiday_global_flag,
    next_day_holiday_flag as pre_holiday_flag,
    previous_day_holiday_flag as post_holiday_flag,
    case
        when not holiday_flag
         and calendar_day_of_week in (1, 2, 3, 4)
         and (previous_day_holiday_flag or next_day_holiday_flag) then true
        else false
    end as bridge_day_flag
from enriched

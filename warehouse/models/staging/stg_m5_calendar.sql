{{ config(tags=["staging", "m5"]) }}

select
    cast(date as date) as dt,
    cast(wm_yr_wk as integer) as wm_yr_wk,
    weekday,
    cast(wday as smallint) as wday,
    cast(month as smallint) as month,
    cast(year as smallint) as year,
    d,
    event_name_1,
    event_type_1,
    event_name_2,
    event_type_2,
    cast(snap_CA as boolean) as snap_ca,
    cast(snap_TX as boolean) as snap_tx,
    cast(snap_WI as boolean) as snap_wi
from {{ source("bronze", "bronze_m5_calendar") }}

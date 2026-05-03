{{ config(tags=["silver", "calendar"], materialized="table") }}

with bounds as (
    select
        min(dt) as min_dt,
        max(dt) + interval {{ env_var('PRAEDIXA_DATE_SPINE_FUTURE_DAYS', '35') | int }} day as max_dt
    from {{ ref('silver_daily_product_demand_training_candidates') }}
)
select
    cast(generated.dt as date) as dt
from bounds
cross join generate_series(bounds.min_dt, bounds.max_dt, interval 1 day) as generated(dt)
where bounds.min_dt is not null
  and bounds.max_dt is not null

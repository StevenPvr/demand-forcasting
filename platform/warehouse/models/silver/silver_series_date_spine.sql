{{ config(tags=["silver", "calendar", "demand"], materialized="table", unique_key=["dataset_source", "location_id", "product_id", "dt"]) }}

with series_bounds as (
    select
        dataset_source,
        location_id,
        product_id,
        max(series_id) as series_id,
        min(dt) as min_dt,
        max(dt) + interval {{ env_var('PRAEDIXA_DATE_SPINE_FUTURE_DAYS', '35') | int }} day as max_dt
    from {{ ref('silver_daily_product_demand_training_candidates') }}
    group by
        dataset_source,
        location_id,
        product_id
)
select
    bounds.dataset_source,
    bounds.location_id,
    bounds.product_id,
    bounds.series_id,
    cast(generated.dt as date) as dt
from series_bounds as bounds
cross join generate_series(bounds.min_dt, bounds.max_dt, interval 1 day) as generated(dt)
where bounds.min_dt is not null
  and bounds.max_dt is not null

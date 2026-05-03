{{ config(tags=["silver", "calendar"], materialized="table", unique_key=["dataset_source", "location_id", "dt"]) }}

with location_bounds as (
    select
        demand.dataset_source,
        demand.location_id,
        min(demand.dt) as min_dt,
        max(demand.dt) + interval {{ env_var("PRAEDIXA_DATE_SPINE_FUTURE_DAYS", "35") | int }} day as max_dt
    from {{ ref("silver_daily_product_demand_training_candidates") }} as demand
    group by
        demand.dataset_source,
        demand.location_id
),
dense_location_dates as (
    select
        bounds.dataset_source,
        bounds.location_id,
        cast(generated.dt as date) as dt
    from location_bounds as bounds
    cross join generate_series(bounds.min_dt, bounds.max_dt, interval 1 day) as generated(dt)
    where bounds.min_dt is not null
      and bounds.max_dt is not null
)
select
    dense.dataset_source,
    dense.location_id,
    dense.dt,
    metadata.country_code,
    metadata.region_code,
    metadata.school_zone
from dense_location_dates as dense
left join {{ ref("silver_open_location_metadata") }} as metadata
  on dense.dataset_source = metadata.dataset_source
 and dense.location_id = metadata.location_id

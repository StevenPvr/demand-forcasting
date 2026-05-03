{{ config(tags=["silver", "quality"], materialized="table") }}

select
    coalesce(silver_run_id, 'unknown') as silver_run_id,
    dataset_source,
    min(dt) as start_date,
    max(dt) as end_date,
    count(*) as row_count,
    count(distinct series_id) as series_count,
    min(source_loaded_at) as first_source_loaded_at,
    max(source_loaded_at) as last_source_loaded_at
from {{ ref("silver_daily_product_demand_training_candidates") }}
group by
    coalesce(silver_run_id, 'unknown'),
    dataset_source

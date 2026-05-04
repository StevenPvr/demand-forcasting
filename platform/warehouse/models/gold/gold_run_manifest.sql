{{ config(tags=["gold", "d1"], materialized="view", schema=env_var('PRAEDIXA_DUCKDB_GOLD_SCHEMA', 'gold')) }}
{% set gold_run_id = env_var('PRAEDIXA_GOLD_RUN_ID', 'manual') %}

select
    '{{ gold_run_id }}' as gold_run_id,
    current_timestamp as generated_at,
    dataset_source,
    split_bucket,
    min(dt) as min_dt,
    max(dt) as max_dt,
    count(*) as row_count,
    sum(case when target_demand_qty_d_plus_1 is null then 1 else 0 end) as null_target_rows,
    avg(sample_weight_source) as avg_sample_weight_source,
    avg(sample_weight_business) as avg_sample_weight_business
from {{ ref('gold_training_matrix_d1') }}
group by
    dataset_source,
    split_bucket

{{ config(tags=["gold", "d1"], materialized="table", schema=env_var("PRAEDIXA_DUCKDB_GOLD_SCHEMA", "gold")) }}

with source_counts as (
    select split_bucket, dataset_source, count(*) as row_count
    from {{ ref("gold_feature_panel_d1") }}
    group by split_bucket, dataset_source
),
split_averages as (
    select
        split_bucket,
        avg(row_count) as average_row_count
    from source_counts
    group by split_bucket
)
select
    source_counts.split_bucket,
    source_counts.dataset_source,
    split_averages.average_row_count / nullif(source_counts.row_count, 0) as sample_weight_source
from source_counts
inner join split_averages using (split_bucket)

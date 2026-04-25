{{ config(tags=["gold"]) }}

with source_splits as (
    select
        dataset_source,
        count(distinct split_bucket) as split_count,
        min(case when split_bucket = 'train' then dt end) as train_min_dt,
        max(case when split_bucket = 'train' then dt end) as train_max_dt,
        min(case when split_bucket = 'val' then dt end) as val_min_dt,
        max(case when split_bucket = 'val' then dt end) as val_max_dt,
        min(case when split_bucket = 'test' then dt end) as test_min_dt,
        max(case when split_bucket = 'test' then dt end) as test_max_dt
    from {{ ref("gold_feature_panel_d1") }}
    group by dataset_source
)
select *
from source_splits
where split_count <> 3
   or train_max_dt >= val_min_dt
   or val_max_dt >= test_min_dt
   or train_min_dt is null
   or val_min_dt is null
   or test_min_dt is null

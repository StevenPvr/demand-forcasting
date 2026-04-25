{{ config(tags=["gold"]) }}

with source_splits as (
    select
        dataset_source,
        count(distinct split_bucket) as split_count,
        sum(case when split_bucket = 'train' then 1 else 0 end) as train_rows,
        sum(case when split_bucket = 'val' then 1 else 0 end) as val_rows,
        sum(case when split_bucket = 'test' then 1 else 0 end) as test_rows,
        min(case when split_bucket = 'train' then dt end) as train_min_dt,
        max(case when split_bucket = 'train' then dt end) as train_max_dt,
        min(case when split_bucket = 'val' then dt end) as val_min_dt,
        max(case when split_bucket = 'val' then dt end) as val_max_dt,
        min(case when split_bucket = 'test' then dt end) as test_min_dt,
        max(case when split_bucket = 'test' then dt end) as test_max_dt
    from {{ ref("gold_feature_panel_d1") }}
    group by dataset_source
),
bakery_bounds as (
    select max(dt) as max_dt
    from {{ ref("silver_bakery_daily_product_demand") }}
),
violations as (
    select
        'freshretail_train_val_contract' as violation,
        source_splits.*
    from source_splits
    where dataset_source in ('freshretail', 'freshretail_lt')
      and (
        split_count <> 2
        or train_rows = 0
        or val_rows = 0
        or test_rows <> 0
        or train_max_dt >= val_min_dt
        or train_min_dt is null
        or val_min_dt is null
      )

    union all

    select
        'bakery_test_holdout_contract' as violation,
        source_splits.*
    from source_splits
    cross join bakery_bounds
    where dataset_source = 'bakery'
      and (
        split_count <> 1
        or train_rows <> 0
        or val_rows <> 0
        or test_rows = 0
        or test_min_dt <= bakery_bounds.max_dt - interval {{ env_var("PRAEDIXA_GOLD_BAKERY_TEST_MONTHS", "3") | int }} month
      )

    union all

    select
        'unexpected_dataset_source' as violation,
        source_splits.*
    from source_splits
    where dataset_source not in ('freshretail', 'freshretail_lt', 'bakery')
)
select *
from violations

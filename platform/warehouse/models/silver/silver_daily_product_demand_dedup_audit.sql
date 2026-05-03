{{ config(tags=["silver", "quality", "demand"], materialized="table") }}

with unioned as (
    select
        10 as source_priority,
        dataset_source,
        dt,
        location_id,
        product_id,
        source_partition,
        source_run_id,
        silver_run_id,
        source_policy_id,
        source_file_path,
        source_loaded_at,
        data_origin
    from {{ ref('silver_bakery_daily_product_demand') }}

    union all

    select
        20 as source_priority,
        dataset_source,
        dt,
        location_id,
        product_id,
        source_partition,
        source_run_id,
        silver_run_id,
        source_policy_id,
        source_file_path,
        source_loaded_at,
        data_origin
    from {{ ref('silver_synthetic_foodservice_daily_product_demand') }}
),
ranked as (
    select
        *,
        row_number() over (
            partition by dataset_source, dt, location_id, product_id
            order by source_priority, source_loaded_at desc, source_partition, source_run_id, silver_run_id
        ) as source_rank,
        count(*) over (
            partition by dataset_source, dt, location_id, product_id
        ) as duplicate_count
    from unioned
)
select
    dataset_source,
    cast(dt as date) as dt,
    location_id,
    product_id,
    duplicate_count,
    source_rank,
    source_priority,
    source_partition,
    source_run_id,
    silver_run_id,
    source_policy_id,
    source_file_path,
    source_loaded_at,
    data_origin
from ranked
where source_rank > 1

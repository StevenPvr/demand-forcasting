{{ config(tags=["silver", "quality", "demand"], materialized="table") }}

with unioned as (
    select
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
source_policy as (
    select
        dataset_source,
        min(source_priority) as source_priority
    from {{ ref('silver_source_registry') }}
    where source_kind = 'dataset'
      and dataset_source is not null
    group by dataset_source
),
normalized as (
    select
        unioned.*,
        coalesce(source_policy.source_priority, 999) as source_priority
    from unioned
    left join source_policy
      on unioned.dataset_source = source_policy.dataset_source
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
    from normalized
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

{{ config(tags=["silver", "profiles"], materialized="table", unique_key=["dataset_source", "product_id"]) }}

with product_rollup as (
    select
        dataset_source,
        product_id,
        max(category_level_1) as category_level_1,
        max(category_level_2) as category_level_2,
        max(category_level_3) as category_level_3
    from {{ ref("silver_daily_product_demand_all") }}
    group by
        dataset_source,
        product_id
)
select
    dataset_source,
    product_id,
    coalesce(category_level_1, dataset_source) as product_family,
    category_level_2 as product_subfamily,
    false as bundle_flag,
    dataset_source = 'bakery' as core_menu_flag,
    false as add_on_flag,
    false as beverage_flag,
    false as dessert_flag,
    dataset_source = 'bakery' as breakfast_flag,
    false as lunch_flag,
    'demand_derived_fallback' as product_profile_source
from product_rollup

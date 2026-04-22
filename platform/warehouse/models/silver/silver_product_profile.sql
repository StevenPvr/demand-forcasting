{{ config(tags=["silver", "profiles"], materialized="table", unique_key=["dataset_source", "product_id"]) }}

with product_rollup as (
    select
        dataset_source,
        product_id,
        max(category_level_1) as category_level_1,
        max(category_level_2) as category_level_2,
        max(category_level_3) as category_level_3,
        avg(avg_selling_price) as avg_selling_price_mean
    from {{ ref("silver_daily_product_demand") }}
    group by
        dataset_source,
        product_id
)
select
    dataset_source,
    product_id,
    coalesce(category_level_1, dataset_source) as product_family,
    category_level_2 as product_subfamily,
    case
        when dataset_source = 'bakery' then 'food_core'
        when dataset_source in ('freshretail', 'freshretail_lt') then 'grocery'
        when dataset_source = 'first_party_daily' then 'client_catalog'
        else 'unknown'
    end as menu_role,
    case
        when avg_selling_price_mean is null then 'unknown'
        when avg_selling_price_mean < 5 then 'low'
        when avg_selling_price_mean < 15 then 'mid'
        else 'high'
    end as price_band,
    false as bundle_flag,
    dataset_source = 'bakery' as core_menu_flag,
    false as add_on_flag,
    false as beverage_flag,
    false as dessert_flag,
    dataset_source = 'bakery' as breakfast_flag,
    false as lunch_flag
from product_rollup

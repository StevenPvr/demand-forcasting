{{ config(tags=["gold"]) }}

with bakery_bounds as (
    select
        max(dt) as max_dt
    from {{ ref("silver_bakery_daily_product_demand") }}
),
violations as (
    select panel.*
    from {{ ref("gold_feature_panel_d1") }} as panel
    cross join bakery_bounds
    where panel.dataset_source = 'bakery'
      and panel.dt <= bakery_bounds.max_dt - interval {{ env_var("PRAEDIXA_GOLD_BAKERY_TEST_MONTHS", "3") | int }} month
)
select *
from violations

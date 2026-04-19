{{ config(tags=["silver"], materialized="table") }}

select * from {{ ref("silver_freshretail_daily_product_demand") }}
union all
select * from {{ ref("silver_commercial_external_daily_product_demand") }}
union all
select * from {{ ref("silver_bakery_daily_product_demand") }}

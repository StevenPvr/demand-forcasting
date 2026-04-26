{{ config(tags=["gold", "d1"], materialized="table", schema=env_var("PRAEDIXA_DUCKDB_GOLD_SCHEMA", "gold")) }}

select * from {{ ref("gold_feature_bakery_d1") }}
union all
select * from {{ ref("gold_feature_freshretail_d1") }}

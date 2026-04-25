{{ config(tags=["gold", "d1", "training"], materialized="view", schema=env_var("PRAEDIXA_DUCKDB_GOLD_SCHEMA", "gold")) }}

select
    *
from {{ ref("gold_daily_product_forecast_panel_d1") }}

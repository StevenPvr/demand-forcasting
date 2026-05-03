{{ config(tags=["gold", "d1", "training"], materialized="table", schema=env_var('PRAEDIXA_DUCKDB_GOLD_SCHEMA', 'gold')) }}

select *
from {{ ref('gold_model_training_panel_d1') }}

{{ config(tags=["gold", "d1"], materialized="table", schema=env_var("PRAEDIXA_DUCKDB_GOLD_SCHEMA", "gold")) }}

{{ praedixa_gold_feature_slice(
    ref("gold_base_panel_d1"),
    "dataset_source in ('synthetic_foodservice_qsr', 'synthetic_foodservice_bakery', 'synthetic_foodservice_restaurant')",
    "dataset_source in ('synthetic_foodservice_qsr', 'synthetic_foodservice_bakery', 'synthetic_foodservice_restaurant')",
    "train_only"
) }}

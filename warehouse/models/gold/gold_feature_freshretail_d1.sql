{{ config(tags=["gold", "d1"], materialized="table", schema=env_var("PRAEDIXA_DUCKDB_GOLD_SCHEMA", "gold")) }}

{{ praedixa_gold_feature_slice(
    ref("gold_base_panel_d1"),
    "dataset_source <> 'bakery'",
    "dataset_source <> 'bakery'",
    "chrono_60_40"
) }}

{{ config(tags=["gold", "d1"], materialized="table", schema=env_var("PRAEDIXA_DUCKDB_GOLD_SCHEMA", "gold")) }}

{{ praedixa_gold_feature_slice(
    ref("gold_base_panel_d1"),
    "dataset_source in ('first_party_daily', 'm5_forecasting_accuracy', 'uci_online_retail_ii', 'uci_online_retail', 'restaurant_sales_report', 'perishable_goods_management')",
    "dataset_source in ('first_party_daily', 'm5_forecasting_accuracy', 'uci_online_retail_ii', 'uci_online_retail', 'restaurant_sales_report', 'perishable_goods_management')",
    "chrono_60_40"
) }}

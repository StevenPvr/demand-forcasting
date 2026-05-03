{{ config(tags=["gold", "d1", "inference"], materialized="table", schema=env_var("PRAEDIXA_DUCKDB_GOLD_SCHEMA", "gold")) }}

select
    * exclude (
        target_demand_qty_d_plus_1,
        target_residual_field_blend_lag_1_lag_7_d_plus_1,
        target_delta_log_wow_d_plus_1,
        target_semantics,
        censor_flag,
        target_source,
        label_quality_score,
        usable_for_training_flag,
        target_true_zero_demand_flag,
        split_bucket
    ),
    'inference' as split_bucket,
    cast(null as varchar) as target_semantics,
    false as usable_for_training_flag
from {{ ref("gold_model_training_panel_d1") }}
where false

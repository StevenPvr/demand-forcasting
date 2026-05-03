{{ config(tags=["gold"]) }}

select *
from {{ ref('gold_model_training_panel_d1') }}
where decision_timestamp >= cast(target_dt as timestamp)

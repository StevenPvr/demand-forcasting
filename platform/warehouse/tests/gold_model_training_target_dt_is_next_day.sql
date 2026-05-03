{{ config(tags=["gold"]) }}

select *
from {{ ref('gold_model_training_panel_d1') }}
where target_dt <> dt + interval 1 day

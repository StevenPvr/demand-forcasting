{{ config(tags=["gold"]) }}

select *
from {{ ref("gold_model_training_panel_d1") }}
where split_bucket in ('train', 'val')
  and coalesce(usable_for_training_flag, false)
  and (
    coalesce(censor_flag, false)
    or coalesce(label_quality_score, 0.0) < 0.75
    or target_source in ('closed_or_missing_observation', 'dense_calendar_zero_fill')
  )

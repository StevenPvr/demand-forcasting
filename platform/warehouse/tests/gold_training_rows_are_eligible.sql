{{ config(tags=["gold"]) }}

select *
from {{ ref('gold_training_matrix_d1') }}
where split_bucket in ('train', 'val')
  and coalesce(usable_for_training_flag, false)
  and (
    coalesce(label_quality_score, 0.0) < 0.75
    or target_source in ('closed_or_missing_observation', 'dense_calendar_zero_fill')
    or (
      coalesce(censor_flag, false)
      and coalesce(target_semantics, '') <> 'latent_demand_estimated'
    )
  )

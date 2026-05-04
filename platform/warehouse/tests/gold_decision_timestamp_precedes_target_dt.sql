{{ config(tags=["gold"]) }}

select *
from {{ ref('gold_training_matrix_d1') }}
where decision_timestamp >= cast(target_dt as timestamp)

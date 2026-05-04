{{ config(tags=["gold"]) }}

select *
from {{ ref('gold_training_matrix_d1') }}
where target_dt <> dt + interval 1 day

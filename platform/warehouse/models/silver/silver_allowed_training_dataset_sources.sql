{{ config(tags=["silver", "governance"], materialized="table") }}

select
    dataset_source
from {{ ref('silver_source_registry') }}
where source_kind = 'dataset'
  and dataset_source is not null
  and commercial_use_allowed
  and ml_training_allowed
  and include_in_training
  and review_status = 'allowed'

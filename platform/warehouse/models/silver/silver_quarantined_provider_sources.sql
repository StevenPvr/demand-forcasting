{{ config(tags=["silver", "governance"], materialized="table") }}

select
    source_id,
    provider_name,
    review_status,
    notes
from {{ ref("silver_source_registry") }}
where source_kind = 'provider'
  and review_status <> 'allowed'

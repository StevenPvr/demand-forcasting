{{ config(tags=["silver", "governance"], materialized="table") }}

select
    source_id,
    provider_name,
    review_status,
    legal_basis,
    license_type,
    commercial_use_allowed,
    ml_training_allowed,
    caching_allowed,
    attribution_required,
    attribution_text
from {{ ref("silver_source_registry") }}
where source_kind = 'provider'
  and review_status = 'allowed'
  and coalesce(commercial_use_allowed, false)
  and coalesce(caching_allowed, false)

{{ config(tags=["staging", "governance"]) }}

select
    cast(source_id as varchar) as source_id,
    cast(source_kind as varchar) as source_kind,
    nullif(cast(dataset_source as varchar), '') as dataset_source,
    cast(provider_name as varchar) as provider_name,
    cast(legal_basis as varchar) as legal_basis,
    cast(license_type as varchar) as license_type,
    cast(commercial_use_allowed as boolean) as commercial_use_allowed,
    cast(ml_training_allowed as boolean) as ml_training_allowed,
    cast(caching_allowed as boolean) as caching_allowed,
    cast(redistribution_mode as varchar) as redistribution_mode,
    cast(attribution_required as boolean) as attribution_required,
    nullif(cast(attribution_text as varchar), '') as attribution_text,
    cast(contract_required as boolean) as contract_required,
    cast(review_status as varchar) as review_status,
    cast(include_in_training as boolean) as include_in_training,
    cast(review_owner as varchar) as review_owner,
    cast(reviewed_at as date) as reviewed_at,
    cast(notes as varchar) as notes
from {{ ref('source_registry') }}

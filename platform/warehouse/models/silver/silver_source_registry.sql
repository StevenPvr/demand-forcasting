{{ config(tags=["silver", "governance"], materialized="table") }}

select
    nullif(trim(cast(source_id as varchar)), '') as source_id,
    nullif(trim(cast(source_kind as varchar)), '') as source_kind,
    nullif(trim(cast(dataset_source as varchar)), '') as dataset_source,
    nullif(trim(cast(provider_name as varchar)), '') as provider_name,
    nullif(trim(cast(legal_basis as varchar)), '') as legal_basis,
    nullif(trim(cast(license_type as varchar)), '') as license_type,
    coalesce(try_cast(commercial_use_allowed as boolean), false) as commercial_use_allowed,
    coalesce(try_cast(ml_training_allowed as boolean), false) as ml_training_allowed,
    coalesce(try_cast(caching_allowed as boolean), false) as caching_allowed,
    nullif(trim(cast(redistribution_mode as varchar)), '') as redistribution_mode,
    coalesce(try_cast(attribution_required as boolean), false) as attribution_required,
    nullif(trim(cast(attribution_text as varchar)), '') as attribution_text,
    coalesce(try_cast(contract_required as boolean), false) as contract_required,
    case
        when lower(trim(cast(review_status as varchar))) in ('allowed', 'quarantine', 'rejected', 'unreviewed')
        then lower(trim(cast(review_status as varchar)))
        when lower(trim(cast(review_status as varchar))) = 'quarantined' then 'quarantine'
        else 'unreviewed'
    end as review_status,
    coalesce(try_cast(include_in_training as boolean), false) as include_in_training,
    coalesce(try_cast(medallion_enabled as boolean), false) as medallion_enabled,
    case
        when lower(trim(cast(training_scope as varchar))) in ('train', 'reference_eval', 'supplemental', 'none')
        then lower(trim(cast(training_scope as varchar)))
        else 'none'
    end as training_scope,
    coalesce(nullif(trim(cast(split_strategy as varchar)), ''), 'none') as split_strategy,
    coalesce(try_cast(source_priority as integer), 999) as source_priority,
    case
        when lower(trim(cast(source_role as varchar))) in ('benchmark', 'synthetic', 'supplemental', 'provider', 'excluded')
        then lower(trim(cast(source_role as varchar)))
        else 'excluded'
    end as source_role,
    nullif(trim(cast(review_owner as varchar)), '') as review_owner,
    try_cast(reviewed_at as date) as reviewed_at,
    nullif(trim(cast(notes as varchar)), '') as notes
from {{ ref('stg_source_registry') }}

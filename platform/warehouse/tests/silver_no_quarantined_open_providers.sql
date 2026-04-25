{{ config(tags=["silver"]) }}

with observed_open_sources as (
    select
        'weather_daily' as source_family,
        coalesce(source_policy_id, source_name) as provider_source_id,
        source_name
    from {{ ref("stg_open_weather_daily") }}
    union all
    select
        'school_holidays' as source_family,
        coalesce(source_policy_id, source_name) as provider_source_id,
        source_name
    from {{ ref("stg_open_school_holidays") }}
    union all
    select
        'public_holidays' as source_family,
        coalesce(source_policy_id, source_name) as provider_source_id,
        source_name
    from {{ ref("stg_open_public_holidays") }}
    union all
    select
        'location_metadata' as source_family,
        coalesce(source_policy_id, source_name) as provider_source_id,
        source_name
    from {{ ref("stg_open_location_metadata") }}
    union all
    select
        'location_catchment' as source_family,
        coalesce(source_policy_id, source_name) as provider_source_id,
        source_name
    from {{ ref("stg_open_location_catchment") }}
    union all
    select
        'macro_annual' as source_family,
        coalesce(source_policy_id, source_name) as provider_source_id,
        source_name
    from {{ ref("stg_open_macro_annual") }}
    union all
    select
        'macro_timeseries' as source_family,
        coalesce(source_policy_id, source_name) as provider_source_id,
        source_name
    from {{ ref("stg_open_macro_timeseries") }}
)
select observed_open_sources.*
from observed_open_sources
left join {{ ref("silver_allowed_provider_sources") }} as allowed
  on observed_open_sources.provider_source_id = allowed.source_id
   or observed_open_sources.source_name = allowed.source_id
where observed_open_sources.provider_source_id is not null
  and allowed.source_id is null

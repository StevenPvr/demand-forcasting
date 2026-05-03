{{ config(tags=["silver", "exogenous"], materialized="table", unique_key=["dataset_source", "location_id"]) }}

with source_rows as (
    select
        stg.*,
        {{ praedixa_canonical_source_id('stg.source_policy_id', 'stg.source_name') }} as canonical_source_id
    from {{ ref('stg_open_location_metadata') }} as stg
    inner join {{ ref('silver_allowed_provider_sources') }} as allowed
      on {{ praedixa_canonical_source_id('stg.source_policy_id', 'stg.source_name') }} = allowed.source_id
)
select
    dataset_source,
    location_id,
    max(country_code) as country_code,
    case
        when max(country_code) = 'FR' then 'Europe/Paris'
        when max(country_code) = 'GB' then 'Europe/London'
        else 'UTC'
    end as timezone,
    max(region_code) as region_code,
    max(city_name) as city_name,
    max(latitude) as latitude,
    max(longitude) as longitude,
    max(school_zone) as school_zone,
    max(weather_location_label) as weather_location_label,
    max(assumption_source) as assumption_source,
    bool_or(coalesce(drive_through_flag, false)) as drive_through_flag,
    bool_or(coalesce(delivery_flag, false)) as delivery_flag,
    bool_or(coalesce(pickup_flag, false)) as pickup_flag,
    bool_or(coalesce(mall_flag, false)) as mall_flag,
    bool_or(coalesce(transit_hub_flag, false)) as transit_hub_flag,
    bool_or(coalesce(tourism_flag, false)) as tourism_flag,
    max(source_name) as source_name,
    max(canonical_source_id) as source_policy_id,
    count(distinct canonical_source_id) as provider_count,
    count(distinct latitude) > 1
        or count(distinct longitude) > 1
        or count(distinct country_code) > 1
        or count(distinct region_code) > 1
        as metadata_conflict_flag,
    max(canonical_source_id) as selected_source_policy_id
from source_rows
group by
    dataset_source,
    location_id

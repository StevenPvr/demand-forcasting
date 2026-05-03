{{ config(tags=["silver", "quality"], materialized="table") }}

with demand as (
    select *
    from {{ ref('silver_daily_product_demand_training_candidates') }}
),
base_incidents as (
    select
        dataset_source,
        silver_run_id,
        source_loaded_at,
        location_id,
        product_id,
        dt,
        'high' as severity,
        'non_negative_demand' as rule_name,
        'negative_value' as incident_type,
        cast(observed_demand_qty as varchar) as observed_value,
        'observed_demand_qty >= 0' as expected_range_or_rule
    from demand
    where observed_demand_qty < 0

    union all

    select
        dataset_source,
        silver_run_id,
        source_loaded_at,
        location_id,
        product_id,
        dt,
        'high' as severity,
        'observed_demand_parse' as rule_name,
        'parse_error' as incident_type,
        cast(observed_demand_qty as varchar) as observed_value,
        'observed_demand_qty must parse as numeric when present' as expected_range_or_rule
    from demand
    where observed_demand_parse_error_flag

    union all

    select
        dataset_source,
        silver_run_id,
        source_loaded_at,
        location_id,
        product_id,
        dt,
        'high' as severity,
        'target_semantics_accepted_values' as rule_name,
        'invalid_value' as incident_type,
        coalesce(target_semantics_raw, 'null') as observed_value,
        'target_semantics in observed_sales, latent_demand_estimated' as expected_range_or_rule
    from demand
    where target_semantics_parse_error_flag
       or target_semantics is null

    union all

    select
        dataset_source,
        silver_run_id,
        source_loaded_at,
        location_id,
        product_id,
        dt,
        'medium' as severity,
        'label_quality_score_range' as rule_name,
        'out_of_range' as incident_type,
        cast(label_quality_score as varchar) as observed_value,
        '0 <= label_quality_score <= 1' as expected_range_or_rule
    from demand
    where label_quality_score is null
       or label_quality_score < 0
       or label_quality_score > 1

    union all

    select
        dataset_source,
        silver_run_id,
        source_loaded_at,
        location_id,
        product_id,
        dt,
        'medium' as severity,
        'required_keys_not_null' as rule_name,
        'missing_value' as incident_type,
        concat(
            'dataset_source=', coalesce(dataset_source, 'null'),
            '|dt=', coalesce(cast(dt as varchar), 'null'),
            '|location_id=', coalesce(location_id, 'null'),
            '|product_id=', coalesce(product_id, 'null')
        ) as observed_value,
        'dataset_source, dt, location_id, product_id are not null' as expected_range_or_rule
    from demand
    where dataset_source is null
       or dt is null
       or location_id is null
       or product_id is null
)
select
    md5(
        coalesce(dataset_source, 'null') || '|'
        || coalesce(location_id, 'null') || '|'
        || coalesce(product_id, 'null') || '|'
        || coalesce(cast(dt as varchar), 'null') || '|'
        || rule_name
    ) as incident_id,
    silver_run_id,
    dataset_source as source_system,
    'daily_product_demand' as entity_type,
    concat(coalesce(dataset_source, 'null'), ':', coalesce(location_id, 'null'), ':', coalesce(product_id, 'null')) as entity_key,
    dt,
    severity,
    rule_name,
    incident_type,
    observed_value,
    expected_range_or_rule,
    'open' as status,
    coalesce(source_loaded_at, cast('1970-01-01 00:00:00' as timestamp)) as first_seen_at,
    coalesce(source_loaded_at, cast('1970-01-01 00:00:00' as timestamp)) as last_seen_at,
    cast(null as timestamp) as resolved_at
from base_incidents

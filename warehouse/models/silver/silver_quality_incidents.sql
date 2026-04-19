{{ config(tags=["silver", "quality"], materialized="view") }}

select
    md5(dataset_source || '|' || location_id || '|' || product_id || '|' || coalesce(cast(dt as varchar), 'null') || '|negative_demand') as incident_id,
    silver_run_id,
    dataset_source as source_system,
    'daily_product_demand' as entity_type,
    concat(dataset_source, ':', location_id, ':', product_id) as entity_key,
    dt,
    'high' as severity,
    'non_negative_demand' as rule_name,
    'negative_value' as incident_type,
    cast(observed_demand_qty as varchar) as observed_value,
    'observed_demand_qty >= 0' as expected_range_or_rule,
    'open' as status,
    now() as created_at
from {{ ref("silver_daily_product_demand") }}
where observed_demand_qty < 0

union all

select
    md5(dataset_source || '|' || location_id || '|' || product_id || '|' || coalesce(cast(dt as varchar), 'null') || '|missing_date') as incident_id,
    silver_run_id,
    dataset_source as source_system,
    'daily_product_demand' as entity_type,
    concat(dataset_source, ':', location_id, ':', product_id) as entity_key,
    dt,
    'medium' as severity,
    'dt_not_null' as rule_name,
    'missing_value' as incident_type,
    'null' as observed_value,
    'dt is not null' as expected_range_or_rule,
    'open' as status,
    now() as created_at
from {{ ref("silver_daily_product_demand") }}
where dt is null

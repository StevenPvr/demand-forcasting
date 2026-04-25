{{ config(tags=["silver", "catchment"], materialized="table", unique_key=["dataset_source", "location_id"]) }}

with source_rows as (
    select stg.*
    from {{ ref("stg_open_location_catchment") }} as stg
    inner join {{ ref("silver_allowed_provider_sources") }} as allowed
      on coalesce(stg.source_policy_id, stg.source_name) = allowed.source_id
       or stg.source_name = allowed.source_id
)
select
    dataset_source,
    location_id,
    max(population_1km) as population_1km,
    max(population_3km) as population_3km,
    max(office_poi_count_1km) as office_poi_count_1km,
    max(school_poi_count_1km) as school_poi_count_1km,
    max(transit_station_count_1km) as transit_station_count_1km,
    max(mall_poi_count_1km) as mall_poi_count_1km,
    max(tourism_poi_count_1km) as tourism_poi_count_1km,
    max(competitor_count_500m) as competitor_count_500m,
    max(competitor_count_1km) as competitor_count_1km,
    max(parking_score) as parking_score,
    max(source_name) as source_name,
    max(coalesce(source_policy_id, source_name)) as source_policy_id
from source_rows
group by
    dataset_source,
    location_id

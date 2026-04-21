{{ config(tags=["silver", "catchment"], materialized="table", unique_key=["dataset_source", "location_id"]) }}

select
    dataset_source,
    location_id,
    population_1km,
    population_3km,
    office_poi_count_1km,
    school_poi_count_1km,
    transit_station_count_1km,
    mall_poi_count_1km,
    tourism_poi_count_1km,
    competitor_count_500m,
    competitor_count_1km,
    parking_score
from {{ ref("stg_open_location_catchment") }}

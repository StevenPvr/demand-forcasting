{{ config(tags=["silver", "exogenous"]) }}

select
    cast(dataset_source as varchar) as dataset_source,
    cast(location_id as varchar) as location_id,
    cast(population_1km as double) as population_1km,
    cast(population_3km as double) as population_3km,
    cast(office_poi_count_1km as integer) as office_poi_count_1km,
    cast(school_poi_count_1km as integer) as school_poi_count_1km,
    cast(transit_station_count_1km as integer) as transit_station_count_1km,
    cast(mall_poi_count_1km as integer) as mall_poi_count_1km,
    cast(tourism_poi_count_1km as integer) as tourism_poi_count_1km,
    cast(competitor_count_500m as integer) as competitor_count_500m,
    cast(competitor_count_1km as integer) as competitor_count_1km,
    nullif(cast(parking_score as varchar), '') as parking_score,
    nullif(cast(source_name as varchar), '') as source_name,
    source_file_path,
    loaded_at
from {{ source("bronze", "bronze_open_location_catchment") }}

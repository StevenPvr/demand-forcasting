{{ config(tags=["silver", "freshretail"], materialized="table", unique_key=["dataset_source", "dt", "location_id", "product_id"]) }}

select * from {{ ref("silver_freshretail_native") }}

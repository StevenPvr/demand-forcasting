{{ config(tags=["m5", "legacy"], unique_key=["dataset_source", "dt", "location_id", "product_id"]) }}

select * from {{ ref("silver_m5_native_daily") }}

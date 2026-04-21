{{ config(tags=["silver", "governance"], materialized="table") }}

select
    *
from {{ ref("stg_source_registry") }}

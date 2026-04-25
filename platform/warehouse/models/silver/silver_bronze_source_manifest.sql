{{ config(tags=["silver", "governance"], materialized="table") }}

select
    source_run_id,
    source_name,
    coalesce(source_policy_id, source_name) as source_policy_id,
    table_name,
    source_path,
    required,
    allow_empty,
    size_bytes,
    mtime_ns,
    sha256,
    loaded_rows,
    loaded_at
from {{ ref("stg_bronze_source_manifest") }}

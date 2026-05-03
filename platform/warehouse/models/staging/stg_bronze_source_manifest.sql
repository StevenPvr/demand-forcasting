{{ config(tags=["staging", "governance"]) }}

select
    cast(source_run_id as varchar) as source_run_id,
    cast(source_name as varchar) as source_name,
    cast(source_policy_id as varchar) as source_policy_id,
    cast(table_name as varchar) as table_name,
    cast(source_path as varchar) as source_path,
    cast(required as boolean) as required,
    cast(allow_empty as boolean) as allow_empty,
    cast(size_bytes as bigint) as size_bytes,
    cast(mtime_ns as bigint) as mtime_ns,
    cast(sha256 as varchar) as sha256,
    cast(loaded_rows as bigint) as loaded_rows,
    cast(loaded_at as timestamp) as loaded_at
from {{ source('bronze', 'bronze_source_manifest') }}

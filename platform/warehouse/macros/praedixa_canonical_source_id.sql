{% macro praedixa_canonical_source_id(source_policy_id_sql, source_name_sql) -%}
coalesce(
    nullif(trim(cast({{ source_policy_id_sql }} as varchar)), ''),
    nullif(trim(cast({{ source_name_sql }} as varchar)), '')
)
{%- endmacro %}

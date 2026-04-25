{% test praedixa_required_columns_not_null(model, required_columns) %}
with null_counts as (
    select
{% for column_name in required_columns %}
        sum(case when {{ adapter.quote(column_name) }} is null then 1 else 0 end) as {{ column_name }}__null_count{{ "," if not loop.last }}
{% endfor %}
    from {{ model }}
),
failures as (
{% for column_name in required_columns %}
    select
        '{{ column_name }}' as column_name,
        {{ column_name }}__null_count as null_row_count
    from null_counts
    where {{ column_name }}__null_count > 0
{% if not loop.last %}
    union all
{% endif %}
{% endfor %}
)
select *
from failures
{% endtest %}

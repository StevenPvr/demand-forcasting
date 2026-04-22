from __future__ import annotations

from praedixa.demand_forecast.training.sampling_common import resolve_sampling_order_columns
from praedixa.demand_forecast.training.sampling_models import (
    GoldSplitSamplingSpec,
    RelationSamplingQuery,
    RelationSamplingSpec,
)


def build_gold_split_sampling_query(
    *,
    gold_table: str,
    split_bucket: str,
    date_col: str,
    dataset_source_col: str,
    sample_store_col: str,
    sample_fraction: float,
    selected_columns: list[str] | None = None,
) -> str:
    select_list = "*" if selected_columns is None else ", ".join(selected_columns)
    return f"""
with scoped as (
    select {select_list}
    from {gold_table}
    where split_bucket = '{split_bucket}'
),
ranked as (
    select
        *,
        row_number() over (
            partition by {dataset_source_col}, {date_col}, {sample_store_col}
            order by hash(
                coalesce(cast(product_id as varchar), ''),
                coalesce(cast(series_id as varchar), '')
            )
        ) as rn_sample,
        count(*) over (
            partition by {dataset_source_col}, {date_col}, {sample_store_col}
        ) as stratum_row_count
    from scoped
)
select *
from ranked
where rn_sample <= greatest(1, cast(ceil(stratum_row_count * {sample_fraction:.12f}) as bigint))
"""


def _sampling_hash_inputs(selected_columns: list[str], *, spec: RelationSamplingSpec) -> list[str]:
    hash_inputs: list[str] = []
    for candidate in ("product_id", "series_id", spec.sample_store_col, spec.date_col):
        if candidate in selected_columns:
            hash_inputs.append(f"coalesce(cast({candidate} as varchar), '')")
    return hash_inputs


def _sampling_query_parts(query: RelationSamplingQuery) -> tuple[str, str, str, str]:
    ordering_columns = resolve_sampling_order_columns(
        query.selected_columns,
        date_col=query.spec.date_col,
        sample_store_col=query.spec.sample_store_col,
    )
    select_list = ", ".join(query.selected_columns)
    qualified_topup_select_list = ", ".join(f"topup_candidates.{column}" for column in query.selected_columns)
    hash_expression = ", ".join(_sampling_hash_inputs(query.selected_columns, spec=query.spec))
    return ", ".join(ordering_columns), select_list, qualified_topup_select_list, hash_expression


def build_sampling_query_for_relation(*, query: RelationSamplingQuery) -> str:
    final_order_by, select_list, qualified_topup_select_list, hash_expression = _sampling_query_parts(query)
    if not query.spec.allow_top_up:
        return _sampling_query_without_topup(
            query=query,
            select_list=select_list,
            hash_expression=hash_expression,
        )
    return _sampling_query_sql(
        query=query,
        select_list=select_list,
        qualified_topup_select_list=qualified_topup_select_list,
        final_order_by=final_order_by,
        hash_expression=hash_expression,
    )


def _sampling_query_without_topup(
    *,
    query: RelationSamplingQuery,
    select_list: str,
    hash_expression: str,
) -> str:
    return f"""
with scoped as (
    select {select_list}
    from ({query.spec.relation_sql}) as relation_source
),
ranked as (
    select
        *,
        row_number() over (
            partition by {query.spec.dataset_source_col}, {query.spec.date_col}, {query.spec.sample_store_col}
            order by hash({hash_expression})
        ) as rn_sample,
        count(*) over (
            partition by {query.spec.dataset_source_col}, {query.spec.date_col}, {query.spec.sample_store_col}
        ) as stratum_row_count
    from scoped
)
select {select_list}
from ranked
where rn_sample <= greatest(1, cast(ceil(stratum_row_count * {query.spec.sample_fraction:.12f}) as bigint))
"""


def _sampling_query_sql(
    *,
    query: RelationSamplingQuery,
    select_list: str,
    qualified_topup_select_list: str,
    final_order_by: str,
    hash_expression: str,
) -> str:
    return (
        _sampling_query_scoped_ctes(
            query=query,
            select_list=select_list,
            hash_expression=hash_expression,
        )
        + _sampling_query_limit_ctes(query=query)
        + _sampling_query_topup_ctes(
            query=query,
            select_list=select_list,
            qualified_topup_select_list=qualified_topup_select_list,
            final_order_by=final_order_by,
        )
        + f"""
select {select_list}
from sampled
"""
    )


def _sampling_query_scoped_ctes(
    *,
    query: RelationSamplingQuery,
    select_list: str,
    hash_expression: str,
) -> str:
    return f"""
with scoped as (
    select
        row_number() over () as __row_id,
        {select_list}
    from ({query.spec.relation_sql}) as relation_source
),
ranked as (
    select
        *,
        row_number() over (
            partition by {query.spec.dataset_source_col}, {query.spec.date_col}, {query.spec.sample_store_col}
            order by hash({hash_expression})
        ) as rn_sample,
        count(*) over (
            partition by {query.spec.dataset_source_col}, {query.spec.date_col}, {query.spec.sample_store_col}
        ) as stratum_row_count
    from scoped
),
primary_sample as (
    select __row_id, {select_list}
    from ranked
    where rn_sample <= greatest(1, cast(ceil(stratum_row_count * {query.spec.sample_fraction:.12f}) as bigint))
),
"""


def _sampling_query_limit_ctes(*, query: RelationSamplingQuery) -> str:
    return f"""
dataset_limits as (
    select
        {query.spec.dataset_source_col} as dataset_source,
        count(*) as original_rows,
        least(
            count(*),
            greatest(
                cast(ceil(count(*) * {query.spec.sample_fraction:.12f}) as bigint),
                {int(query.spec.min_samples_per_dataset)}
            )
        ) as target_rows
    from scoped
    group by 1
),
dataset_primary_counts as (
    select
        {query.spec.dataset_source_col} as dataset_source,
        count(*) as sampled_rows
    from primary_sample
    group by 1
),
"""


def _sampling_query_topup_ctes(
    *,
    query: RelationSamplingQuery,
    select_list: str,
    qualified_topup_select_list: str,
    final_order_by: str,
) -> str:
    return f"""
topup_candidates as (
    select
        scoped.__row_id,
        {select_list},
        row_number() over (
            partition by scoped.{query.spec.dataset_source_col}
            order by {final_order_by}
        ) as rn_topup
    from scoped
    where scoped.__row_id not in (select __row_id from primary_sample)
),
topup as (
    select topup_candidates.__row_id, {qualified_topup_select_list}
    from topup_candidates
    inner join dataset_limits
      on topup_candidates.{query.spec.dataset_source_col} = dataset_limits.dataset_source
    left join dataset_primary_counts
      on topup_candidates.{query.spec.dataset_source_col} = dataset_primary_counts.dataset_source
    where topup_candidates.rn_topup <= greatest(
        0,
        dataset_limits.target_rows - coalesce(dataset_primary_counts.sampled_rows, 0)
    )
),
sampled as (
    select * from primary_sample
    union all
    select * from topup
)
"""


def build_gold_sampling_spec(
    *,
    gold_table: str,
    split_bucket: str,
    date_col: str,
    dataset_source_col: str,
    sample_store_col: str,
    sample_fraction: float,
) -> GoldSplitSamplingSpec:
    return GoldSplitSamplingSpec(
        gold_table=gold_table,
        split_bucket=split_bucket,
        date_col=date_col,
        dataset_source_col=dataset_source_col,
        sample_store_col=sample_store_col,
        sample_fraction=sample_fraction,
    )


__all__ = [
    "build_gold_sampling_spec",
    "build_gold_split_sampling_query",
    "build_sampling_query_for_relation",
]

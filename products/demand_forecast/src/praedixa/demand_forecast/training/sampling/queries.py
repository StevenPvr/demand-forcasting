from __future__ import annotations

from praedixa.demand_forecast.training.sampling.common import (
    resolve_sampling_order_columns,
    series_key_sql,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES,
)
from praedixa.demand_forecast.training.sampling.dataset_filters import (
    dataset_source_not_in_filter,
)
from praedixa.demand_forecast.training.validation.eligibility import (
    DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE,
    NON_TRAINABLE_TARGET_SOURCES,
)
from praedixa.demand_forecast.training.sampling.models import (
    GoldSplitSamplingSpec,
    RelationSamplingQuery,
    RelationSamplingSpec,
)

_TRAINING_ELIGIBILITY_SQL_COLUMNS: set[str] = {
    "usable_for_training_flag",
    "censor_flag",
    "label_quality_score",
    "target_source",
}


def _per_stratum_sample_target_sql(sample_fraction: float) -> str:
    return f"cast(floor(stratum_row_count * {sample_fraction:.12f}) as bigint)"


def _per_series_stratum_sample_target_sql(sample_fraction: float) -> str:
    return (
        "least("
        "stratum_row_count, "
        f"greatest(1, cast(ceil(stratum_row_count * {sample_fraction:.12f}) as bigint))"
        ")"
    )


def _sql_tuple(values: tuple[str, ...]) -> str:
    quoted_values = ["'" + value.replace("'", "''") + "'" for value in values]
    return "(" + ", ".join(quoted_values) + ")"


def _gold_training_eligibility_filter(selected_columns: list[str] | None) -> str:
    if selected_columns is not None and not _TRAINING_ELIGIBILITY_SQL_COLUMNS.issubset(
        set(selected_columns)
    ):
        return "true"
    return f"""
(
    coalesce(usable_for_training_flag, false)
    and not coalesce(censor_flag, false)
    and coalesce(label_quality_score, 0.0) >= {DEFAULT_MIN_TRAINING_LABEL_QUALITY_SCORE:.6f}
    and coalesce(target_source, '') not in {_sql_tuple(NON_TRAINABLE_TARGET_SOURCES)}
)""".strip()


def build_gold_split_sampling_query(
    *,
    gold_table: str,
    split_bucket: str,
    date_col: str,
    dataset_source_col: str,
    sample_store_col: str,
    sample_fraction: float,
    selected_columns: list[str] | None = None,
    excluded_dataset_sources: tuple[
        str, ...
    ] = DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES,
) -> str:
    select_list = "*" if selected_columns is None else ", ".join(selected_columns)
    training_eligibility_filter = (
        _gold_training_eligibility_filter(selected_columns)
        if split_bucket in {"train", "val"}
        else "true"
    )
    dataset_scope_filter = dataset_source_not_in_filter(
        dataset_source_col,
        excluded_dataset_sources,
    )
    series_key = series_key_sql(
        selected_columns,
        sample_store_col=sample_store_col,
    )
    return f"""
with scoped as (
    select {select_list}
    from {gold_table}
    where split_bucket = '{split_bucket}'
      and {dataset_scope_filter}
      and {training_eligibility_filter}
),
series_population as (
    select
        {dataset_source_col} as __sample_dataset_source,
        {sample_store_col} as __sample_store,
        {series_key} as __sample_series_key
    from scoped
    group by 1, 2, 3
),
ranked_series as (
    select
        *,
        row_number() over (
            partition by __sample_dataset_source, __sample_store
            order by hash(__sample_series_key)
        ) as rn_sample,
        count(*) over (
            partition by __sample_dataset_source, __sample_store
        ) as stratum_row_count
    from series_population
),
selected_series as (
    select *
    from ranked_series
    where rn_sample <= {_per_series_stratum_sample_target_sql(sample_fraction)}
)
select
    scoped.*,
    selected_series.rn_sample,
    selected_series.stratum_row_count
from scoped
inner join selected_series
  on scoped.{dataset_source_col} = selected_series.__sample_dataset_source
 and scoped.{sample_store_col} = selected_series.__sample_store
 and {series_key} = selected_series.__sample_series_key
"""


def _sampling_hash_inputs(
    selected_columns: list[str], *, spec: RelationSamplingSpec
) -> list[str]:
    hash_inputs: list[str] = []
    for candidate in ("product_id", "client_id", spec.sample_store_col, spec.date_col):
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
    qualified_topup_select_list = ", ".join(
        f"topup_candidates.{column}" for column in query.selected_columns
    )
    hash_expression = ", ".join(
        _sampling_hash_inputs(query.selected_columns, spec=query.spec)
    )
    return (
        ", ".join(ordering_columns),
        select_list,
        qualified_topup_select_list,
        hash_expression,
    )


def build_sampling_query_for_relation(*, query: RelationSamplingQuery) -> str:
    final_order_by, select_list, qualified_topup_select_list, hash_expression = (
        _sampling_query_parts(query)
    )
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


def build_complete_series_sampling_query_for_relation(
    *,
    query: RelationSamplingQuery,
    series_col: str,
) -> str:
    select_list = ", ".join(query.selected_columns)
    series_key = f"coalesce(cast({series_col} as varchar), '')"
    scoped_series_key = f"coalesce(cast(scoped.{series_col} as varchar), '')"
    return f"""
with scoped as (
    select {select_list}
    from ({query.spec.relation_sql}) as relation_source
),
series_population as (
    select
        {query.spec.dataset_source_col} as __sample_dataset_source,
        {query.spec.sample_store_col} as __sample_store,
        {series_key} as __sample_series_key
    from scoped
    group by 1, 2, 3
),
ranked_series as (
    select
        *,
        row_number() over (
            partition by __sample_dataset_source, __sample_store
            order by hash(__sample_series_key)
        ) as rn_sample,
        count(*) over (
            partition by __sample_dataset_source, __sample_store
        ) as stratum_row_count
    from series_population
),
selected_series as (
    select *
    from ranked_series
    where rn_sample <= {_per_series_stratum_sample_target_sql(query.spec.sample_fraction)}
)
select {select_list}
from scoped
inner join selected_series
  on scoped.{query.spec.dataset_source_col} = selected_series.__sample_dataset_source
 and scoped.{query.spec.sample_store_col} = selected_series.__sample_store
 and {scoped_series_key} = selected_series.__sample_series_key
"""


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
where rn_sample <= {_per_stratum_sample_target_sql(query.spec.sample_fraction)}
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
    where rn_sample <= {_per_stratum_sample_target_sql(query.spec.sample_fraction)}
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
    excluded_dataset_sources: tuple[
        str, ...
    ] = DEFAULT_OPTIMISATION_HOLDOUT_DATASET_SOURCES,
) -> GoldSplitSamplingSpec:
    return GoldSplitSamplingSpec(
        gold_table=gold_table,
        split_bucket=split_bucket,
        date_col=date_col,
        dataset_source_col=dataset_source_col,
        sample_store_col=sample_store_col,
        sample_fraction=sample_fraction,
        excluded_dataset_sources=excluded_dataset_sources,
    )


__all__ = [
    "build_complete_series_sampling_query_for_relation",
    "build_gold_sampling_spec",
    "build_gold_split_sampling_query",
    "build_sampling_query_for_relation",
]

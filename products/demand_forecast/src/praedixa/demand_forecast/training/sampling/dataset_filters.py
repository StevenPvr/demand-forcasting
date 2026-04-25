from __future__ import annotations


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def dataset_source_not_in_filter(
    dataset_source_col: str,
    excluded_dataset_sources: tuple[str, ...],
) -> str:
    resolved_sources = tuple(source for source in excluded_dataset_sources if source)
    if not resolved_sources:
        return "true"
    excluded_literals = ", ".join(sql_string_literal(source) for source in resolved_sources)
    return f"{dataset_source_col} not in ({excluded_literals})"


__all__ = [
    "dataset_source_not_in_filter",
    "sql_string_literal",
]

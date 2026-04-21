from __future__ import annotations

import re

import pandas as pd
import polars as pl

from praedixa.platform.datasets.standardization.commercial_external_standardizers import (
    DEFAULT_SILVER_RUN_ID,
    DEFAULT_SOURCE_RUN_ID,
    aggregate_transaction_lines,
)


def _consume_sql_escape(char: str, escaped: bool) -> tuple[bool, bool]:
    if escaped:
        return True, False
    if char == "\\":
        return True, True
    return False, False


def _consume_sql_quote(char: str, in_quote: bool) -> tuple[bool, bool]:
    if char != "'":
        return False, in_quote
    return True, not in_quote


def _open_tuple_depth(
    *,
    index: int,
    depth: int,
    start_index: int | None,
) -> tuple[int, int | None]:
    next_start_index = index + 1 if depth == 0 else start_index
    return depth + 1, next_start_index


def _close_tuple_depth(
    *,
    index: int,
    depth: int,
    start_index: int | None,
    tuples: list[str],
    values_blob: str,
) -> tuple[int, int | None]:
    next_depth = depth - 1
    if next_depth == 0 and start_index is not None:
        tuples.append(values_blob[start_index:index])
        return next_depth, None
    return next_depth, start_index


def _update_tuple_depth(
    *,
    char: str,
    index: int,
    depth: int,
    start_index: int | None,
    tuples: list[str],
    values_blob: str,
) -> tuple[int, int | None]:
    if char == "(":
        return _open_tuple_depth(index=index, depth=depth, start_index=start_index)
    if char != ")":
        return depth, start_index
    return _close_tuple_depth(
        index=index,
        depth=depth,
        start_index=start_index,
        tuples=tuples,
        values_blob=values_blob,
    )


def _split_sql_tuples(values_blob: str) -> list[str]:
    tuples: list[str] = []
    start_index: int | None = None
    depth = 0
    in_quote = False
    escaped = False
    for index, char in enumerate(values_blob):
        consumed_escape, escaped = _consume_sql_escape(char, escaped)
        if consumed_escape:
            continue
        consumed_quote, in_quote = _consume_sql_quote(char, in_quote)
        if consumed_quote:
            continue
        if in_quote:
            continue
        depth, start_index = _update_tuple_depth(
            char=char,
            index=index,
            depth=depth,
            start_index=start_index,
            tuples=tuples,
            values_blob=values_blob,
        )
    return tuples


def _split_sql_fields(tuple_blob: str) -> list[str]:
    fields: list[str] = []
    buffer: list[str] = []
    in_quote = False
    escaped = False
    for char in tuple_blob:
        consumed, escaped, in_quote, buffer = _consume_sql_field_char(
            char=char,
            fields=fields,
            buffer=buffer,
            escaped=escaped,
            in_quote=in_quote,
        )
        if consumed:
            continue
        buffer.append(char)
    fields.append("".join(buffer).strip())
    return fields


def _flush_sql_field_buffer(fields: list[str], buffer: list[str]) -> list[str]:
    fields.append("".join(buffer).strip())
    return []


def _escaped_field_transition(
    *,
    char: str,
    buffer: list[str],
    escaped: bool,
    in_quote: bool,
) -> tuple[bool, bool, bool, list[str]] | None:
    if not escaped:
        return None
    buffer.append(char)
    return True, False, in_quote, buffer


def _control_field_transition(
    *,
    char: str,
    fields: list[str],
    buffer: list[str],
    in_quote: bool,
) -> tuple[bool, bool, bool, list[str]] | None:
    if char == "\\":
        buffer.append(char)
        return True, True, in_quote, buffer
    if char == "'":
        buffer.append(char)
        return True, False, not in_quote, buffer
    if char == "," and not in_quote:
        return True, False, in_quote, _flush_sql_field_buffer(fields, buffer)
    return None


def _consume_sql_field_char(
    *,
    char: str,
    fields: list[str],
    buffer: list[str],
    escaped: bool,
    in_quote: bool,
) -> tuple[bool, bool, bool, list[str]]:
    escaped_transition = _escaped_field_transition(
        char=char,
        buffer=buffer,
        escaped=escaped,
        in_quote=in_quote,
    )
    if escaped_transition is not None:
        return escaped_transition
    control_transition = _control_field_transition(
        char=char,
        fields=fields,
        buffer=buffer,
        in_quote=in_quote,
    )
    if control_transition is not None:
        return control_transition
    return False, False, in_quote, buffer


def _parse_sql_scalar(raw_value: str) -> object:
    if raw_value.upper() == "NULL":
        return None
    if raw_value.startswith("'") and raw_value.endswith("'"):
        inner = raw_value[1:-1]
        inner = inner.replace("\\'", "'").replace("\\\\", "\\")
        return inner
    return raw_value


def _sql_insert_pattern(table_name: str) -> re.Pattern[str]:
    return re.compile(
        rf"INSERT INTO\s+`{re.escape(table_name)}`\s*\((?P<columns>.*?)\)\s*VALUES\s*(?P<values>.*?);",
        re.IGNORECASE | re.DOTALL,
    )


def _sql_insert_columns(match: re.Match[str]) -> list[str]:
    return [column.strip().strip("`") for column in match.group("columns").split(",")]


def _sql_row_from_tuple(columns: list[str], tuple_blob: str) -> dict[str, object] | None:
    field_values = [_parse_sql_scalar(value) for value in _split_sql_fields(tuple_blob)]
    if len(field_values) != len(columns):
        return None
    return dict(zip(columns, field_values, strict=False))


def _iter_insert_rows(sql_text: str, *, table_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for match in _sql_insert_pattern(table_name).finditer(sql_text):
        columns = _sql_insert_columns(match)
        for tuple_blob in _split_sql_tuples(match.group("values")):
            row = _sql_row_from_tuple(columns, tuple_blob)
            if row is not None:
                rows.append(row)
    return rows


def _resolve_pharmacy_column(frame: pd.DataFrame, uppercase: str, lowercase: str) -> str:
    return uppercase if uppercase in frame.columns else lowercase


def _resolve_pharmacy_unit_price_column(frame: pd.DataFrame) -> str | None:
    if "HJ" in frame.columns:
        return "HJ"
    if "hj" in frame.columns:
        return "hj"
    return None


def standardize_mendeley_pharmacy_sql_text(
    sql_text: str,
    *,
    source_partition: str = "historical",
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.DataFrame:
    """Standardize the Indonesian pharmacy SQL dump into daily medicine demand."""

    transaction_rows = _iter_insert_rows(sql_text, table_name="transaction")
    if not transaction_rows:
        raise ValueError("No INSERT rows were found for the `transaction` table in the pharmacy SQL dump.")
    frame = pd.DataFrame(transaction_rows)
    return aggregate_transaction_lines(
        frame,
        dataset_source="mendeley_pharmacy_id",
        source_partition=source_partition,
        location_id="indonesia_pharmacy_1",
        dt_column=_resolve_pharmacy_column(frame, "TGL", "tgl"),
        product_column=_resolve_pharmacy_column(frame, "KD_OBAT", "kd_obat"),
        qty_column=_resolve_pharmacy_column(frame, "QTY", "qty"),
        revenue_column=None,
        unit_price_column=_resolve_pharmacy_unit_price_column(frame),
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )

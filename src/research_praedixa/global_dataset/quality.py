from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl


GRAIN_COLUMNS = ["dataset_source", "dt", "location_id", "product_id"]
SERIES_KEY_SEPARATOR = "__"
REQUIRED_NON_NULL_COLUMNS = [
    "dataset_source",
    "source_partition",
    "source_run_id",
    "series_id",
    "dt",
    "location_id",
    "product_id",
    "observed_demand_qty",
]


@dataclass(frozen=True)
class DataQualityIssue:
    """One medallion data quality issue detected on a canonical dataset."""

    severity: str
    code: str
    message: str
    row_count: int
    sample: list[dict[str, object]]


@dataclass(frozen=True)
class DatasetQualityReport:
    """Compact data quality report for one canonical dataset frame."""

    dataset_name: str
    row_count: int
    series_count: int
    issues: list[DataQualityIssue]

    @property
    def error_count(self) -> int:
        """Return the number of blocking contract violations."""

        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        """Return the number of non-blocking quality warnings."""

        return sum(1 for issue in self.issues if issue.severity == "warning")

    def to_manifest_dict(self) -> dict[str, object]:
        """Serialize the report into a manifest-friendly dictionary."""

        payload = asdict(self)
        payload["error_count"] = self.error_count
        payload["warning_count"] = self.warning_count
        return payload


def _build_issue(
    *,
    severity: str,
    code: str,
    message: str,
    failing_rows: pl.DataFrame,
    sample_columns: list[str],
) -> DataQualityIssue:
    sample_frame = failing_rows.select([column for column in sample_columns if column in failing_rows.columns]).head(5)
    return DataQualityIssue(
        severity=severity,
        code=code,
        message=message,
        row_count=int(failing_rows.height),
        sample=sample_frame.to_dicts(),
    )


def _expected_series_id_expr() -> pl.Expr:
    return pl.concat_str(
        [pl.col("location_id").cast(pl.Utf8), pl.col("product_id").cast(pl.Utf8)],
        separator=SERIES_KEY_SEPARATOR,
    )


def _null_required_column_issues(frame: pl.DataFrame) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    for column_name in REQUIRED_NON_NULL_COLUMNS:
        failing_rows = frame.filter(pl.col(column_name).is_null())
        if failing_rows.height:
            issues.append(
                _build_issue(
                    severity="error",
                    code=f"null_{column_name}",
                    message=f"Required canonical column `{column_name}` contains null values.",
                    failing_rows=failing_rows,
                    sample_columns=GRAIN_COLUMNS + [column_name],
                )
            )
    return issues


def _duplicate_grain_issue(frame: pl.DataFrame) -> DataQualityIssue | None:
    duplicate_rows = frame.group_by(GRAIN_COLUMNS).len().filter(pl.col("len") > 1)
    if not duplicate_rows.height:
        return None
    return _build_issue(
        severity="error",
        code="duplicate_grain",
        message="Canonical grain is not unique on dataset_source x dt x location_id x product_id.",
        failing_rows=duplicate_rows,
        sample_columns=GRAIN_COLUMNS + ["len"],
    )


def _negative_demand_issue(frame: pl.DataFrame) -> DataQualityIssue | None:
    negative_demand_rows = frame.filter(pl.col("observed_demand_qty") < 0)
    if not negative_demand_rows.height:
        return None
    return _build_issue(
        severity="error",
        code="negative_observed_demand_qty",
        message="Observed demand quantity must stay non-negative in canonical datasets.",
        failing_rows=negative_demand_rows,
        sample_columns=GRAIN_COLUMNS + ["observed_demand_qty"],
    )


def _negative_revenue_issue(frame: pl.DataFrame) -> DataQualityIssue | None:
    negative_revenue_rows = frame.filter(pl.col("observed_revenue_net").is_not_null() & (pl.col("observed_revenue_net") < 0))
    if not negative_revenue_rows.height:
        return None
    return _build_issue(
        severity="warning",
        code="negative_observed_revenue_net",
        message="Observed revenue is negative on some rows; confirm returns/cancellations business rules.",
        failing_rows=negative_revenue_rows,
        sample_columns=GRAIN_COLUMNS + ["observed_revenue_net", "observed_demand_qty"],
    )


def _invalid_series_issue(frame: pl.DataFrame) -> DataQualityIssue | None:
    invalid_series_rows = frame.filter(pl.col("series_id") != _expected_series_id_expr())
    if not invalid_series_rows.height:
        return None
    return _build_issue(
        severity="error",
        code="invalid_series_id",
        message="series_id must equal location_id__product_id for auditability and stable joins.",
        failing_rows=invalid_series_rows,
        sample_columns=GRAIN_COLUMNS + ["series_id"],
    )


def _zero_demand_non_zero_revenue_issue(frame: pl.DataFrame) -> DataQualityIssue | None:
    inconsistent_revenue_rows = frame.filter(
        pl.col("observed_demand_qty").eq(0)
        & pl.col("observed_revenue_net").is_not_null()
        & pl.col("observed_revenue_net").ne(0)
    )
    if not inconsistent_revenue_rows.height:
        return None
    return _build_issue(
        severity="warning",
        code="zero_demand_non_zero_revenue",
        message="Rows with zero demand but non-zero revenue should be explained before modelling.",
        failing_rows=inconsistent_revenue_rows,
        sample_columns=GRAIN_COLUMNS + ["observed_demand_qty", "observed_revenue_net"],
    )


def _stockout_flag_availability_issue(frame: pl.DataFrame) -> DataQualityIssue | None:
    inconsistent_stockout_rows = frame.filter(
        pl.col("observed_stockout_available").eq(False) & pl.col("observed_stockout_flag").eq(True)
    )
    if not inconsistent_stockout_rows.height:
        return None
    return _build_issue(
        severity="warning",
        code="stockout_flag_without_availability",
        message="observed_stockout_flag is true while observed_stockout_available is false.",
        failing_rows=inconsistent_stockout_rows,
        sample_columns=GRAIN_COLUMNS + ["observed_stockout_available", "observed_stockout_flag"],
    )


def _append_issue_if_present(issues: list[DataQualityIssue], issue: DataQualityIssue | None) -> None:
    if issue is not None:
        issues.append(issue)


def validate_canonical_frame(frame: pl.DataFrame, *, dataset_name: str) -> DatasetQualityReport:
    """Validate one canonical silver/gold-like frame before trusting it downstream."""

    issues = _null_required_column_issues(frame)
    _append_issue_if_present(issues, _duplicate_grain_issue(frame))
    _append_issue_if_present(issues, _negative_demand_issue(frame))
    _append_issue_if_present(issues, _negative_revenue_issue(frame))
    _append_issue_if_present(issues, _invalid_series_issue(frame))
    _append_issue_if_present(issues, _zero_demand_non_zero_revenue_issue(frame))
    _append_issue_if_present(issues, _stockout_flag_availability_issue(frame))
    row_count = int(frame.height)
    series_count = int(frame.select(pl.col("series_id").n_unique()).item()) if row_count else 0

    return DatasetQualityReport(
        dataset_name=dataset_name,
        row_count=row_count,
        series_count=series_count,
        issues=issues,
    )


def raise_on_error_issues(report: DatasetQualityReport) -> None:
    """Fail fast when medallion contract violations are detected."""

    error_issues = [issue for issue in report.issues if issue.severity == "error"]
    if not error_issues:
        return

    issue_summaries = ", ".join(f"{issue.code}={issue.row_count}" for issue in error_issues)
    raise ValueError(
        f"Canonical dataset quality gate failed for `{report.dataset_name}`: {issue_summaries}."
    )

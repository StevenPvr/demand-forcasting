from __future__ import annotations

"""Prepare un dataset journalier regulier a horizon calendrier J+1."""

import json
import logging
import time
import unicodedata
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.data_cleaning.constants import (
    AFTERNOON_END_MINUTE,
    CSV_ENCODING,
    DAILY_FREQUENCY,
    EXOG_SALES_PREFIX,
    EXOG_SALES_SUFFIX,
    FORECAST_HORIZON_DAYS,
    INVALID_ARTICLES,
    LUNCH_END_MINUTE,
    MORNING_END_MINUTE,
    ORIGIN_DATE_COLUMN,
    ORIGIN_MISSING_FLAG_COLUMN,
    TARGET_ARTICLE,
    TARGET_COLUMN,
    TARGET_DATE_COLUMN,
    TARGET_MISSING_FLAG_COLUMN,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
SALES_CATEGORY_NAMES: tuple[str, ...] = (
    "bread",
    "viennoiserie",
    "pastry",
    "sandwich",
    "beverage",
    "other",
)


def _drop_export_index(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Supprime la colonne d'index exportee par erreur dans le CSV source."""

    keep_columns: list[str] = [
        str(column)
        for column in raw_df.columns
        if str(column).strip() and not str(column).lower().startswith("unnamed:")
    ]
    if keep_columns and keep_columns[0].lower() == "index":
        keep_columns = keep_columns[1:]
    return raw_df.loc[:, keep_columns].copy()


def load_sales_dataset(input_csv: Path) -> pd.DataFrame:
    """Charge et normalise les colonnes utiles du dataset brut."""

    raw_df: pd.DataFrame = pd.read_csv(input_csv)
    nan_before: int = int(raw_df.isna().sum().sum())
    sales_df: pd.DataFrame = _drop_export_index(raw_df)
    sales_df["date"] = pd.to_datetime(sales_df["date"], format="%Y-%m-%d").dt.strftime(
        "%Y-%m-%d"
    )
    sales_df["time"] = pd.to_datetime(sales_df["time"], format="%H:%M").dt.strftime(
        "%H:%M"
    )
    sales_df["article"] = sales_df["article"].astype("string").str.strip()
    sales_df["ticket_number"] = sales_df["ticket_number"].astype("string").str.strip()
    sales_df["Quantity"] = sales_df["Quantity"].astype(float)
    sales_df["unit_price"] = _parse_unit_price_series(cast(pd.Series, sales_df["unit_price"]))
    nan_after: int = int(sales_df.isna().sum().sum())
    LOGGER.info("Loaded %d rows from %s", len(sales_df), input_csv)
    LOGGER.info("NaN count before cleanup: %d", nan_before)
    LOGGER.info("NaN count after cleanup: %d", nan_after)
    return sales_df.loc[:, ["date", "time", "ticket_number", "article", "Quantity", "unit_price"]]


def _parse_unit_price_series(price_series: pd.Series) -> pd.Series:
    """Convertit les prix unitaires string du CSV en float."""

    normalized_series = (
        price_series.astype("string")
        .fillna("0")
        .str.replace("€", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    numeric_series = cast(pd.Series, pd.to_numeric(normalized_series, errors="coerce"))
    return cast(pd.Series, numeric_series.fillna(0.0).astype(float))


def _cancel_negative_sales_for_article(article_df: pd.DataFrame) -> pd.DataFrame:
    """Compense chaque quantite negative avec les ventes positives passees les plus proches."""

    working_df: pd.DataFrame = article_df.copy()
    positive_indices: list[int] = []
    for row_index in working_df.index:
        quantity: float = float(working_df.at[row_index, "Quantity"])
        if quantity > 0:
            positive_indices.append(row_index)
            continue
        remaining_to_cancel: float = abs(quantity)
        while remaining_to_cancel > 0 and positive_indices:
            previous_index: int = positive_indices[-1]
            previous_quantity: float = float(working_df.at[previous_index, "Quantity"])
            cancelled_quantity: float = min(previous_quantity, remaining_to_cancel)
            working_df.at[previous_index, "Quantity"] = previous_quantity - cancelled_quantity
            remaining_to_cancel -= cancelled_quantity
            if float(working_df.at[previous_index, "Quantity"]) <= 0:
                positive_indices.pop()
        working_df.at[row_index, "Quantity"] = 0.0
    return working_df


def _remove_negative_cancellations(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les annulations negatives en annulant d'abord le positif precedent."""

    preserved_columns: list[str] = [str(column) for column in sales_df.columns]
    ordered_df: pd.DataFrame = sales_df.copy()
    ordered_df["row_order"] = range(len(ordered_df))
    ordered_df = ordered_df.sort_values(["article", "date", "time", "row_order"])
    cleaned_articles: list[pd.DataFrame] = []
    negative_rows: int = int((ordered_df["Quantity"] < 0).sum())
    for _, article_df in ordered_df.groupby("article", sort=False):
        cleaned_articles.append(_cancel_negative_sales_for_article(article_df))
    cleaned_df: pd.DataFrame = pd.concat(cleaned_articles, ignore_index=True)
    cleaned_df = cleaned_df.loc[cleaned_df["Quantity"] > 0].copy()
    cleaned_df = cleaned_df.sort_values(["date", "time", "row_order"]).reset_index(drop=True)
    if negative_rows > 0:
        LOGGER.info("Removed %d negative cancellation rows", negative_rows)
    return cleaned_df.loc[:, preserved_columns]


def remove_negative_cancellations(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les annulations negatives via l'API publique de nettoyage partagee."""

    return _remove_negative_cancellations(sales_df)


def _filter_invalid_articles(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Retire les libelles parasites qui ne doivent pas devenir des exogenes."""

    article_series: pd.Series = cast(pd.Series, sales_df["article"])
    invalid_articles: tuple[str, ...] = tuple(INVALID_ARTICLES)
    valid_mask: pd.Series = cast(pd.Series, ~article_series.isin(invalid_articles))
    filtered_df: pd.DataFrame = cast(pd.DataFrame, sales_df.loc[valid_mask].copy())
    removed_rows: int = int((~valid_mask).sum())
    if removed_rows > 0:
        LOGGER.info("Removed %d rows with invalid articles", removed_rows)
    return filtered_df


def filter_invalid_articles(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Retire les articles invalides via l'API publique de nettoyage partagee."""

    return _filter_invalid_articles(sales_df)


def _aggregate_sales_by_day(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Agrege les quantites vendues par jour et par article."""

    grouped_df: pd.DataFrame = cast(
        pd.DataFrame,
        sales_df.groupby(["date", "article"], as_index=False)["Quantity"].sum(),
    )
    wide_df: pd.DataFrame = cast(
        pd.DataFrame,
        grouped_df.pivot(index="date", columns="article", values="Quantity").fillna(0.0),
    )
    wide_df.columns.name = None
    wide_df.index = cast(pd.DatetimeIndex, pd.to_datetime(wide_df.index, format="%Y-%m-%d"))
    return wide_df.sort_index()


def aggregate_sales_by_day(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Agrege les ventes journalieres via l'API publique partagee."""

    return _aggregate_sales_by_day(sales_df)


def _ensure_transaction_columns(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Garantit la presence des colonnes transactionnelles utiles aux features."""

    prepared_df: pd.DataFrame = sales_df.copy()
    if "ticket_number" not in prepared_df.columns:
        prepared_df["ticket_number"] = [f"ticket_{index}" for index in range(len(prepared_df))]
    if "unit_price" not in prepared_df.columns:
        prepared_df["unit_price"] = 0.0
    prepared_df["ticket_number"] = prepared_df["ticket_number"].astype("string").fillna("missing_ticket")
    numeric_unit_price = cast(pd.Series, pd.to_numeric(prepared_df["unit_price"], errors="coerce"))
    prepared_df["unit_price"] = cast(pd.Series, numeric_unit_price.fillna(0.0).astype(float))
    return prepared_df


def _ascii_lower(value: str) -> str:
    """Normalise un texte en ASCII minuscule pour des heuristiques robustes."""

    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", errors="ignore").decode("ascii").lower()


def _sales_category(article_name: str) -> str:
    """Affecte une famille produit large a partir du libelle article."""

    normalized_name = _ascii_lower(article_name)
    if any(keyword in normalized_name for keyword in ("sandwich", "formule", "plat", "traiteur", "triangle")):
        return "sandwich"
    if any(keyword in normalized_name for keyword in ("boisson", "cafe", "eau", "the")):
        return "beverage"
    if any(
        keyword in normalized_name
        for keyword in (
            "croissant",
            "pain au chocolat",
            "pain aux raisins",
            "chausson",
            "kouign",
            "brioche",
            "pepito",
            "viennoiserie",
            "palmier",
            "pain choco amandes",
        )
    ):
        return "viennoiserie"
    if any(
        keyword in normalized_name
        for keyword in (
            "tarte",
            "eclair",
            "cookie",
            "flan",
            "far",
            "paris brest",
            "milles feuilles",
            "tropezienne",
            "fraisier",
            "savarin",
            "macaron",
            "fondant",
            "chantilly",
            "religieuse",
            "st honore",
            "royal",
            "meringue",
            "brownies",
            "crumble",
            "entremets",
            "nantais",
            "breton",
            "gal ",
            "galette",
            "sucette",
        )
    ):
        return "pastry"
    if any(
        keyword in normalized_name
        for keyword in (
            "baguette",
            "banette",
            "pain ",
            "boule",
            "campagne",
            "complet",
            "moisson",
            "ficelle",
            "seigle",
            "bread",
            "viennoise",
            "gache",
            "polka",
        )
    ) or normalized_name.startswith("pain"):
        return "bread"
    return "other"


def _minute_of_day(time_value: str) -> int:
    """Convertit une heure HH:MM en minute de la journee."""

    hour_text, minute_text = str(time_value).split(":")
    return int(hour_text) * 60 + int(minute_text)


def _sales_daypart(minute_of_day: int) -> str:
    """Retourne le segment de journee associe a une minute donnee."""

    if minute_of_day < MORNING_END_MINUTE:
        return "morning"
    if minute_of_day < LUNCH_END_MINUTE:
        return "lunch"
    if minute_of_day < AFTERNOON_END_MINUTE:
        return "afternoon"
    return "evening"


def _augment_sales_with_context_columns(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes transactionnelles necessaires aux agrégats journaliers."""

    prepared_df = _ensure_transaction_columns(sales_df)
    enriched_df = prepared_df.copy()
    enriched_df["line_revenue"] = enriched_df["Quantity"] * enriched_df["unit_price"]
    enriched_df["minute_of_day"] = [
        _minute_of_day(time_value)
        for time_value in cast(pd.Series, enriched_df["time"]).astype(str)
    ]
    enriched_df["sales_daypart"] = [
        _sales_daypart(minute)
        for minute in cast(pd.Series, enriched_df["minute_of_day"]).astype(int)
    ]
    enriched_df["sales_category"] = [
        _sales_category(article_name)
        for article_name in cast(pd.Series, enriched_df["article"]).astype(str)
    ]
    return enriched_df


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divise deux series en evitant les divisions par zero."""

    safe_denominator = denominator.replace(0.0, pd.NA)
    divided_series = cast(pd.Series, numerator.divide(safe_denominator))
    return cast(pd.Series, divided_series.fillna(0.0).astype(float))


def _aggregate_daypart_shares(context_df: pd.DataFrame, total_quantity: pd.Series) -> pd.DataFrame:
    """Construit les parts de quantite par segment horaire."""

    grouped = cast(
        pd.DataFrame,
        context_df.groupby(["date", "sales_daypart"], as_index=False)["Quantity"].sum(),
    )
    pivot_df = cast(
        pd.DataFrame,
        grouped.pivot(index="date", columns="sales_daypart", values="Quantity").fillna(0.0),
    )
    for column in ("morning", "lunch", "afternoon", "evening"):
        if column not in pivot_df.columns:
            pivot_df[column] = 0.0
    pivot_df = pivot_df.loc[:, ["morning", "lunch", "afternoon", "evening"]]
    pivot_df.columns = [f"exog_sales_{column}_quantity_share_lag1" for column in pivot_df.columns]
    return pivot_df.apply(lambda column: _safe_divide(column, total_quantity))


def _aggregate_category_metrics(
    context_df: pd.DataFrame,
    metric_name: str,
    metric_label: str,
) -> pd.DataFrame:
    """Agrege un metrique par famille produit et par jour."""

    grouped = cast(
        pd.DataFrame,
        context_df.groupby(["date", "sales_category"], as_index=False)[metric_name].sum(),
    )
    pivot_df = cast(
        pd.DataFrame,
        grouped.pivot(index="date", columns="sales_category", values=metric_name).fillna(0.0),
    )
    for category_name in SALES_CATEGORY_NAMES:
        if category_name not in pivot_df.columns:
            pivot_df[category_name] = 0.0
    ordered_df = pivot_df.loc[:, list(SALES_CATEGORY_NAMES)].copy()
    ordered_df.columns = [
        f"exog_sales_category_{category_name}_{metric_label}_lag1"
        for category_name in SALES_CATEGORY_NAMES
    ]
    return ordered_df


def _aggregate_sales_context_by_day(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Construit des agrégats journaliers riches a partir des transactions brutes."""

    context_df = _augment_sales_with_context_columns(sales_df)
    daily_group = context_df.groupby("date")
    base_df = cast(
        pd.DataFrame,
        daily_group.agg(
            total_quantity=("Quantity", "sum"),
            total_revenue=("line_revenue", "sum"),
            ticket_count=("ticket_number", "nunique"),
            unique_article_count=("article", "nunique"),
            first_sale_minute=("minute_of_day", "min"),
            last_sale_minute=("minute_of_day", "max"),
        ),
    )
    total_quantity_series = cast(pd.Series, base_df["total_quantity"])
    total_revenue_series = cast(pd.Series, base_df["total_revenue"])
    ticket_count_series = cast(pd.Series, base_df["ticket_count"])
    base_df["avg_items_per_ticket"] = _safe_divide(total_quantity_series, ticket_count_series)
    base_df["avg_revenue_per_ticket"] = _safe_divide(total_revenue_series, ticket_count_series)
    base_df["weighted_unit_price"] = _safe_divide(total_revenue_series, total_quantity_series)
    base_df["sales_span_minutes"] = (
        base_df["last_sale_minute"] - base_df["first_sale_minute"]
    ).clip(lower=0.0)
    renamed_base_df = base_df.rename(
        columns={
            "total_quantity": "exog_sales_total_quantity_lag1",
            "total_revenue": "exog_sales_total_revenue_lag1",
            "ticket_count": "exog_sales_ticket_count_lag1",
            "unique_article_count": "exog_sales_unique_article_count_lag1",
            "avg_items_per_ticket": "exog_sales_avg_items_per_ticket_lag1",
            "avg_revenue_per_ticket": "exog_sales_avg_revenue_per_ticket_lag1",
            "weighted_unit_price": "exog_sales_weighted_unit_price_lag1",
            "first_sale_minute": "exog_sales_first_sale_minute_lag1",
            "last_sale_minute": "exog_sales_last_sale_minute_lag1",
            "sales_span_minutes": "exog_sales_sales_span_minutes_lag1",
        }
    )
    daypart_df = _aggregate_daypart_shares(
        context_df,
        cast(pd.Series, renamed_base_df["exog_sales_total_quantity_lag1"]),
    )
    category_quantity_df = _aggregate_category_metrics(context_df, "Quantity", "quantity")
    category_revenue_df = _aggregate_category_metrics(context_df, "line_revenue", "revenue")
    category_share_df = category_quantity_df.apply(
        lambda column: _safe_divide(
            cast(pd.Series, column),
            cast(pd.Series, renamed_base_df["exog_sales_total_quantity_lag1"]),
        )
    )
    category_share_df.columns = [
        column.replace("_quantity_lag1", "_quantity_share_lag1") for column in category_quantity_df.columns
    ]
    feature_df = pd.concat(
        [renamed_base_df, daypart_df, category_quantity_df, category_revenue_df, category_share_df],
        axis=1,
    )
    feature_df.index = pd.to_datetime(feature_df.index, format="%Y-%m-%d")
    return feature_df.sort_index()


def _regularize_daily_sales_grid(
    daily_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reindexe la serie sur tous les jours calendaires et trace les jours manquants."""

    daily_index: pd.DatetimeIndex = cast(pd.DatetimeIndex, daily_df.index)
    full_index = pd.date_range(daily_index.min(), daily_index.max(), freq=DAILY_FREQUENCY)
    regularized_df = daily_df.reindex(full_index, fill_value=0.0)
    missing_mask = pd.Series(
        (~regularized_df.index.isin(daily_index)).astype(int),
        index=regularized_df.index,
        dtype=int,
    )
    first_date: pd.Timestamp = cast(pd.Timestamp, regularized_df.index.min())
    last_date: pd.Timestamp = cast(pd.Timestamp, regularized_df.index.max())
    LOGGER.info(
        "Regularized daily grid from %s to %s with %d missing dates filled to zero",
        first_date.date(),
        last_date.date(),
        int(missing_mask.sum()),
    )
    return regularized_df, missing_mask


def regularize_daily_sales_grid(
    daily_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Regularise la grille journaliere via l'API publique partagee."""

    return _regularize_daily_sales_grid(daily_df)


def _build_sales_exog_mapping(feature_articles: list[str]) -> dict[str, str]:
    """Construit un mapping stable entre colonnes exogenes et produits."""

    return {
        f"{EXOG_SALES_PREFIX}{index}{EXOG_SALES_SUFFIX}": article
        for index, article in enumerate(feature_articles, start=1)
    }


def _training_rows(
    daily_df: pd.DataFrame,
    sales_context_df: pd.DataFrame,
    missing_mask: pd.Series,
    sales_exog_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """Construit les lignes du dataset horizon-aware origin_date -> target_date."""

    rows: list[dict[str, Any]] = []
    daily_index: pd.DatetimeIndex = cast(pd.DatetimeIndex, daily_df.index)
    for origin_date in daily_index[:-FORECAST_HORIZON_DAYS]:
        origin_timestamp: pd.Timestamp = cast(pd.Timestamp, origin_date)
        target_date: pd.Timestamp = cast(
            pd.Timestamp,
            origin_timestamp + pd.Timedelta(days=FORECAST_HORIZON_DAYS),
        )
        row: dict[str, Any] = {
            ORIGIN_DATE_COLUMN: origin_timestamp.strftime("%Y-%m-%d"),
            TARGET_DATE_COLUMN: target_date.strftime("%Y-%m-%d"),
        }
        for column_name, article in sales_exog_mapping.items():
            row[column_name] = float(daily_df.at[origin_timestamp, article])
        for column_name in sales_context_df.columns:
            row[column_name] = float(sales_context_df.at[origin_timestamp, column_name])
        row[ORIGIN_MISSING_FLAG_COLUMN] = int(missing_mask.loc[origin_timestamp])
        row[TARGET_MISSING_FLAG_COLUMN] = int(missing_mask.loc[target_date])
        row[TARGET_COLUMN] = float(daily_df.at[target_date, TARGET_ARTICLE])
        rows.append(row)
    return rows


def build_daily_baguette_training_frame(
    sales_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Transforme les ventes article par article en dataset J+1 regulier."""

    filtered_sales_df: pd.DataFrame = _filter_invalid_articles(sales_df)
    filtered_sales_df = _remove_negative_cancellations(filtered_sales_df)
    daily_df: pd.DataFrame = _aggregate_sales_by_day(filtered_sales_df)
    sales_context_df: pd.DataFrame = _aggregate_sales_context_by_day(filtered_sales_df)
    if TARGET_ARTICLE not in daily_df.columns:
        raise ValueError(f"Target article not found in dataset: {TARGET_ARTICLE}")
    regularized_df, missing_mask = _regularize_daily_sales_grid(daily_df)
    regularized_context_df = sales_context_df.reindex(regularized_df.index, fill_value=0.0)
    feature_articles: list[str] = sorted(
        article for article in regularized_df.columns if article != TARGET_ARTICLE
    )
    sales_exog_mapping = _build_sales_exog_mapping(feature_articles)
    training_df = pd.DataFrame(
        _training_rows(
            daily_df=regularized_df,
            sales_context_df=regularized_context_df,
            missing_mask=missing_mask,
            sales_exog_mapping=sales_exog_mapping,
        )
    )
    LOGGER.info("Aggregated to %d regularized daily rows", len(regularized_df))
    LOGGER.info(
        "Prepared %d daily training rows with %d sales exogenous columns",
        len(training_df),
        len(sales_exog_mapping),
    )
    return training_df, sales_exog_mapping


def _write_feature_mapping(output_json: Path, sales_exog_mapping: dict[str, str]) -> None:
    """Ecrit le mapping JSON des colonnes exogenes de ventes."""

    derived_feature_columns: list[str] = [
        "exog_sales_total_quantity_lag1",
        "exog_sales_total_revenue_lag1",
        "exog_sales_ticket_count_lag1",
        "exog_sales_unique_article_count_lag1",
        "exog_sales_avg_items_per_ticket_lag1",
        "exog_sales_avg_revenue_per_ticket_lag1",
        "exog_sales_weighted_unit_price_lag1",
        "exog_sales_first_sale_minute_lag1",
        "exog_sales_last_sale_minute_lag1",
        "exog_sales_sales_span_minutes_lag1",
        "exog_sales_morning_quantity_share_lag1",
        "exog_sales_lunch_quantity_share_lag1",
        "exog_sales_afternoon_quantity_share_lag1",
        "exog_sales_evening_quantity_share_lag1",
    ] + [
        f"exog_sales_category_{category_name}_{metric_name}_lag1"
        for category_name in SALES_CATEGORY_NAMES
        for metric_name in ("quantity", "revenue", "quantity_share")
    ]
    payload: dict[str, Any] = {
        "aggregation_level": "daily",
        "frequency": DAILY_FREQUENCY,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "target_article": TARGET_ARTICLE,
        "target_column": TARGET_COLUMN,
        "origin_date_column": ORIGIN_DATE_COLUMN,
        "target_date_column": TARGET_DATE_COLUMN,
        "sales_exog_mapping": sales_exog_mapping,
        "derived_sales_feature_columns": derived_feature_columns,
        "flag_columns": [ORIGIN_MISSING_FLAG_COLUMN, TARGET_MISSING_FLAG_COLUMN],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding=CSV_ENCODING,
    )


def prepare_daily_baguette_dataset(
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
) -> dict[str, int | str]:
    """Prepare et sauvegarde le dataset journalier et son mapping JSON."""

    start_time: float = time.perf_counter()
    sales_df: pd.DataFrame = load_sales_dataset(input_csv)
    training_df, sales_exog_mapping = build_daily_baguette_training_frame(sales_df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(output_csv, index=False, encoding=CSV_ENCODING)
    _write_feature_mapping(output_json, sales_exog_mapping)
    elapsed_seconds: float = time.perf_counter() - start_time
    LOGGER.info("Saved daily training dataset to %s", output_csv)
    LOGGER.info("Saved sales exogenous mapping to %s", output_json)
    LOGGER.info("Preparation completed in %.3f seconds", elapsed_seconds)
    feature_count: int = len(
        [
            column
            for column in training_df.columns
            if column.startswith("exog_") and not column.startswith("exog_flag_")
        ]
    )
    return {
        "row_count": len(training_df),
        "feature_count": feature_count,
        "output_csv": str(output_csv),
        "output_json": str(output_json),
    }

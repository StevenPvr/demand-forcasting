from __future__ import annotations

"""Analyse descriptive, decomposition et autocorrelation de la target."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
_CACHE_DIR: Path = Path(tempfile.gettempdir()) / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir())))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

from src.data_analyse.constants import (
    CSV_ENCODING,
    DEFAULT_MAX_LAGS,
    TARGET_COLUMN,
    TARGET_DATE_COLUMN,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def _configure_plot_style() -> None:
    """Applique un style lisible et colorblind-friendly pour les figures."""

    sns.set_theme(style="ticks", context="paper", palette="colorblind")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def load_target_series(input_csv: Path) -> pd.DataFrame:
    """Charge la serie cible avec son horodatage cible."""

    raw_df = pd.read_csv(input_csv)
    date_column = TARGET_DATE_COLUMN if TARGET_DATE_COLUMN in raw_df.columns else "date"
    target_df: pd.DataFrame = raw_df.loc[:, [date_column, TARGET_COLUMN]].copy()
    target_df["timestamp"] = pd.to_datetime(target_df[date_column], format="%Y-%m-%d")
    ordered_df: pd.DataFrame = target_df.sort_values("timestamp").reset_index(drop=True)
    LOGGER.info("Loaded %d target rows from %s", len(ordered_df), input_csv)
    return ordered_df


def infer_seasonal_period(target_df: pd.DataFrame) -> int:
    """Retourne une saisonnalite hebdomadaire par defaut pour la serie journaliere."""

    if len(target_df) < 7:
        return max(len(target_df), 2)
    return 7


def _safe_max_lags(row_count: int, requested_lags: int) -> int:
    """Borne le nombre de lags pour rester valide avec PACF."""

    pacf_limit: int = max((row_count // 2) - 1, 1)
    return max(min(requested_lags, pacf_limit), 1)


def _summary_payload(
    target_df: pd.DataFrame,
    seasonal_period: int,
    max_lags: int,
) -> dict[str, Any]:
    """Construit le resume numerique de l'analyse."""

    target_series: pd.Series = cast(pd.Series, target_df[TARGET_COLUMN]).astype(float)
    ljung_box = acorr_ljungbox(target_series, lags=[max_lags], return_df=True)
    acf_values = acf(target_series, nlags=max_lags, fft=True)
    pacf_values = pacf(target_series, nlags=max_lags, method="ywm")
    acf_array: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(acf_values, dtype=float)
    pacf_array: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(pacf_values, dtype=float)
    return {
        "row_count": int(len(target_df)),
        "seasonal_period": int(seasonal_period),
        "max_lags": int(max_lags),
        "target_mean": float(target_series.mean()),
        "target_std": cast(float, target_series.std()),
        "target_zero_share": float((target_series == 0).mean()),
        "ljung_box_stat": float(ljung_box["lb_stat"].iloc[0]),
        "ljung_box_pvalue": float(ljung_box["lb_pvalue"].iloc[0]),
        "acf_lag_1": float(acf_array[1]),
        "pacf_lag_1": float(pacf_array[1]),
    }


def _plot_target_timeseries(target_df: pd.DataFrame, output_dir: Path) -> None:
    """Trace la serie temporelle de la target."""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(target_df["timestamp"], target_df[TARGET_COLUMN], color="#0072B2", linewidth=1)
    ax.set_title("Daily baguette demand over time")
    ax.set_xlabel("Target date")
    ax.set_ylabel("Demand on next day")
    fig.tight_layout()
    fig.savefig(output_dir / "target_timeseries.png", bbox_inches="tight")
    plt.close(fig)


def _plot_target_distribution(target_df: pd.DataFrame, output_dir: Path) -> None:
    """Trace la distribution de la target."""

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        x=target_df[TARGET_COLUMN].to_numpy(dtype=float),
        bins=30,
        kde=True,
        color="#009E73",
        ax=ax,
    )
    ax.set_title("Target distribution")
    ax.set_xlabel("Demand on next day")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", bbox_inches="tight")
    plt.close(fig)


def _plot_target_decomposition(
    target_df: pd.DataFrame,
    output_dir: Path,
    seasonal_period: int,
) -> None:
    """Trace une decomposition STL de la target."""

    fig = _build_target_decomposition_figure(target_df, seasonal_period)
    fig.savefig(output_dir / "target_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def _build_target_decomposition_figure(
    target_df: pd.DataFrame,
    seasonal_period: int,
) -> Figure:
    """Construit une decomposition STL avec des dates lisibles sur l'axe X."""

    target_series = pd.Series(
        target_df[TARGET_COLUMN].to_numpy(),
        index=pd.DatetimeIndex(target_df["timestamp"]),
        dtype=float,
    )
    stl_result = STL(target_series, period=seasonal_period, robust=True).fit()
    fig = stl_result.plot()
    fig.set_size_inches(10, 8)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    for axis in fig.axes:
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)
    fig.axes[-1].set_xlabel("Target date")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _plot_target_acf_pacf(
    target_df: pd.DataFrame,
    output_dir: Path,
    max_lags: int,
) -> None:
    """Trace les graphiques ACF et PACF de la target."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(target_df[TARGET_COLUMN], lags=max_lags, ax=axes[0], zero=False)
    plot_pacf(target_df[TARGET_COLUMN], lags=max_lags, ax=axes[1], zero=False, method="ywm")
    axes[0].set_title("Autocorrelation function")
    axes[1].set_title("Partial autocorrelation function")
    fig.tight_layout()
    fig.savefig(output_dir / "target_acf_pacf.png", bbox_inches="tight")
    plt.close(fig)


def analyze_target_series(
    input_csv: Path,
    output_dir: Path,
    seasonal_period: int | None = None,
    max_lags: int | None = None,
) -> dict[str, int]:
    """Produit les graphiques et le resume de la target."""

    start_time: float = time.perf_counter()
    _configure_plot_style()
    target_df: pd.DataFrame = load_target_series(input_csv)
    resolved_period: int = seasonal_period or infer_seasonal_period(target_df)
    resolved_lags: int = _safe_max_lags(
        len(target_df),
        max_lags or min(DEFAULT_MAX_LAGS, resolved_period),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_payload(target_df, resolved_period, resolved_lags)
    (output_dir / "target_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding=CSV_ENCODING,
    )
    _plot_target_timeseries(target_df, output_dir)
    _plot_target_distribution(target_df, output_dir)
    _plot_target_decomposition(target_df, output_dir, resolved_period)
    _plot_target_acf_pacf(target_df, output_dir, resolved_lags)
    LOGGER.info("Saved target analysis to %s", output_dir)
    LOGGER.info("Analysis completed in %.3f seconds", time.perf_counter() - start_time)
    return {
        "row_count": int(len(target_df)),
        "seasonal_period": int(resolved_period),
        "max_lags": int(resolved_lags),
    }

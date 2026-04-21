from __future__ import annotations

from praedixa.demand_forecast.contracts.targets import DEFAULT_VARIATION_TARGET_COL
from praedixa.platform.runtime.paths import LOCAL_DUCKDB_PATH
from praedixa.platform.runtime.paths import OPTIMISATION_DIR


DEFAULT_DUCKDB_PATH = LOCAL_DUCKDB_PATH
DEFAULT_GOLD_TABLE = "gold.gold_daily_product_forecast_panel_d1"
DEFAULT_TRAIN_INPUT_PATH = None
DEFAULT_TUNING_INPUT_PATH = None
DEFAULT_OUTPUT_DIR = OPTIMISATION_DIR
DEFAULT_TARGET_COL = DEFAULT_VARIATION_TARGET_COL
DEFAULT_DATE_COL = "dt"
DEFAULT_DATASET_SOURCE_COL = "dataset_source"
DEFAULT_SAMPLE_STORE_COL = "location_id"
DEFAULT_TRAIN_SAMPLE_FRACTION = 0.10
DEFAULT_TUNING_SAMPLE_FRACTION = 0.10
DEFAULT_MIN_SAMPLES_PER_DATASET = 1000
DEFAULT_MAX_PARALLEL_FOLD_WORKERS = 5
DEFAULT_TARGET_TRANSFORM = "log1p"
DEFAULT_IDENTIFIER_FEATURE_COLS = (
    "series_id",
    "location_id",
    "product_id",
    "client_id",
    "gold_run_id",
)
DEFAULT_EXCLUDED_RISKY_FEATURE_COLS = (
    "current_day_demand_qty",
    "observed_revenue_net",
    "target_dt",
    "sample_weight_source",
    "sample_weight_business",
    "is_primary_eval_dataset",
    "source_legal_basis",
    "source_license_type",
    "source_review_status",
    "source_legal_status_snapshot",
)

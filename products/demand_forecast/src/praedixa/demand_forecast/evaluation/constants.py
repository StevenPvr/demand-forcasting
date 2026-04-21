from __future__ import annotations

from praedixa.demand_forecast.contracts.targets import DEFAULT_VARIATION_TARGET_COL
from praedixa.platform.runtime.paths import EVALUATION_DIR
from praedixa.platform.runtime.paths import OPTIMISATION_DIR
from praedixa.platform.runtime.paths import SOURCES_DIR


DEFAULT_TRAIN_SELECTION_INPUT_PATH = None
DEFAULT_TRAIN_TUNING_INPUT_PATH = None
DEFAULT_VAL_INPUT_PATH = None
DEFAULT_BEST_PARAMS_PATH = OPTIMISATION_DIR / "best_optuna_params.json"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR
DEFAULT_REQUESTED_TARGET_COL = DEFAULT_VARIATION_TARGET_COL
DEFAULT_BAKERY_REFERENCE_TRAIN_CSV = SOURCES_DIR / "bakery_sales" / "data" / "data_preprocessing" / "daily_product_arima_train_2021_2022.csv"
DEFAULT_BAKERY_REFERENCE_VAL_CSV = SOURCES_DIR / "bakery_sales" / "data" / "data_preprocessing" / "daily_product_arima_val_2021_2022.csv"
DEFAULT_BAKERY_REFERENCE_TEST_CSV = SOURCES_DIR / "bakery_sales" / "data" / "data_preprocessing" / "daily_product_arima_test_2021_2022.csv"
DEFAULT_MODEL_FAMILY = "FOUNDATION_TFT"
DEFAULT_EVALUATION_EXCLUDED_HELPER_FEATURE_COLS = (
    "rn_sample",
    "stratum_row_count",
)

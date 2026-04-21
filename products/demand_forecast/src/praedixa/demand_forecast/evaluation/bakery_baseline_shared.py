from __future__ import annotations

from collections.abc import Callable

import pandas as pd


HistoryForecaster = Callable[[list[float], pd.Timestamp], float]
BlendForecaster = Callable[[dict[str, float]], float]
FIELD_BASELINE_NAME = "blend_lag_1_lag_7_50_50"
DEFAULT_UNIT_COST_EUR = 1.0

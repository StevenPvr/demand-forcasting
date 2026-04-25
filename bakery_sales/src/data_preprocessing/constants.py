from __future__ import annotations

"""Constantes reutilisables pour le preprocessing journalier."""

CSV_ENCODING: str = "utf-8"
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15
TARGET_COLUMN: str = "target_baguette_t_plus_1"
ORIGIN_DATE_COLUMN: str = "origin_date"
TARGET_DATE_COLUMN: str = "target_date"
DAILY_FREQUENCY: str = "D"
SEASONAL_PERIOD_WEEKLY: int = 7
ADF_PVALUE_THRESHOLD: float = 0.05
DAY_OF_WEEK_PERIOD: int = 7
DAY_OF_MONTH_PERIOD: int = 31
DAY_OF_YEAR_PERIOD: int = 366
YEARLY_FOURIER_HARMONICS: tuple[int, ...] = (2, 3)
WEEK_OF_YEAR_PERIOD: int = 53
WEEK_OF_MONTH_PERIOD: int = 6
MONTH_PERIOD: int = 12
QUARTER_PERIOD: int = 4
TARGET_AUTOREGRESSIVE_LAGS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 14, 21, 30)
TARGET_ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 28)
PAYDAY_WINDOW_DAYS: tuple[int, ...] = (28, 29, 30, 31, 1, 2, 3, 4, 5)
DEFAULT_SCHOOL_ZONE: str = "B"
DEFAULT_SCHOOL_HOLIDAYS_URL: str = (
    "https://www.data.gouv.fr/api/1/datasets/r/c3781037-dffb-4789-9af9-15a955336771"
)
DEFAULT_PUBLIC_HOLIDAYS_URL: str = (
    "https://etalab.github.io/jours-feries-france-data/json/metropole.json"
)
DEFAULT_WEATHER_API_BASE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_WEATHER_CITY_NAME: str = "Guerande"
DEFAULT_WEATHER_LATITUDE: float = 47.32829
DEFAULT_WEATHER_LONGITUDE: float = -2.42934
DEFAULT_WEATHER_TIMEZONE: str = "Europe/Paris"
WEATHER_DAILY_VARIABLES: tuple[str, ...] = (
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_mean",
    "rain_sum",
    "snowfall_sum",
    "precipitation_sum",
    "precipitation_hours",
    "sunshine_duration",
    "shortwave_radiation_sum",
    "weather_code",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "cloud_cover_mean",
    "relative_humidity_2m_mean",
)
HOLIDAY_DISTANCE_FILL_VALUE: float = 999.0
HOT_DAY_TEMPERATURE_THRESHOLD: float = 25.0
COLD_DAY_TEMPERATURE_THRESHOLD: float = 5.0
HEAVY_RAIN_MM_THRESHOLD: float = 5.0
SUNNY_DAY_HOURS_THRESHOLD: float = 8.0
WINDY_DAY_KMH_THRESHOLD: float = 30.0

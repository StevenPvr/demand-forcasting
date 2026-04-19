from __future__ import annotations

"""Chemins centralises pour les artefacts d'analyse."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_ANALYSE_DIR: Path = DATA_DIR / "data_analyse"
DAILY_INPUT_CSV: Path = DATA_DIR / "data_cleaning" / "daily_baguette_2021_2022.csv"

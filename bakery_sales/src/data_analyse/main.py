from __future__ import annotations

"""Point d'entree executable pour l'analyse de la target."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_analyse.analyze_target_series import analyze_target_series
from src.data_analyse.paths import DAILY_INPUT_CSV, DATA_ANALYSE_DIR


def main() -> None:
    """Execute l'analyse descriptive et temporelle de la target."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    analyze_target_series(
        input_csv=DAILY_INPUT_CSV,
        output_dir=DATA_ANALYSE_DIR,
    )


if __name__ == "__main__":
    main()

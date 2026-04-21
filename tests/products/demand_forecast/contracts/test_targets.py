from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.contracts.targets import (  # noqa: E402
    DEFAULT_VARIATION_TARGET_COL,
    ensure_learning_target_column,
    reconstruct_absolute_predictions,
    resolve_target_contract,
)


class TargetUtilsTests(unittest.TestCase):
    def test_resolve_target_contract_prefers_delta_log_wow_when_anchor_exists(self) -> None:
        frame = pd.DataFrame(
            {
                "target": [10.0, 12.0, 14.0],
                "sale_amount_lag_7": [8.0, 10.0, 11.0],
            }
        )

        target_contract = resolve_target_contract(frame, frame, requested_target_col=DEFAULT_VARIATION_TARGET_COL)

        self.assertEqual(target_contract.learning_target_col, DEFAULT_VARIATION_TARGET_COL)
        self.assertEqual(target_contract.absolute_target_col, "target")
        self.assertEqual(target_contract.target_mode, "delta_log_wow")
        self.assertEqual(target_contract.reconstruction_anchor_col, "sale_amount_lag_7")

    def test_ensure_learning_target_column_builds_delta_log_wow_from_absolute_target(self) -> None:
        frame = pd.DataFrame(
            {
                "target": [10.0, 12.0, 14.0],
                "sale_amount_lag_7": [8.0, 10.0, 11.0],
            }
        )
        target_contract = resolve_target_contract(frame, frame, requested_target_col=DEFAULT_VARIATION_TARGET_COL)

        enriched = ensure_learning_target_column(frame, target_contract)
        expected = np.log1p(frame["target"]) - np.log1p(frame["sale_amount_lag_7"])

        np.testing.assert_allclose(
            enriched[DEFAULT_VARIATION_TARGET_COL].to_numpy(dtype=float),
            np.asarray(expected, dtype=float),
        )

    def test_reconstruct_absolute_predictions_recovers_absolute_scale_from_delta_log_wow(self) -> None:
        frame = pd.DataFrame(
            {
                "target": [10.0, 12.0, 14.0],
                "sale_amount_lag_7": [8.0, 10.0, 11.0],
            }
        )
        target_contract = resolve_target_contract(frame, frame, requested_target_col=DEFAULT_VARIATION_TARGET_COL)
        enriched = ensure_learning_target_column(frame, target_contract)

        reconstructed = reconstruct_absolute_predictions(
            enriched[DEFAULT_VARIATION_TARGET_COL].to_numpy(dtype=float),
            enriched,
            target_contract,
        )

        np.testing.assert_allclose(reconstructed, frame["target"].to_numpy(dtype=float))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.contracts.targets import (  # noqa: E402
    DEFAULT_ABSOLUTE_TARGET_COL,
    DEFAULT_FIELD_BASELINE_COL,
    DEFAULT_FIELD_BASELINE_RESIDUAL_TARGET_COL,
    ensure_learning_target_column,
    reconstruct_absolute_predictions,
    resolve_target_contract,
)


class TargetUtilsTests(unittest.TestCase):
    def test_resolve_target_contract_defaults_to_absolute_identity(self) -> None:
        frame = pd.DataFrame(
            {
                DEFAULT_ABSOLUTE_TARGET_COL: [10.0, 12.0, 14.0],
                "sale_amount_lag_7": [8.0, 10.0, 11.0],
            }
        )

        target_contract = resolve_target_contract(frame, frame)

        self.assertEqual(
            target_contract.learning_target_col, DEFAULT_ABSOLUTE_TARGET_COL
        )
        self.assertEqual(
            target_contract.absolute_target_col, DEFAULT_ABSOLUTE_TARGET_COL
        )
        self.assertEqual(target_contract.target_mode, "identity")
        self.assertIsNone(target_contract.reconstruction_anchor_col)

    def test_ensure_learning_target_column_returns_materialized_absolute_target(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                DEFAULT_ABSOLUTE_TARGET_COL: [10.0, 12.0, 14.0],
                "sale_amount_lag_7": [8.0, 10.0, 11.0],
            }
        )
        target_contract = resolve_target_contract(frame, frame)

        enriched = ensure_learning_target_column(frame, target_contract)

        np.testing.assert_allclose(
            enriched[DEFAULT_ABSOLUTE_TARGET_COL].to_numpy(dtype=float),
            frame[DEFAULT_ABSOLUTE_TARGET_COL].to_numpy(dtype=float),
        )

    def test_reconstruct_absolute_predictions_is_identity_for_absolute_target(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                DEFAULT_ABSOLUTE_TARGET_COL: [10.0, 12.0, 14.0],
                "sale_amount_lag_7": [8.0, 10.0, 11.0],
            }
        )
        target_contract = resolve_target_contract(frame, frame)
        enriched = ensure_learning_target_column(frame, target_contract)

        reconstructed = reconstruct_absolute_predictions(
            enriched[DEFAULT_ABSOLUTE_TARGET_COL].to_numpy(dtype=float),
            enriched,
            target_contract,
        )

        np.testing.assert_allclose(
            reconstructed,
            frame[DEFAULT_ABSOLUTE_TARGET_COL].to_numpy(dtype=float),
        )

    def test_resolve_target_contract_never_applies_implicit_log_transform(self) -> None:
        train_frame = pd.DataFrame({DEFAULT_ABSOLUTE_TARGET_COL: [1.0, 2.0, 3.0]})
        tuning_frame = pd.DataFrame({DEFAULT_ABSOLUTE_TARGET_COL: [5.0, 6.0, 7.0]})

        target_contract = resolve_target_contract(train_frame, tuning_frame)

        self.assertEqual(target_contract.target_mode, "identity")

    def test_resolve_target_contract_supports_field_baseline_residual(self) -> None:
        frame = pd.DataFrame(
            {
                DEFAULT_ABSOLUTE_TARGET_COL: [10.0, 12.0, 14.0],
                DEFAULT_FIELD_BASELINE_COL: [8.0, 9.0, 10.0],
                DEFAULT_FIELD_BASELINE_RESIDUAL_TARGET_COL: [2.0, 3.0, 4.0],
            }
        )

        target_contract = resolve_target_contract(
            frame,
            frame,
            requested_target_col=DEFAULT_FIELD_BASELINE_RESIDUAL_TARGET_COL,
        )

        self.assertEqual(
            target_contract.learning_target_col,
            DEFAULT_FIELD_BASELINE_RESIDUAL_TARGET_COL,
        )
        self.assertEqual(target_contract.target_mode, "additive_residual")
        self.assertEqual(
            target_contract.reconstruction_anchor_col, DEFAULT_FIELD_BASELINE_COL
        )
        reconstructed = reconstruct_absolute_predictions(
            np.array([2.0, 3.0, 4.0]),
            frame,
            target_contract,
        )
        np.testing.assert_allclose(reconstructed, [10.0, 12.0, 14.0])

    def test_resolve_target_contract_rejects_unavailable_requested_target(self) -> None:
        frame = pd.DataFrame({DEFAULT_ABSOLUTE_TARGET_COL: [1.0, 2.0]})

        with self.assertRaisesRegex(ValueError, "Requested learning target"):
            resolve_target_contract(
                frame,
                frame,
                requested_target_col=DEFAULT_FIELD_BASELINE_RESIDUAL_TARGET_COL,
            )


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
from typing import cast
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.utils.memory import downcast_pandas_frame  # noqa: E402


class MemoryUtilsTests(unittest.TestCase):
    def test_downcast_pandas_frame_preserves_nullable_boolean_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "nullable_flag": pd.Series([True, False, None], dtype="boolean"),
                "float_col": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            }
        )

        compact = downcast_pandas_frame(frame)

        self.assertEqual(str(compact["nullable_flag"].dtype), "boolean")
        self.assertTrue(pd.isna(cast(bool | None, compact.loc[2, "nullable_flag"])))
        self.assertIn(str(compact["float_col"].dtype), {"float32", "Float32"})


if __name__ == "__main__":
    unittest.main()

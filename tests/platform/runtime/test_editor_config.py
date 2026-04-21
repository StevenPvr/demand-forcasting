from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class EditorConfigTests(unittest.TestCase):
    def test_pyright_extra_paths_only_reference_src_roots(self) -> None:
        config_path = PROJECT_ROOT / "pyrightconfig.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["extraPaths"],
            [
                "platform/python/src",
                "products/demand_forecast/src",
            ],
        )

    def test_vscode_python_analysis_paths_only_reference_src_roots(self) -> None:
        settings_path = PROJECT_ROOT / ".vscode" / "settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8"))

        self.assertEqual(
            settings["python.analysis.extraPaths"],
            [
                "${workspaceFolder}/platform/python/src",
                "${workspaceFolder}/products/demand_forecast/src",
            ],
        )
        self.assertEqual(
            settings["python.autoComplete.extraPaths"],
            [
                "${workspaceFolder}/platform/python/src",
                "${workspaceFolder}/products/demand_forecast/src",
            ],
        )


if __name__ == "__main__":
    unittest.main()

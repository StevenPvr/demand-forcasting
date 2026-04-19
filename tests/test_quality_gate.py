from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_GATE_PATH = PROJECT_ROOT / "praedixa-ecc" / "scripts" / "quality_gate.py"


def load_quality_gate_module():
    spec = spec_from_file_location("praedixa_quality_gate", QUALITY_GATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load quality_gate module.")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QualityGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quality_gate = load_quality_gate_module()

    def test_collect_test_targets_maps_source_to_existing_test(self) -> None:
        source_path = PROJECT_ROOT / "src" / "research_praedixa" / "feature_engineering" / "pipeline.py"
        targets = self.quality_gate.collect_test_targets([source_path])
        relative_targets = [path.relative_to(PROJECT_ROOT) for path in targets]

        self.assertIn(Path("tests/test_feature_engineering_pipeline.py"), relative_targets)

    def test_collect_test_targets_keeps_test_file_itself(self) -> None:
        test_path = PROJECT_ROOT / "tests" / "test_python_doctor.py"
        targets = self.quality_gate.collect_test_targets([test_path])

        self.assertEqual(targets, [test_path])

    def test_module_name_from_test_path_returns_unittest_module_name(self) -> None:
        test_path = PROJECT_ROOT / "tests" / "test_quality_gate.py"
        module_name = self.quality_gate.module_name_from_test_path(test_path)

        self.assertEqual(module_name, "tests.test_quality_gate")

    def test_gate_python_files_uses_explicit_paths(self) -> None:
        source_path = PROJECT_ROOT / "praedixa-ecc" / "scripts" / "quality_gate.py"
        files = self.quality_gate.gate_python_files([str(source_path)], repo_scope=False)

        self.assertEqual(files, [source_path.resolve()])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DOCTOR_PATH = PROJECT_ROOT / "praedixa-ecc" / "scripts" / "python_doctor.py"


def load_python_doctor_module():
    spec = spec_from_file_location("praedixa_python_doctor", PYTHON_DOCTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load python_doctor module.")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PythonDoctorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.python_doctor = load_python_doctor_module()

    def test_scan_source_file_reports_core_violations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "src" / "research_praedixa" / "sample.py"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(
                "\n".join(
                    [
                        "def build_frame(frame):",
                        "    print('debug')",
                        "    try:",
                        "        return frame",
                        "    except:",
                        "        return None",
                    ]
                ),
                encoding="utf-8",
            )

            findings = self.python_doctor.scan_python_file(source_path)
            codes = {finding.code for finding in findings}

            self.assertIn("future-annotations", codes)
            self.assertIn("missing-annotations", codes)
            self.assertIn("print-in-prod", codes)
            self.assertIn("bare-except", codes)

    def test_scan_test_file_skips_future_annotations_and_print_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "tests" / "test_sample.py"
            test_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.write_text(
                "\n".join(
                    [
                        "def test_it():",
                        "    print('debug')",
                        "    assert True",
                    ]
                ),
                encoding="utf-8",
            )

            findings = self.python_doctor.scan_python_file(test_path)
            codes = {finding.code for finding in findings}

            self.assertNotIn("future-annotations", codes)
            self.assertNotIn("print-in-prod", codes)

    def test_expected_test_candidates_match_repo_pattern(self) -> None:
        source_path = PROJECT_ROOT / "src" / "research_praedixa" / "feature_engineering" / "pipeline.py"
        candidates = self.python_doctor.expected_test_candidates(source_path)
        relative_candidates = [path.relative_to(PROJECT_ROOT) for path in candidates]

        self.assertIn(Path("tests/test_feature_engineering_pipeline.py"), relative_candidates)

    def test_iter_python_files_expands_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            first_file = root / "a.py"
            nested_file = root / "nested" / "b.py"
            nested_file.parent.mkdir(parents=True, exist_ok=True)
            first_file.write_text("from __future__ import annotations\n", encoding="utf-8")
            nested_file.write_text("from __future__ import annotations\n", encoding="utf-8")

            files = self.python_doctor.iter_python_files([str(root)], repo_scope=False)
            relative_files = {path.relative_to(root) for path in files}

            self.assertEqual(relative_files, {Path("a.py"), Path("nested/b.py")})


if __name__ == "__main__":
    unittest.main()

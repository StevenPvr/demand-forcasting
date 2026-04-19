from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_ROOT = REPO_ROOT / "praedixa-ecc"
CODEX_TEMPLATE = FRAMEWORK_ROOT / "codex-template"
SKILLS_SOURCE = FRAMEWORK_ROOT / "skills"
CODEX_TARGET = REPO_ROOT / ".codex"
SKILLS_TARGET = REPO_ROOT / ".agents" / "skills"


def copy_tree_contents(source: Path, target: Path) -> None:
    for source_path in sorted(source.rglob("*")):
        relative_path = source_path.relative_to(source)
        target_path = target / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def main() -> None:
    copy_tree_contents(CODEX_TEMPLATE, CODEX_TARGET)
    copy_tree_contents(SKILLS_SOURCE, SKILLS_TARGET)
    print(f"Synced Codex surface into {CODEX_TARGET}")
    print(f"Synced project skills into {SKILLS_TARGET}")


if __name__ == "__main__":
    main()

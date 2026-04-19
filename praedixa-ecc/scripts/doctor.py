from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_ROOT = REPO_ROOT / "praedixa-ecc"
SKILLS_SOURCE = FRAMEWORK_ROOT / "skills"
ACTIVE_CODEX = REPO_ROOT / ".codex"
ACTIVE_SKILLS = REPO_ROOT / ".agents" / "skills"


def skill_names() -> list[str]:
    return sorted(path.name for path in SKILLS_SOURCE.iterdir() if path.is_dir())


def main() -> None:
    missing: list[str] = []
    if not FRAMEWORK_ROOT.exists():
        missing.append("praedixa-ecc/")
    if not ACTIVE_CODEX.exists():
        missing.append(".codex/")
    if not ACTIVE_SKILLS.exists():
        missing.append(".agents/skills/")

    for skill_name in skill_names():
        skill_path = ACTIVE_SKILLS / skill_name
        if not skill_path.exists():
            missing.append(str(skill_path.relative_to(REPO_ROOT)))

    if missing:
        print("Doctor status: FAIL")
        for item in missing:
            print(f"- missing: {item}")
        raise SystemExit(1)

    print("Doctor status: OK")
    print(f"- framework root: {FRAMEWORK_ROOT}")
    print(f"- active codex surface: {ACTIVE_CODEX}")
    print(f"- active skills: {ACTIVE_SKILLS}")
    print(f"- managed skills: {len(skill_names())}")


if __name__ == "__main__":
    main()

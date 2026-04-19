from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_ROOT = REPO_ROOT / "praedixa-ecc"
SKILLS_ROOT = FRAMEWORK_ROOT / "skills"
RULES_ROOT = FRAMEWORK_ROOT / "rules"
TEMPLATES_ROOT = FRAMEWORK_ROOT / "templates"
CODEX_TEMPLATE = FRAMEWORK_ROOT / "codex-template"


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_skill(skill_dir: Path, errors: list[str]) -> None:
    skill_md = skill_dir / "SKILL.md"
    openai_yaml = skill_dir / "agents" / "openai.yaml"
    require(skill_md.exists(), f"{skill_dir.name}: missing SKILL.md", errors)
    require(openai_yaml.exists(), f"{skill_dir.name}: missing agents/openai.yaml", errors)
    if skill_md.exists():
        text = skill_md.read_text(encoding="utf-8")
        require(text.startswith("---\n"), f"{skill_dir.name}: missing YAML frontmatter", errors)
        require("\nname:" in text, f"{skill_dir.name}: missing frontmatter name", errors)
        require("\ndescription:" in text, f"{skill_dir.name}: missing frontmatter description", errors)
    if openai_yaml.exists():
        text = openai_yaml.read_text(encoding="utf-8")
        require("display_name:" in text, f"{skill_dir.name}: missing display_name", errors)
        require("short_description:" in text, f"{skill_dir.name}: missing short_description", errors)


def main() -> None:
    errors: list[str] = []

    require(CODEX_TEMPLATE.joinpath("AGENTS.md").exists(), "codex-template/AGENTS.md missing", errors)
    require(CODEX_TEMPLATE.joinpath("config.toml").exists(), "codex-template/config.toml missing", errors)
    require(CODEX_TEMPLATE.joinpath("agents").exists(), "codex-template/agents missing", errors)
    require(FRAMEWORK_ROOT.joinpath("scripts", "python_doctor.py").exists(), "scripts/python_doctor.py missing", errors)
    require(FRAMEWORK_ROOT.joinpath("scripts", "quality_gate.py").exists(), "scripts/quality_gate.py missing", errors)

    for required_rule in [
        RULES_ROOT / "common" / "foundation.md",
        RULES_ROOT / "python" / "coding-style.md",
        RULES_ROOT / "python" / "patterns.md",
        RULES_ROOT / "python" / "testing.md",
        RULES_ROOT / "ml" / "experiment-integrity.md",
        RULES_ROOT / "timeseries" / "data-contracts.md",
        RULES_ROOT / "timeseries" / "leakage.md",
        RULES_ROOT / "timeseries" / "backtesting.md",
        RULES_ROOT / "timeseries" / "metrics.md",
        RULES_ROOT / "praedixa" / "business-framing.md",
        RULES_ROOT / "praedixa" / "retail-perishable-demand.md",
    ]:
        require(required_rule.exists(), f"missing rule: {required_rule.relative_to(FRAMEWORK_ROOT)}", errors)

    for required_template in [
        TEMPLATES_ROOT / "dataset-manifest.md",
        TEMPLATES_ROOT / "experiment-spec.md",
        TEMPLATES_ROOT / "backtest-report.md",
        TEMPLATES_ROOT / "business-translation-memo.md",
        TEMPLATES_ROOT / "decision-log.md",
    ]:
        require(required_template.exists(), f"missing template: {required_template.relative_to(FRAMEWORK_ROOT)}", errors)

    for skill_dir in sorted(path for path in SKILLS_ROOT.iterdir() if path.is_dir()):
        validate_skill(skill_dir, errors)

    if errors:
        print("Framework validation: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Framework validation: OK")
    print(f"- skills: {len([path for path in SKILLS_ROOT.iterdir() if path.is_dir()])}")
    print(f"- rules root: {RULES_ROOT}")
    print(f"- templates root: {TEMPLATES_ROOT}")


if __name__ == "__main__":
    main()

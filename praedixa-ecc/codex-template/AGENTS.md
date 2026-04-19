# Praedixa ECC for Codex

Ce fichier complete le `AGENTS.md` racine.
Il sert surtout de routeur vers les bons fichiers du framework selon la situation.

## Mission

Utiliser Praedixa ECC comme cadre par defaut pour le travail de data science, machine learning et series temporelles dans ce repo.

## Boucles de travail

Identifier explicitement la boucle dominante :

- `forecasting core loop`
- `ml research loop`
- `Praedixa operating loop`

## Fichiers a utiliser selon la situation

- qualite du code Python :
  - `praedixa-ecc/skills/python-code-quality/SKILL.md`
- contrats de donnees time series :
  - `praedixa-ecc/skills/timeseries-dataset-contracts/SKILL.md`
- feature engineering time series :
  - `praedixa-ecc/skills/timeseries-feature-engineering/SKILL.md`
- leakage / disponibilite de l'information :
  - `praedixa-ecc/skills/anti-leakage-audit/SKILL.md`
- backtesting / walk-forward / splits temporels :
  - `praedixa-ecc/skills/temporal-backtesting/SKILL.md`
- baselines :
  - `praedixa-ecc/skills/forecast-baselines/SKILL.md`
- evaluation / metriques / interpretation business :
  - `praedixa-ecc/skills/forecast-evaluation/SKILL.md`
  - `praedixa-ecc/skills/praedixa-roi-translation/SKILL.md`
- experimentation ML :
  - `praedixa-ecc/skills/ml-experiment-loop/SKILL.md`
- wedge demande Praedixa :
  - `praedixa-ecc/skills/praedixa-demand-forecast/SKILL.md`
- wedge staffing Praedixa :
  - `praedixa-ecc/skills/praedixa-staffing-forecast/SKILL.md`
- signaux internes / externes :
  - `praedixa-ecc/skills/praedixa-retail-signals/SKILL.md`

## Verification

Pour une modification Python non triviale :

```bash
.venv/bin/python praedixa-ecc/scripts/quality_gate.py <python-paths...>
```

Pour verifier la structure du framework :

```bash
.venv/bin/python praedixa-ecc/scripts/validate_framework.py
.venv/bin/python praedixa-ecc/scripts/doctor.py
```

## Agents

Les roles Codex vivent dans `.codex/agents/`.

- `forecast-reviewer` : revue critique leakage / baselines / metriques / operationalite
- `experiment-auditor` : rigueur experimentale et qualite de preuve
- `docs-researcher` : verification documentaire et API

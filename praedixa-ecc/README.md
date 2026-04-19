# Praedixa ECC

Framework Codex-only interne pour les workflows data science, machine learning et series temporelles de Praedixa.

Le framework est structure en deux couches :

- un noyau generique pour l'integrite ML / forecasting
- une surcouche Praedixa pour la demande, les effectifs et la lecture ROI

## Structure

- `codex-template/` : surface `.codex/` a projeter dans le repo actif
- `skills/` : source de verite des skills projet-locales
- `rules/` : garde-fous de reference pour Codex
- `templates/` : artefacts standardises pour les experiments et livrables
- `scripts/` : installation, sync, doctor et validation
- `docs/` : architecture et conventions du framework

## Boucles cibles

- `forecasting core loop`
- `ml research loop`
- `Praedixa operating loop`

## Commandes utiles

```bash
.venv/bin/python praedixa-ecc/scripts/validate_framework.py
.venv/bin/python praedixa-ecc/scripts/sync_codex_surface.py
.venv/bin/python praedixa-ecc/scripts/doctor.py
.venv/bin/python praedixa-ecc/scripts/python_doctor.py src/research_praedixa/feature_engineering/pipeline.py
.venv/bin/python praedixa-ecc/scripts/quality_gate.py src/research_praedixa/feature_engineering/pipeline.py
```

## Surface active

Le framework prend vie dans le repo via :

- `.codex/`
- `.agents/skills/`

La source de verite reste `praedixa-ecc/`.

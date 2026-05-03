# Invariants métier (validation post-génération)

## Liste codée

Les règles exploitables par une suite de tests sont définies dans :

[`validator_rules.py`](../../../platform/python/src/praedixa/platform/datasets/synthetic_foodservice/spec/validator_rules.py) (`VALIDATION_INVARIANTS`)

Chaque entrée comporte :

- `id` — identifiant stable (ex. `INV-04-price-window`)
- `description` — intention fonctionnelle
- `severity` — `error` (bloquant) ou `warning` (contrôle qualité)
- `applies_to_layers` — couches où la règle **doit** passer (`internal_clean`, `ops_realistic`) ; la couche `pos_synth_export` est excluse sauf tests « soft » sur sous-ensemble corrigé

## Tableau récapitulatif

| ID | Objet |
|----|--------|
| INV-01 | Fenêtre de vie produit vs ventes |
| INV-02 | Somme des lignes vs total ticket |
| INV-03 | Cohérence fermetures |
| INV-04 | Prix effectifs vs table `prices` |
| INV-05 | Promotions actives vs timestamp |
| INV-06 | Blocage vente sous rupture sans substitut |
| INV-07 | Conservation de stock théorique |
| INV-08 | Calendrier fériés / zone |
| INV-09 | Saisonnalité produit |
| INV-10 | Ventes observées ≤ demande latente (sans substitution) |

## Usage

```python
from praedixa.platform.datasets.synthetic_foodservice.spec.validator_rules import VALIDATION_INVARIANTS

assert any(inv.id == "INV-02-line-sum-vs-ticket-total" for inv in VALIDATION_INVARIANTS)
```

Les implémentations concrètes de checks (SQL, polars, pandera) vivent dans un module futur `validation_runner.py` ; ce dépôt ne les impose pas encore.

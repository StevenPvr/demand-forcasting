# TODO - Optimisation Economique Asymetrique Finalisee

Contexte: sur la run XGBoost bakery reference du 2026-04-30, le biais global est
positif (`+2.185426104083398`). En operation alimentaire perissable, une
sur-prevision produit des invendus/gaspillage. Une sous-prevision legere peut
etre moins couteuse si le client substitue vers un autre produit.

Objectif: ne plus optimiser seulement l'erreur moyenne. Integrer une fonction
de cout decisionnelle qui penalise davantage la sur-production que la
sous-production legere, tout en empechant les ruptures fortes.

## Principe Produit

- Biais positif: a penaliser fortement, car il pousse a jeter.
- Biais legerement negatif: acceptable si substitution client probable.
- Biais trop negatif: a penaliser, car il cree perte de chiffre d'affaires,
  frustration client et baisse de niveau de service.

La cible n'est donc pas "forcer le biais negatif", mais:

```text
biais cible legerement <= 0
avec forte penalite sur sur-prevision
et garde-fou contre sous-prevision excessive
```

## Etat D'Implementation

### 1. Formaliser La Loss Economique

- [x] Centraliser les couts dans une config explicite.
- [x] Definir au minimum:
  - cout de sur-production par unite;
  - cout de sous-production legere par unite;
  - cout de sous-production forte par unite;
  - seuil de sous-production legere;
  - biais cible global ou par famille produit.
- [x] Documenter que ces couts sont des hypotheses produit tant qu'ils ne sont
  pas calibres client.

Forme possible:

```text
economic_loss =
    over_forecast_units * waste_cost
  + mild_under_forecast_units * substitution_adjusted_stockout_cost
  + severe_under_forecast_units * lost_sales_cost
```

### 2. Ajouter Un Score HPO Oriente Decision

- [x] Ne plus choisir le meilleur trial uniquement sur MAE/WAPE.
- [x] Ajouter un score validation composite:

```text
hpo_score =
    economic_loss
  + positive_bias_penalty
  + severe_negative_bias_penalty
  + wape_guardrail_penalty
```

- [x] Garder MAE/WAPE en metriques de controle, pas comme unique objectif.
- [x] Logger separement:
  - `validation_economic_loss`;
  - `validation_bias`;
  - `validation_positive_bias_penalty`;
  - `validation_severe_negative_bias_penalty`;
  - `validation_wape`.

### 3. Penaliser Explicitement Le Biais Positif

- [x] Ajouter une penalite nulle ou faible si le biais est dans une zone cible
  legerement negative.
- [x] Ajouter une penalite forte si le biais est positif.
- [x] Ajouter une penalite progressive si le biais devient trop negatif.

Exemple conceptuel:

```text
if bias > 0:
    penalty = bias * positive_bias_weight
elif bias < max_allowed_negative_bias:
    penalty = abs(bias - max_allowed_negative_bias) * severe_stockout_weight
else:
    penalty = 0
```

### 4. Calibrer Les Predictions Apres Modele

- [x] Ajouter une calibration post-XGBoost optionnelle orientee loss
  economique.
- [x] Tester des offsets multiplicatifs/additifs sur validation:
  - `prediction * factor`;
  - `prediction + intercept`;
  - combinaison slope/intercept.
- [x] Choisir la calibration qui minimise la loss economique sous garde-fous
  WAPE/biais.
- [x] Sauvegarder les parametres de calibration dans les artefacts.

### 5. Segmenter Par Produit Ou Famille

- [x] Ne pas appliquer le meme compromis a tous les produits.
- [x] Introduire une segmentation:
  - produit d'appel;
  - forte marge;
  - substituable;
  - peu substituable;
  - fort risque gaspillage.
- [x] Commencer par famille produit si le volume par produit est insuffisant.
- [x] Autoriser des couts differents par segment.

### 6. Adapter Les Artefacts

- [x] Ajouter aux metrics JSON:
  - loss economique validation/test;
  - biais global;
  - biais par produit;
  - sur-production totale;
  - sous-production totale;
  - cout sur-production;
  - cout sous-production;
  - version de la config economique.
- [x] Ajouter une section dans la model card:
  - objectif d'optimisation;
  - couts utilises;
  - interpretation du biais;
  - limites business.

### 7. Tests Anti-Regression

- [x] Test unitaire: biais positif augmente le score plus que biais legerement
  negatif a erreur absolue comparable.
- [x] Test unitaire: biais trop negatif est penalise.
- [x] Test unitaire: la loss economique est stable et deterministe.
- [x] Test integration: HPO choisit le trial a meilleure loss economique quand
  la MAE seule favoriserait un autre trial.
- [x] Test artefact: la config economique et les composantes de cout sont
  serialisees.

## Critere De Promotion

Un run ne doit etre promu que si:

- la loss economique test bat la baseline;
- le biais positif est reduit ou justifie par segment;
- la degradation MAE/WAPE reste dans un garde-fou accepte;
- les gains restent robustes sous au moins une analyse de sensibilite des couts.

## Point De Vigilance

Ne pas presenter un biais negatif comme toujours desirable. La bonne promesse
Praedixa est une decision economiquement meilleure sous contraintes, pas une
sous-production systematique.

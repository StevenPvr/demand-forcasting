# Audit qualite des donnees synthetic foodservice

Date: 2026-04-26

## Cadrage retenu

Objectif court terme: verifier si un modele pre-entraine sur des verticales
foodservice peut generaliser vers `bakery` sans etre entraine sur `bakery`.

La cible du pipeline est donc volontairement simple et homogene:

- predire les ventes observees;
- garder les lignes de vente a 0 quand elles correspondent a ce qui est observe
  dans le POS;
- conserver `censor_flag` / `observed_stockout_flag` comme features ou metadata
  descriptives;
- exclure uniquement les observations vraiment absentes, fermees, incompletes ou
  low-quality.

La demande latente, la reconstruction de ventes perdues et le modele binaire de
rupture de stock sont des sujets separes. Ils ne doivent pas piloter ce protocole
de pretraining / zero-shot bakery.

## Verdict

Le pipeline est maintenant aligne avec le test a mener: les lignes marquees
stockout/contrainte ne sont plus supprimees par defaut du training tant que le
label de vente observee est complet.

Le bon contrat est:

```text
target_semantics = observed_sales
target_source = observed_sales
censor_flag = signal descriptif
observed_stockout_flag = signal descriptif
usable_for_training_flag = observation POS complete et cible exploitable
```

Cela evite de construire des datasets non homogenes entre M5, FreshRetail et
synthetic foodservice. M5 ne porte pas les memes signaux de stock; le training
principal doit donc apprendre une cible commune: la vente observee.

## Changement de politique

Avant:

- une ligne `censor_flag=true` pouvait etre degradee a `label_quality_score=0.5`;
- elle pouvait devenir `usable_for_training_flag=false`;
- le gold et les requetes d'echantillonnage pouvaient filtrer `censor_flag`.

Maintenant:

- `censor_flag` n'est plus un critere d'exclusion training;
- `label_quality_score` mesure la completude du label observe, pas l'existence
  d'une contrainte operationnelle;
- `usable_for_training_flag` depend de la qualite d'observation, des fermetures
  et des sources explicitement non entrainables;
- les ruptures restent disponibles pour analyses, features ou futur modele
  binaire.

## Points de controle qualite

### 1. Homogeneite de cible

Le pretraining doit melanger uniquement des lignes comparables:

- ventes observees quotidiennes;
- split temporel strict;
- pas de fuite oracle;
- pas de donnees bakery dans le train si le test vise le zero-shot bakery.

### 2. FreshRetail

A verifier avant une evaluation definitive:

- confirmer que les lignes de vente a 0 supprimees ou marquees non utilisables
  sont seulement des observations incompletes, fermees ou non fiables;
- ne pas retirer une ligne simplement parce qu'elle correspond a une rupture ou
  a une contrainte si le POS observe bien 0 vente ce jour-la;
- garder le signal stockout pour diagnostic ou modele separe.

### 3. Synthetic foodservice

Le generateur doit rester utile pour tester la robustesse, mais sans changer la
cible:

- les colonnes oracle restent hors export model-facing;
- `lost_sales_qty_debug` reste dans le sidecar debug;
- le modele principal ne voit que les ventes observees et les signaux autorises;
- les jours fermes ou incomplets sont non entrainables.

### 4. Evaluation zero-shot bakery

Le protocole attendu:

1. Entrainer le modele pretraining sans `synthetic_foodservice_bakery`.
2. Evaluer sur bakery uniquement en test out-of-sample.
3. Comparer a un baseline saisonnier simple et a un modele non pre-entraine.
4. Lire WAPE, MAE, bias et erreurs par horizon.
5. Verifier que le gain ne vient pas d'une fuite de vertical, de produit ou de
   date.

## Decision

Pour le run actuel, on ne cherche pas a estimer la demande latente. On cherche a
savoir si le pretraining transfere vers la boulangerie sur une cible de ventes
observees. Les flags de stockout/contrainte restent utiles, mais ils n'excluent
plus le dataset principal.

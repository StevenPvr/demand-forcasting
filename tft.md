# TFT pour Praedixa

Document de référence pour concevoir, implémenter, entraîner, évaluer et servir un backend **Temporal Fusion Transformer (TFT)** sérieux pour Praedixa.

Ce document est écrit pour le repo actuel :

- backend TFT encore **placeholder**
- pipeline `bronze -> silver -> gold` déjà en place
- orchestration `training` / `evaluation` déjà structurée
- cible Praedixa prioritaire : **prévision exploitable de la demande et des besoins opérationnels**

Il ne faut pas lire ce document comme une célébration abstraite du TFT. Il faut le lire comme un **plan d’architecture concret** pour brancher un vrai backend deep forecasting dans ce repo, sans dériver hors du wedge actuel.

---

## 1. Résumé exécutif

Pour Praedixa, un TFT "state-of-the-art" ne veut pas dire :

- prendre l’architecture la plus grosse possible
- ajouter des couches sans discipline
- se raconter que le deep learning compense une cible bancale

Pour Praedixa, un TFT state-of-the-art veut dire :

- une **sémantique de cible irréprochable**
- une **taxonomie stricte des covariables** connue / inconnue / statique
- un **dataset contract** stable entre train / validation / test / inference
- des **prévisions probabilistes** utilisables en décision
- une **évaluation par horizon, segment et coût business**
- une **reproductibilité forte** sur les runs de référence
- un **bundle d’artefacts** fiable pour le serving
- un **fallback baseline** quand le contrat de features casse

Le TFT est un bon candidat pour Praedixa parce qu’il est précisément conçu pour :

- le multi-horizon
- le multi-séries
- les covariables hétérogènes
- les signaux futurs connus
- l’interprétabilité opérationnelle

Mais il n’est crédible que si :

- on reste discipliné sur la cible
- on ne fuit jamais temporellement
- on ne confond jamais interprétabilité et causalité
- on l’évalue contre des baselines simples mais robustes

---

## 2. Position Praedixa

Le wedge Praedixa reste :

- prévision de la demande
- prévision des besoins en effectifs

Le TFT doit donc être au service de décisions très concrètes :

- produire plus juste
- acheter plus juste
- réduire le gaspillage
- réduire les ruptures
- mieux ajuster les effectifs
- améliorer la marge

Le backend TFT n’est pas la vision produit entière. C’est un **moteur prédictif probabiliste** qui doit plus tard nourrir une couche d’aide à la décision.

Autrement dit :

- aujourd’hui : mieux anticiper
- demain : mieux arbitrer

---

## 3. Sources de référence utilisées

Ce document s’appuie sur des sources primaires et à jour consultées le **20 avril 2026** :

- papier TFT publié dans *International Journal of Forecasting* via Google Research
- source officielle `sktime/pytorch-forecasting`
- documentation officielle `pytorch-forecasting`
- documentation officielle Lightning
- documentation officielle PyTorch
- métadonnées PyPI pour la compatibilité de versions

Principales URLs :

- `https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/`
- `https://github.com/sktime/pytorch-forecasting/blob/main/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py`
- `https://github.com/sktime/pytorch-forecasting/blob/main/pytorch_forecasting/models/temporal_fusion_transformer/sub_modules.py`
- `https://github.com/sktime/pytorch-forecasting/blob/main/pytorch_forecasting/data/timeseries/_timeseries.py`
- `https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.data.timeseries._timeseries.TimeSeriesDataSet.html`
- `https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.encoders.GroupNormalizer.html`
- `https://pytorch-forecasting.readthedocs.io/en/v1.4.0/api/pytorch_forecasting.metrics.quantile.QuantileLoss.html`
- `https://lightning.ai/docs/pytorch/latest/common/trainer.html`
- `https://docs.pytorch.org/docs/stable/notes/randomness`
- `https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html`
- `https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html`
- `https://docs.pytorch.org/docs/stable/generated/torch.compile.html`
- `https://pypi.org/project/pytorch-forecasting/`
- `https://pypi.org/project/lightning/`
- `https://pypi.org/project/torch/`

---

## 4. Ce que "state-of-the-art" veut dire ici

Il faut distinguer deux choses :

### 4.1 State-of-the-art absolu en forecasting

Sur certains benchmarks, d’autres familles de modèles peuvent battre TFT selon :

- l’horizon
- la densité des covariables
- la structure de panel
- la taille des datasets

Donc TFT n’est pas "meilleur partout, toujours".

### 4.2 State-of-the-art de mise en œuvre TFT pour Praedixa

Ici, l’objectif est :

- un **TFT implémenté proprement**
- compatible avec le besoin Praedixa de covariables riches
- probabiliste
- audit-able
- industrialisable

Pour Praedixa, cela signifie :

1. utiliser une stack moderne et compatible `Python 3.13`
2. fiabiliser `TimeSeriesDataSet`
3. définir un vrai contrat de features et de targets
4. utiliser Lightning pour un training loop propre
5. produire quantiles + artefacts + diagnostics
6. ne promouvoir que si le gain business est réel

---

## 5. Compatibilité de stack recommandée

Au 20 avril 2026, les versions observées côté écosystème sont :

- `torch` : `2.11.0`
- `lightning` : `2.6.1`
- `pytorch-forecasting` : `1.7.0`
- `Python` : supporté jusqu’à `3.13` dans les packages observés, et `pytorch-forecasting` annonce `>=3.10,<3.15`

### 5.1 Recommandation pour ce repo

Pour une première intégration robuste :

- `python == 3.13`
- `torch ~= 2.11`
- `lightning ~= 2.6`
- `pytorch-forecasting ~= 1.7`
- `torchmetrics` via dépendances Lightning / Forecasting

### 5.2 Pourquoi cette recommandation

- cohérence avec `.python-version`
- compatibilité macOS arm64 et Linux moderne
- documentation récente
- présence d’API stables pour `Trainer`, `inference_mode`, `precision`, `deterministic`
- support Python 3.13 confirmé sur PyPI pour `torch` et `pytorch-forecasting`

### 5.3 Ce qu’il faut éviter

- mélanger une vieille version `pytorch-forecasting` avec un `torch` très récent sans matrice testée
- faire du "latest latest latest" sans lockfile
- coder un backend from scratch avant d’avoir une baseline framework-based

Règle recommandée :

- commencer par **brancher un backend basé sur `pytorch-forecasting`**
- stabiliser les contrats
- ensuite seulement, décider si un `from scratch` apporte un vrai gain

---

## 6. État actuel du repo

Le repo est déjà structuré comme si un backend modèle unique devait exister.

### 6.1 Ce qui existe déjà

Sous `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/` :

- `backend.py`
- `model_utils.py`

Aujourd’hui :

- `get_tft_backend_availability()` annonce explicitement que le backend n’est pas prêt
- `fit_tft_model()`, `predict_with_tft_model()` et `save_tft_model()` lèvent volontairement une exception

### 6.2 Ce que le repo attend implicitement

Les pipelines `training` et `evaluation` sont déjà écrits comme si ces fonctions existaient réellement.

Exemples :

- `training/tuning.py` appelle `fit_tft_model()` et `predict_with_tft_model()`
- `evaluation/modeling.py` appelle `fit_tft_model()`, `predict_with_tft_model()` et `save_tft_model()`
- `training/orchestrator.py` et `evaluation/orchestrator.py` construisent déjà les artefacts métier

### 6.3 Conclusion pratique

Le sujet n’est pas "inventer l’architecture globale".

Le sujet est :

- brancher un vrai backend TFT
- **sans casser** l’architecture existante
- en respectant :
  - `TargetContract`
  - les dossiers d’artefacts
  - les invariants train / tuning / evaluation du repo

---

## 7. Contrat cible Praedixa

Le repo contient déjà une abstraction très utile : `TargetContract`.

Elle encode :

- `learning_target_col`
- `absolute_target_col`
- `target_mode`
- `reconstruction_anchor_col`

### 7.1 Modes déjà prévus par le code

#### Mode `delta_log_wow`

Le repo sait déjà travailler avec :

- cible d’apprentissage = delta log contre une ancre `lag_7`
- reconstruction des prédictions absolues via :
  - ancre
  - transformation inverse

Ce mode est intéressant pour Praedixa car :

- il absorbe une partie de la saisonnalité hebdomadaire
- il peut rendre l’apprentissage plus stationnaire
- il se rapproche parfois de ce qu’on veut réellement piloter : déviation attendue contre base hebdomadaire

Mais il ajoute un contrat fort :

- l’ancre doit exister en entraînement **et** en serving
- la reconstruction absolue doit être parfaitement testée

#### Mode `log1p`

Si la cible est non négative, le repo retombe naturellement vers un mode log.

Avantage :

- très classique
- stable
- simple à auditer

#### Mode `identity`

À garder pour :

- cible pouvant être négative
- phase de debug
- benchmark simple

### 7.2 Recommandation stratégique

Pour Praedixa, il faut supporter **deux backends cibles** dès la V1 :

1. **backend absolu**
   - apprentissage direct sur `target_demand_qty_d_plus_1`
   - normalisation de groupe
   - quantiles absolus

2. **backend delta-log-wow**
   - apprentissage sur delta versus `lag_7`
   - reconstruction absolue
   - promotion seulement si gain réel

La recommandation de départ est :

- benchmarker d’abord le mode absolu
- garder le mode WoW comme challenger sérieux

Pourquoi :

- le mode absolu est plus simple à expliquer
- le mode WoW est plus subtil mais potentiellement plus robuste sur des patterns hebdomadaires forts

---

## 8. Quand TFT est un bon choix pour Praedixa

Le TFT est particulièrement pertinent si les conditions suivantes sont vraies :

- beaucoup de séries `site x produit`
- forte hétérogénéité d’amplitude
- présence de signaux futurs connus
- besoin de probabilisme
- besoin de diagnostics d’importance relative des variables

Dans Praedixa, cela correspond bien à :

- météo prévue
- calendrier
- promotions planifiées
- prix planifiés
- fermetures planifiées
- variables structurelles de site et produit

### 8.1 Quand il ne faut pas forcer TFT

Si un des cas suivants domine, il faut être sceptique :

- très peu de séries
- panel trop court
- quasi aucune covariable utile
- cible mal définie
- pipeline de données pas stabilisé

Dans ces cas :

- un baseline saisonnier
- un gradient boosting tabulaire
- un modèle plus simple type N-HiTS ou LGBM par horizon

peuvent être meilleurs, moins chers et plus lisibles.

---

## 9. Architecture conceptuelle du TFT

Le papier TFT pose une idée simple :

- combiner **traitement local** et **dépendances longues**
- traiter proprement des covariables de natures différentes
- garder une forme d’interprétabilité

Le cœur logique est :

1. représenter les variables
2. sélectionner les plus pertinentes
3. conditionner la dynamique par le contexte statique
4. modéliser la dynamique locale via LSTM
5. lire des dépendances plus longues via attention
6. produire une sortie probabiliste

La force du TFT n’est donc pas "l’attention" seule.

La force du TFT est la **combinaison disciplinée** :

- embeddings
- Variable Selection Networks
- Gated Residual Networks
- LSTM encodeur / décodeur
- attention interprétable
- gating partout

---

## 10. Différences entre le papier et l’implémentation `pytorch-forecasting`

Le point important pour Praedixa : la référence de fait n’est pas seulement le papier, mais aussi la façon dont `pytorch-forecasting` l’implémente.

La classe officielle `TemporalFusionTransformer` annonce plusieurs améliorations par rapport au papier :

- variables statiques continues supportées
- résumés de variables catégorielles multiples
- longueurs encodeur / décodeur variables par échantillon
- embeddings catégoriels non retraités inutilement par VSN
- dimensions de variables ajustées par interpolation pour réduire les paramètres
- possibilité de partager certains sous-réseaux entre encodeur et décodeur
- contraintes monotones héritées du base model

Conséquence :

- si Praedixa s’appuie sur `pytorch-forecasting`, le comportement réel sera celui de cette implémentation, pas un copier-coller pur du papier

---

## 11. Lecture ligne par ligne de l’implémentation importante

Cette section résume les lignes importantes de la classe officielle sans recopier le code source.

### 11.1 Signature de la classe

La classe `TemporalFusionTransformer` expose notamment :

- `hidden_size`
- `lstm_layers`
- `dropout`
- `output_size`
- `loss`
- `attention_head_size`
- `max_encoder_length`
- familles de covariables
- `hidden_continuous_size`
- `embedding_sizes`
- `learning_rate`
- `reduce_on_plateau_patience`
- `monotone_constraints`
- `share_single_variable_networks`
- `causal_attention`
- `mask_bias`

### 11.2 Hyperparamètres officiellement signalés comme structurants

La docstring officielle insiste particulièrement sur :

- `hidden_size` comme hyperparamètre principal
- `lstm_layers` où `2` est souvent bon
- `attention_head_size` avec `4` comme bon défaut
- `mask_bias=-inf` à préférer si on veut faciliter le mixed precision

Pour Praedixa, cela veut dire :

- ne pas tuner au hasard
- commencer par `hidden_size`, `encoder_length`, `dropout`, `learning_rate`

---

## 12. Bloc 1 : embeddings et projections d’entrée

### 12.1 Catégorielles

L’implémentation utilise `MultiEmbedding`.

But :

- encoder les variables catégorielles dans un espace latent
- partager une logique de gestion des catégories
- respecter les tailles d’embeddings dérivées du dataset

Exemples Praedixa :

- `series_id`
- `location_id`
- `product_id`
- `day_of_week`
- `holiday_flag`
- `promo_type`

### 12.2 Variables réelles

Les variables continues ne sont pas injectées "brutes". L’implémentation crée des **prescalers** :

- une projection linéaire par variable réelle
- `1 -> hidden_continuous_size` ou taille spécifique par variable

Pourquoi c’est important :

- cela évite de concaténer naïvement des scalaires de nature très différente
- cela met continues et embeddings dans un espace latent comparable

Exemples Praedixa :

- `forecast_temperature`
- `planned_price`
- `same_dow_mean_4w`
- `lag_1`
- `lag_7`

---

## 13. Bloc 2 : Variable Selection Networks

Les VSN sont au cœur du TFT.

L’implémentation en construit trois :

- sélection statique
- sélection encodeur
- sélection décodeur

### 13.1 Ce qu’elles font réellement

Pour chaque famille de variables :

- chaque variable est d’abord passée dans un sous-réseau individuel
- les variables sont aussi concaténées dans une représentation aplatie
- un GRN "flattened" calcule des poids normalisés
- ces poids servent à pondérer les sorties de chaque variable
- on somme ensuite les contributions pondérées

En pseudo-code :

```text
pour chaque variable:
  projeter -> transformer -> obtenir une représentation latente

concaténer toutes les représentations brutes
calculer des poids softmax
pondérer chaque représentation
sommer
```

### 13.2 Microdétail important

Quand il n’y a **qu’une seule variable** dans une famille :

- le module court-circuite la vraie sélection
- il n’y a pas d’arbitrage réel entre variables
- les poids retournés valent simplement 1

Conséquence pour Praedixa :

- les graphes d’importance n’ont du sens que s’il y a réellement plusieurs variables concurrentes

### 13.3 Microdétail important sur les catégorielles

Les embeddings catégoriels ne repassent pas par le même type de GRN que les réelles.

L’implémentation traite les embeddings comme déjà "riches" et évite un retraitement redondant.

### 13.4 `share_single_variable_networks`

Option utile mais à manier avec prudence :

- si activée, certains sous-réseaux individuels sont partagés entre encodeur et décodeur
- réduit le nombre de paramètres
- peut améliorer la généralisation
- peut aussi rigidifier inutilement le modèle

Recommandation Praedixa :

- `False` au départ
- benchmarker `True` seulement après baseline stable

---

## 14. Bloc 3 : contexte statique

Après sélection statique, le TFT calcule plusieurs contextes spécialisés à partir d’une représentation statique unique.

L’implémentation crée plusieurs GRN distincts pour :

- guider la sélection de variables temporelles
- initialiser l’état caché du LSTM
- initialiser l’état cellule du LSTM
- enrichir les représentations post-LSTM

Ce point est fondamental.

Le contexte statique ne sert pas seulement à "ajouter du contexte". Il sert à **conditionner tout le reste du réseau**.

En pratique, cela permet au modèle de traiter différemment :

- un coffee shop urbain
- une boulangerie de centre commercial
- un site drive
- un produit météo-sensible
- un produit de routine

---

## 15. Bloc 4 : LSTM encodeur et décodeur

Le TFT ne remplace pas tout par l’attention.

Il garde :

- un LSTM pour l’historique
- un LSTM pour la partie future

### 15.1 Pourquoi ce choix est intelligent

Les LSTM restent bons pour :

- capter l’inertie locale
- stabiliser les séquences courtes à moyennes
- transmettre une mémoire immédiate

Dans Praedixa :

- rythmes hebdomadaires
- inertie à court terme
- effets locaux après promo
- retours progressifs à la normale

sont souvent bien captés par cette brique.

### 15.2 Initialisation non triviale

L’état caché initial et l’état cellule initial ne sont pas nuls.

Ils sont dérivés du contexte statique.

Conséquence :

- le LSTM démarre déjà avec un "prior" sur le type de série

### 15.3 Microdétail de code intéressant

Dans l’implémentation officielle, certains modules de gate / add-norm post-LSTM sont **partagés** entre encodeur et décodeur.

Ce partage n’est pas toujours intuitif quand on lit le modèle pour la première fois.

Implication :

- la symétrie encodeur / décodeur est plus forte qu’on pourrait le croire

---

## 16. Bloc 5 : gating et résiduels

Le TFT utilise massivement :

- `GatedLinearUnit`
- `AddNorm`
- `GateAddNorm`
- `GatedResidualNetwork`

### 16.1 Pourquoi c’est vital

Le gating sert à :

- laisser passer l’information utile
- étouffer l’information inutile
- stabiliser les chemins résiduels
- éviter qu’un bloc dégrade systématiquement la représentation

### 16.2 Gated Linear Unit

Le GLU :

- projette l’entrée
- la coupe en deux
- utilise une partie comme porte pour l’autre

Effet :

- filtrage adaptatif

### 16.3 AddNorm

Combine :

- branche principale
- skip connection
- normalisation

Quand dimensions différentes :

- l’implémentation rééchantillonne la branche skip par interpolation

Ce microdétail est important :

- il permet au réseau de garder des skip connections même lorsque tailles d’entrée et de sortie diffèrent

### 16.4 GatedResidualNetwork

Le GRN :

- projection linéaire
- éventuelle injection de contexte
- activation ELU
- nouvelle projection
- gate + résiduel + layer norm

C’est la brique de fusion universelle du TFT.

---

## 17. Bloc 6 : static enrichment

Après le LSTM, le modèle réinjecte le contexte statique dans les représentations temporelles.

But :

- spécialiser le même pattern temporel selon le type de série

Exemple Praedixa :

- un pic du samedi n’a pas la même lecture selon :
  - centre-ville
  - retail park
  - gare
  - quartier business

Le static enrichment permet précisément ce type d’adaptation.

---

## 18. Bloc 7 : attention interprétable

Le module d’attention officiel est `InterpretableMultiHeadAttention`.

### 18.1 Ce qu’il fait

- requêtes `q` sur la partie **décodeur**
- clés `k` et valeurs `v` sur **tout le contexte encodeur + décodeur**
- masquage causal ou non selon configuration

### 18.2 Microdétail très important

Dans cette implémentation :

- chaque tête a ses propres projections `q` et `k`
- la projection `v` est partagée
- les sorties des têtes sont **moyennées** avant projection finale

Cette moyenne contribue au caractère "interpretable" :

- on évite un simple concat de têtes opaques
- on garde une lecture agrégée des patterns d’attention

### 18.3 `causal_attention`

Par défaut :

- `True`

Donc :

- le décodeur n’assiste pas à des pas futurs indisponibles

Si `False` :

- le modèle peut regarder des pas futurs du décodeur autorisés par les covariables disponibles

Recommandation Praedixa :

- garder `True` au départ
- n’ouvrir `False` que si un cas d’usage précis le justifie et après test anti-fuite rigoureux

### 18.4 `mask_bias`

Le masque d’attention applique une très grande valeur négative aux positions interdites.

Point pratique issu de la source :

- défaut historique : `-1e9`
- pour mixed precision, mieux vaut souvent `-float("inf")`

Recommandation Praedixa :

- en `32-true` ou CPU : `-1e9` acceptable
- en `bf16-mixed` / `16-mixed` : préférer `-inf` si validé numériquement

---

## 19. Bloc 8 : tête de sortie

La sortie dépend de `output_size`.

### 19.1 Cas mono-cible

Le plus probable pour Praedixa au départ :

- une seule cible
- une couche linéaire finale

### 19.2 Cas multi-cible

L’implémentation supporte aussi :

- une liste de sorties
- une couche par cible

Ce n’est pas la priorité Praedixa V1.

### 19.3 Quantiles

Avec `QuantileLoss` :

- `output_size = nombre_de_quantiles`

Exemple :

- `[0.1, 0.5, 0.9]` -> `output_size = 3`

Le point forecast central lu par la loss est alors :

- le quantile `0.5`

---

## 20. Lecture pas à pas du `forward()`

Le `forward()` officiel peut se résumer ainsi :

1. concaténer encodeur et décodeur dans le temps
2. construire embeddings catégoriels + vecteurs continus
3. sélectionner les variables statiques
4. dériver le contexte statique
5. sélectionner les variables temporelles encodeur
6. sélectionner les variables temporelles décodeur
7. initialiser le LSTM via le contexte statique
8. faire tourner LSTM encodeur
9. faire tourner LSTM décodeur
10. appliquer skip + gate + addnorm post-LSTM
11. enrichir statiquement
12. appliquer l’attention sur la fenêtre de prédiction
13. appliquer gate + feed-forward
14. projeter en sortie
15. dénormaliser via `target_scale`
16. retourner aussi les tenseurs d’interprétation :
    - attention encodeur
    - attention décodeur
    - poids de variables statiques
    - poids de variables encodeur
    - poids de variables décodeur

Ce n’est donc pas un modèle qui renvoie seulement `prediction`.

Il renvoie aussi un vrai **paquet d’artefacts interprétables**.

---

## 21. Interprétation interne : ce que fait vraiment `interpret_output()`

Cette méthode vaut la peine d’être comprise car elle conditionne tout le volet audit / diagnostics.

### 21.1 Gestion des longueurs variables

L’implémentation :

- remet à niveau les tenseurs d’attention quand les séquences n’ont pas toutes la même longueur
- masque les zones padding
- réaligne l’attention encodeur pour que le passé récent soit à droite

### 21.2 Agrégation des poids de variables

Pour encodeur et décodeur :

- les poids sont sommés dans le temps
- puis normalisés par les longueurs effectives

Conséquence :

- on obtient une importance moyenne par variable, pas une importance brute non comparable

### 21.3 Attention finale

L’attention utilisée pour l’interprétation :

- moyenne les têtes
- prend un horizon de prédiction donné
- concatène partie encodeur et partie décodeur

### 21.4 Ce que cela veut dire pour Praedixa

L’interprétation la plus utile ne sera pas :

- un graphique global unique

mais plutôt :

- importance par famille de sites
- importance par famille produit
- attention par régimes météo
- attention par périodes commerciales

---

## 22. `TimeSeriesDataSet` : la vraie pièce centrale

Dans la pratique, le composant le plus risqué n’est pas le modèle. C’est `TimeSeriesDataSet`.

Pour Praedixa, la discipline dataset est plus importante que le choix de `hidden_size`.

### 22.1 Ce que gère `TimeSeriesDataSet`

La classe gère :

- l’encodage des catégories
- le scaling
- la normalisation de cible
- les longueurs encodeur / décodeur
- la génération des sous-séquences
- les datasets train / val / test / inference

### 22.2 Ce que cela implique

Le dataset train devient :

- la source de vérité des normalizers
- la source de vérité des encoders
- la source de vérité du contrat de reconstruction

---

## 23. Contrat minimal de dataset Praedixa

Pour un TFT Praedixa, le panel canonique doit contenir au minimum :

### 23.1 Colonnes structurantes

- `dt`
- `time_idx`
- `series_id`
- `location_id`
- `product_id`

### 23.2 Cible

Selon le run :

- `target_demand_qty_d_plus_1`
- ou `target_delta_log_wow_d_plus_1`

### 23.3 Ancre éventuelle

Si mode WoW :

- `target_lag_7` ou équivalent

### 23.4 Diagnostics de label

- `target_semantics`
- `censor_flag`
- `target_source`
- `label_quality_score`
- `usable_for_training_flag`

### 23.5 Variables futures connues

Exemples :

- `day_of_week`
- `month`
- `holiday_flag`
- `school_holiday_flag`
- `planned_promo_flag`
- `planned_discount_pct`
- `forecast_temperature`
- `forecast_rain_mm`
- `planned_open_flag`

### 23.6 Variables historiques inconnues

Exemples :

- `observed_stockout_flag`
- `observed_service_time`
- `observed_sales`
- `lag_1`
- `lag_7`
- `same_dow_mean_4w`

---

## 24. Mapping Praedixa -> familles de variables TFT

### 24.1 `group_ids`

Recommandation de départ :

```python
group_ids = ["series_id"]
```

Pourquoi :

- `series_id` identifie déjà `site x produit`
- simplifie la normalisation de groupe
- laisse `location_id` et `product_id` être des variables statiques séparées

Règle d'implémentation :

- la source exécutable de vérité du mapping TFT est `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/feature_mapping.py`
- aucun rôle TFT ne doit être inféré par heuristique de dtype, cardinalité, nommage ou constance intra-groupe
- toute nouvelle colonne du panel `gold` doit recevoir un rôle explicite avant d'entrer dans le backend TFT
- si une colonne n'est pas mappée explicitement, le pipeline doit échouer

### 24.2 `static_categoricals`

Recommandation de départ :

- `location_id`
- `product_id`
- `client_id` si stable
- `product_family`
- `site_format`
- `city_cluster`

### 24.3 `static_reals`

Seulement si vraiment stables et effectivement disponibles dans le contrat de features.

### 24.4 `time_varying_known_categoricals`

- `day_of_week`
- `week_of_year_bucket`
- `month`
- `holiday_flag`
- `school_holiday_flag`
- `promo_flag_planned`
- `event_flag_planned`

### 24.5 `time_varying_known_reals`

- `planned_price`
- `planned_discount_pct`
- `forecast_temperature`
- `forecast_rain_mm`
- `relative_time_idx` si activé

### 24.6 `time_varying_unknown_categoricals`

- `observed_stockout_flag`
- `observed_closure_flag`
- `channel_disabled_flag`

### 24.7 `time_varying_unknown_reals`

- cible absolue si apprentissage direct
- cible transformée si apprentissage transformé

Attention :

- dans l'implémentation TFT actuelle de Praedixa, les lags ponctuels bruts (`lag_1`, `lag_7`, `target_lag_*`, `*_lag_*`) sont exclus explicitement du backend
- la logique est de laisser le TFT exploiter sa fenêtre encodeur pour les points historiques bruts, et de ne conserver comme raccourcis explicites que des résumés lissés et agrégés
- dans le contrat courant, ces résumés historiques restent mappés en `time_varying_known_reals` parce qu'ils sont disponibles à la date de décision représentée par la ligne
- exemples conservés dans `known_reals` : `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`, `rolling_std_28`, `same_dow_mean_4w`, `promo_rate_7`, `promo_rate_28`, `activity_rate_7`, `activity_rate_28`, `observed_stockout_rate_7`, `observed_stockout_rate_28`
- les variables conservées doivent rester strictement causales et disponibles à la date de décision

---

## 25. `time_idx` : règles Praedixa

Le `TimeSeriesDataSet` attend un **index entier**.

### 25.1 Règles

- entier monotone
- cohérent par `series_id`
- si pas de trous métier : incrément de `+1`

### 25.2 Si trous

Les trous doivent être explicités.

Un trou n’est pas forcément :

- zéro vente

Il peut signifier :

- fermeture
- produit inactif
- absence d’observation
- jour non exploitable

### 25.3 Recommandation Praedixa

Deux approches acceptables :

1. panel dense reconstruit
   - un jour = une ligne
   - trous reconstruits
   - colonnes de statut explicites

2. panel avec trous + `allow_missing_timesteps=True`
   - seulement si la sémantique est parfaitement documentée

Par défaut :

- préférer **panel dense reconstruit**

---

## 26. `allow_missing_timesteps` : piège de compréhension

La doc officielle est explicite :

- `allow_missing_timesteps=True` gère les **trous de lignes**
- pas les **NA dans les colonnes**

Cela veut dire :

- si une valeur de `forecast_temperature` est NA, ce flag ne résout rien
- il faut remplir ou exclure explicitement

Recommandation Praedixa :

- conserver cette option comme exception, pas comme refuge
- logger à part :
  - trous de panel
  - NA de colonnes

---

## 27. `predict_mode`, `from_dataset`, `from_parameters`

### 27.1 `from_dataset`

La doc officielle indique :

- construit un nouveau dataset à partir d’un dataset existant
- conserve encoders, scalers, etc.
- appelle `from_parameters()` sous le capot

### 27.2 `get_parameters`

Retourne :

- les paramètres du dataset
- utilisables pour reconstruire exactement le même contrat

### 27.3 `from_parameters`

C’est la méthode clé pour le serving fiable :

- nouveau dataframe
- mêmes encoders
- mêmes scalers
- même contrat structurel

### 27.4 Règle Praedixa

Le backend doit suivre strictement :

```text
training_dataset = TimeSeriesDataSet(train_df, ...)
dataset_parameters = training_dataset.get_parameters()
validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_df, stop_randomization=True)
test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_df, stop_randomization=True)
inference_dataset = TimeSeriesDataSet.from_parameters(dataset_parameters, future_df, predict=True)
```

Jamais :

- refit silencieux des normalizers sur `val`, `test` ou `future_df`

---

## 28. `predict_mode`

La doc officielle dit :

- `predict_mode=False` : fenêtres glissantes, adapté à l’entraînement
- `predict_mode=True` : une seule séquence finale par groupe, adapté à la prédiction

Recommandation Praedixa :

- train / tuning : `predict_mode=False`
- inference batch de production : `predict_mode=True`

---

## 29. Normalisation de cible : choix recommandés

### 29.1 `GroupNormalizer`

La doc officielle :

- scale par groupe
- peut utiliser `standard` ou `robust`
- peut prendre une transformation

C’est le meilleur point de départ pour Praedixa si l’on apprend directement la cible absolue.

Recommandation :

```python
GroupNormalizer(groups=["series_id"], method="standard")
```

ou :

```python
GroupNormalizer(groups=["series_id"], method="robust")
```

si forte présence d’outliers.

### 29.2 `EncoderNormalizer`

La doc officielle précise :

- il est fit sur chaque séquence encodeur

Avantages :

- utile pour séries très hétérogènes

Inconvénients :

- plus difficile à auditer
- plus difficile à comparer strictement
- moins bon candidat pour les tests de non-régression forts

Recommandation Praedixa :

- éviter `EncoderNormalizer` en première version
- ne l’introduire qu’en challenger

### 29.3 Piège `log1p` du normalizer

La doc `GroupNormalizer` mentionne un point important :

- si on utilise une transformation `log1p`, l’inverse standard reste `exp`, pas `expm1`

C’est un microdétail critique pour ce repo, car le `TargetContract` reconstruit explicitement avec `np.expm1`.

Conséquence :

- si on choisit un mode TFT où le repo gère lui-même `log1p` / `expm1`, il faut **éviter de doubler** cette logique dans le normalizer

Recommandation Praedixa :

- si `TargetContract.target_mode == "log1p"` gère déjà la transformation :
  - garder le `target_normalizer` sans transformation log
- si on veut déléguer le log au normalizer :
  - alors il faut réaligner toute la reconstruction du repo

Le plus simple :

- laisser le `TargetContract` gérer la transformation
- garder le normalizer en simple standardisation de groupe

---

## 30. Catégories inconnues et cold start

La doc `NaNLabelEncoder` précise :

- avec `add_nan=True`, les classes inconnues et `nan` peuvent être encodées en `0`

Pour Praedixa, c’est indispensable sur certaines colonnes :

- `product_family`
- `promo_type`
- `event_type`

Sur `series_id`, `location_id`, `product_id`, le vrai sujet n’est pas seulement l’encodage inconnu.

Le vrai sujet est :

- veut-on scorer un nouvel objet jamais vu ?

### 30.1 Cas à distinguer

#### Nouveau `series_id`, site et produit déjà vus séparément

Possiblement gérable si :

- `location_id` et `product_id` sont présents comme statiques
- le modèle apprend des régularités cross-series

#### Nouveau `location_id`

Plus risqué.

#### Nouveau `product_id`

Plus risqué.

#### Nouveau site + nouveau produit

Cas le plus risqué.

### 30.2 Règle Praedixa

Le backend doit tester explicitement :

- unseen categories
- new product
- new site
- new series_id

Et définir le fallback :

- TFT si contrat satisfait
- sinon baseline / heuristique contrôlée

---

## 31. QuantileLoss et sorties probabilistes

### 31.1 Ce que dit la doc

La `QuantileLoss` :

- est une `MultiHorizonMetric`
- utilise la loss pinball par quantile
- convertit naturellement le quantile `0.5` en point forecast central

### 31.2 Recommandation Praedixa de départ

Jeu minimal :

- `[0.1, 0.5, 0.9]`

Jeu plus fin si calibration forte nécessaire :

- `[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]`

### 31.3 Règle de promotion

Ne jamais promouvoir un TFT quantile sur :

- seule `val_loss`

Toujours regarder :

- pinball loss par quantile
- couverture empirique
- largeur moyenne d’intervalle
- WAPE / MAE / Bias absolus
- coût rupture / surstock

---

## 32. Hyperparamètres réellement importants

### 32.1 Les plus importants

Pour Praedixa, ordre de priorité pratique :

1. `max_encoder_length`
2. `max_prediction_length`
3. `hidden_size`
4. `dropout`
5. `learning_rate`
6. `batch_size`
7. `hidden_continuous_size`
8. `attention_head_size`
9. `lstm_layers`

### 32.2 `hidden_size`

Effet :

- capacité globale
- taille des représentations
- coût mémoire

Point de départ recommandé :

- `32` ou `64`

### 32.3 `hidden_continuous_size`

Très important si beaucoup de covariables réelles.

Règle pratique :

- souvent `<= hidden_size`

Point de départ recommandé :

- `8`, `16`, `32`

### 32.4 `attention_head_size`

Point de départ :

- `4`

Si dataset plus petit :

- `1` ou `2`

### 32.5 `lstm_layers`

Point de départ :

- `1`

Challenger :

- `2`

### 32.6 `dropout`

Zone raisonnable :

- `0.1` à `0.3`

### 32.7 `share_single_variable_networks`

Départ :

- `False`

### 32.8 `causal_attention`

Départ :

- `True`

### 32.9 `monotone_constraints`

L’implémentation supporte ces contraintes mais la docstring indique aussi qu’elles ralentissent fortement l’entraînement.

Recommandation Praedixa :

- ne pas les activer au départ
- les réserver à des covariables où une contrainte monotone est réellement défendable

---

## 33. Longueurs encodeur et horizon

### 33.1 `max_encoder_length`

Dans Praedixa, le bon choix dépend :

- de la saisonnalité utile
- du rythme hebdomadaire
- des événements
- de la mémoire longue réellement exploitable

Point de départ recommandé :

- `28`
- `56`

Si données très riches et stables :

- `84`

mais pas par réflexe.

### 33.2 `max_prediction_length`

Pour le wedge actuel :

- `1`
- `7`
- `14`

sont les horizons les plus naturels.

### 33.3 Recommandation V1

Deux configurations à benchmarker :

1. `encoder=28`, `prediction=7`
2. `encoder=56`, `prediction=7`

Puis challenger :

3. `encoder=56`, `prediction=14`

---

## 34. Training loop Lightning recommandé

### 34.1 Reproductibilité forte

La doc Lightning recommande :

- `seed_everything(..., workers=True)`
- `Trainer(deterministic=True)`

La doc PyTorch rappelle que :

- la reproductibilité parfaite n’est jamais garantie entre versions / plateformes
- mais on peut fortement réduire la variance

### 34.2 Référence Praedixa pour runs de référence

```python
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import torch

seed_everything(config.random_seed, workers=True)
torch.use_deterministic_algorithms(True)
```

Et côté trainer :

- `deterministic=True`
- `benchmark=False`
- `inference_mode=True`
- `gradient_clip_val` non nul

### 34.3 Callbacks minimum

- `EarlyStopping`
- `ModelCheckpoint(save_top_k=3, save_last=True)`
- `LearningRateMonitor`

### 34.4 `detect_anomaly`

La doc Lightning est claire :

- utile pour debug
- trop coûteux pour runs normaux

Donc :

- `True` seulement en debug
- `False` sinon

### 34.5 `precision`

La doc Lightning distingue :

- `32-true`
- `16-mixed`
- `bf16-mixed`
- etc.

Recommandation Praedixa :

- run de référence / CPU / MPS fragile :
  - `32-true`
- entraînement GPU moderne avec validation de parité :
  - `bf16-mixed`

### 34.6 `gradient_clip_val`

Obligatoire dans ce projet.

Zone de départ :

- `0.1`
- `0.3`
- `0.5`
- `1.0`

---

## 35. `torch.use_deterministic_algorithms`

La doc PyTorch précise :

- cette option force l’usage d’algorithmes déterministes quand disponibles
- sinon l’opération peut lever une erreur

Elle rappelle aussi :

- cela ralentit souvent
- cela ne suffit pas seul à rendre tout parfaitement reproductible

Recommandation Praedixa :

### 35.1 Mode référence

```python
torch.use_deterministic_algorithms(True)
```

### 35.2 Mode exploration

```python
torch.use_deterministic_algorithms(False)
```

ou éventuellement :

- `warn_only=True` en debug exploratoire

### 35.3 Règle projet

Les runs servant :

- de non-régression
- de comparaison de checkpoints
- de promotion

doivent être en **mode déterministe de référence**.

---

## 36. `torch.inference_mode` et serving

La doc PyTorch indique que `inference_mode` :

- va plus loin que `no_grad`
- enlève du coût supplémentaire lié à autograd
- impose aussi plus de contraintes

Et surtout :

- `inference_mode` ne remplace pas `model.eval()`

Règle Praedixa :

```python
model.eval()
with torch.inference_mode():
    preds = ...
```

Toujours les deux.

---

## 37. `torch.compile`

La doc PyTorch 2.x présente `torch.compile()` comme la voie moderne d’accélération.

Mais elle rappelle implicitement :

- optimisation possible seulement si le graphe se compile proprement
- présence potentielle de graph breaks

Recommandation Praedixa :

### 37.1 Ce qu’il ne faut pas faire

- activer `torch.compile()` dès la première intégration
- diagnostiquer qualité modèle et compile en même temps

### 37.2 Stratégie recommandée

1. backend stable en eager mode
2. parité numérique validée
3. benchmark inference/train
4. `torch.compile()` seulement si gain mesuré

---

## 38. Dataloaders : bonnes pratiques

Le `TimeSeriesDataSet.to_dataloader()` gère :

- padding
- collate
- séparation encodeur / décodeur

### 38.1 Recommandations

- `train=True` pour train
- `train=False` pour val / test / predict
- `shuffle=True` seulement côté train
- `drop_last=True` au train si utile

### 38.2 `batch_sampler="synchronized"`

La doc mentionne cette option pour aligner les échantillons dans le temps.

À utiliser seulement si :

- on a un vrai besoin d’alignement temporel explicite
- et pas de trous incompatibles

Ce n’est pas un besoin par défaut pour Praedixa V1.

---

## 39. Bundle d’artefacts Praedixa

Le repo décrit déjà un bundle promouvable minimal. Il faut le respecter.

Bundle recommandé :

```text
artifacts/run_YYYYMMDD_HHMMSS/
  model.ckpt
  dataset_parameters.json
  encoders.pkl
  feature_manifest.yaml
  input_schema.json
  output_schema.json
  split_manifest.json
  metrics.json
  config.yaml
  model_card.md
  git_sha.txt
  data_hashes.json
```

### 39.1 Artefacts absolument critiques pour TFT

- `model.ckpt`
- `dataset_parameters.json`
- `feature_manifest.yaml`
- `target_contract.json`
- `quantiles.json`
- `metrics_by_horizon.json`

### 39.2 Pourquoi `dataset_parameters.json` est vital

Parce qu’il fixe :

- encoders
- scalers
- familles de variables
- longueurs

Sans lui, l’inférence n’est pas fiable.

---

## 40. Contrat d’entrée / sortie du backend

Le backend TFT à brancher dans ce repo doit fournir au minimum :

### 40.1 `fit_tft_model(...)`

Responsabilité :

- construire dataset train / valid
- instancier le modèle
- lancer `trainer.fit`
- retourner :
  - objet modèle
  - chemin checkpoint
  - métriques de validation
  - dataset parameters

### 40.2 `predict_with_tft_model(...)`

Responsabilité :

- reconstruire le dataset d’inférence avec `from_parameters`
- lancer `predict`
- retourner :
  - point forecast
  - quantiles
  - éventuellement sorties brutes si demandées

### 40.3 `save_tft_model(...)`

Responsabilité :

- sauver checkpoint
- sauver bundle dataset / manifest / target contract

### 40.4 `load_tft_model(...)`

Cette fonction n’existe pas encore mais devrait exister.

Responsabilité :

- charger checkpoint
- charger dataset parameters
- charger manifest
- valider compatibilité d’entrée

---

## 41. Proposition de modules backend à ajouter

Structure recommandée :

```text
products/demand_forecast/src/praedixa/demand_forecast/backends/tft/
  backend.py
  model_utils.py
  config.py
  dataset.py
  module.py
  metrics.py
  artifacts.py
  inference.py
  manifests.py
```

### 41.1 `config.py`

Dataclasses / modèles :

- `TFTTrainingConfig`
- `TFTDatasetConfig`
- `TFTInferenceConfig`

### 41.2 `dataset.py`

Fonctions :

- construire `TimeSeriesDataSet`
- valider taxonomie des variables
- sauver / recharger `dataset_parameters`

### 41.3 `module.py`

Fonctions :

- construire `TemporalFusionTransformer.from_dataset(...)`
- configurer loss / metrics

### 41.4 `artifacts.py`

Fonctions :

- sauver bundle
- calculer hash de données
- écrire manifests JSON/YAML

### 41.5 `inference.py`

Fonctions :

- reconstruire dataset futur
- lancer prediction
- reconstituer sortie métier

---

## 42. Plan d’intégration avec `training/orchestrator.py`

Le pipeline actuel attend déjà :

- sélection de features
- folds
- tuning
- rapports Optuna

### 42.1 Ce qu’il faut garder

- structure des dossiers
- `TargetContract`
- `feature_audit`
- log des poids de datasets

### 42.2 Ce qu’il faudra adapter

Aujourd’hui, `training/tuning.py` sample des hyperparamètres de style boosting :

- `max_depth`
- `min_samples_leaf`
- etc.

Cela n’a évidemment pas de sens pour TFT.

Il faudra remplacer par un espace Optuna TFT, par exemple :

- `hidden_size`
- `hidden_continuous_size`
- `attention_head_size`
- `dropout`
- `learning_rate`
- `lstm_layers`
- `gradient_clip_val`
- `batch_size`
- `max_encoder_length`

---

## 43. Espace de recherche Optuna recommandé

### 43.1 Petit espace initial

```text
hidden_size: [16, 32, 64]
hidden_continuous_size: [8, 16, 32]
attention_head_size: [1, 2, 4]
lstm_layers: [1, 2]
dropout: [0.05, 0.1, 0.2, 0.3]
learning_rate: loguniform(1e-4, 3e-2)
gradient_clip_val: [0.1, 0.3, 0.5, 1.0]
batch_size: [64, 128, 256]
max_encoder_length: [28, 56]
```

### 43.2 Ce qu’il ne faut pas tuner trop tôt

- `share_single_variable_networks`
- `causal_attention`
- `monotone_constraints`
- quantiles trop fins

---

## 44. Métrique d’optimisation

Le piège classique serait :

- optimiser uniquement `val_loss` pinball

Pour Praedixa, la promotion doit se faire sur un paquet de métriques.

### 44.1 Pour l’optimisation Optuna

Optimiser en priorité :

- pinball / val loss

car c’est differentiable / standard.

### 44.2 Pour la sélection finale

Lire aussi :

- WAPE absolu
- MAE absolu
- Bias
- couverture P10/P90
- largeur moyenne d’intervalle
- métriques par horizon
- métriques par segment
- métriques business

### 44.3 Règle de promotion

Le modèle champion n’est pas forcément celui avec le meilleur `val_loss`.

C’est celui qui :

- garde une bonne calibration
- réduit l’erreur absolue utile
- améliore les métriques business

---

## 45. Métriques business Praedixa

Le wedge Praedixa impose des métriques orientées opérations.

À suivre au minimum :

- `stockout_cost_proxy`
- `waste_cost_proxy`
- `material_cost_error`
- `service_level_proxy`
- `labor_alignment_proxy` si dérivé en aval

### 45.1 Règle critique

Une amélioration de pinball loss sans amélioration économique n’est pas suffisante pour promouvoir le modèle.

---

## 46. Calibration probabiliste

Le TFT sera utilisé pour fournir des quantiles. Il faut donc mesurer :

### 46.1 Couverture empirique

Exemple :

- le vrai se trouve-t-il environ 80% du temps entre P10 et P90 ?

### 46.2 Largeur d’intervalle

Un intervalle trop large :

- est peu utile

### 46.3 Sharpness vs coverage

Il faut arbitrer :

- précision
- couverture
- largeur

### 46.4 Lecture par horizon

Toujours regarder :

- `coverage@h=1`
- `coverage@h=2`
- ...

---

## 47. Intermittence et all-zero

Certains produits alimentaires peuvent être :

- faiblement vendus
- intermittents
- parfois à zéro longtemps

Le TFT peut les gérer, mais il faut tester explicitement :

- séries all-zero
- séries quasi all-zero
- séries ultra intermittentes

Dans certains cas, un fallback ou une segmentation produit est plus robuste.

---

## 48. Cohérence hiérarchique

Praedixa opère naturellement à plusieurs niveaux :

- produit
- catégorie
- site
- réseau

Le TFT de base ne garantit pas la cohérence hiérarchique.

### 48.1 Conséquence

On peut avoir :

- somme des SKU != total site

### 48.2 Recommandation Praedixa

V1 :

- accepter l’absence de réconciliation intégrée
- mesurer l’écart de cohérence

V2 :

- ajouter une couche de reconciliation post-forecast si besoin opérationnel réel

---

## 49. Pièges spécifiques Praedixa

### 49.1 Météo observée injectée comme connue

Interdit.

### 49.2 Stockout observé utilisé comme feature future

Interdit.

### 49.3 `log1p` géré deux fois

Très dangereux dans ce repo :

- une fois dans `TargetContract`
- une fois dans `GroupNormalizer`

### 49.4 Catégories non vues qui crashent en prod

Interdit.

### 49.5 Dataset refit au test

Interdit.

### 49.6 Promotion sur val_loss seul

Interdit.

### 49.7 Interprétation vendue comme causalité

Interdit.

### 49.8 Pas de fallback si features futures manquantes

Interdit.

---

## 50. Recommandation d’architecture cible pour Praedixa V1

### 50.1 Cible V1

Backend TFT framework-based, non from-scratch.

### 50.2 Mode de cible recommandé

Deux pistes parallèles :

- **Champion candidat A** : cible absolue
- **Champion candidat B** : cible `delta_log_wow`

### 50.3 Sortie

- `P10`
- `P50`
- `P90`

### 50.4 Datasets

- `group_ids=["series_id"]`
- `location_id`, `product_id` en statiques catégorielles
- `GroupNormalizer(groups=["series_id"])`
- reconstruction train / val / test / future via `from_dataset` / `from_parameters`

### 50.5 Trainer

- Lightning
- `deterministic=True`
- `benchmark=False`
- `gradient_clip_val` activé
- `EarlyStopping`
- `ModelCheckpoint(save_top_k=3, save_last=True)`
- `LearningRateMonitor`

### 50.6 Serving

- `model.eval()`
- `torch.inference_mode()`
- validation schéma d’entrée
- fallback baseline

---

## 51. Configuration V1 de départ

Configuration raisonnable, non dogmatique :

```yaml
python: "3.13"
torch: "2.11.x"
lightning: "2.6.x"
pytorch_forecasting: "1.7.x"

group_ids:
  - "series_id"

static_categoricals:
  - "location_id"
  - "product_id"
  - "product_family"
  - "site_format"

static_reals: []

time_varying_known_categoricals:
  - "day_of_week"
  - "month"
  - "holiday_flag"
  - "promo_flag_planned"

time_varying_known_reals:
  - "forecast_temperature"
  - "forecast_rain_mm"
  - "planned_price"

time_varying_unknown_categoricals:
  - "observed_stockout_flag"

time_varying_unknown_reals:
  - "target_demand_qty_d_plus_1"
  - "lag_1"
  - "lag_7"
  - "same_dow_mean_4w"

max_encoder_length: 56
max_prediction_length: 7
hidden_size: 32
hidden_continuous_size: 16
attention_head_size: 4
lstm_layers: 1
dropout: 0.1
learning_rate: 1e-3
gradient_clip_val: 0.1
loss: "QuantileLoss"
quantiles: [0.1, 0.5, 0.9]
target_normalizer: "GroupNormalizer(groups=['series_id'])"
causal_attention: true
share_single_variable_networks: false
precision: "32-true"
mask_bias: -1000000000.0
```

Version mixed precision à challenger plus tard :

- `precision: "bf16-mixed"`
- `mask_bias: -inf`

---

## 52. Blueprint de code attendu pour `fit_tft_model()`

Pseudo-code de ce que doit faire le backend :

```text
1. valider le manifest de features
2. valider le TargetContract
3. préparer train_df / valid_df sans fuite
4. construire training_dataset
5. extraire dataset_parameters
6. construire validation_dataset via from_dataset(training_dataset, ...)
7. créer dataloaders
8. instancier TemporalFusionTransformer.from_dataset(training_dataset, ...)
9. configurer Trainer Lightning
10. trainer.fit(...)
11. récupérer meilleur checkpoint
12. sauver:
   - model.ckpt
   - dataset_parameters.json
   - feature_manifest.yaml
   - metrics.json
   - split_manifest.json
13. retourner un objet de backend propre
```

---

## 53. Blueprint de code attendu pour `predict_with_tft_model()`

```text
1. charger checkpoint + dataset parameters + feature manifest
2. valider schéma d’entrée
3. construire future_df au grain dt x series_id
4. reconstruire inference_dataset = from_parameters(...)
5. lancer predict
6. convertir la sortie brute en:
   - P10
   - P50
   - P90
7. si target transformée:
   - reconstruire absolu proprement
8. valider monotonicité des quantiles
9. retourner dataframe métier
```

---

## 54. Tests obligatoires

### 54.1 Dataset contracts

- même train params => même encodage
- `from_parameters()` == contrat train
- `predict_mode=True` retourne une séquence par groupe

### 54.2 Targets

- reconstruction `delta_log_wow` correcte
- reconstruction `log1p` correcte
- quantiles absolus cohérents après reconstruction

### 54.3 Cold start

- catégories inconnues encodées proprement
- site nouveau
- produit nouveau
- série nouvelle

### 54.4 Checkpoint

- round-trip checkpoint => même prédiction à tolérance fixée

### 54.5 Déterminisme

- même seed => métriques identiques à tolérance fixée

### 54.6 Robustesse données

- séries vides
- all-zero
- trous de calendrier
- NA dans covariables
- features futures manquantes

### 54.7 Business

- calcul des métriques économiques
- fallback baseline si contrat cassé

---

## 55. Monitoring en production

À suivre au minimum :

### 55.1 Santé des features

- `null_rate`
- `out_of_range_rate`
- `unseen_category_rate`
- `stale_known_future_rate`

### 55.2 Santé des sorties

- distribution des quantiles
- crossing de quantiles
- variance de `P90 - P10`
- saturation anormale à 0

### 55.3 Santé des performances

- WAPE / MAE / Bias par horizon
- couverture empirique
- drift par segment

### 55.4 Infra

- latence `p95`
- taux d’échec
- mémoire CPU / GPU
- batch size effectif

---

## 56. Ce qu’il ne faut pas faire maintenant

- réécrire TFT from scratch avant d’avoir validé la stack framework
- brancher un modèle sans bundle `dataset_parameters`
- tuner 50 hyperparamètres avant d’avoir une baseline propre
- promouvoir un modèle sans métriques business
- parler de demande latente si le label est en fait une vente observée

---

## 57. Ordre d’implémentation recommandé

1. verrouiller dépendances `torch` / `lightning` / `pytorch-forecasting`
2. écrire `feature_manifest` exécutable
3. implémenter `dataset.py`
4. implémenter `module.py`
5. implémenter `fit_tft_model()`
6. implémenter `predict_with_tft_model()`
7. implémenter `save/load` artefacts
8. brancher `training/tuning.py`
9. brancher `evaluation/modeling.py`
10. ajouter tests de non-régression
11. seulement ensuite lancer tuning sérieux

---

## 58. Décisions techniques recommandées aujourd’hui

Si je devais fixer les choix de départ pour Praedixa sans coder encore le backend, je recommanderais :

### 58.1 Choix backend

- `pytorch-forecasting` d’abord
- from-scratch plus tard si besoin démontré

### 58.2 Choix cible

- benchmark absolu + benchmark delta-log-wow

### 58.3 Choix normalisation

- `GroupNormalizer(groups=["series_id"])`
- sans transformation log interne si le repo gère déjà `log1p`

### 58.4 Choix probabiliste

- `QuantileLoss([0.1, 0.5, 0.9])`

### 58.5 Choix training

- Lightning
- `deterministic=True`
- `benchmark=False`
- `gradient_clip_val` obligatoire

### 58.6 Choix inference

- `model.eval()`
- `torch.inference_mode()`
- fallback baseline

### 58.7 Choix roadmap

- V1 = fiable, lisible, mesurable
- V2 = optimisation décisionnelle et réconciliation hiérarchique

---

## 59. Règle stratégique finale

Le vrai sujet n’est pas "avoir un TFT".

Le vrai sujet est d’avoir un backend qui :

- respecte le temps
- respecte la sémantique métier
- respecte la logique économique
- respecte le contrat d’inférence

Si ces quatre choses sont vraies, le TFT peut devenir un excellent moteur probabiliste pour Praedixa.

Si elles ne le sont pas, même le meilleur TFT du monde donnera un faux sentiment de sophistication.

---

## 60. Checklist de lancement backend TFT Praedixa

- [ ] dépendances verrouillées
- [ ] manifest de covariables TFT écrit
- [ ] politique de cible absolue vs WoW écrite
- [ ] dataset train validé
- [ ] `from_dataset()` / `from_parameters()` testés
- [ ] catégories inconnues gérées
- [ ] quantiles P10/P50/P90 disponibles
- [ ] métriques par horizon disponibles
- [ ] bundle d’artefacts complet
- [ ] fallback baseline codé
- [ ] tests de round-trip checkpoint présents
- [ ] test déterminisme présent
- [ ] monitoring prod défini

---

## 61. Note finale sur la vision Praedixa

Le TFT n’est pas la destination.

C’est une brique.

Une très bonne brique si :

- elle produit des prévisions probabilistes fiables
- elle expose des diagnostics utiles
- elle nourrit ensuite des décisions de production, achat, staffing et arbitrage

La bonne trajectoire Praedixa est donc :

- d’abord un TFT robuste, probabiliste, rigoureux
- ensuite une couche de décision orientée coût / niveau de service / capacité

Le wedge reste la prévision.
La vision reste la décision.

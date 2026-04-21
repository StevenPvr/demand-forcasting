# Bonnes Pratiques -- praedixa-demand-forecast

> Document de reference pour le projet Praedixa de prevision de demande.
> Cible analytique ideale : **demande latente** quand les signaux de censure sont disponibles.
> Cible de repli : **ventes observees** quand on ne peut pas reconstruire proprement la demande latente.
> Chaque regle est concrete et actionnable. Les exemples DO/DON'T s'appuient sur le code existant.

---

## Table des matieres

1. [Data Science generales](#1-data-science-generales)
2. [Machine Learning](#2-machine-learning)
3. [Forecasting demande latente](#3-forecasting-demande-latente)
4. [Series temporelles](#4-series-temporelles)
5. [Regles specifiques au projet](#5-regles-specifiques-au-projet)

---

## 1. Data Science generales

### 1.1 Reproductibilite

**Seeds** -- Fixer toutes les sources d'aleatoire. Le projet utilise deja `random_seed: int = 7` dans `PipelineConfig` -- propager ce pattern a tout le pipeline.

```python
# DO -- seed explicite, passe via la config
np.random.seed(config.random_seed)
sample = data.sample(frac=0.05, random_state=config.random_seed)

# DON'T -- seed implicite ou absent
sample = data.sample(frac=0.05)  # non reproductible
```

**Versioning des donnees** -- Nommer les fichiers avec une convention lisible (`commercial_external_daily.parquet`, `train_selection_70_selected.parquet`, etc.). Ajouter un hash SHA-256 ou `manifest.json` pour detecter les corruptions. Ne jamais committer les donnees volumineuses dans git (utiliser DVC ou stockage externe).

**Environnements** -- Verrouiller les dependances avec `uv.lock` (fait). Travailler dans `.venv/bin/python`. Epingler Python dans `.python-version` (fait : `3.13`).

### 1.1.b Reproductibilite Deep Learning

Pour un backend TFT de reference, la reproductibilite ne doit pas dependre seulement de `numpy` ou `pandas`.

```python
from lightning.pytorch import Trainer, seed_everything

seed_everything(config.random_seed, workers=True)
torch.use_deterministic_algorithms(True)

trainer = Trainer(
    deterministic=True,
    benchmark=False,
)
```

- [ ] Logger `python`, `torch`, `lightning`, `cuda`, `cudnn`, type de GPU et nombre de devices.
- [ ] Interdire `DataLoader(..., in_order=False)` pour les runs de reference.
- [ ] Geler l'ordre des series et des batches pour les backtests de reference.
- [ ] Ajouter un test `meme seed => memes metriques a tolerance fixee`.

### 1.1.c DISCIPLINE D'EXECUTION -- ARRET DES ITERATIONS COUTEUSES

- [ ] AVANT TOUT RUN LONG, FAIRE UNE PHASE DE RAISONNEMENT COMPLETE : LIRE LE CODE IMPACTE, IDENTIFIER LES DEPENDANCES, VERIFIER LE SCHEMA, ET PREPARER UN PLAN DE MODIFICATION COHERENT.
- [ ] INTERDICTION DE LANCER DES RUNS DE 10-30 MINUTES EN BOUCLE POUR DECOUVRIR LE PROBLEME PAR ESSAIS SUCCESSIFS.
- [ ] SI UN RUN LONG EST NECESSAIRE, IL DOIT ETRE JUSTIFIE PAR UNE HYPOTHESE CLAIRE ET ETRE PRECEDE D'UN MAXIMUM DE VERIFICATIONS RAPIDES : TESTS UNITAIRES, `dbt parse`, LECTURE DU SQL COMPILE OU INSPECTION CIBLEE.
- [ ] APRES UN ECHEC, NE PAS RELANCER IMMEDIATEMENT UN AUTRE RUN LONG. PRENDRE LE TEMPS D'ANALYSER LA CAUSE, VALIDER LE PATCH LE PLUS COMPLET POSSIBLE, PUIS RELANCER UNE SEULE FOIS.
- [ ] L'OBJECTIF EST DE FAIRE LE TRAVAIL CORRECTEMENT DES LA PREMIERE ITERATION UTILE, PAS D'UTILISER LE PIPELINE COMPLET COMME OUTIL DE DEBUG PRINCIPAL.
- [ ] EN CAS DE DOUTE SUR L'IMPACT D'UN CHANGEMENT TFT/DBT, PRIVILEGIER D'ABORD LES PREUVES LOCALES ET PEU COUTEUSES, PUIS SEULEMENT ENSUITE LE RUN COMPLET.

### 1.2 Gestion des donnees manquantes

```python
# DO -- strategie explicite, documentee, tracable
df["target_demand_qty_d_plus_1"] = df.groupby("series_id")["target_demand_qty_d_plus_1"].transform(
    lambda s: s.ffill(limit=5)  # forward-fill max 5 jours
)
LOGGER.info("NaN restants apres ffill(5) : %d", df["target_demand_qty_d_plus_1"].isna().sum())

# DON'T -- fillna silencieux sans limite
df["target_demand_qty_d_plus_1"].fillna(method="ffill", inplace=True)  # propage indefiniment
```

- [ ] Compter et logger les NaN avant et apres traitement.
- [ ] Distinguer NaN "donnees absentes" des NaN "site ferme / produit inactif / donnee manquante".
- [ ] Documenter la strategie dans un docstring. Ne jamais interpoler entre series metier distinctes.

### 1.2.b Contrats de donnees et de config

- [ ] Chaque entree/sortie de pipeline a un schema executable `pandera`.
- [ ] Chaque config a un modele `Pydantic` avec `extra='forbid'`.
- [ ] Aucun champ sensible ne doit apparaitre en clair dans les logs.
- [ ] Chaque schema porte une `schema_version`.

### 1.3 Documentation du pipeline

Chaque module (`data_fetching`, `data_cleaning`, `data_preprocessing`) doit contenir un docstring de module dans `__init__.py`, et chaque fonction publique doit documenter ses parametres, retour et effets de bord.

```python
# DO
def build_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Construit le panel de prevision complet.

    Etapes : 1) Charge les sources  2) Aligne le schema canonique  3) Controle la qualite temporelle.

    Args:
        config: Configuration du pipeline (dates, chemins, retries).
    Returns:
        DataFrame trie par (dt, series_id).
    """
```

### 1.4 Separation des responsabilites

| Etape | Module | Entree | Sortie |
|---|---|---|---|
| Bronze | `apps/warehouse/load_bronze/main.py` | sources brutes / exports / open data | tables bronze |
| Silver | `apps/warehouse/run_silver/main.py` | bronze + enrichissements | tables silver |
| Gold | `apps/warehouse/run_gold/main.py` | silver | panel gold canonique |
| Global dataset | `platform/python/src/praedixa/platform/datasets/standardization/` | sources compatibles commerce | parquet canonique |
| Feature prep | `products/demand_forecast/src/praedixa/demand_forecast/feature_screening/` | panel nettoye | jeux de travail locaux |
| Optimize | `products/demand_forecast/src/praedixa/demand_forecast/training/` | jeux train / tuning | artefacts d'optimisation |
| Evaluate | `products/demand_forecast/src/praedixa/demand_forecast/evaluation/` | predictions + labels + baselines | metriques, diagnostics, model card |

Chaque etape lit un fichier et en produit un autre. Ne pas tout mettre dans une seule fonction.

### 1.5 Formats de stockage

| Format | Usage | Pourquoi |
|---|---|---|
| **Parquet** | Stockage principal | Typage fort, compression snappy, lecture par colonnes, 10-50x plus rapide que CSV |
| **CSV** | Debug / echantillons | Lisible humainement, utile pour un `sample_5pct` de verification |

- [ ] Ne jamais stocker de dates en string dans un parquet (utiliser `datetime64[ns]`).
- [ ] Compresser avec snappy (defaut pyarrow) ou zstd pour l'archivage long terme.

### 1.6 Logging et monitoring

Le projet utilise deja `logging` (`LOGGER = logging.getLogger(__name__)`) -- bien.

```python
# DO -- niveaux de log coherents
LOGGER.info("Fetching %d series", len(series_ids))             # progression
LOGGER.warning("One source batch failed (%s/%s)", attempts, n)  # recuperable
LOGGER.error("No usable dataset rows retrieved")                # echec

# DON'T
print(f"Fetching {len(series_ids)} rows")  # invisible en prod, non filtrable
```

- [ ] `logging` toujours, `print` jamais en production.
- [ ] Logger le nombre de lignes/series a chaque etape.
- [ ] Logger les temps d'execution (`time.perf_counter()`).

### 1.7 Typage strict

**Tout le code doit etre type.** Chaque variable, parametre et retour de fonction porte une annotation de type.

```python
# DO -- typage complet
def _parse_date(value: str | dt.date | dt.datetime) -> dt.date:
    ...

def _fetch_prices(
    series_ids: list[str],
    start_date: str,
    end_date: str,
    config: PipelineConfig,
) -> dict[str, pd.Series]:
    aliases: dict[str, str] = _load_aliases(config.series_aliases_csv)
    results: dict[str, pd.Series] = {}
    ...

LOGGER: logging.Logger = logging.getLogger(__name__)

# DON'T -- pas de type
def _parse_date(value):
    ...

results = {}
```

- [ ] `from __future__ import annotations` en premiere ligne de chaque fichier (union `X | Y` compatible Python 3.10+).
- [ ] Installer les stubs pour les librairies tierces (`pandas-stubs`, `types-requests`, `types-pytz`).
- [ ] Pour les librairies sans stubs, creer des stubs dans `typings/` si on depend vraiment d'elles.
- [ ] Utiliser `X | None` au lieu de `Optional[X]`.
- [ ] Typer les variables locales importantes (DataFrames, dictionnaires, listes).
- [ ] **Jamais** de `# type: ignore` nu -- utiliser `cast()` ou `Any` quand le type checker ne peut pas inferer, ou un `# type: ignore[code]` cible avec justification courte.
- [ ] **Zero erreur Pylance** dans tous les fichiers.

```python
from typing import Any, cast

# DO -- cast explicite quand le type checker ne peut pas inferer
last_valid: pd.Timestamp | None = cast(pd.Timestamp | None, series.last_valid_index())

# DO -- Any pour les retours de librairies non typees si cast impossible
raw_result: Any = external_lib.some_call()
data: pd.DataFrame = raw_result

# DON'T
data = external_provider.fetch(...)  # type: ignore[union-attr]
```

### 1.8 Limites de taille

- [ ] **50 lignes max par fonction.** Si une fonction depasse 50 lignes, la decouper en sous-fonctions.
- [ ] **500 lignes max par fichier.** Si un fichier depasse 500 lignes, le scinder en modules.

---

## 2. Machine Learning

### 2.1 Split temporel strict

**C'est la regle la plus importante du projet.**

```python
# DO -- split par date, sans aucune fuite du futur
train = df[df["date"] < "2020-01-01"]
val   = df[(df["date"] >= "2020-01-01") & (df["date"] < "2022-01-01")]
test  = df[df["date"] >= "2022-01-01"]

# DON'T -- JAMAIS de split aleatoire sur des series temporelles
train, test = train_test_split(df, test_size=0.2)  # INTERDIT : fuite temporelle
```

- [ ] Le set de test ne doit jamais etre touche avant l'evaluation finale.
- [ ] Les dates de coupure doivent etre dans la config, pas en dur.

### 2.2 Walk-forward validation

```python
# Expanding window -- le train grandit, le test avance
splits = [
    ("2004-01-01", "2015-12-31", "2016-01-01", "2016-12-31"),
    ("2004-01-01", "2016-12-31", "2017-01-01", "2017-12-31"),
    ("2004-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
]
```

- Utiliser `TimeSeriesSplit` comme point de depart, mais verifier les frontieres par **date** (pas par index).
- Laisser un **gap** d'au moins 1-5 jours entre train et validation pour eviter la fuite via lags/rolling.

### 2.2.b Taxonomie TFT des variables

Chaque feature doit porter un role explicite compatible TFT.

```yaml
price_planned:
  role: time_varying_known_real
  available_at_prediction: true
  source_system: pricing
  max_publication_lag_hours: 0

weather_observed:
  role: time_varying_unknown_real
  available_at_prediction: false
  source_system: weather_observed

promo_flag:
  role: time_varying_known_categorical
  available_at_prediction: true

series_id:
  role: static_categorical
  available_at_prediction: true
```

- [ ] Aucune feature n'entre dans le modele sans `role`.
- [ ] Aucune feature n'entre dans le modele sans `available_at_prediction`.
- [ ] Une meteo observee n'est jamais declaree `known`.
- [ ] La cible reelle doit etre dans `time_varying_unknown_reals`.

### 2.3 Prevention du data leakage

| Source de fuite | Exemple | Prevention |
|---|---|---|
| Feature calculee sur tout le dataset | `StandardScaler().fit(df_complet)` | Fit uniquement sur train |
| Information future dans les features | rolling mean sans `min_periods` | Toujours `min_periods=window` |
| Target qui fuit dans les features | `target_d_plus_1` accessible a `t` | Verifier l'alignement temporel |
| Donnees publiees apres coup | meteo observee ou KPI revise | Donnees point-in-time |

```python
# DO -- pipeline sklearn avec fit sur train uniquement
pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
pipe.fit(X_train, y_train)       # scaler apprend mu/sigma du train
preds = pipe.predict(X_test)     # applique les params du train au test

# DON'T -- normalisation sur tout le dataset
df_scaled = StandardScaler().fit_transform(df_all)  # FUITE
```

### 2.3.b Dataset train/val/test/inference pour TFT

Le dataset train fait foi pour les encoders, normalizers et parametres de reconstruction.

```python
training = TimeSeriesDataSet(...)
validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)

dataset_parameters: dict[str, Any] = training.get_parameters()
inference_ds = TimeSeriesDataSet.from_parameters(
    dataset_parameters,
    future_df,
    predict=True,
)
```

- [ ] `training = TimeSeriesDataSet(train_df, ...)`
- [ ] `validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)`
- [ ] `test = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)`
- [ ] L'inference reconstruit avec `from_parameters()`, jamais avec un nouveau fit.
- [ ] `allow_missing_timesteps=True` n'est autorise que si la semantique business des trous est documentee.

### 2.3.c Normalisation, cold start et categories non vues

- [ ] `GroupNormalizer` par defaut pour la cible.
- [ ] `EncoderNormalizer` uniquement avec justification ecrite.
- [ ] Toute categorie inconnue en prod doit tomber dans un bucket explicite.
- [ ] Le comportement cold start doit etre teste pour `new_product`, `new_site`, `new_series_id`.

### 2.4 Metriques adaptees au forecasting operationnel

Les metriques classiques (MSE, RMSE) ne suffisent pas. Ajouter :

| Metrique | Mesure | Usage |
|---|---|---|
| **WAPE** | Erreur absolue ponderee par le volume | Metrique principale cross-series |
| **MAE** | Erreur absolue moyenne | Lecture directe en unites |
| **Bias** | Sur/sous-prevision moyenne | Controle du drift operationnel |
| **Stockout-weighted error** | Erreur sur jours tendus | Important si la demande latente est visee |
| **Waste-weighted error** | Erreur sur jours de surproduction | Important pour le cout matiere |

```python
def wape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-8) -> float:
    denom: float = float(y_true.abs().sum())
    if denom < eps:
        return float("nan")
    return float(y_true.sub(y_pred).abs().sum() / denom)

def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_pred - y_true).mean())
```

- [ ] Metriques calculees sur le set de test out-of-sample uniquement.
- [ ] Comparer a un baseline naif saisonnier ou last-value, pas seulement a un modele complexe.

### 2.4.b Forecast probabiliste et calibration

- [ ] Le backend TFT doit produire au minimum `P10`, `P50`, `P90`.
- [ ] Les metriques obligatoires deviennent : pinball loss par quantile, couverture empirique des intervalles, largeur moyenne d'intervalle, WAPE, MAE, Bias, cout rupture, cout surstock.
- [ ] Toutes les metriques doivent etre lues par horizon `h=1..H`, pas seulement en agrege.
- [ ] TFT n'est jamais promu seul : il doit battre `Baseline`, un naif saisonnier, et un challenger plus simple.

### 2.5 Calibration et overfitting

```python
# DO -- commencer simple
model = Ridge(alpha=1.0)  # baseline lineaire regulariee

# DON'T -- complexite prematuree
model = ComplexBoostingModel(...)  # surapprentissage probable
```

- [ ] Regularisation systematique (L1/L2, dropout, early stopping).
- [ ] Limiter les features a ~10-20x le nombre d'observations independantes.
- [ ] Un gain massif offline sans robustesse par split temporel = signal d'overfitting.
- [ ] Documenter le nombre de combinaisons testees et les hypotheses de selection.

### 2.6 Feature importance et interpretabilite

- [ ] Calculer l'importance (permutation importance, SHAP) apres chaque entrainement.
- [ ] Eliminer les features negligeables -- elles ajoutent du bruit.
- [ ] Verifier que les features importantes ont un sens economique.

### 2.6.b Interpretabilite propre au TFT

- [ ] Sauvegarder les sorties de `interpret_output()` pour le meilleur checkpoint.
- [ ] Logger les poids de selection de variables par segment et par horizon.
- [ ] Ne jamais presenter attention/importance comme une causalite.

### 2.7 Production training loop Lightning

```python
callbacks = [
    EarlyStopping(
        monitor="val_wape",
        mode="min",
        patience=config.patience,
        strict=True,
        check_finite=True,
    ),
    ModelCheckpoint(
        monitor="val_wape",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="{epoch:03d}-{val_wape:.4f}",
    ),
    LearningRateMonitor(logging_interval="epoch"),
]
```

- [ ] `EarlyStopping` obligatoire sur une metrique metier de validation.
- [ ] `ModelCheckpoint(save_top_k=3, save_last=True)` obligatoire.
- [ ] `LearningRateMonitor` obligatoire.
- [ ] `gradient_clip_val` obligatoire via config.
- [ ] `detect_anomaly=True` seulement en debug.
- [ ] `DeviceStatsMonitor` active sur les runs GPU.

---

## 3. Forecasting demande latente

### 3.1 Demande latente vs ventes observees

La cible ideale du projet est la **demande latente** :

- ce que le client aurait voulu vendre ou servir sans rupture, fermeture partielle ou contrainte de capacite
- ce que l'on cherche a approcher quand les signaux de censure existent

Quand ce n'est pas possible, la cible de repli reste la **vente observee**.

- [ ] Toujours documenter explicitement si la cible d'un run est `demande latente estimee` ou `vente observee`.
- [ ] Ne jamais presenter une prediction de ventes observees comme une prediction de demande latente.

### 3.2 Censure operationnelle

Les signaux suivants doivent etre traites comme des indices de censure potentielle :

- `observed_stockout_flag`
- `location_closed_flag`
- `product_active_flag = False`
- `channel_disabled_flag`
- `kitchen_saturation_flag`
- `assortment_restriction_flag`

- [ ] Si ces signaux existent, les exploiter dans la lecture de la cible et dans l'evaluation.
- [ ] Si ces signaux n'existent pas, documenter que le pipeline predit des ventes observees et non une demande pure.

### 3.2.b Qualite de label et censure

Colonnes canoniques a ajouter ou maintenir dans les datasets de travail :

- `target_semantics` = `latent_demand_estimated` | `observed_sales`
- `censor_flag`
- `target_source`
- `label_quality_score`
- `usable_for_training_flag`

- [ ] Un run `latent demand` ne doit jamais apprendre silencieusement sur des ventes observees censurees sans masque, ponderation ou strategie explicite.
- [ ] Les jours censures doivent etre separes dans l'evaluation.
- [ ] Les agregats de performance doivent toujours distinguer `clean`, `censored`, `estimated-latent`.

### 3.3 Point-in-time data

Les donnees doivent refleter ce qui etait **reellement disponible** au moment de la decision.

```python
# DO -- n'utiliser que les signaux disponibles au moment de la prediction
frame["weather_known_at_decision"] = weather_forecast["temperature"]

# DON'T -- utiliser une observation publiee ou mesuree apres coup comme si elle etait connue
frame["weather_known_at_decision"] = weather_observed["temperature"]
```

### 3.4 Evaluation realiste -- checklist

- [ ] Aucun look-ahead bias.
- [ ] Distinction explicite entre jours normaux et jours censures.
- [ ] Comparaison a des baselines statistiques simples.
- [ ] Reporting separe pour les segments a forte valeur operationnelle.
- [ ] Lecture economique des erreurs : gaspillage, ruptures, cout matiere, niveau de service.

---

## 4. Series temporelles

### 4.1 Stationnarite

La demande brute n'est pas stationnaire ; les variations, residus ou cibles transformees le sont souvent davantage. Tester avant de modeliser.

```python
from statsmodels.tsa.stattools import adfuller, kpss

# ADF : H0 = non-stationnaire. p < 0.05 => stationnaire.
adf_pval = adfuller(series.dropna())[1]

# KPSS : H0 = stationnaire. p < 0.05 => non-stationnaire.
kpss_pval = kpss(series.dropna(), regression="c")[1]
```

| ADF (p < 0.05) | KPSS (p > 0.05) | Conclusion |
|---|---|---|
| Oui | Oui | Stationnaire |
| Non | Non | Non-stationnaire -- differencier |
| Oui | Non | Trend-stationary |
| Non | Oui | Racine unitaire avec break structurel |

### 4.2 Autocorrelation

- L'autocorrelation dans la demande et les ventes est souvent forte a court terme.
- L'autocorrelation dans les residus du modele = specification incomplete ou leakage.
- Les erreurs et les metriques doivent etre lues par regime, pas seulement en moyenne globale.

### 4.3 Feature engineering temporel

```python
# Lags
for lag in [1, 2, 7, 14, 28]:
    df[f"demand_lag_{lag}"] = df.groupby("series_id")["target"].shift(lag)

# Rolling statistics
for w in [7, 14, 28]:
    df[f"demand_mean_{w}d"] = df.groupby("series_id")["target"].transform(
        lambda s: s.rolling(w, min_periods=w).mean()
    )
    df[f"demand_std_{w}d"] = df.groupby("series_id")["target"].transform(
        lambda s: s.rolling(w, min_periods=w).std()
    )

# Calendrier
df["day_of_week"] = df["date"].dt.dayofweek     # 0=lundi
df["month"] = df["date"].dt.month                # effet janvier
df["quarter_end"] = df["date"].dt.is_quarter_end
```

- [ ] Toujours `min_periods=window` dans les rolling.
- [ ] Toujours `shift(1)` le target ou les features derivees du target.
- [ ] Features de calendrier = jours operationnels reels, pas interpretation calendaire naive.

### 4.4 Jours feries et weekends

Le projet doit raisonner avec le calendrier operationnel reel du site, pas seulement avec un calendrier ouvrable generique.

```python
# DO -- calendrier metier precis
site_calendar = build_site_calendar(site_id, start, end)

# DON'T
index = pd.date_range(start, end)  # ignore ouvertures, fermetures et jours speciaux
```

### 4.5 Frequence et alignement temporel

```python
# Merger des donnees de frequences differentes (demandes journalieres + macro mensuelle)
macro["date"] = macro["date"] + pd.offsets.MonthEnd(0)
df = df.merge(macro, on="date", how="left")
df["macro_feat"] = df.groupby("series_id")["macro_feat"].ffill()

# DON'T -- merger sans aligner => perd 95% des lignes
df = prices.merge(macro, on="date")
```

### 4.6 Saisonnalite

- La saisonnalite dans la demande est souvent forte et utile, surtout en alimentaire perissable.
- Les patterns jour de semaine, vacances, fériés, météo et événements sont prioritaires.
- Tester la significativite statistique avant d'integrer une composante saisonniere.

---

## 5. Regles specifiques au projet

### 5.0 Etat courant du repo

- Le repo est centré sur la prevision de demande Praedixa.
- La couche canonique est le pipeline medaillon `bronze -> silver -> gold` dans `platform/warehouse/`.
- Les points d'entree locaux actifs sont :
  1. `apps/warehouse/run_silver/main.py`
  2. `apps/warehouse/run_gold/main.py`
  3. `apps/platform/build_global_dataset/main.py`
- `features_selection_lag` reste utile pour la preparation et l'analyse locale, mais ce n'est pas la vision finale du backend modele.
- `optimisation` et `evaluation` doivent etre pensees pour un backend TFT unique, meme si ce backend n'est pas encore branche completement. Tant que ce backend n'est pas connecte, ces etapes doivent se comporter comme des placeholders explicites.
- La cible ideale reste la demande latente. Quand elle n'est pas reconstructible proprement, le repo doit parler explicitement de ventes observees.

### 5.1 Convention de nommage des fichiers

```
var/{etape}/{description}.{format}

Exemples :
  var/warehouse/praedixa.duckdb
  var/datasets/global_dataset/commercial_external_daily.parquet
  var/experiments/demand_forecast/feature_screening/train_selection_70_selected.parquet
```

- [ ] Minuscules, underscores. Plage de dates dans le nom. Parquet principal, CSV pour debug.

### 5.2 Structure des DataFrames

Le panel principal suit un format long (tidy) :

| Colonne | Type | Description |
|---|---|---|
| `dt` | `datetime64[ns]` | Date de decision |
| `series_id` | `str` | Identifiant serie site x produit |
| `target_demand_qty_d_plus_1` | `float64` | Cible absolue D+1 |

Colonnes frequentes ensuite : `location_id`, `product_id`, `lag_1`, `lag_7`, `rolling_mean_7`, `holiday_flag`, `weather_temperature`, `observed_stockout_flag`.

```python
class SchemaValidationError(ValueError):
    """Erreur de validation de schema pour les DataFrames canoniques."""


EXPECTED_DTYPES: dict[str, str] = {
    "dt": "datetime64[ns]",
    "series_id": "object",
    "target_demand_qty_d_plus_1": "float64",
}


def validate_schema(df: pd.DataFrame) -> None:
    missing: list[str] = [col for col in EXPECTED_DTYPES if col not in df.columns]
    if missing:
        raise SchemaValidationError(f"Colonnes manquantes: {missing}")

    bad_types: dict[str, tuple[str, str]] = {
        col: (str(df[col].dtype), expected)
        for col, expected in EXPECTED_DTYPES.items()
        if str(df[col].dtype) != expected
    }
    if bad_types:
        raise SchemaValidationError(f"Dtypes invalides: {bad_types}")
```

- [ ] Ne jamais utiliser l'index pour stocker des donnees (toujours `reset_index`).

### 5.2.b Bundle d'artefacts, securite et serialisation

Le bundle minimal d'un run promouvable doit etre :

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

- [ ] Chargement par defaut via `state_dict` / `weights_only=True`.
- [ ] Jamais de refit d'encoder au chargement.
- [ ] Signatures d'entree/sortie versionnees.
- [ ] Si modele pre-entraine/fine-tune : stocker aussi `parent_checkpoint_id`, couches gelees, strategie de fine-tuning`.

### 5.3 Gestion de la censure et des contraintes

Le repo doit distinguer autant que possible :

- demande latente
- ventes observees
- ventes censurees par rupture, fermeture ou contrainte operationnelle

- [ ] Utiliser les flags de censure quand ils existent.
- [ ] Documenter quand une cible est seulement une vente observee.
- [ ] Ne pas melanger silencieusement des regimes normaux et censes dans l'analyse.

### 5.3.b Serving, inference et performance

- [ ] `model.eval()` + `torch.inference_mode()` obligatoires en prediction.
- [ ] Benchmark obligatoire entre eager, AMP, et eventuellement `torch.compile`.
- [ ] Export seulement apres test de parite numerique.
- [ ] Toute route d'inference a un fallback baseline si contrat de features casse.

### 5.4 Activation des pipelines via `main.py`

**Chaque module/etape du pipeline doit avoir un fichier `main.py`** qui sert de point d'entree unique. Ce fichier doit etre directement executable en cliquant dessus dans l'IDE (pas besoin de terminal).

```python
# platform/python/src/praedixa/platform/datasets/standardization/pipeline.py

def main() -> None:
    build_global_daily_standardization()
```

```python
# apps/platform/build_global_dataset/main.py

from praedixa.platform.datasets.standardization.main import main

if __name__ == "__main__":
    main()
```

- [ ] **Tout** pipeline s'active via son `main.py`, jamais en important directement un module.
- [ ] Le `sys.path` n'est autorise que dans un wrapper mince si l'execution locale le rend necessaire.
- [ ] Le `if __name__ == "__main__"` est obligatoire.

### 5.4.c Packaging et entrypoints

- [ ] Les hacks `sys.path` sont tolerables uniquement dans des wrappers minces, jamais dans la logique importable sous `src/`.
- [ ] La logique metier vit dans `pipeline.py`, `service.py` ou `app.py`.
- [ ] `__main__.py` et `apps/` servent uniquement de wrappers d'execution.

### 5.4.b Modularite stricte du pipeline

Le fichier `main.py` doit rester un **orchestrateur**: chargement, enchainement des etapes, sauvegarde.
Toute logique metier (cleaning, outliers, feature engineering, validation schema) doit vivre dans des modules dedies.

```python
# DO -- main orchestration only
from praedixa.platform.datasets.standardization.outlier_pipeline import apply_outlier_flags

def main() -> None:
    df = load_raw_dataset(INPUT_PATH)
    cleaned = apply_outlier_flags(df)
    save_cleaned(cleaned, OUTPUT_PATH, SAMPLE_PATH)

# DON'T -- logique metier lourde dans main.py
def main() -> None:
    ...  # 300+ lignes de logique outliers, rules, rolling stats, etc.
```

- [ ] `main.py` < 150 lignes, sans regles metier complexes.
- [ ] Un module metier par responsabilite (ex: `outlier_pipeline.py`, `nan_policy.py`).
- [ ] Les tests suivent le miroir des modules (`test_outlier_pipeline.py`, etc.).

### 5.5 Fichiers `paths.py` et `constants.py`

Ces fichiers se placent **a la racine du module data actif** quand ils deviennent necessaires. Pour le repo actuel, la logique de chemins vit surtout dans `platform/python/src/praedixa/platform/runtime/` et dans les modules produit sous `products/demand_forecast/src/praedixa/demand_forecast/`.

**`paths.py`** -- centralise tous les chemins du pipeline.

```python
# platform/python/src/praedixa/platform/runtime/paths.py

from __future__ import annotations
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[6]
VAR_DIR: Path = PROJECT_ROOT / "var"
WAREHOUSE_DIR: Path = VAR_DIR / "warehouse"
GLOBAL_DATASET_DIR: Path = VAR_DIR / "datasets" / "global_dataset"
FEATURE_SELECTION_DIR: Path = VAR_DIR / "experiments" / "demand_forecast" / "feature_screening"
```

**`constants.py`** -- centralise les constantes metier reutilisables.

```python
# platform/python/src/praedixa/platform/runtime/constants.py

from __future__ import annotations

DEFAULT_TARGET_COL: str = "target_demand_qty_d_plus_1"
DEFAULT_DATE_COL: str = "dt"
SAMPLE_FRAC: float = 0.05
RANDOM_SEED: int = 7
MAX_RETRIES: int = 3
RETRY_SLEEP: float = 2.0
```

- [ ] **Jamais** de chemins en dur dans les fonctions -- tout passe par `paths.py`.
- [ ] **Jamais** de constantes magiques dans le code -- tout passe par `constants.py`.
- [ ] Les autres fichiers du module importent depuis `paths` et `constants`.

```python
# DO
from praedixa.platform.runtime.paths import GLOBAL_DATASET_DIR
from praedixa.platform.runtime.constants import MAX_RETRIES

# DON'T
output_path = Path("var/datasets/global_dataset/commercial_external_daily.parquet")  # chemin en dur
max_retries = 3  # constante magique
```

### 5.6 Tests unitaires -- architecture miroir

Les tests dans `tests/` doivent suivre au plus près l'arborescence logique des packages sous `platform/python/src/` et `products/demand_forecast/src/` :

```
platform/python/src/praedixa/platform/datasets/standardization/pipeline.py
tests/platform/datasets/standardization/test_pipeline.py

products/demand_forecast/src/praedixa/demand_forecast/training/pipeline.py
tests/products/demand_forecast/training/test_pipeline.py

products/demand_forecast/src/praedixa/demand_forecast/evaluation/pipeline.py
tests/products/demand_forecast/evaluation/test_pipeline.py
```

- [ ] Un fichier source = un fichier de test correspondant.
- [ ] Le fichier de test est nomme `test_{nom_du_fichier_source}.py`.
- [ ] Ecrire les tests **immediatement** apres l'ecriture d'une fonction ou d'un fichier, pas apres coup.

### 5.6.b CI, lint, typing et tests de non-regression

- [ ] Interdire `# type: ignore` nu.
- [ ] Autoriser uniquement `# type: ignore[code]` avec justification courte et ticket.
- [ ] Activer `warn-unused-ignores`.
- [ ] Ajouter un test anti-leakage sur toutes les features `known future`.
- [ ] Ajouter un test round-trip checkpoint => meme prediction.
- [ ] Ajouter un test `from_parameters()` => meme encodage que le train.
- [ ] Ajouter un test cold start categories non vues.
- [ ] Ajouter un test series vides / all-zero / trous de calendrier.
- [ ] Ajouter un test non-regression des metriques sur un mini backtest fige.

Commandes de gate :

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict platform/python/src products/demand_forecast/src
.venv/bin/python -m unittest discover -s tests
```

### 5.7 Tests directement executables

Chaque fichier de test doit etre executable de **deux facons** : individuellement (clic dans l'IDE) et via `unittest`.

```python
# tests/platform/datasets/standardization/test_pipeline.py

import sys
from pathlib import Path

# Remonter jusqu'a la racine du projet
PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
import pandas as pd

from praedixa.demand_forecast.contracts.targets import ensure_learning_target_column


class ExamplePipelineTests(unittest.TestCase):
    def test_learning_target_column_is_created(self) -> None:
        ...

    def test_empty_price_map_raises(self) -> None:
        # Exemple avec config qui pointe vers un CSV vide
        with self.assertRaisesRegex(RuntimeError, "No price data"):
            ...


# --- Execution directe (clic dans l'IDE) ---
if __name__ == "__main__":
    unittest.main()
```

- [ ] Framework : `unittest`. Lancer tout avec `.venv/bin/python -m unittest discover -s tests`.
- [ ] Lancer un seul fichier : clic direct ou `.venv/bin/python tests/platform/datasets/standardization/test_pipeline.py`.
- [ ] Le bloc `if __name__ == "__main__": unittest.main()` est **obligatoire** dans chaque fichier de test.
- [ ] Le `sys.path` en haut du fichier est **obligatoire** pour l'execution directe.
- [ ] Fixtures legeres dans `conftest.py` ou dans les tests.
- [ ] **Jamais** d'appels API reels dans les tests.

### 5.8 Monitoring prod, registry et promotion

- [ ] `null rate` et `out-of-range` par feature.
- [ ] `unseen category rate`.
- [ ] `stale known-future covariates rate`.
- [ ] WAPE / Bias par horizon et segment.
- [ ] Couverture empirique P50/P90 quand les labels arrivent.
- [ ] `p95 latency`, taux d'echec, taille de batch, memoire GPU/CPU.
- [ ] Un alias `candidate`.
- [ ] Un alias `champion`.
- [ ] Rollback possible sans refit.

### 5.9 Model card, datasheet et coherence hierarchique

- [ ] Le model card doit contenir `target_semantics`, fenetre d'entrainement, segments/horizons evalues, performance `clean` vs `censored`, covariables `known` vs `observed`, limites connues et fallback.
- [ ] Si les decisions existent a plusieurs niveaux (`SKU`, `site`, `categorie`, `reseau`), le repo doit soit reconcilier les previsions, soit documenter explicitement pourquoi il ne le fait pas.
- [ ] Les metriques de promotion doivent etre calculees aussi aux niveaux d'agregation reellement utilises par l'operationnel.

---

## Checklist avant chaque commit

```
[ ] Les tests passent (.venv/bin/python -m unittest discover -s tests)
[ ] Pas de print() en dehors des notebooks
[ ] Les seeds sont fixes et passent par la config
[ ] Les nouvelles features utilisent min_periods et shift correctement
[ ] Le schema du DataFrame de sortie est documente et valide
[ ] Le logging trace le nombre de lignes a chaque etape
[ ] Les fichiers de donnees ne sont pas commites dans git
[ ] Si le changement touche le pipeline medaillon, `apps/warehouse/run_silver/main.py` et `apps/warehouse/run_gold/main.py` ont ete verifies
[ ] Chaque nouveau fichier source a son fichier de test miroir
[ ] Les main.py et fichiers de test sont directement executables (sys.path + __main__)
[ ] Les chemins passent par paths.py, les constantes par constants.py
```

## Verification

Pour une modification Python non triviale, utiliser :

```bash
.venv/bin/python -m compileall praedixa platform/python/src products/demand_forecast/src apps
.venv/bin/python -m unittest discover -s tests
```

Pour verifier le warehouse local :

```bash
dbt parse --project-dir platform/warehouse --profiles-dir platform/warehouse
```

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

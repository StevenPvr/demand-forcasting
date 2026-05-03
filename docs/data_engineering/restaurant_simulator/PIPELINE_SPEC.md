# Pipeline : demande latente → ventes observées → exports

## Références code

- Ordre structuré : [`pipeline.py`](../../../../platform/python/src/praedixa/platform/datasets/synthetic_foodservice/spec/pipeline.py) (`PIPELINE_STEPS`)
- Tolérance d’arrondi ticket/lignes par défaut : **0,02 €** (`ROUNDING_TOLERANCE_EUR_DEFAULT`)

## Ordre nominal (cascade)

Les étapes **1–10** construisent la vérité simulée puis les **coquilles** ; les étapes **11–13** matérialisent les trois couches d’export (voir [EXPORT_LAYERS.md](./EXPORT_LAYERS.md)).

1. **Intervalles ouvrables** — croisement `opening_hours`, `closures`, fuseaux et DST. Produire une distinction explicite entre **jour sans vente** (établissement fermé), **export_gap** (pas de fichier / coupure POS) et **ventes nulles** (ouvert mais flux nul).

2. **Grille exogène** — attacher météo, événements, indicateurs `holidays` à chaque cellule temporelle simulée.

3. **Intensité latente** — modèle log-multiplicatif sur \(\lambda\) site puis ventilation produit / canal : baseline × saisonnalités × météo × événements × prix relatifs × promos × cycle de vie.

4. **Tirages latents** — Poisson / binomiale négative / ZINB ; composition panier (menus, compléments). Remplissage de `latent_demand_internal` (**ne jamais écraser** après cette étape).

5. **Capacité staff** — agrégation `staff_schedules` → plafonds cuisine et salle ; files d’attente et **abandons** (réduction de débit observé sans toucher \(D^{\text{latent}}\)).

6. **Évolution stocks** — mouvements théoriques : réceptions, `production_batches`, consommation latente **proposée**, `waste` anticipé, ajustements.

7. **Ruptures et substitution** — clip des ventes à la disponibilité ; `stockouts` ; réaffectation vers un substitut du même `substitution_group_id` ; pertes en `estimated_lost_sales_internal` si pas de substitut.

8. **Prix et promotions** — appliquer SCD2 `prices` et fenêtres `promotions` ; cannibalisation éventuelle déjà reflétée en amont sur \(\lambda\) ou ici sur allocations ligne selon règle de scénario.

9. **Erreurs caisse** — bruit humain (quantités ±1, lignes oubliées, `promotion_id` erroné).

10. **Invendus et snapshots** — finaliser `waste` ; éventuels `stock_snapshots` en fin de journée.

11–13. **Exports** — `internal_clean`, `ops_realistic`, `pos_synth_export`.

## Identités de conservation

### Stock (hors erreur d’inventaire volontaire)

\[
S_{t^+} = S_{t^-} + \text{Inbound} + \text{Produit} - \text{Vendu}_{\text{obs}} - \text{Gaspillage} - \text{Ajustements}
\]

### Censure avec substitution

Pour un groupe de substitution \(g\), en première approximation :

\[
\sum_{p \in g} D^{\text{obs}}_{p} \;\le\; \sum_{p \in g} D^{\text{latent}}_{p} + \varepsilon
\]

où \(\varepsilon\) est du bruit opérationnel (vols non enregistrés, dons, etc.) si le scénario l’active.

### Cohérence ticket

\[
\sum_{\text{lignes}} \text{line\_amount\_ttc} \approx \text{total\_amount\_ttc} - \text{total\_discount}
\]

Les violations intentionnelles doivent porter `correction_flag` ou un motif dans `data_anomaly_flags`.

## Implémentation Python suggérée

Modules futurs (non exigés par ce dépôt aujourd’hui mais alignés sur ce plan) :

- `calendar.py` — étape 1
- `latent.py` — étapes 3–4
- `constraints.py` — étapes 5–7
- `inventory.py` — étapes 6, 10
- `export_corrupt.py` — étape 13

Chaque module prend des tables **immutables en entrée** et produit de nouvelles frames ; les grain commits journaliers ou 15 min restent paramétrables via `simulation_parameters`.

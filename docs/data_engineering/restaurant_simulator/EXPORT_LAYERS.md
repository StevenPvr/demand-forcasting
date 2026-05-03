# Export layers & corruption (MCAR / MAR)

## Source normative

- Paramètres chiffrés : [`export_layers.yaml`](../../../platform/python/src/praedixa/platform/datasets/synthetic_foodservice/spec/export_layers.yaml)
- Chargement : `load_export_layers()` depuis `praedixa.platform.datasets.synthetic_foodservice.spec.loader`

## Les trois couches

| Couche | Usage | Corruption |
|--------|--------|------------|
| `internal_clean` | Audit simulateur, invariants stricts, diagnostic substitution / censure | Aucune volontaire sur les timestamps et montants ; `latent_demand_internal` dans un **bundle séparé** |
| `ops_realistic` | Comportement back-office crédible (écarts d’inventaire, ruptures partiellement détectées) | Légère ; encore exploitable pour contrôle de gestion |
| `pos_synth_export` | « Export client » CSV/Parquet | Défauts MCAR/MAR selon `sites.data_quality_level` |

## Profils `data_quality_level`

Les taux du YAML sont des **cibles** (probabilités annuelles par cellule ou par ticket selon la clé) :

- **high** — intégration correcte, erreurs résiduelles humaines.
- **medium** — promos mal taguées, quelques trous d’export.
- **medium_low** — ponts Excel, DST, promos manquantes fréquentes.
- **low** — migrations POS, IDs instables, gros trous.

## MCAR vs MAR

- **MCAR** : `missing_rate_cell.mcar` — manquants indépendants des valeurs sous-jacentes (ex. fichier tronqué).
- **MAR** : combiné avec `mar_weight` et les **crochetés** listés sous `mar_hooks` dans le YAML (ex. météo plus absente en village si qualité basse ; trous sur jours de fermeture mal déclarée).

## Techniques de dégradation (liste opérationnelle)

- Doublons de `ticket_id` avec micro-décalage temporel.
- Timestamps local/UTC inversés ; passages heure d’été mal gérés.
- `promotion_id` NULL malgré remise sur la ligne.
- `product_id` : application incohérente du mapping entre fichiers (`product_id_mappings`).
- Journées partiellement exportées (`partial_export_day_rate`).
- Tickets `cancelled_flag` mais lignes résiduelles.

L’implémentation doit journaliser le **tirage** des corruptions (seed dérivée) dans `simulation_parameters` pour rejouer le même export sale.

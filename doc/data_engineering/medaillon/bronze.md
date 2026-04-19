# Couche Bronze Praedixa

## Statut actuel

Aujourd'hui, Praedixa n'a **pas encore** de vraie pipeline `raw JSON -> bronze tabulaire` en production.

Les API et formats JSON des logiciels de caisse ne sont pas encore connus ou stabilises.

Donc, dans l'etat actuel du repo :

- il n'y a pas encore de couche bronze de production au sens "ingestion POS reelle"
- il existe en revanche une **zone bronze technique** dans DuckDB
- cette zone bronze est alimentee a partir des fichiers locaux dans `data/`
- cette zone bronze sert de **substitut de bronze tabulaire** pour developper la silver

La formulation la plus juste est donc :

- **pas encore de vraie pipeline bronze**
- **oui a une base/schema bronze locale de substitution**

## Cible d'architecture

La cible reste :

- `raw_landing`
  - payloads JSON bruts
  - stockage immutable pour audit et replay
- `bronze`
  - projection tabulaire proche source
  - flatten des JSON
  - typage technique minimal
  - fidelite maximale a la source
- `silver`
  - normalisation metier
  - standardisation des schemas
  - flags qualite, anomalies, manquants
  - consolidation au grain journalier
- `gold`
  - panel de modelling
  - jointures de signaux
  - feature engineering leak-safe

## Ce qu'on a reellement aujourd'hui

Pour la V1 locale :

- les datasets sous `data/` jouent le role d'inputs bronze
- `scripts/load_bronze_duckdb.py` charge ces donnees dans un schema `bronze` DuckDB
- la silver SQL lit ensuite ce schema `bronze`

Sources actuellement chargees en bronze de substitution :

- `data/bakery_sales/data_train.parquet`
- `data/bakery_sales/data_val.parquet`
- `data/bakery_sales/Bakery sales.csv`
- `data/m5/calendar.csv`
- `data/m5/sell_prices.csv`
- `data/m5/sales_train_validation.csv`

Tables bronze de substitution :

- `bronze.bronze_freshretail_daily`
- `bronze.bronze_bakery_order_lines`
- `bronze.bronze_m5_calendar`
- `bronze.bronze_m5_sell_prices`
- `bronze.bronze_m5_sales_long`

## Pourquoi cette approche est saine

Cette approche permet de :

- coder la silver tout de suite
- figer un contrat tabulaire que la silver sait lire
- ne pas bloquer l'architecture sur des APIs POS encore inconnues
- separer clairement :
  - le probleme d'ingestion JSON
  - le probleme de standardisation silver

Autrement dit :

- la **forme finale de la vraie bronze** reste ouverte
- le **contrat tabulaire lu par la silver** existe deja

## Regle de travail a retenir

Tant que les APIs POS ne sont pas connues :

- on ne pretend pas avoir une vraie bronze de production
- on travaille avec une **bronze database/schema de substitution**
- la silver doit etre ecrite comme si elle lisait une vraie bronze tabulaire

Quand les POS seront connus plus tard, il faudra seulement ajouter :

- la brique `raw JSON -> bronze tabulaire`

La silver, elle, ne devra pas etre repensee en profondeur si le contrat bronze tabulaire reste coherent.

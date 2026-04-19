PYTHON := .venv/bin/python
CONFIG := config/distributed.yaml

.PHONY: setup-pro setup-air sync-air start-postgres start-cluster stop-cluster cluster-status smoke-distributed prepare-distributed feature-select-distributed hpo-distributed backtest-distributed train-distributed predict-distributed check

setup-pro:
	bash scripts/distributed/bootstrap_pro.sh

setup-air:
	ssh steven@Macbook-Air-de-Steven.local 'cd /Users/steven/Programmation/research_praedixa && bash scripts/distributed/bootstrap_air.sh' || \
	ssh steven@10.188.88.36 'cd /Users/steven/Programmation/research_praedixa && bash scripts/distributed/bootstrap_air.sh'

sync-air:
	$(PYTHON) scripts/distributed/sync_to_air.py --config $(CONFIG)

start-postgres:
	$(PYTHON) scripts/distributed/configure_postgres.py --config $(CONFIG)

start-cluster:
	$(PYTHON) scripts/distributed/start_cluster.py --config $(CONFIG)

stop-cluster:
	$(PYTHON) scripts/distributed/stop_cluster.py --config $(CONFIG)

cluster-status:
	$(PYTHON) scripts/distributed/cluster_status.py --config $(CONFIG)

smoke-distributed:
	$(PYTHON) scripts/distributed/smoke_test_cluster.py --config $(CONFIG)

prepare-distributed:
	PYTHONPATH=src $(PYTHON) -m research_praedixa.distributed.cli --config $(CONFIG) prepare --sync-to-air

feature-select-distributed:
	PYTHONPATH=src $(PYTHON) -m research_praedixa.distributed.cli --config $(CONFIG) feature-select

hpo-distributed:
	PYTHONPATH=src $(PYTHON) -m research_praedixa.distributed.cli --config $(CONFIG) hpo

backtest-distributed:
	PYTHONPATH=src $(PYTHON) -m research_praedixa.distributed.cli --config $(CONFIG) backtest

train-distributed: hpo-distributed backtest-distributed

predict-distributed:
	PYTHONPATH=src $(PYTHON) -m research_praedixa.distributed.cli --config $(CONFIG) predict --input-path data/distributed_runtime/evaluation/test.parquet

check:
	PYTHONPATH=src $(PYTHON) -m unittest tests.test_memory_utils tests.test_optimisation_pipeline tests.test_distributed_config

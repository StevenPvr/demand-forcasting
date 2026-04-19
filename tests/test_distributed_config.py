from __future__ import annotations

import unittest

from research_praedixa.distributed.budgets import build_runtime_budget
from research_praedixa.distributed.config import load_distributed_config


class DistributedConfigTests(unittest.TestCase):
    def test_load_distributed_config_resolves_known_hosts(self) -> None:
        config = load_distributed_config("config/distributed.yaml")
        self.assertEqual(config.scheduler.host, "MacBook-Pro-de-Steven.local")
        self.assertEqual(config.air_workers.ssh_host, "steven@Macbook-Air-de-Steven.local")
        self.assertEqual(config.air_workers.fallback_host, "steven@10.188.88.36")
        self.assertTrue(str(config.project.resolve_python_bin()).endswith(".venv/bin/python"))

    def test_runtime_budget_matches_worker_pools(self) -> None:
        config = load_distributed_config("config/distributed.yaml")
        budget = build_runtime_budget(config)
        self.assertEqual(budget.pro_task_slots, 3)
        self.assertEqual(budget.air_task_slots, 2)
        self.assertEqual(budget.cluster_task_slots, 5)
        self.assertEqual(budget.optuna_parallel_tasks, 5)
        self.assertEqual(budget.xgboost_threads_per_task, 2)


if __name__ == "__main__":
    unittest.main()

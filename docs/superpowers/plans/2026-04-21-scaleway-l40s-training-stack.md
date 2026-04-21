# Scaleway L40S Training Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mettre en place une chaîne complète, reproductible et performante pour préparer les datasets, optimiser les hyperparamètres TFT et entraîner/évaluer le modèle sur une instance Scaleway `L40S-1-48G`.

**Architecture:** Le plan sépare strictement 1) la construction de données canoniques et des bundles d’entraînement, 2) la politique runtime GPU/TFT, 3) l’optimisation hyperparamétrique single-GPU, 4) la persistance/traçabilité des artefacts. Les jobs GPU ne doivent plus reconstruire `bronze/silver/gold` ni recalculer inutilement les bundles à chaque essai Optuna.

**Tech Stack:** Python 3.13, `uv`, PyTorch, Lightning, `pytorch-forecasting`, DuckDB, Parquet, Optuna, TensorBoard, NVIDIA CUDA sur Scaleway L40S.

---

## Execution Strategy (User Preference Locked)

Le plan doit etre execute en **deux phases strictes** :

### Phase A - Tout le code en local d'abord

Objectif :
- faire **tous** les changements de code localement
- stabiliser les tests
- valider le contrat dataset / TFT / HPO / artefacts
- ne **rien** configurer sur Scaleway tant que le logiciel n'est pas pret

Ce qui est autorise dans cette phase :
- nouvelles commandes `apps/`
- nouveaux modules Python
- refactor runtime TFT
- redesign HPO single-GPU
- bundles d'entrainement
- artefacts / checkpoints / loaders / runbooks
- tests unitaires et tests de non-regression locaux

Ce qui est interdit dans cette phase :
- bootstrap machine distante
- installation distante
- benchmark sur la L40S
- ajustements ad hoc "pour que ca marche sur le serveur"

### Phase B - Configuration Scaleway a la toute fin

Objectif :
- une fois le code fige localement, seulement alors :
  - ouvrir l'instance
  - connecter VS Code en Remote SSH
  - cloner/synchroniser le repo
  - verifier GPU/CUDA
  - lancer le premier run reel

Regle :
- **aucune decision d'architecture ne doit dependre du serveur avant la fin**
- la machine distante ne sert qu'a executer un pipeline deja pret

---

## Current State

- Le backend TFT existe déjà dans [`products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py`](../../../products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py).
- Le pipeline `training` produit aujourd’hui des artefacts d’optimisation (`best_params`, rapports Optuna, métadonnées), mais pas encore une chaîne remote/GPU robuste et benchmarkée.
- Le backend TFT est aujourd’hui structuré pour CPU par défaut (`accelerator="cpu"`, `devices=1`) dans [`model_utils.py`](../../../products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py).
- Le repo n’a pas encore de couche `ops/scaleway/` ni de runbook GPU.
- Le `feature_screening` et les orchestrateurs `training/evaluation` existent déjà et doivent devenir les consommateurs de bundles versionnés, pas des producteurs ad hoc de données à chaque run.

## Scaleway Facts vs Assumptions

### Facts

- Instance cible : `L40S-1-48G`
- Zone : `PAR 2`
- CPU : `8 cores`
- RAM : `96 GB`
- GPU : `1 x NVIDIA L40S`, `48 GB VRAM`
- IP publique : `51.159.162.168`
- Commande SSH : `ssh root@51.159.162.168`
- Image : `Ubuntu Noble GPU OS 13 passthrough`

### Assumptions to Verify First

- Le driver NVIDIA et CUDA sont déjà installés et fonctionnels sur l’image Scaleway.
- Les deux volumes sont correctement montés ou montables pour séparer code, datasets, checkpoints et logs.
- Le pare-feu / security group autorise au minimum SSH et, si retenu, TensorBoard via tunnel SSH uniquement.

## File Map

### Create

- `ops/scaleway/bootstrap_l40s.sh`
- `ops/scaleway/check_gpu_env.sh`
- `ops/scaleway/run_training_remote.sh`
- `ops/scaleway/sync_training_bundle.sh`
- `ops/scaleway/README.md`
- `apps/demand_forecast/build_training_bundle/main.py`
- `products/demand_forecast/src/praedixa/demand_forecast/training_bundle/main.py`
- `products/demand_forecast/src/praedixa/demand_forecast/training_bundle/bundle_builder.py`
- `products/demand_forecast/src/praedixa/demand_forecast/training_bundle/manifest.py`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/runtime_profile.py`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/system_info.py`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/dataloader_profile.py`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/checkpoint_io.py`
- `tests/products/demand_forecast/training_bundle/test_bundle_builder.py`
- `tests/products/demand_forecast/backends/tft/test_runtime_profile.py`
- `tests/products/demand_forecast/backends/tft/test_checkpoint_io.py`
- `docs/ops/scaleway-l40s-runbook.md`

### Modify

- `apps/demand_forecast/run_training/main.py`
- `apps/demand_forecast/run_evaluation/main.py`
- `products/demand_forecast/src/praedixa/demand_forecast/training/main.py`
- `products/demand_forecast/src/praedixa/demand_forecast/training/orchestrator.py`
- `products/demand_forecast/src/praedixa/demand_forecast/training/tuning.py`
- `products/demand_forecast/src/praedixa/demand_forecast/training/constants.py`
- `products/demand_forecast/src/praedixa/demand_forecast/evaluation/orchestrator.py`
- `products/demand_forecast/src/praedixa/demand_forecast/evaluation/modeling.py`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py`
- `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/artifacts.py`
- `tests/products/demand_forecast/backends/tft/test_model_utils.py`
- `tests/products/demand_forecast/training/test_pipeline.py`
- `tests/products/demand_forecast/evaluation/test_pipeline.py`

---

### Task 1: Bootstrap and Validate the Scaleway L40S Host (Final Phase Only)

**Files:**
- Create: `ops/scaleway/bootstrap_l40s.sh`
- Create: `ops/scaleway/check_gpu_env.sh`
- Create: `ops/scaleway/README.md`
- Test: manual checks over SSH

- [ ] **Step 1: Write the host bootstrap script**

Script responsibilities:
- apt update/upgrade minimal packages
- install `git`, `curl`, `build-essential`, `tmux`, `htop`, `nvtop`, `jq`
- install `uv` if absent
- create directories:
  - `/srv/praedixa/app`
  - `/srv/praedixa/data`
  - `/srv/praedixa/artifacts`
  - `/srv/praedixa/logs`

- [ ] **Step 2: Write the GPU environment check script**

Checks to include:
- `nvidia-smi`
- `python -c "import torch; print(torch.cuda.is_available())"`
- CUDA device name, device count, total VRAM
- BF16 support / CUDA capability if available
- mounted volumes via `lsblk` and `df -h`

- [ ] **Step 3: Verify the host manually**

Run:

```bash
ssh root@51.159.162.168
bash /srv/praedixa/app/ops/scaleway/check_gpu_env.sh
```

Expected:
- `nvidia-smi` returns one L40S
- `torch.cuda.is_available()` is `True`
- volume layout is visible and writable

- [ ] **Step 4: Document the verified environment**

Add to `ops/scaleway/README.md`:
- exact driver version
- exact CUDA runtime version
- exact `torch` install command used
- mount points retained for datasets and artifacts

- [ ] **Step 5: Commit**

```bash
git add ops/scaleway/bootstrap_l40s.sh ops/scaleway/check_gpu_env.sh ops/scaleway/README.md
git commit -m "ops: add scaleway l40s bootstrap and gpu checks"
```

---

### Task 2: Materialize a Versioned Training Bundle Before Any GPU Job

**Files:**
- Create: `apps/demand_forecast/build_training_bundle/main.py`
- Create: `products/demand_forecast/src/praedixa/demand_forecast/training_bundle/main.py`
- Create: `products/demand_forecast/src/praedixa/demand_forecast/training_bundle/bundle_builder.py`
- Create: `products/demand_forecast/src/praedixa/demand_forecast/training_bundle/manifest.py`
- Test: `tests/products/demand_forecast/training_bundle/test_bundle_builder.py`

- [ ] **Step 1: Write the failing tests for bundle creation**

Validate that a bundle contains:
- `train.parquet`
- `tuning.parquet`
- `valid.parquet` or `test_reference.parquet` when needed
- `feature_manifest.json`
- `target_contract.json`
- `bundle_manifest.json`
- SHA-256 hashes and row counts

- [ ] **Step 2: Implement a deterministic bundle builder**

Behavior:
- read only from canonical outputs (`gold` table or selected parquet inputs)
- freeze exact split boundaries and feature list
- coerce numerics to `float32` when safe
- keep categoricals as strings / categoricals
- never rebuild `bronze/silver/gold` inside the GPU training loop

- [ ] **Step 3: Add a thin app entrypoint**

`apps/demand_forecast/build_training_bundle/main.py` should call the bundle builder and print the artifact paths.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m unittest tests.products.demand_forecast.training_bundle.test_bundle_builder
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add apps/demand_forecast/build_training_bundle/main.py products/demand_forecast/src/praedixa/demand_forecast/training_bundle tests/products/demand_forecast/training_bundle/test_bundle_builder.py
git commit -m "feat: add versioned training bundle builder"
```

---

### Task 3: Add Explicit GPU Runtime Profiles for Single-L40S Training

**Files:**
- Create: `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/runtime_profile.py`
- Create: `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/dataloader_profile.py`
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py`
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/training/constants.py`
- Test: `tests/products/demand_forecast/backends/tft/test_runtime_profile.py`

- [ ] **Step 1: Write failing tests for GPU profile resolution**

Cases:
- CPU debug profile
- Scaleway L40S training profile
- unsupported accelerator configuration rejected cleanly

- [ ] **Step 2: Implement runtime profiles**

Recommended L40S defaults:
- `accelerator="gpu"`
- `devices=1`
- `precision="bf16-mixed"` if supported, fallback explicit path if not
- `torch.set_float32_matmul_precision("high")`
- `num_workers=4` or `6` (bounded by 8 vCPU)
- `pin_memory=True`
- `persistent_workers=True` only when `num_workers > 0`
- `batch_size` profile ranges for benchmark, not hardcoded as final truth

- [ ] **Step 3: Keep `torch.compile()` behind a measured flag**

Do not enable by default before benchmarking. Add a boolean runtime flag and require a benchmark report before defaulting it on.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m unittest tests.products.demand_forecast.backends.tft.test_runtime_profile tests.products.demand_forecast.backends.tft.test_model_utils
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add products/demand_forecast/src/praedixa/demand_forecast/backends/tft/runtime_profile.py products/demand_forecast/src/praedixa/demand_forecast/backends/tft/dataloader_profile.py products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py products/demand_forecast/src/praedixa/demand_forecast/training/constants.py tests/products/demand_forecast/backends/tft/test_runtime_profile.py
git commit -m "feat: add l40s gpu runtime profiles for tft"
```

---

### Task 4: Redesign Hyperparameter Optimization for One GPU, Not Many CPU Trials

**Files:**
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/training/tuning.py`
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/training/orchestrator.py`
- Test: `tests/products/demand_forecast/training/test_pipeline.py`

- [ ] **Step 1: Write failing tests around GPU HPO policy**

Assertions:
- GPU trial concurrency = `1`
- fold execution on GPU does not spawn multiple concurrent GPU fits
- pruning configuration is serialized into metadata

- [ ] **Step 2: Implement a staged Optuna policy**

Required design:
- Stage A: short budget search (`8-12` epochs, broad search)
- Stage B: top-K narrowed search (`20-30` epochs)
- Stage C: final selected config for full train

Required constraints:
- one GPU trial at a time
- keep CPU-only baseline evaluation parallel if useful
- use pruning (MedianPruner or Hyperband/SuccessiveHalving)

- [ ] **Step 3: Persist HPO runtime metadata**

Metadata must include:
- host name / instance type
- GPU name
- CUDA version
- trial duration
- epochs completed
- pruned/completed status

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m unittest tests.products.demand_forecast.training.test_pipeline
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add products/demand_forecast/src/praedixa/demand_forecast/training/tuning.py products/demand_forecast/src/praedixa/demand_forecast/training/orchestrator.py tests/products/demand_forecast/training/test_pipeline.py
git commit -m "feat: redesign tft hpo for single gpu execution"
```

---

### Task 5: Promote Artifact Handling from “Best Params” to Full Reproducible Training Runs

**Files:**
- Create: `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/system_info.py`
- Create: `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/checkpoint_io.py`
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/artifacts.py`
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py`
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/evaluation/orchestrator.py`
- Test: `tests/products/demand_forecast/backends/tft/test_checkpoint_io.py`

- [ ] **Step 1: Write failing tests for artifact round-trip**

Need coverage for:
- save model payload
- reload payload
- same prediction at tolerance fixed on a tiny frame

- [ ] **Step 2: Extend the artifact payload**

Bundle must contain:
- `state_dict`
- `dataset_parameters`
- `history_frame`
- feature scalers
- target scaler
- runtime profile used
- GPU/system info
- git SHA
- bundle manifest / data hashes

- [ ] **Step 3: Add explicit load/reload utilities**

The repo currently saves a TFT payload but lacks a real round-trip loader contract. Add it before remote training becomes “production-like”.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m unittest tests.products.demand_forecast.backends.tft.test_checkpoint_io tests.products.demand_forecast.backends.tft.test_model_utils tests.products.demand_forecast.evaluation.test_pipeline
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add products/demand_forecast/src/praedixa/demand_forecast/backends/tft/artifacts.py products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py products/demand_forecast/src/praedixa/demand_forecast/backends/tft/system_info.py products/demand_forecast/src/praedixa/demand_forecast/backends/tft/checkpoint_io.py tests/products/demand_forecast/backends/tft/test_checkpoint_io.py
git commit -m "feat: add reproducible tft checkpoint bundle"
```

---

### Task 6: Add a Real GPU Training Entrypoint for Scaleway

**Files:**
- Modify: `apps/demand_forecast/run_training/main.py`
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/training/main.py`
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/training/orchestrator.py`
- Create: `docs/ops/scaleway-l40s-runbook.md`

- [ ] **Step 1: Add CLI/runtime parameters**

Required flags:
- `--bundle-dir`
- `--runtime-profile`
- `--output-dir`
- `--max-trials`
- `--stage-budget`
- `--tensorboard-logdir`

- [ ] **Step 2: Make the app consume bundles, not ad hoc local state**

The GPU training entrypoint should accept a versioned bundle path and should not assume the local Mac layout.

- [ ] **Step 3: Write the runbook**

The runbook must include:
- `git clone` / `rsync`
- `uv sync`
- bundle sync to `/srv/praedixa/data`
- training launch command
- log locations
- artifact locations
- retrieval instructions back to local

- [ ] **Step 4: Run unit tests**

Run:

```bash
.venv/bin/python -m unittest tests.products.demand_forecast.training.test_pipeline tests.products.demand_forecast.evaluation.test_pipeline
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add apps/demand_forecast/run_training/main.py products/demand_forecast/src/praedixa/demand_forecast/training/main.py products/demand_forecast/src/praedixa/demand_forecast/training/orchestrator.py docs/ops/scaleway-l40s-runbook.md
git commit -m "feat: add scaleway gpu training entrypoint and runbook"
```

---

### Task 7: Optimize the Dataset Path for GPU Throughput, Not Just Correctness

**Files:**
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/training_bundle/bundle_builder.py`
- Modify: `products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py`
- Test: `tests/products/demand_forecast/training_bundle/test_bundle_builder.py`

- [ ] **Step 1: Add typed bundle optimization rules**

Rules:
- continuous reals -> `float32`
- booleans/status encoded compactly but semantically preserved
- categoricals preserved as strings/codes with manifest
- no expensive conversions inside every Optuna trial

- [ ] **Step 2: Cache everything reusable outside the trial loop**

Cache candidates:
- bundle parquet files
- feature manifest
- target contract
- dataset parameters seeds where possible
- fold definitions

- [ ] **Step 3: Add throughput benchmark output**

Collect:
- rows/sec during dataloader
- epoch duration
- peak VRAM
- peak RAM
- loader worker count

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m unittest tests.products.demand_forecast.training_bundle.test_bundle_builder tests.products.demand_forecast.backends.tft.test_model_utils
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add products/demand_forecast/src/praedixa/demand_forecast/training_bundle/bundle_builder.py products/demand_forecast/src/praedixa/demand_forecast/backends/tft/model_utils.py tests/products/demand_forecast/training_bundle/test_bundle_builder.py
git commit -m "perf: optimize training bundle and dataloader path for gpu"
```

---

### Task 8: Benchmark the L40S and Freeze the Winning Runtime Profile

**Files:**
- Create: `ops/scaleway/run_training_remote.sh`
- Create: `ops/scaleway/sync_training_bundle.sh`
- Modify: `docs/ops/scaleway-l40s-runbook.md`

- [ ] **Step 1: Write the remote execution wrapper**

Wrapper responsibilities:
- sync repo or pull latest commit
- sync bundle
- activate `.venv`
- launch training with tee’d logs
- write outputs to `/srv/praedixa/artifacts/run_<timestamp>/`

- [ ] **Step 2: Run benchmark matrix on the real L40S**

Benchmark matrix:
- precision: `bf16-mixed` vs fallback mode
- batch size ladder: `128`, `256`, `384`, `512` or until OOM
- dataloader workers: `2`, `4`, `6`
- `torch.compile`: off vs on (only after baseline)

- [ ] **Step 3: Freeze the winning profile**

Decision criteria:
- no OOM
- stable validation loss
- best throughput per euro/hour
- reproducible wall-clock

- [ ] **Step 4: Record benchmark results in the runbook**

Include:
- exact chosen profile
- why it won
- what failed
- recovery steps for OOM / divergence / bad checkpoint

- [ ] **Step 5: Commit**

```bash
git add ops/scaleway/run_training_remote.sh ops/scaleway/sync_training_bundle.sh docs/ops/scaleway-l40s-runbook.md
git commit -m "ops: add l40s remote training workflow and benchmark profile"
```

---

## Acceptance Criteria

- A fresh SSH session on `root@51.159.162.168` can validate the GPU and environment in one command.
- A versioned training bundle can be built once and reused for HPO/training/evaluation.
- The TFT backend can train on `gpu:0` with a dedicated runtime profile for `L40S-1-48G`.
- Hyperparameter search uses a single-GPU-safe execution policy and logs pruning/runtime metadata.
- Final training artifacts are reproducible and include scalers, dataset parameters, system info and hashes.
- A round-trip checkpoint test proves same-prediction tolerance on a fixed mini example.
- The runbook is sufficient for a fresh operator to launch and retrieve a full training run.

## Recommended Execution Order

### Phase A - Local Code First

1. Task 2 — freeze the training bundle contract
2. Task 3 — add runtime profiles
3. Task 4 — redesign GPU HPO
4. Task 5 — make artifacts reproducible
5. Task 6 — add the real remote entrypoint and local CLI contract
6. Task 7 — optimize dataset throughput

### Phase B - Remote Configuration Last

7. Task 1 — bootstrap and verify the Scaleway L40S host
8. Task 8 — benchmark and freeze the production profile on the real machine

## Explicit Non-Goals for the First Pass

- Multi-GPU distributed training
- Spot/preemptible orchestration
- Full MLOps platform integration (`W&B`, MLflow, etc.) unless later justified
- Automatic `torch.compile` by default before benchmark proof
- Rebuilding `bronze/silver/gold` inside each remote training run
- Faire dependre les choix de code locaux de la machine distante avant la phase finale

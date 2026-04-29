# 🏗️ Architecture Overview

---

## 🤔 Why MLflow?

RL training is hard to reproduce — even small differences in hyperparameters, environment configs, or reward functions can dramatically change results. After training ends, you need to be able to answer "what configuration did this run use?" to enable cross-experiment comparison and reproduction.

| Feature | TensorBoard | MLflow |
|---|:---:|:---:|
| Training curve visualization | ✅ | ✅ |
| Hyperparameter storage | limited | ✅ |
| Environment config file (yaml) | ❌ | ✅ |
| Reward function code storage | ❌ | ✅ |
| Checkpoint file storage | ❌ | ✅ |
| Cross-experiment parameter comparison | ❌ | ✅ |
| Run search / filtering | ❌ | ✅ |

NEXUS uses `DualLogger` to run both tools simultaneously. TensorBoard handles real-time training curve monitoring; MLflow stores the full experiment record.

---

## ⚠️ Infrastructure Constraints

```
[GPU Server]                        [MLflow Server]
  No internet access                  Internet accessible
  SCP/SSH only                        MLflow Tracking UI
  Isaac Lab training                  http://<server>:5000
```

> The GPU Server **cannot** make outbound HTTP calls to the MLflow server. All data transfer happens exclusively via SCP/SSH.

---

## 🔀 Two Logging Pipelines

### 🅰️ Pipeline A — Direct MLflow Logging *(recommended, scheduled sync)*

```
PPO.write_stats()
      │
      ▼
  DualLogger.add_scalar()
  ┌─────────────────────────────────────────────┐
  │  TBLogger          │  MLflowLogger          │
  │  → tfevents (disk) │  → local MLflow :5100  │
  └─────────────────────────────────────────────┘
                           │
                  sync_mlflow_to_server.sh
                  (every N minutes via cron)
                           │
                  ┌────────┴────────┐
                  │ export_delta.py │  (query local MLflow,
                  │                 │   serialize only new steps per tag)
                  └────────┬────────┘
                           │ SCP delta.json
                           ▼
                  ┌─────────────────┐
                  │ import_delta.py │  (get_or_create run,
                  │                 │   log_batch new metrics)
                  └────────┬────────┘
                           │
                           ▼
                    [MLflow Server :5000]
                    Central experiment DB
```

> ✅ **Used when:** training code uses `make_logger()` and the run is actively training (or long-running).
>
> 🔁 **Incremental:** per-run, per-tag last-synced step is tracked in `~/.nexus/sync_state/{experiment}.json`.

---

### 🅱️ Pipeline B — TensorBoard Post-Upload *(one-shot, no code changes)*

```
PPO.write_stats()
      │
      ▼
  SummaryWriter
  → tfevents (disk)
        │
        │ (after training ends — manual, one-time)
        ▼
  post_upload/upload_tb.py
        │ parse tfevents → log_batch()
        ▼
  [MLflow Server :5000]
```

> ✅ **Used when:** training code is unchanged (no `make_logger()` integration yet), or to back-fill completed legacy runs.
>
> ⚠️ **Not scheduled:** this is a one-shot batch upload. For ongoing sync during training, use Pipeline A.

---

## 🔄 Complete Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    PPO Training Process                 │
│                                                         │
│  __init__()                                             │
│  ├── make_logger(params=agent_cfg)                      │
│  │                                                      │
│  train()  ──── write_stats() ──── add_scalar(tag, val)  │
│  │                                                      │
│  │  [epoch end]  log_checkpoint(path, kind="last")      │
│  │  [new best]   log_checkpoint(path, kind="best")      │
│  │                                                      │
│  └── close()                                            │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│                      DualLogger                          │
│                                                          │
│   add_scalar()  ──────┬──────────────────────────────┐   │
│   log_checkpoint()    │                              │   │
│   log_artifact()      │                              │   │
│   close()             │                              │   │
└───────────────────────┼──────────────────────────────┼───┘
                        │                              │
           ┌────────────▼──────────┐   ┌───────────────▼────────────┐
           │      TBLogger         │   │       MLflowLogger         │
           │                       │   │                            │
           │  SummaryWriter        │   │  MlflowClient              │
           │  └── tfevents/        │   │  ├── log_batch() (metrics) │
           │                       │   │  ├── log_batch() (params)  │
           │  log_checkpoint() →   │   │  └── log_artifact()        │
           │    (silently skipped) │   │                            │
           └────────────┬──────────┘   └──────────────┬─────────────┘
                        │                             │
                        ▼                             ▼
           ┌───────────────────────┐   ┌────────────────────────────┐
           │  Local tfevents       │   │  Local MLflow Server       │
           │  ~/runs/TASK/         │   │  127.0.0.1:5100            │
           │  (real-time view)     │   │  (experiment store)        │
           └───────────────────────┘   └──────────────┬─────────────┘
                                                      │
                                            (scheduled_sync)
                                                      │
                                                      ▼
                                       ┌────────────────────────────┐
                                       │  Central MLflow Server     │
                                       │  <server-ip>:5000          │
                                       │  (team shared dashboard)   │
                                       └────────────────────────────┘
```

---

## 📁 Repository Structure

```
nexus/
│
├── nexus/                          # Importable Python package (`from nexus.logger import ...`)
│   └── logger/                     # Unified logging package
│       ├── __init__.py             # make_logger() factory (core exports only)
│       ├── dual_logger.py          # TensorBoard + MLflow simultaneously
│       ├── mlflow_logger.py        # MLflow-only logger
│       ├── tb_logger.py            # TensorBoard wrapper (legacy compat)
│       ├── git_utils.py            # Git commit/dirty-state capture (auto-called at run start)
│       ├── sweep_logger.py         # [Advanced] HP sweep parent run
│       ├── model_registry.py       # [Advanced] Model Registry operations
│       ├── rl_metrics.py           # [Advanced] RL diagnostic metric helpers
│       └── system_metrics.py       # [Advanced] Background CPU/GPU logging
│
├── post_upload/                    # Pipeline B — upload after training
│   ├── upload_tb.py                # Full tfevents → MLflow batch upload
│   ├── verify_tb.py                # Numeric validation vs. TB source
│   └── upload_eval.py              # Attach evaluation artifacts (mp4 / report) to a run
│
├── scheduled_sync/                 # Pipeline A — sync while training runs (air-gapped SCP)
│   ├── start_local_mlflow.sh       # [GPU Server] start local MLflow server
│   ├── sync_mlflow_to_server.sh    # [GPU Server] delta export → SCP → import
│   ├── export_delta.py             # [GPU Server] serialize new metrics only
│   └── import_delta.py             # [MLflow server] import delta JSON
│
├── chart_settings/                 # Persist MLflow UI column/chart settings
│   ├── chart_settings.json         # Team-standard column and chart configuration
│   └── apply_chart_settings.py     # CLI: apply / show / bookmarklet
│
├── tests/
│   └── smoke_test.py               # End-to-end local validation script
│
├── docs/                           # Filename prefix conveys reading order (00 → 30)
│   ├── 00_PRINCIPLES.md            # Canonical team-agreed rules + engineering invariants
│   ├── 10_ARCHITECTURE.md          # ← You are here. System design & component map
│   ├── 11_LOGGER_SETUP.md          # Pipeline A — logger integration (step-by-step diff)
│   ├── 12_SCHEDULED_SYNC.md        # Pipeline A — cron sync wiring (config, validate, multi-user)
│   ├── 13_POST_UPLOAD.md           # Pipeline B — upload_tb / verify_tb / upload_eval CLIs in depth
│   ├── 20_MLFLOW_SERVER_SETUP.md   # Operator — central MLflow server install (incl. local PC verify)
│   ├── 21_AIRGAPPED_GPU_SERVER_SETUP.md  # Operator — GPU node offline bring-up (pip wheel / Docker, incl. verify)
│   ├── 30_ADVANCED_FEATURES.md     # Opt-in — SweepLogger, RL metrics, Model Registry
│   ├── 31_CHART_SETTINGS_GUIDE.md  # Opt-in — persist MLflow chart/column settings
│   └── ko/                         # 🇰🇷 Korean onboarding & operational policy track
│       ├── README.md               # Korean track index + reading order
│       ├── 01_INTRO.md             # 동기·두 파이프라인 개념·FAQ
│       └── 02_EXPERIMENT_STANDARD.md  # 명명 규칙·태그·라이프사이클·Confluence 템플릿
│
├── brand.py                        # ASCII art, sigils, color constants
├── setup.sh
└── github_init.sh
```

The next section maps each runtime component (factory, loggers, sync scripts) to its on-disk location, with one-line role descriptions.

---

## 🧩 Component Map

| Component | Location | Purpose |
|---|---|---|
| `make_logger()` | `nexus/logger/` | 🏭 Factory: returns DualLogger / MLflowLogger / TBLogger |
| `DualLogger` | `nexus/logger/dual_logger.py` | 🔀 Forwards calls to TB + MLflow simultaneously |
| `MLflowLogger` | `nexus/logger/mlflow_logger.py` | 📊 Buffers + flushes to local MLflow via `log_batch()` |
| `TBLogger` | `nexus/logger/tb_logger.py` | 📈 Thin `SummaryWriter` wrapper |
| `start_local_mlflow.sh` | `scheduled_sync/` | 🚀 Starts local MLflow on GPU Server (loopback) |
| `sync_mlflow_to_server.sh` | `scheduled_sync/` | 🔄 Orchestrates delta export → SCP → import |
| `export_delta.py` | `scheduled_sync/` | 📦 Serializes only new metric points per run/tag |
| `import_delta.py` | `scheduled_sync/` | ⬆️ Reads delta JSON, logs new metrics to central MLflow |
| `upload_tb.py` | `post_upload/` | 📤 Manual full upload after training |
| `verify_tb.py` | `post_upload/` | ✅ Validates upload against TB source |
| `upload_eval.py` | `post_upload/` | 🎬 Attaches eval artifacts (mp4/report) to an existing run |

---

## 🔄 MLflow Run Lifecycle *(Pipeline A)*

```
Training starts
    │
    ▼
make_logger(mode="dual") called
    ├─ TBLogger:     creates tfevents file
    └─ MLflowLogger: creates MLflow run (or resumes if run_name exists)
           │ tags:   researcher, seed, task, hardware, isaac_lab_version
           │ params: all agent_cfg hyperparameters (logged once at start)
    │
    ▼ (each epoch)
write_stats() → add_scalar() × N
    ├─ TBLogger:     writes to tfevents
    └─ MLflowLogger: buffers metrics in memory
           │ when step changes → log_batch() to local MLflow (:5100)
    │
    ▼ (every 5 min, via cron)
sync_mlflow_to_server.sh
    ├─ export_delta.py   → delta JSON (only new steps since last sync)
    ├─ SCP               → remote inbox
    └─ SSH import_delta.py on central server (log_batch new metrics)
    │
    ▼
Training ends
    │
writer.close() called
    ├─ TBLogger:     flush + close tfevents
    └─ MLflowLogger: flush remaining buffer + set run status = FINISHED
```

---

## 🗃️ MLflow Run Internal Structure

One training run stored in MLflow:

```
Run: "ShadowHand_PPO_seed42_20240315_143022"
│
├── Parameters (logged once at run start)
│   ├── lr                     0.0003
│   ├── gamma                  0.99
│   ├── num_envs               4096
│   ├── horizon_length         16
│   ├── mini_epochs            8
│   ├── env.obs_dim            157
│   ├── env.action_dim         22
│   └── ...                    (full agent_cfg; nested keys flattened with ".")
│
├── Tags (logged once at run start)
│   ├── experiment             "robot_hand_rl"
│   ├── researcher             "jongho"
│   ├── task                   "ShadowHandOver"
│   └── hardware               "robot_22dof"
│
├── Metrics (logged every step)
│   ├── losses/actor_loss      [step 0..N]
│   ├── losses/critic_loss     [step 0..N]
│   ├── losses/entropy         [step 0..N]
│   ├── performance/RLTrainFPS [step 0..N]
│   ├── episode/reward_mean    [step 0..N]
│   └── ...
│
└── Artifacts (file storage)
    └── checkpoints/
        ├── best.pth           ← best checkpoint (overwritten whenever score improves)
        └── last.pth           ← latest epoch checkpoint (overwritten each epoch)
```

---

## 💾 Checkpoint Management Policy

```
Epoch 1   → last.pth saved  (score=0.12)
Epoch 2   → last.pth saved  (score=0.18) → best.pth saved
Epoch 3   → last.pth saved  (score=0.15)
Epoch 4   → last.pth saved  (score=0.24) → best.pth updated
   ...
Epoch N   → last.pth saved  (score=0.31) → best.pth updated

MLflow artifacts/checkpoints/ always contains exactly 2 files:
  best.pth  ← highest score across the entire training run
  last.pth  ← most recent epoch
```

By keeping only two checkpoints instead of stacking every epoch, storage waste is avoided while still covering both use cases: resuming training and deployment.

---

## 🗂️ MLflow Experiment Structure

```
MLflow Central Server
├── Experiment: "robot_hand_rl"
│   ├── Run: "ppo_baseline_v1"           ← Pipeline A
│   │   ├── Tags:      researcher, seed, task, hardware, isaac_lab_version
│   │   ├── Params:    lr, gamma, e_clip, batch_size, ...
│   │   ├── Metrics:   losses/*, performance/*, info/*, episode_rewards/*
│   │   └── Artifacts: checkpoints/best.pth, checkpoints/ep_100_...pth
│   │
│   └── Run: "legacy_run_001"            ← Pipeline B (one-shot post-upload)
│       ├── Tags:    source=tensorboard_import
│       └── Metrics: (same metric tags, uploaded once after training)
│
└── Experiment: "robot_hand_sim2real_eval"
    └── Run: "real_robot_20250416"
        └── Tags: sim_run_id=<run_id>    ← links back to training run
```

> ⚠️ **Always set `sim_run_id`** on real-robot eval runs. This is the only way to trace a Sim-to-Real failure back to its training origin.

---

## 🔍 Experiment Comparison Workflow

```
MLflow UI (Central Server :5000)
│
├── Experiments
│   └── robot_hand_rl
│       ├── Run A  seed=42  reward_fn_v1  lr=3e-4  → best_score=0.31
│       ├── Run B  seed=42  reward_fn_v2  lr=3e-4  → best_score=0.44  ← reward fn change effect
│       ├── Run C  seed=42  reward_fn_v2  lr=1e-4  → best_score=0.41
│       └── Run D  seed=7   reward_fn_v2  lr=3e-4  → best_score=0.43
│
└── Compare Runs (A vs B)
    ├── Parameters diff: agent_cfg keys side by side (lr, gamma, e_clip, ...)
    └── Metrics chart:   episode/reward_mean curves side by side
```

---

## 🗺️ Recommended Migration Path

| Phase | Mode | Description | Goal |
|:---:|:---:|---|---|
| **1** *(now)* | `"dual"` | TensorBoard works as before + MLflow gets the data | Team learns MLflow UI at their own pace |
| **2** *(~1 month)* | `"dual"` | MLflow becomes primary tool; TensorBoard for local debug only | MLflow-first habit formed |
| **3** *(optional)* | `"mlflow"` | TensorBoard removed; MLflow is single source of truth | Full centralization |

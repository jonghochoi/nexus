# 🧪 Advanced Features

> **Purpose:** Opt-in extensions to the core `make_logger()` workflow — hyperparameter sweeps, post-training eval artifact uploads, RL-specific diagnostics, MLflow Model Registry helpers, background system metrics, automatic git tracking, persistent chart settings, and the advanced smoke-test mode.
>
> The standard workflow (Pipeline A or B) requires **no knowledge of anything in this doc**. Come back when the basics feel natural and you want to add one of these features.

---

## Table of Contents

- [Import pattern](#import-pattern)
- [1. SweepLogger — hyperparameter sweep management](#1-sweeplogger--hyperparameter-sweep-management)
- [2. Model Registry](#2-model-registry)
- [3. EvalLogger — post-training eval artifact upload](#3-evallogger--post-training-eval-artifact-upload)
- [4. SystemMetricsLogger — background resource logging](#4-systemmetricslogger--background-resource-logging)
- [5. Git commit tracking](#5-git-commit-tracking)
- [6. Chart settings — persistent column layout](#6-chart-settings--persistent-column-layout)
- [7. Smoke test — advanced mode](#7-smoke-test--advanced-mode)
- [Next steps](#next-steps)

---

## Import pattern

Advanced features are **not** exported from `nexus.logger` by default. Each module must be imported explicitly:

```python
# Core — always available
from nexus.logger import make_logger, MLflowLogger, DualLogger

# Advanced — explicit import required
from nexus.logger.sweep_logger   import SweepLogger
from nexus.logger.model_registry import ModelRegistry
from nexus.logger.eval_logger    import EvalLogger
from nexus.logger.system_metrics import SystemMetricsLogger
```

---

## 1. SweepLogger — hyperparameter sweep management

Groups multiple training runs under a single parent run in the MLflow UI tree. Useful when running a grid search or Optuna sweep.

Both `tracking_uri` and `experiment_name` are required — there are no defaults, so the sweep is always placed exactly where you intend.

### ── Basic usage — context manager (recommended)

```python
from nexus.logger.sweep_logger import SweepLogger
from nexus.logger import make_logger

with SweepLogger(
    sweep_name="ppo_lr_sweep",
    tracking_uri="http://127.0.0.1:5100",
    experiment_name="robot_hand_rl",
    sweep_params={"lr_range": "[1e-4, 1e-3]", "n_trials": "3"},
) as sweep:
    for lr in [1e-4, 3e-4, 1e-3]:
        logger = make_logger(
            mode="mlflow",
            run_name=f"ppo_lr_{lr}",
            tracking_uri="http://127.0.0.1:5100",
            experiment_name="robot_hand_rl",
            params={"lr": lr},
            parent_run_id=sweep.parent_run_id,   # ← links child to parent
        )
        run_training(logger)
        logger.close()

    sweep.log_summary(
        best_params={"lr": 3e-4},
        best_metrics={"reward": 95.2},
    )
```

The context manager marks the parent run **FINISHED** on clean exit and **FAILED** automatically if an exception escapes the block — no orphaned `RUNNING` runs in the MLflow UI.

**What you see in the MLflow UI:** a collapsible parent run `ppo_lr_sweep` with child runs nested underneath, each showing its own metric curves.

**Works without Hydra.** `SweepLogger` is pure MLflow — no Hydra dependency. Use it with a plain Python `for` loop, Optuna callbacks, or any other sweep tool.

### ── Optuna integration

Pass `sweep.parent_run_id` inside the Optuna objective to nest every trial under the sweep run. Call `sweep.log_summary()` after `study.optimize()` to record the winner on the parent run.

```python
import optuna
from nexus.logger.sweep_logger import SweepLogger
from nexus.logger import make_logger

with SweepLogger(
    sweep_name="optuna_ppo_lr",
    tracking_uri="http://127.0.0.1:5100",
    experiment_name="robot_hand_rl",
    sweep_params={"n_trials": "20", "sampler": "TPE"},
) as sweep:

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        entropy = trial.suggest_float("entropy_coef", 0.0, 0.05)
        logger = make_logger(
            mode="mlflow",
            run_name=f"trial_{trial.number}",
            tracking_uri="http://127.0.0.1:5100",
            experiment_name="robot_hand_rl",
            params={"lr": lr, "entropy_coef": entropy},
            parent_run_id=sweep.parent_run_id,
        )
        reward = run_training(logger, lr=lr, entropy_coef=entropy)
        logger.close()
        return reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    sweep.log_summary(
        best_params=study.best_params,
        best_metrics={"reward": study.best_value},
    )
```

Optuna is not a dependency of `nexus-logger` — install it separately (`pip install optuna`) only on machines that run sweeps.

### ── Manual lifecycle (alternative)

If you need to manage the lifecycle outside a `with` block, wrap in `try/finally` to avoid orphaned runs:

```python
sweep = SweepLogger(sweep_name=..., tracking_uri=..., experiment_name=...)
try:
    ...
    sweep.log_summary(...)
finally:
    sweep.close()          # accepts status="FINISHED" | "FAILED" | "KILLED"
```

---

## 2. Model Registry

> **Where to register, and when** — Model Registry entries write to whatever tracking URI the caller is configured for, and `scheduled_sync` does **not** propagate registry rows. Two distinct paths exist; pick the right one for your workflow:
>
> | Use case | API | Server it writes to |
> |---|---|---|
> | Single-machine workflow / smoke test (logger and registry on the same MLflow) | `MLflowLogger.register_checkpoint()` (in-loop) | The logger's `tracking_uri` |
> | Multi-node NEXUS deployment — register *after* training on central | `post_upload/register_model.py` CLI **or** `ModelRegistry.register_from_run_name()` | Central MLflow |
>
> In a GPU-server / central-server topology, calling `register_checkpoint()` from a `DualLogger` puts the version on the *local* `5100` server only — it will **not** sync to central. The recommended workflow there is: train (with `log_checkpoint()`) → wait for `scheduled_sync` → evaluate the synced runs → run `post_upload/register_model.py` against central for the runs you choose to promote.

### ── In-loop registration (single-machine / smoke test)

After uploading a checkpoint with `log_checkpoint()`, register it on the *same* server the logger is writing to:

```python
logger.log_checkpoint("/path/to/best.pth", kind="best")

version = logger.register_checkpoint(
    model_name="shadow_hand_ppo",
    kind="best",
    description="PPO v3 — in-hand reorientation, seed 42",
)
logger.promote_model("shadow_hand_ppo", version=version, stage="Production")
```

This API is on `MLflowLogger`. It is intentionally **not** forwarded by `DualLogger` — the asymmetry between local and central registries makes the in-loop call ambiguous in a NEXUS deployment, so the central path below is the default.

### ── Post-hoc registration on central (typical NEXUS flow)

After training is finished and `scheduled_sync` has copied the run to central, register from any host that can reach central MLflow:

```bash
python post_upload/register_model.py \
    --tracking_uri http://nexus-server:5000 \
    --experiment shadow_hand_rl \
    --run_name exp_v3_seed42 \
    --kind best \
    --model_name shadow_hand_ppo \
    --description "PPO v3 — 87% success on real hand" \
    --stage Staging
```

Or from Python (notebook, eval script):

```python
from nexus.logger.model_registry import ModelRegistry

registry = ModelRegistry(tracking_uri="http://nexus-server:5000")
result = registry.register_from_run_name(
    experiment="shadow_hand_rl",
    run_name="exp_v3_seed42",
    model_name="shadow_hand_ppo",
    kind="best",
    description="PPO v3 — 87% success on real hand",
    stage="Staging",
)
```

The new version is stamped with `nexus.sourceRunName=<run_name>` so the registry entry is traceable back to the source run. See [`docs/13_POST_UPLOAD.md`](13_POST_UPLOAD.md#step-8--register_modelpy--register-a-checkpoint-as-a-model-version) for the full CLI reference.

### ── `ModelRegistry` — querying the registry

```python
from nexus.logger.model_registry import ModelRegistry

registry = ModelRegistry(tracking_uri="http://127.0.0.1:5100")

# Get current production model info
prod = registry.get_production_model("shadow_hand_ppo")
# {"version": "3", "run_id": "...", "sim_run_id": "...", ...}

# List all versions
for v in registry.list_versions("shadow_hand_ppo"):
    print(v["version"], v["stage"], v["run_id"])

# Sim-to-Real traceability: tag which sim run produced this policy
registry.set_sim_to_real_link(
    model_name="shadow_hand_ppo",
    version="3",
    sim_run_id="abc123def456",
)
```

**Sim-to-Real link:** when a real-robot evaluation fails, look up the Production model version → find `sim_run_id` → open that sim run in MLflow to inspect training curves, hyperparameters, and reward function used.

---

## 3. EvalLogger — post-training eval artifact upload

Attaches post-training evaluation outputs (mp4 rollouts, GIF previews, reports, score JSONs) to an existing MLflow run as artifacts under `eval/<eval_id>/`. The run is resolved by `run_name` — the same identity key used by `MLflowLogger` and Pipeline B.

`EvalLogger` is designed for the external-repo case: the training repo already uses `make_logger()`, which writes a `.nexus_run.json` sidecar into the output directory. The eval step reads that sidecar to recover run identity without re-passing config.

### ── Why eval uploads target central directly

In a NEXUS deployment a training run lives in two places: the GPU-node-local relay (`http://127.0.0.1:5100`, where `MLflowLogger` writes during training) and the central MLflow (`http://nexus-server:5000`, where `scheduled_sync` mirrors the run on a cron cycle). `EvalLogger.from_run_info()` defaults to **uploading directly to central**, not to the local relay. Three reasons:

1. **Eval bundles are large and one-shot.** A typical eval ships hundreds of MB of mp4/gif rollouts. `scheduled_sync` is tuned for fast metric-delta replication; routing big artifacts through it bloats every sync cycle and delays the next one. A direct upload bypasses that queue entirely.
2. **Immediate visibility.** Eval is a human-in-the-loop step — you want the result on the team-shared central UI as soon as it finishes, not after the next sync tick (which can be minutes away). Direct upload makes the run appear instantly in the same view everyone else uses.
3. **Run UUIDs differ between local and central.** The local relay and central server assign their own MLflow run UUIDs, so the `run_id` in the sidecar is local-only. `EvalLogger` resolves the target run by `tags.mlflow.runName` (which is identical on both sides), so a central upload Just Works without needing a UUID translation table.

The trade-off is that the run must already have been synced to central at least once before eval runs (`EvalLogger` will raise `ValueError` if no run with that `run_name` exists on the target server). The intended order is: training finishes → at least one `scheduled_sync` cycle completes → eval runs.

For this default to work, the trainer must let the sidecar know the central URI. Pass `central_tracking_uri="http://nexus-server:5000"` to `make_logger()`:

```python
self.writer = make_logger(
    mode="dual",
    tb_dir=output_dir,
    run_name=run_name,
    tracking_uri="http://127.0.0.1:5100",            # local relay (training writes here)
    central_tracking_uri="http://nexus-server:5000", # NEXUS central (eval reads from sidecar)
    experiment_name="robot_hand_rl",
    agent_params=agent_cfg,
    env_params=env_cfg,
)
```

This writes both URIs into `.nexus_run.json`. Sidecars where `central_tracking_uri` is omitted (older trainer, or `make_logger()` called without the argument) will get an instructive error from `from_run_info()` pointing to the migration — see _Resolution order_ below.

### ── Recommended eval_dir layout

`EvalLogger` walks `eval_dir` recursively and uploads everything it finds. A flat layout is fine; subdirectories are preserved under `eval/<eval_id>/` on the server.

```
eval_outputs/<run_name>/
├── rollout.mp4            ← full-resolution video — auto-embedded in index.html
├── rollout_preview.gif    ← short preview — also rendered inline by MLflow
├── report.md              ← human-readable summary
├── metrics.json           ← machine-readable scores (auto-promoted via --metrics-from)
└── success_rate.png       ← any other plots
```

After upload the MLflow run gains an `eval/<eval_id>/` artifact folder with all of the above plus the auto-generated `index.html`. The `eval.last_id` tag on the run is always stamped with the most recent eval_id so downstream consumers can find the latest bundle without scanning artifacts.

### ── Basic usage

```python
from nexus.logger.eval_logger import EvalLogger

# make_logger() writes .nexus_run.json into output_dir during training.
# Pass the same output_dir here to pick up run_name / experiment / tracking_uri.
ev = EvalLogger.from_run_info(output_dir)

eval_id = ev.upload(
    eval_dir=output_dir / "eval",
    metrics={"success_rate": 0.87, "mean_return": 132.4},
    metrics_from=output_dir / "eval" / "metrics.json",   # auto-flatten JSON scalars
    tags={"observer_commit": "abc123"},
)
# → artifacts/eval/<eval_id>/ on the MLflow run
```

The same run can receive multiple eval bundles over time — each lands in its own `eval/<eval_id>/` subdir so they never collide.

### ── Explicit construction

When the `.nexus_run.json` sidecar is not available (e.g. the run was created by Pipeline B's `upload_tb.py`), pass all three params directly:

```python
ev = EvalLogger(
    run_name="ppo_v3_seed0",
    tracking_uri="http://nexus-server:5000",
    experiment="robot_hand_rl",
)
ev.upload(eval_dir="./eval_outputs/ppo_v3_seed0/")
```

### ── Resolution order for tracking_uri

`from_run_info()` picks the destination MLflow URI in this order:

1. **Explicit `tracking_uri=` argument** — wins unconditionally. Use this for one-off overrides (e.g. mirror the same eval bundle to a staging server).
2. **`target="central"` (default)** → the sidecar's `central_tracking_uri`. Recommended path; see _Why eval uploads target central directly_ above. Raises `ValueError` when the sidecar has no `central_tracking_uri` (i.e. `make_logger()` was called without `central_tracking_uri=`). The error message points to the three migration options.
3. **`target="local"`** → the sidecar's `tracking_uri` (the GPU-node-local relay). Useful when debugging an in-progress training run before it has been synced to central:

```python
ev = EvalLogger.from_run_info(output_dir, target="local")  # upload to 127.0.0.1:5100
```

Or pass an explicit override:

```python
ev = EvalLogger.from_run_info(
    output_dir,
    tracking_uri="http://staging-mlflow:5000",   # one-off override, ignores target
)
```

### ── Auto-generated index.html

MLflow 2.13's artifact viewer renders HTML inline but not `.mp4`. `EvalLogger` auto-generates an `index.html` next to any video it finds, embedding it in a `<video controls>` tag. Open `eval/<eval_id>/index.html` in the Artifacts pane to play rollouts in-browser.

Suppress with `generate_index=False` if your eval tool already ships its own page:

```python
ev.upload(eval_dir=..., generate_index=False)
```

### ── Silent / programmatic mode

Pass `verbose=False` to suppress all console output — useful when `EvalLogger` is called from a training script rather than interactively:

```python
ev = EvalLogger.from_run_info(output_dir, verbose=False)
ev.upload(eval_dir=..., metrics={"sr": 0.9})
```

### ── metrics_from — auto-promote a JSON file

If the eval tool writes a `metrics.json`, pass the path to `metrics_from` and `EvalLogger` flattens it automatically (dotted-key for nested dicts). Explicit `metrics` dict wins on key conflict:

```python
ev.upload(
    eval_dir=...,
    metrics_from="eval/metrics.json",          # {"locomotion": {"speed": 1.2}} → eval/locomotion.speed
    metrics={"success_rate": 0.87},            # explicit key wins if present in JSON too
)
```

### ── eval_id — stable name vs. timestamp

`eval_id` defaults to a `YYYYmmdd_HHMMSS` timestamp, producing a new subdir on every call. Pass a fixed string to overwrite a previous bundle (e.g. during iterative debugging) or to give the folder a human-readable name:

```python
ev.upload(eval_dir=..., eval_id="checkpoint_500")
# → artifacts/eval/checkpoint_500/
```

### ── dry_run — preview without uploading

Pass `dry_run=True` to resolve the run and list what would be uploaded without touching MLflow:

```python
eval_id = ev.upload(eval_dir=..., metrics={"sr": 0.87}, dry_run=True)
# Prints file list and metric preview; no artifacts or metrics written
```

---

## 4. SystemMetricsLogger — background resource logging

Spawns a daemon thread that periodically logs CPU, RAM, and GPU metrics to MLflow without blocking training.

```python
from nexus.logger.system_metrics import SystemMetricsLogger

logger = make_logger(mode="mlflow", ...)
sys_logger = SystemMetricsLogger(logger, interval_seconds=30, gpu_index=3)

sys_logger.start()

for step in training_loop():
    ...  # normal training

sys_logger.stop()
logger.close()
```

**Metrics logged (when available):**

| Key | Requires |
|---|---|
| `system/cpu_percent` | `psutil` |
| `system/ram_gb` | `psutil` |
| `system/gpu_memory_mb` | `gpu_index` set + `nvidia-ml-py` or `nvidia-smi` |
| `system/gpu_util_percent` | `gpu_index` set + `nvidia-ml-py` or `nvidia-smi` |

Silently skips any metric whose dependency is not installed. The thread is a daemon — it will not prevent process exit if `stop()` is not called explicitly.

Optional installs:
```bash
pip install psutil nvidia-ml-py
```

### ── GPU metrics: explicit opt-in via `gpu_index`

GPU collection is **off by default**. Pass `gpu_index=N` to enable it. The chosen index is stamped onto the run tag `system.gpu_index` at `start()` time so the active device is visible in the MLflow UI.

There is no auto-detection. On multi-GPU hosts, frameworks like PyTorch routinely create a small stray CUDA context on GPU 0 (during default-device initialisation, `torch.cuda.device_count()` calls, or library imports) before the actual training tensors move to `cuda:N`. Any PID-based scan would then attribute metrics to the wrong device, so the caller must specify the index explicitly.

#### ▸ How to pick `gpu_index`

The value is interpreted exactly as `nvidia-smi -i=N` interprets it — i.e. an **NVML index**, not a CUDA device index. The two differ when `CUDA_VISIBLE_DEVICES` is in use:

| Launch pattern | What `cuda:0` maps to | `gpu_index` to pass |
|---|---|---|
| `python train.py` with `torch.device("cuda:3")` | physical GPU 3 | `3` |
| `CUDA_VISIBLE_DEVICES=3 python train.py` (code uses `cuda:0`) | physical GPU 3 | `3` |
| Container with `NVIDIA_VISIBLE_DEVICES=3` (NVML inside container only sees one device) | the only visible GPU | `0` |

> ⚠️ `CUDA_VISIBLE_DEVICES` does **not** remap NVML/`nvidia-smi` indices — it only restricts what the CUDA runtime exposes. NVML still uses physical indices, so the value you pass to `SystemMetricsLogger` must be the **physical** index even when your training code references `cuda:0`. The container case is different because `NVIDIA_VISIBLE_DEVICES` (NVIDIA Container Toolkit) does isolate the NVML view.

#### ▸ Recommended pattern

For unambiguous attribution on multi-GPU hosts, prefer `CUDA_VISIBLE_DEVICES=N` over hard-coding `torch.device("cuda:N")` in code — the former prevents the framework from creating stray contexts on other GPUs, and the same value works for both the runtime and the metrics logger:

```bash
CUDA_VISIBLE_DEVICES=3 python train.py --gpu-metrics-index 3
```

```python
# train.py
import argparse, torch
from nexus.logger import make_logger
from nexus.logger.system_metrics import SystemMetricsLogger

p = argparse.ArgumentParser()
p.add_argument("--gpu-metrics-index", type=int, default=None,
               help="NVML/nvidia-smi index of the GPU to track (omit to skip GPU metrics)")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:0 inside CVD mask

logger = make_logger(mode="mlflow", run_name="my_run", ...)
sys_logger = SystemMetricsLogger(logger, gpu_index=args.gpu_metrics_index)
sys_logger.start()
```

If you don't need per-GPU memory/utilisation in MLflow at all, simply omit `gpu_index` — only CPU and RAM will be logged.

---

## 5. Git commit tracking

`MLflowLogger` automatically captures the git state of the training code at run start. No extra code is needed — it is on by default.

### ── What gets recorded

| MLflow location | Key | Value |
|---|---|---|
| Tags tab | `git_commit` | Full SHA of HEAD — e.g. `54696cb326bb...` |
| Tags tab | `git_dirty` | `"false"` when tree is clean, `"true"` when there are uncommitted changes |
| Artifacts | `git/git_patch.html` | Full `git diff HEAD` rendered as a self-contained HTML page with line-level colouring — **only present when `git_dirty = "true"`** |

### ── Inspecting a dirty-tree run

If training was launched with uncommitted changes, open `git/git_patch.html` directly in the MLflow Artifacts tab to view the diff inline (additions in green, deletions in red, hunk headers in blue). The page is self-contained — no external assets needed. For byte-exact reproducibility, commit before training so `git_dirty = "false"` and `git_commit` alone identifies the source state.

### ── Opting out

Pass `track_git=False` to suppress all git tags and artifacts — useful when training outside a git repo or in CI environments where the working tree is intentionally untracked:

```python
logger = MLflowLogger(
    run_name="my_run",
    ...
    track_git=False,
)
```

`make_logger()` forwards `track_git` transparently:

```python
logger = make_logger(mode="mlflow", ..., track_git=False)
```

---

## 6. Chart settings — persistent column layout

MLflow stores the runs-table column visibility (which tags, params, and metrics are shown) in the **browser's localStorage**. This means the layout resets whenever you open a fresh browser or switch machines.

`chart_settings/` solves this by storing the desired layout as MLflow **experiment tags** — permanently, on the server — and providing a one-click bookmarklet to restore it in any browser.

```bash
# 1. Edit chart_settings/chart_settings.json to define your column layout
# 2. Save it to the MLflow server (run once after every edit)
python chart_settings/apply_chart_settings.py apply

# 3. Generate a browser bookmarklet to restore localStorage
python chart_settings/apply_chart_settings.py bookmarklet

# 4. Verify what is currently stored on the server
python chart_settings/apply_chart_settings.py show
```

The bookmarklet fetches the stored settings from the MLflow API and writes them to localStorage, then reloads the page. Paste it into the browser console (F12 > Console), or save it as a browser bookmark for one-click restore.

→ Full workflow and persistence details: [`31_CHART_SETTINGS_GUIDE.md`](31_CHART_SETTINGS_GUIDE.md)

---

## 7. Smoke test — advanced mode

The smoke test (`tests/smoke_test.py`) runs only core tests by default. Pass `--advanced` to also validate the features described in this document.

```bash
python tests/smoke_test.py --advanced
# Tests 1–5: core (always run)
# Test 6: OmegaConf DictConfig flatten
# Test 7: scheduled_sync round-trip
# Test 8: SweepLogger parent-child runs
# Test 9: EvalLogger artifact upload
```

---

## Next steps

- **Persistent MLflow chart/column layout** → [`31_CHART_SETTINGS_GUIDE.md`](31_CHART_SETTINGS_GUIDE.md)
- **Architecture detail (pipelines, run lifecycle, registry)** → [`10_ARCHITECTURE.md`](10_ARCHITECTURE.md)
- **Required tags + Sim-to-Real linkage** → [`00_PRINCIPLES.md`](00_PRINCIPLES.md)

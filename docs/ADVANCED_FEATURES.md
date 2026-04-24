# Advanced Features Guide

> **Note:** These features are opt-in. The standard `make_logger()` workflow requires no knowledge of anything here. Come back when the basics feel natural.

---

## Import Pattern

Advanced features are **not** exported from `logger` by default. Each module must be imported explicitly:

```python
# Core — always available
from logger import make_logger, MLflowLogger, DualLogger

# Advanced — explicit import required
from logger.sweep_logger   import SweepLogger
from logger.model_registry import ModelRegistry
from logger.system_metrics import SystemMetricsLogger
from logger                import rl_metrics          # module, not a class
```

---

## 1. SweepLogger — Hyperparameter Sweep Management

Groups multiple training runs under a single parent run in the MLflow UI tree. Useful when running a grid search or Optuna sweep.

```python
from logger.sweep_logger import SweepLogger
from logger import make_logger

sweep = SweepLogger(
    sweep_name="ppo_lr_sweep",
    tracking_uri="http://127.0.0.1:5100",
    experiment_name="robot_hand_rl",
    sweep_params={"lr_range": "[1e-4, 1e-3]", "n_trials": "5"},
)

for lr in [1e-4, 3e-4, 1e-3]:
    logger = make_logger(
        mode="mlflow",
        log_dir="/tmp/tb",
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
sweep.close()
```

**What you see in the MLflow UI:** a collapsible parent run `ppo_lr_sweep` with child runs nested underneath, each showing its own metric curves.

**Works without Hydra.** `SweepLogger` is pure MLflow — no Hydra dependency. Use it with a plain Python `for` loop, Optuna callbacks, or any other sweep tool.

---

## 2. RL Diagnostic Metrics

### 2a. `log_rl_metrics()` — per-step logging

`MLflowLogger` and `DualLogger` both expose `log_rl_metrics()`. It logs standard PPO/RL diagnostics under the `rl/` key namespace.

```python
logger.log_rl_metrics(
    step,
    explained_variance=ev,     # float | None — skip by passing None
    approx_kl=kl,
    clip_fraction=cf,
    grad_norm=gn,
    entropy=ent,
    success_rate=sr,
)
```

Keys logged to MLflow: `rl/explained_variance`, `rl/approx_kl`, `rl/clip_fraction`, `rl/grad_norm`, `rl/entropy`, `rl/success_rate`. Any keyword left as `None` is silently skipped.

### 2b. `rl_metrics` module — pure NumPy helpers

Compute the values before logging them:

```python
from logger import rl_metrics
import numpy as np

ev = rl_metrics.explained_variance(value_preds, returns)
kl = rl_metrics.approx_kl(old_log_probs, new_log_probs)
cf = rl_metrics.clip_fraction(prob_ratios, clip_eps=0.2)
gn = rl_metrics.grad_norm(model.parameters())   # torch or numpy
```

No MLflow dependency — safe to import anywhere in a training codebase.

| Function | Input | Returns |
|---|---|---|
| `explained_variance(values, returns)` | 1-D numpy arrays | `float` (NaN if var≈0) |
| `approx_kl(log_probs_old, log_probs_new)` | 1-D numpy arrays | `float` |
| `clip_fraction(ratios, clip_eps=0.2)` | 1-D numpy array | `float` in [0, 1] |
| `grad_norm(parameters)` | torch param iterator or list of numpy arrays | `float` |

---

## 3. Model Registry

### 3a. Registering a checkpoint

After uploading a checkpoint with `log_checkpoint()`, register it in the MLflow Model Registry:

```python
logger.log_checkpoint("/path/to/best.pth", kind="best")

version = logger.register_checkpoint(
    model_name="shadow_hand_ppo",
    kind="best",
    description="PPO v3 — in-hand reorientation, seed 42",
)
print(f"Registered as version {version}")
```

### 3b. Promoting a model to Production

```python
logger.promote_model(
    model_name="shadow_hand_ppo",
    version=version,
    stage="Production",   # "Staging" | "Production" | "Archived"
)
```

### 3c. `ModelRegistry` — querying the registry

```python
from logger.model_registry import ModelRegistry

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

## 4. SystemMetricsLogger — Background Resource Logging

Spawns a daemon thread that periodically logs CPU, RAM, and GPU memory to MLflow without blocking training.

```python
from logger.system_metrics import SystemMetricsLogger

logger = make_logger(mode="mlflow", ...)
sys_logger = SystemMetricsLogger(logger, interval_seconds=30)

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
| `system/gpu_memory_mb` | `pynvml` or `nvidia-smi` |

Silently skips any metric whose dependency is not installed. The thread is a daemon — it will not prevent process exit if `stop()` is not called explicitly.

Optional installs:
```bash
pip install psutil pynvml
```

---

## 5. Git Commit Tracking

`MLflowLogger` automatically captures the git state of the training code at run
start. No extra code is needed — it is on by default.

### What gets recorded

| MLflow location | Key | Value |
|---|---|---|
| Tags tab | `git_commit` | Full SHA of HEAD — e.g. `54696cb326bb...` |
| Tags tab | `git_dirty` | `"false"` when tree is clean, `"true"` when there are uncommitted changes |
| Artifacts | `git/git_patch.diff` | Full `git diff HEAD` output — **only present when `git_dirty = "true"`** |

### Restoring a dirty-tree run

If training was launched with uncommitted changes, the exact state can be
recovered from any machine that has the base commit:

```bash
git checkout <git_commit>
# download git_patch.diff from the MLflow run's Artifacts tab
git apply git_patch.diff
```

### Opting out

Pass `track_git=False` to suppress all git tags and artifacts — useful when
training outside a git repo or in CI environments where the working tree is
intentionally untracked:

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

## 6. Smoke Test — Advanced Mode

The smoke test (`tests/smoke_test.py`) runs only core tests by default. Pass `--advanced` to also validate the features described in this document.

```bash
python tests/smoke_test.py --advanced
# Tests 1–5: core (always run)
# Test 6: rl_metrics helper accuracy
# Test 7: log_rl_metrics MLflow logging
# Test 8: SweepLogger parent-child runs
```

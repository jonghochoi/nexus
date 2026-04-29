# 🔧 Logger Setup Guide

How to replace `tensorboardX.SummaryWriter` with the unified logger in any training class.

> ✅ Only **3 locations** in your training code need to be modified. `write_stats()` and the rest of the training loop require **zero changes**.

---

## 📋 What Changes

| Location | Change | Lines modified |
|---|---|:---:|
| `import` | Replace `SummaryWriter` with `make_logger` | 1 |
| `__init__()` | Replace writer initialization | ~15 |
| `train()` | Add checkpoint logging + `writer.close()` | ~10 |
| `write_stats()` | **Nothing** | 0 |

---

## Step 1 — Replace the import

```python
# ❌ Before
from tensorboardX import SummaryWriter

# ✅ After
from nexus.logger import make_logger
```

---

## Step 2 — Replace writer initialization in `__init__()`

```python
# ❌ Before
self.writer = SummaryWriter(log_dir=output_dir)

# ✅ After
self.writer = make_logger(
    mode="dual",                         # "dual" | "mlflow" | "tensorboard"
    tb_dir=output_dir,                   # TensorBoard log dir (omit when mode="mlflow")
    run_name=os.path.basename(output_dir),
    tracking_uri="http://127.0.0.1:5100",
    experiment_name=agent_cfg.get("experiment_name", "robot_hand_rl"),
    params=agent_cfg,                    # logs all hyperparams once at run start
    tags={
        "researcher": os.environ.get("USER", "unknown"),
        "task":       agent_cfg.get("task", "unknown"),
        "hardware":   "robot_22dof",
    },
)
```

---

## Step 3 — Add checkpoint logging + `close()` in `train()`

Replace the per-epoch checkpoint block and add a close call:

```python
def train(self) -> None:
    best_score = -float("inf")

    for epoch in range(self.max_epochs):
        # ... training logic (unchanged) ...

        # ── checkpoint: last ──────────────────────────────────────────
        last_path = os.path.join(self.nn_dir, "last.pth")
        self.save(last_path)
        self.writer.log_checkpoint(last_path, kind="last")

        # ── checkpoint: best ──────────────────────────────────────────
        if current_score > best_score:
            best_score = current_score
            best_path = os.path.join(self.nn_dir, "best.pth")
            self.save(best_path)
            self.writer.log_checkpoint(best_path, kind="best")

    print("max steps achieved")
    self.writer.close()
```

`log_checkpoint(path, kind)` renames the file on upload so MLflow always stores exactly two checkpoint files: `checkpoints/best.pth` and `checkpoints/last.pth`. Each call silently overwrites the previous version for that kind.

---

## 🎛️ Switching Modes

Change **only** the `mode` argument. Everything else stays the same.

| `mode` | TensorBoard | MLflow | When to use |
|:---:|:---:|:---:|---|
| `"dual"` | ✅ | ✅ | **Recommended** — transition period |
| `"mlflow"` | ❌ | ✅ | After team is fully on MLflow |
| `"tensorboard"` | ✅ | ❌ | Rollback / no MLflow server available |

In `"tensorboard"` mode, `log_checkpoint()` and `log_artifact()` are silently ignored.

---

## ⚙️ Prerequisite: Local MLflow Server on GPU Server

> Before starting training with `mode="dual"` or `"mlflow"`, the local MLflow server must be running.

```bash
# Run once before any training jobs (stays alive in background)
bash scheduled_sync/start_local_mlflow.sh
```

This starts a local MLflow server on `127.0.0.1:5100`. All GPU processes on the server share this single server via loopback HTTP — no internet required.

---

## 💡 What Gets Stored Where

| Data | When | MLflow path |
|---|---|---|
| Hyperparameters (`agent_cfg`) | Run start | Parameters tab |
| Tags (experiment, researcher, task, hardware …) | Run start | Tags tab |
| `git_commit`, `git_dirty` tags | Run start | Tags tab |
| `git_patch.diff` | Run start *(dirty tree only)* | `git/git_patch.diff` |
| Metrics (loss, FPS, reward …) | Every step | Metrics tab |
| `best.pth` | When best score improves | `checkpoints/best.pth` |
| `last.pth` | Every epoch end | `checkpoints/last.pth` |

---

## 📄 Full Diff *(copy-paste ready)*

```diff
-from tensorboardX import SummaryWriter
+from nexus.logger import make_logger

 class YourTrainer:
     def __init__(self, env, output_dir, agent_cfg, ...):
         ...
-        self.writer = SummaryWriter(log_dir=output_dir)
+        self.writer = make_logger(
+            mode="dual",
+            tb_dir=output_dir,
+            run_name=os.path.basename(output_dir),
+            tracking_uri="http://127.0.0.1:5100",
+            experiment_name=agent_cfg.get("experiment_name", "robot_hand_rl"),
+            params=agent_cfg,
+            tags={
+                "researcher": os.environ.get("USER", "unknown"),
+                "task":       agent_cfg.get("task", "unknown"),
+                "hardware":   "robot_22dof",
+            },
+        )

     def write_stats(self, ...):
         # NO CHANGES HERE

     def train(self):
+        best_score = -float("inf")
         for epoch in range(self.max_epochs):
             # ... existing loop (unchanged) ...
+
+            last_path = os.path.join(self.nn_dir, "last.pth")
+            self.save(last_path)
+            self.writer.log_checkpoint(last_path, kind="last")
+
+            if current_score > best_score:
+                best_score = current_score
+                best_path = os.path.join(self.nn_dir, "best.pth")
+                self.save(best_path)
+                self.writer.log_checkpoint(best_path, kind="best")

         print("max steps achieved")
+        self.writer.close()
```

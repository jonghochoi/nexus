# 📦 Eval Artifact Ingestion

How an external evaluation pipeline lands its outputs (videos, plots, reports, scalar metrics) on the **same** central MLflow run that the trainer is logging to — without the trainer and the evaluator sharing any code.

> 📖 The `EvalLogger` API itself is summarised in [`30_ADVANCED_FEATURES.md`](30_ADVANCED_FEATURES.md). This guide is the **operational** companion: how the two sides hand off, what the contract is, and the metric/tag naming policy that keeps eval data discoverable in the MLflow UI.

---

## Table of contents

- [When to use this](#when-to-use-this)
- [The handoff contract](#the-handoff-contract)
- [`EvalLogger` API](#evallogger-api)
- [Artifact layout on the run](#artifact-layout-on-the-run)
- [Naming asymmetry: metrics vs tags](#naming-asymmetry-metrics-vs-tags)
- [Sourcing metrics from a JSON file](#sourcing-metrics-from-a-json-file)
- [Failure modes and their messages](#failure-modes-and-their-messages)
- [Worked example: a training-repo glue script](#worked-example-a-training-repo-glue-script)
- [Migrating trainers without `central_tracking_uri`](#migrating-trainers-without-central_tracking_uri)

---

## When to use this

The trainer has already opened an MLflow run via `make_logger()`. After (or alongside) training, a **separate** evaluation pipeline produces a directory of files — rollout videos, coverage heatmaps, an HTML report, a `metrics.json` — and you want those files to appear under that same run on central MLflow.

Concrete examples:

- A post-training rollout / scoring step that writes a directory of `mp4` + `metrics.json`.
- A standalone benchmark suite producing CSV/PNG comparison plots.
- A third-party RL eval pipeline (e.g. an external evaluator such as **observer**) called as a subprocess from the training repo. Nexus does not import that package; the training repo glues the two together.

What this is **not** for:

- Logging metrics live during training — that's `make_logger().add_scalar()` in [`11_LOGGER_SETUP.md`](11_LOGGER_SETUP.md).
- Bulk-uploading completed tfevents back-fills — that's Pipeline B (`upload_tb.py`) in [`13_POST_UPLOAD.md`](13_POST_UPLOAD.md).
- Promoting checkpoints into the Model Registry — that's `register_model.py` / `ModelRegistry`.

---

## The handoff contract

### ── The sidecar file

When the trainer calls `make_logger(tb_dir=output_dir, central_tracking_uri="http://nexus-server:5000", ...)`, the logger writes one file into `output_dir`:

```
output_dir/.nexus_run.json
```

Schema v1 (canonical: [`nexus/logger/run_info.py`](../nexus/logger/run_info.py)):

| Key | Required | Purpose |
|---|:---:|---|
| `schema_version` | yes | Integer; bumps only on incompatible changes |
| `run_name` | yes | The MLflow run name — used as the identity key (`tags.mlflow.runName`) on both local and central servers |
| `run_id` | yes | The local MLflow run UUID; not used by `EvalLogger` (the local and central UUIDs differ) |
| `experiment` | yes | Experiment name owning the run |
| `tracking_uri` | yes | The server the trainer is logging to (typically `http://127.0.0.1:5100`) |
| `central_tracking_uri` | optional | The team-shared central MLflow (e.g. `http://nexus-server:5000`); **required** for the recommended `target="central"` flow |
| `created_at` | optional | ISO-8601 UTC timestamp |

The sidecar is the **only** thing the eval step needs from the trainer. Pass the trainer's `output_dir` to `EvalLogger.from_run_info(...)` and you are done — no shared config, no env vars, no parsing of training launch flags.

### ── Why `run_name`, not `run_id`

The local MLflow relay (`:5100`) and the central server (`:5000`) maintain **independent** run UUIDs — even though `scheduled_sync` ships the same logical run to both. `EvalLogger` always resolves the target run by `tags.mlflow.runName` ([`eval_logger.py:368-372`](../nexus/logger/eval_logger.py)) so the same call works against either server, transparently. The local-only `run_id` recorded in the sidecar is **not** consulted by this path.

---

## `EvalLogger` API

```python
from nexus.logger.eval_logger import EvalLogger
```

### ── Construction

```python
# Recommended — read identity from the sidecar.
ev = EvalLogger.from_run_info(output_dir, target="central")
```

`from_run_info(path_or_dir, *, tracking_uri=None, target="central", verbose=True)` — see [`eval_logger.py:174`](../nexus/logger/eval_logger.py).

URI resolution (first non-empty wins):

1. Explicit `tracking_uri=` argument — overrides the sidecar.
2. `target="central"` (default) → sidecar's `central_tracking_uri`. Raises `ValueError` with a migration message if missing.
3. `target="local"` → sidecar's `tracking_uri`. Useful for debugging an in-progress run before the next `scheduled_sync` cycle.

The explicit form is also available when the sidecar is unavailable:

```python
ev = EvalLogger(
    run_name="ppo_v17_seed3",
    tracking_uri="http://nexus-server:5000",
    experiment="robot_hand_rl",
)
```

### ── `upload()`

```python
eval_id = ev.upload(
    eval_dir,                   # local directory; walked recursively
    *,
    eval_id=None,               # default: time.strftime("%Y%m%d_%H%M%S")
    metrics=None,               # {key: numeric}
    metrics_from=None,          # path to a JSON file
    tags=None,                  # {key: str}
    generate_index=True,        # auto-create index.html for video preview
    dry_run=False,              # resolve run + print plan; no MLflow round-trip
) -> str                        # the resolved eval_id
```

Definition at [`eval_logger.py:235`](../nexus/logger/eval_logger.py).

`upload()` may be called multiple times on the same `EvalLogger` instance — each call lands under a fresh `eval/<eval_id>/` subdirectory and never collides with previous bundles.

---

## Artifact layout on the run

Each `upload()` call lands as:

```
artifacts/
└── eval/
    └── <eval_id>/                    ← default: YYYYmmdd_HHMMSS
        ├── metrics.json
        ├── episodes.json
        ├── coverage/
        │   ├── heatmap_roll_pitch.png
        │   └── ...
        ├── videos/
        │   ├── front.mp4
        │   ├── side.mp4
        │   └── combined_grid.mp4
        └── index.html               ← auto-generated unless suppressed
```

Subdirectories of `eval_dir` are preserved verbatim ([`eval_logger.py:493-505`](../nexus/logger/eval_logger.py)).

`index.html` embeds any `.mp4` / `.webm` / `.mov` via `<video>` tags and any `.gif` / `.png` / `.jpg` / `.svg` via `<img>` tags — open it from the MLflow UI's Artifacts pane and rollouts play in-line without download. Suppress with `generate_index=False`. If `eval_dir` already contains an `index.html`, the auto-generator steps aside.

A single run can carry an unbounded number of `eval/<eval_id>/` bundles — there is no overwrite of `checkpoints/best.pth` or `checkpoints/last.pth` (the eval namespace is intentionally separate). The sentinel tag `eval.last_id` (see below) always points to the most recent bundle.

---

## Naming asymmetry: metrics vs tags

`EvalLogger.upload()` automatically prefixes the keys you pass — but **the prefixes are different for metrics and for tags**, and this catches everyone the first time. The asymmetry exists because MLflow's metric tab parses `/` as a chart-namespace separator while the tag panel does not.

| You pass | Becomes on the run | Why |
|---|---|---|
| `metrics={"success_rate": 0.91}` | metric key `eval/success_rate` | Slash → chart namespace. Plotted alongside `train/loss` etc. ([`eval_logger.py:335`](../nexus/logger/eval_logger.py)) |
| `metrics={"failure_distribution.success": 0.62}` | metric key `eval/failure_distribution.success` | Dotted leaf keys (from nested JSON flatten) survive — only the **top-level** prefix is added |
| `tags={"observer_commit": "abc123"}` | tag key `eval.observer_commit` | Dot prefix keeps it grouped with run-level tags ([`namespace_tags`, `eval_logger.py:132-138`](../nexus/logger/eval_logger.py)) |
| `tags={"mlflow.note.content": "..."}` | tag key `mlflow.note.content` *(unchanged)* | Keys already containing `.` pass through verbatim |

The implementation literally writes `f"eval/{k}"` for metrics and `f"eval.{k}"` for tags. **Never pre-prefix** — `metrics={"eval/success_rate": ...}` becomes `eval/eval/success_rate`, which is wrong.

In addition, **every** `upload()` call always stamps `eval.last_id = <eval_id>` ([`eval_logger.py:338`](../nexus/logger/eval_logger.py)) so consumers can find "the latest bundle" without scanning the artifact tree.

The metric step is `int(time.time())` so successive `upload()` calls plot in chronological order along the x-axis.

---

## Sourcing metrics from a JSON file

If your evaluator already writes `metrics.json`, point `EvalLogger` at it directly:

```python
ev.upload(
    eval_dir=result_dir,
    metrics_from=result_dir / "metrics.json",
)
```

`flatten_metrics_json()` ([`eval_logger.py:98-129`](../nexus/logger/eval_logger.py)) walks nested dicts with `.` separators and skips non-numeric leaves silently — so a file like:

```json
{
  "success_rate": 0.91,
  "failure_distribution": {"success": 0.91, "early_drop": 0.02, "late_slip": 0.03},
  "dominant_failure_mode": "success"
}
```

becomes the MLflow metric set:

```
eval/success_rate                       = 0.91
eval/failure_distribution.success       = 0.91
eval/failure_distribution.early_drop    = 0.02
eval/failure_distribution.late_slip     = 0.03
```

The string `dominant_failure_mode` is dropped (record it as a tag instead).

`metrics` and `metrics_from` may be supplied together — the explicit `metrics` dict wins on conflict ([`eval_logger.py:289-293`](../nexus/logger/eval_logger.py)).

---

## Failure modes and their messages

`EvalLogger` raises rather than `sys.exit()`-ing so callers can catch and recover. The error messages below are surfaced verbatim — grep your evaluator's logs for them.

### ── Sidecar issues

| When | Where raised | Message excerpt |
|---|---|---|
| `.nexus_run.json` missing at the path | `read_run_info` | `.nexus_run.json not found at {p} — was the run started with make_logger(tb_dir=...)?` |
| Sidecar lacks `central_tracking_uri` and you used `target="central"` | `from_run_info` | `{path}/.nexus_run.json has no central_tracking_uri — either upgrade the trainer to pass make_logger(central_tracking_uri=...), pass tracking_uri="http://<central>:5000" explicitly to from_run_info(), or call from_run_info(..., target="local") to use the sidecar's local tracking_uri.` |
| Sidecar missing required keys | `read_run_info` | `{p} is missing required keys: [...]` |

### ── Run resolution issues

| When | Message excerpt |
|---|---|
| Experiment does not exist on the target server | `Experiment not found: '{name}'. Create it via upload_tb.py or pass the correct experiment name.` |
| No run with that `run_name` in the experiment | `No run found in '{exp}' with run_name='{name}'.` |
| Multiple runs share the `run_name` | `Multiple runs share run_name='{name}' in '{exp}'. Disambiguate by deleting duplicates. run_ids: ...` |

### ── Eval directory issues

| When | Message excerpt |
|---|---|
| `eval_dir` does not exist | `eval_dir not found: {path}` |
| `eval_dir` is a regular file | `eval_dir is not a directory: {path}` |
| `eval_dir` contains zero files | `eval_dir is empty: {path}` |

`dry_run=True` short-circuits **after** the run is resolved but **before** any artifact upload, so it surfaces the sidecar / run-resolution failure modes above without touching the artifact tree — handy as a smoke check before the first real run.

---

## Worked example: a training-repo glue script

The script below sits in **the training repo** (e.g. `training_repo/scripts/run_eval_and_upload.py`). It imports both `nexus` and an external evaluator package, runs the evaluator on one checkpoint, then forwards the results to nexus. **Nexus and the external evaluator never import each other** — the training repo is the only glue.

```python
"""
training_repo/scripts/run_eval_and_upload.py
============================================
Lives in the TRAINING repo. Imports nexus + a downstream eval package
(here illustrated with ``observer``). Neither package depends on the
other; this script is the only glue.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

from nexus.logger.eval_logger import EvalLogger

# ── downstream eval package — NOT a nexus dependency ──
from observer.configs.eval_config import EvalConfig
from observer.pipeline.orchestrator import PipelineOrchestrator
from observer.pipeline.result_locator import locate_results, read_metrics
# ── end downstream ──


def _eval_commit() -> str:
    env = os.environ.get("OBSERVER_COMMIT")
    if env:
        return env
    try:
        import observer
        repo = Path(observer.__file__).resolve().parent.parent
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--training-output-dir", required=True, type=Path,
                   help="Dir holding .nexus_run.json from make_logger().")
    p.add_argument("--observer-config", required=True, type=Path)
    p.add_argument("--eval-output-dir", type=Path, default=None,
                   help="Defaults to <training-output-dir>/eval.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    eval_root = args.eval_output_dir or (args.training_output_dir / "eval")
    eval_root.mkdir(parents=True, exist_ok=True)

    # Run the downstream evaluator.
    cfg = EvalConfig.from_yaml(str(args.observer_config))
    orch = PipelineOrchestrator(config=cfg, output_root=eval_root)
    result = orch.run_single(args.checkpoint)
    if not result.success:
        print(f"[eval] downstream run_single failed: {result.error_msg}",
              file=sys.stderr)
        return 2

    obs = locate_results(eval_root, result_dir=result.output_dir)
    metrics = read_metrics(obs)  # already-flat dotted keys; do NOT pre-prefix
    if not metrics:
        print("[eval] no metrics produced — aborting upload", file=sys.stderr)
        return 3

    tags = {
        "observer_commit": _eval_commit(),         # → eval.observer_commit
        "checkpoint": args.checkpoint.name,        # → eval.checkpoint
        "skip_video": str(cfg.skip_video).lower(), # → eval.skip_video
    }

    # Hand off to nexus.
    ev = EvalLogger.from_run_info(args.training_output_dir, target="central")
    eval_id = ev.upload(
        eval_dir=result.output_dir,  # uploads recursively under eval/<eval_id>/
        metrics=metrics,             # logged as eval/<key> automatically
        tags=tags,                   # bare keys get eval. prefix
        generate_index=True,
        dry_run=args.dry_run,
    )
    print(f"[eval] uploaded eval_id={eval_id} ({len(metrics)} metrics) "
          f"under run_name={ev.run_name} on {ev.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Invocation:

```bash
python scripts/run_eval_and_upload.py \
    --checkpoint runs/ppo_v17_seed3/best.pth \
    --training-output-dir runs/ppo_v17_seed3 \
    --observer-config configs/eval_config.yaml
```

> ⚠️ This script lives in the training repo, **not** in nexus. We show it here as a reference template — copy it to your own repository and adapt the `observer` import lines if your downstream eval package is something else. The `EvalLogger` half of the contract is what nexus owns.

---

## Migrating trainers without `central_tracking_uri`

If your trainer was set up before `make_logger(central_tracking_uri=...)` was wired through, the sidecar's `central_tracking_uri` field will be absent and `EvalLogger.from_run_info(target="central")` will raise. Three options, in order of preference:

1. **Upgrade the trainer.** Add `central_tracking_uri="http://nexus-server:5000"` to the `make_logger()` call. The next training run writes a complete sidecar, and no eval-side changes are needed.
2. **Override at upload time.** Pass `tracking_uri="http://nexus-server:5000"` to `from_run_info()`. The sidecar's local URI is ignored and central is used directly.
3. **Fall back to local.** Pass `target="local"` to `from_run_info()`. Eval artifacts land on the GPU-node local MLflow (`:5100`); they reach central only after the next `scheduled_sync` cycle copies them over. Use this for debugging an in-progress run.

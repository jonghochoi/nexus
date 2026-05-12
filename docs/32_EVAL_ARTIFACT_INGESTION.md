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
- [Real-world eval bundles](#real-world-eval-bundles)
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

### ── Quick start

```python
from nexus.logger.eval_logger import EvalLogger

# make_logger() writes .nexus_run.json into output_dir during training.
# Pass the same output_dir here to pick up run_name / experiment / tracking_uri.
ev = EvalLogger.from_run_info(output_dir)

eval_id = ev.upload(
    eval_dir=output_dir / "eval",
    metrics={"success_rate": 0.87, "mean_return": 132.4},
    metrics_from=output_dir / "eval" / "metrics.json",
    tags={"observer_commit": "abc123"},
)
# → artifacts/eval/<eval_id>/ on the MLflow run
```

### ── CLI — `post_upload/upload_eval.py`

For shell-driven workflows (post-eval glue scripts, CI jobs, ad-hoc uploads from the operator desktop), `post_upload/upload_eval.py` wraps the same API. Sidecar-driven invocation matches the Python quick start one-for-one — pass the trainer's `tb_dir` to `--run-info` and the eval bundle to `--eval-dir`:

```bash
python post_upload/upload_eval.py \
    --run-info     ./logs/exp1/run_001/PPO \
    --eval-dir     ./eval_results/exp1__ckpt_5000__20260510 \
    --metrics-from ./eval_results/exp1__ckpt_5000__20260510/metrics.json \
    --tag observer.commit=abc123
```

Without a sidecar, supply the explicit triple:

```bash
python post_upload/upload_eval.py \
    --central-tracking-uri http://nexus-server:5000 \
    --experiment           robot_hand_rl \
    --run-name             exp1_run_001 \
    --eval-dir             ./eval_results/exp1__ckpt_5000__20260510
```

`--dry-run` resolves the run and lists what would be uploaded without touching MLflow. `--history` prints recent `upload_eval` invocations (recorded in `~/.nexus/history.json` alongside `upload_tb` / `register_model`).

Bare `--tag KEY=VAL` keys are auto-prefixed with `eval.`; pass a dotted key (e.g. `--tag observer.commit=abc123`) to bypass the prefix.

### ── Recommended eval_dir layout

`EvalLogger` walks `eval_dir` recursively and uploads everything it finds. A flat layout is fine; subdirectories are preserved under `eval/<eval_id>/` on the server.

```
eval_outputs/<run_name>/
├── rollout.mp4            ← full-resolution video — embedded in auto index.html
├── rollout_preview.gif    ← uploaded as-is, viewable from the Artifacts pane
├── report.md              ← uploaded as-is, downloadable
├── metrics.json           ← uploaded as-is (pass via metrics_from= for scalars)
└── success_rate.png       ← uploaded as-is, viewable from the Artifacts pane
```

> 💡 The auto index.html only embeds video files — MLflow already previews images and renders text/JSON inline, so no extra wrapper is needed for those. Inline embedding for additional file types may be added later; for now, ship your own `index.html` if you need a custom layout (the auto-generator steps aside).

### ── Upload options

**`generate_index` — auto-generated index.html**

MLflow 2.13's artifact viewer renders HTML inline but not `.mp4`. `EvalLogger` auto-generates an `index.html` next to any video file it finds, embedding it in a `<video controls>` tag. Open `eval/<eval_id>/index.html` in the Artifacts pane to play rollouts in-browser. If no video file is present, a short placeholder page is generated pointing the user back to the Artifacts pane. If `eval_dir` already contains an `index.html`, the auto-generator steps aside. Suppress explicitly with `generate_index=False`:

```python
ev.upload(eval_dir=..., generate_index=False)
```

> ⚠️ The video is embedded as a `data:video/mp4;base64,...` URI inside `<video src=...>`, not as a relative reference to the sibling mp4. MLflow's HTML preview iframe doesn't resolve relative URLs to sibling artifacts (the player UI appears but no mp4 request fires), so the bytes are inlined directly. Trade-off: HTML ends up ≈ 1.37× the mp4 size and must download fully before playback. Files over **30 MB raw mp4** fall back to a download link — the mp4 is always uploaded as a sibling artifact regardless, so download is the universal fallback. Raise the ceiling at `nexus/logger/eval_logger.py::_MAX_DATA_URI_BYTES` if your central MLflow tolerates larger HTML payloads.

**`eval_id` — stable name vs. timestamp**

`eval_id` defaults to a `YYYYmmdd_HHMMSS` timestamp, producing a new subdir on every call. Pass a fixed string to overwrite a previous bundle (e.g. during iterative debugging) or to give the folder a human-readable name:

```python
ev.upload(eval_dir=..., eval_id="checkpoint_500")
# → artifacts/eval/checkpoint_500/
```

**`verbose=False` — silent / programmatic mode**

Pass `verbose=False` to suppress all console output — useful when `EvalLogger` is called from a script rather than interactively:

```python
ev = EvalLogger.from_run_info(output_dir, verbose=False)
ev.upload(eval_dir=..., metrics={"sr": 0.9})
```

**`dry_run=True` — preview without uploading**

Pass `dry_run=True` to resolve the run and list what would be uploaded without touching MLflow. Surfaces sidecar and run-resolution errors without touching the artifact tree — handy as a smoke check before the first real run:

```python
eval_id = ev.upload(eval_dir=..., metrics={"sr": 0.87}, dry_run=True)
# Prints file list and metric preview; no artifacts or metrics written
```

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

Subdirectories of `eval_dir` are preserved verbatim ([`eval_logger.py`](../nexus/logger/eval_logger.py) — `_upload_artifacts`).

`index.html` embeds any `.mp4` / `.webm` / `.mov` via `<video controls>` tags so rollouts play in-line without download. **Other files (images, reports, JSON, …) are uploaded as-is into the same `eval/<eval_id>/` directory and previewed or downloaded individually from the MLflow Artifacts pane** — they are not embedded in the auto index. Inline rendering for additional file types may be added later; until then, drop your own `index.html` next to the bundle if you need a custom layout (the auto-generator detects it and steps aside). Suppress entirely with `generate_index=False`.

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

## Real-world eval bundles

`EvalLogger` does not distinguish simulator output from real-world capture sessions — it walks `eval_dir` and uploads whatever it finds, and any `.mp4` / `.webm` / `.mov` is embedded in the auto index regardless of how it was recorded ([`eval_logger.py:74-76`](../nexus/logger/eval_logger.py)). To keep **both** kinds of bundle legible on the same run, layer the conventions below on top of the API contract. No code changes are required — these are call-site and folder rules.

### ── `eval_id` prefix: `sim_` / `real_`

The default `eval_id` is `YYYYmmdd_HHMMSS`, which loses the origin as soon as real-world bundles join. Prefix the source onto the timestamp so the Artifacts pane groups them naturally (it sorts lexicographically):

```python
import time
ts = time.strftime("%Y%m%d_%H%M%S")

ev.upload(eval_dir="./sim_eval/",     eval_id=f"sim_{ts}")
ev.upload(eval_dir="./real_capture/", eval_id=f"real_{ts}")
ev.upload(eval_dir="./demo/",         eval_id="real_20260512_lab_demo")  # human-readable
```

Result on the run:

```
artifacts/eval/
├── sim_20260511_140230/
├── real_20260511_153012/
└── real_20260512_lab_demo/
```

The sentinel tag `eval.last_id` ([`eval_logger.py:344`](../nexus/logger/eval_logger.py)) always points to the most recent bundle, so uploading the real-world session last makes "the most recent eval" trivially discoverable.

### ── Tag schema for bundle metadata

`namespace_tags()` ([`eval_logger.py:134`](../nexus/logger/eval_logger.py)) prefixes bare tag keys with `eval.`. Pick a fixed schema so dashboards and `search_runs(filter_string=...)` queries are stable across bundles:

| Tag key | Example value | Purpose |
|---|---|---|
| `eval.source` | `"sim"` / `"real_world"` | Primary filter in MLflow UI search |
| `eval.session_date` | `"2026-05-12"` | Human-friendly capture date (distinct from the timestamp in `eval_id`) |
| `eval.operator` | `"jongho"` | Who ran the session |
| `eval.location` | `"lab_b_rig2"` | Physical rig or environment ID — for real-world bundles |
| `eval.checkpoint` | `"best.pth"` / `"epoch_500"` | Which checkpoint was evaluated (run-internal cross-check) |
| `eval.notes` | `"gripper recalibrated mid-session"` | Short free-form annotation |

```python
ev.upload(
    eval_dir="./real_capture/",
    eval_id=f"real_{ts}",
    tags={
        "source":       "real_world",
        "session_date": "2026-05-12",
        "operator":     "jongho",
        "location":     "lab_b_rig2",
        "checkpoint":   "best.pth",
        "notes":        "10 trials, gripper recalibrated mid-session",
    },
)
```

> ⚠️ Tags are written via `client.set_tag(run_id, k, v)` ([`eval_logger.py:346`](../nexus/logger/eval_logger.py)) — **run-level, not bundle-level**. A subsequent `upload()` that reuses the same key overwrites the previous value, so `eval.source` always reflects the **last** uploaded bundle. Use the manifest pattern below for per-bundle metadata that must never be lost.

### ── `manifest.json` for permanent per-bundle metadata

Because tags overwrite, the durable source of truth for "what was this specific bundle?" is a file inside the bundle itself:

```
real_capture_2026-05-12/
├── manifest.json
├── report.md
├── metrics.json
├── photos/
│   ├── setup.jpg
│   └── failure_mode_3.jpg
└── videos/
    ├── trial_01.mp4
    └── trial_02.mp4
```

`manifest.json` example:

```json
{
  "source":       "real_world",
  "session_date": "2026-05-12",
  "operator":     "jongho",
  "location":     "lab_b_rig2",
  "checkpoint":   "best.pth",
  "trials":       10,
  "notes":        "gripper recalibrated mid-session"
}
```

The manifest lives forever under `artifacts/eval/<eval_id>/manifest.json` and is not affected by later `upload()` calls. Tags remain the **search** surface; the manifest is the **truth** surface.

### ── Metric key grouping: `sim/` vs `real/`

Metric keys are written verbatim with an `eval/` prefix ([`eval_logger.py:341`](../nexus/logger/eval_logger.py)), and MLflow parses `/` as a chart-namespace separator. Embed the origin inside the metric key so simulation and real-world numbers land in **different chart groups** rather than overlapping on the same axis:

```python
ev.upload(eval_dir="./sim_eval/",     metrics={"sim/success_rate":  0.87, "sim/mean_return":  132.4})
ev.upload(eval_dir="./real_capture/", metrics={"real/success_rate": 0.62, "real/n_trials":    10})
```

Result on the run's Metrics tab:

```
eval/sim/success_rate
eval/sim/mean_return
eval/real/success_rate
eval/real/n_trials
```

If you prefer to load scalars from a JSON file via `metrics_from=`, note that `flatten_metrics_json` uses **dots**, not slashes — `{"sim": {"success_rate": ...}}` becomes `eval/sim.success_rate`, which is grouped under `eval/` only, not under `eval/sim/`. For slash-based chart grouping, pass the metrics through `metrics=` explicitly.

### ── Shipping a custom `index.html`

The auto-generated index only embeds video files. Real-world sessions typically want **video + written report + measurement table** on one page. Drop your own `index.html` into the bundle and the auto-generator steps aside ([`eval_logger.py:320`](../nexus/logger/eval_logger.py)):

```html
<!doctype html>
<html><head><meta charset="utf-8"><title>Real eval — 2026-05-12 lab_b_rig2</title></head>
<body>
  <h1>Real-world eval — 2026-05-12 lab_b_rig2</h1>
  <p>Operator: jongho · 10 trials · 6 success.</p>

  <h2>Rollouts</h2>
  <video controls src="videos/trial_01.mp4"></video>
  <video controls src="videos/trial_02.mp4"></video>

  <h2>Report</h2>
  <iframe src="report.html" style="width:100%;height:600px;border:0"></iframe>
</body></html>
```

MLflow's Artifacts pane renders `.md` as raw text, so if you want the report inline, pre-convert it to `.html` and include both in the bundle.

### ── End-to-end example

```python
from nexus.logger.eval_logger import EvalLogger

ev = EvalLogger.from_run_info(training_output_dir)   # uses .nexus_run.json from training

ev.upload(
    eval_dir="./real_capture_2026-05-12/",
    eval_id="real_20260512_lab_demo",
    metrics_from="./real_capture_2026-05-12/metrics.json",
    metrics={"real/success_rate": 0.6, "real/n_trials": 10},
    tags={
        "source":       "real_world",
        "session_date": "2026-05-12",
        "operator":     "jongho",
        "location":     "lab_b_rig2",
    },
)
```

Summary — folder prefixes for grouping, tags for search, manifest for permanence, slash-keyed metrics for chart separation, custom `index.html` when video alone isn't enough. None of this needs a code change in `EvalLogger`; it's a team-level naming policy.

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

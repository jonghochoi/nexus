# 🎛️ NEXUS Preset Guide

> A **preset** captures a recurring comparison view (experiment + filters + metrics + chart options) as a single YAML file. One CLI call then renders it to a standalone Plotly HTML report — no need to rebuild the MLflow UI filter state each time, and the "view" itself becomes a reviewable artifact in git.

---

## 📑 Contents

- [Why presets?](#why-presets)
- [Quick start](#quick-start)
- [Preset YAML reference](#preset-yaml-reference)
- [CLI usage](#cli-usage)
- [Typical workflows](#typical-workflows)
- [Maintenance](#maintenance)
- [Internal layout (swapping renderers)](#internal-layout-swapping-renderers)
- [Troubleshooting](#troubleshooting)

---

## Why presets?

| Situation | Without presets | With presets |
|---|---|---|
| Compare 10 LR-ablation runs | Filter tags in MLflow UI → check 12 run rows → pick 4 metrics → drag smoothing (every time) | `python -m presets render presets/examples/ppo_lr_ablation.yaml --open` (one line) |
| Attach to a weekly report | Multiple screenshots | Ship the generated `.html` (works offline / email / Confluence) |
| Share "the view I was looking at" | Screenshot + URL explanation | Send `presets/*.yaml` — or open a PR |
| Sim-to-Real gap check | Copy sim/real run IDs manually | Just edit `run_ids` in `sim_to_real_gap.yaml` |
| Audit trail of what we compare | None | `git log` on `presets/` |

**Bottom line:** a preset is a **declarative, reusable, version-controlled** statement of "what we want to see in MLflow."

### Preview the output without a server

Don't have an MLflow server to point at yet? Open
[`presets/examples/sample_output.html`](../presets/examples/sample_output.html)
in a browser — it's generated from fabricated training curves and shows
exactly what a rendered preset looks like (layout, interactivity, legend
grouping). Regenerate at any time with:

```bash
python presets/examples/_sample_demo.py
```

---

## Quick start

### 1) Verify install

```bash
source venv/bin/activate
python -c "import plotly, yaml; print('OK')"
```

`setup.sh` already installs `plotly` and `pyyaml`. If you're on an old venv, re-run:

```bash
bash setup.sh
```

### 2) List example presets

```bash
python -m presets list
```

Output:
```
  ppo_lr_ablation                robot_hand_rl         presets/examples/ppo_lr_ablation.yaml
  seed_spread                    robot_hand_rl         presets/examples/seed_spread.yaml
  sim_to_real_gap                robot_hand_rl         presets/examples/sim_to_real_gap.yaml
```

### 3) Render

```bash
python -m presets render presets/examples/ppo_lr_ablation.yaml \
    --tracking_uri http://<nexus-server>:5000 \
    --open
```

```
[NXS] resolving preset 'ppo_lr_ablation' against http://<nexus-server>:5000
[NXS] rendering 8 run(s) × 4 metric(s)
[OK]  wrote preset_outputs/ppo_lr_ablation.html
```

Your browser opens an interactive Plotly report with one subplot per metric.

---

## Preset YAML reference

```yaml
name: ppo_lr_ablation                    # [required] identifier (alphanumeric + underscore)
title: "PPO LR ablation — ..."           # report title (falls back to `name`)
experiment: robot_hand_rl                # [required] MLflow experiment name

description: |                           # free-form note displayed at the top of the report
  Why this preset exists and how to read it.

# ── 1) Filters ─────────────────────────────────────────────
#  Multiple keys are AND'd; list values inside one key are OR'd.
filters:
  tags:
    task: in_hand_reorientation          # single value — pushed down to MLflow filter_string
    researcher: [kim, lee]               # list — OR-filtered client-side
  params:
    "network.hidden_dim": 256            # quote the key if it contains a dot
  status: [FINISHED]                     # RUNNING / FINISHED / FAILED / ...

# ── 2) Run selection ───────────────────────────────────────
select:
  mode: latest                           # latest | all | explicit
  limit: 12                              # used only when mode=latest
  # run_ids: [id1, id2, ...]             # used only when mode=explicit

# ── 3) Metrics (one subplot per entry) ─────────────────────
metrics:
  - key: rl/success_rate
    smoothing: 0.6                       # same EMA as TensorBoard; [0, 1)
    y_label: "success rate"
  - rl/approx_kl                         # bare string uses all defaults
  - rl/explained_variance

# ── 4) Chart options ───────────────────────────────────────
chart:
  x_axis: step                           # step | timestamp
  group_by: params.lr.base               # color / legend key (dotted path into the Run)
  default_smoothing: 0.0                 # fallback when a metric has no `smoothing`
  height_per_plot: 320                   # pixels per subplot
```

### `group_by` paths

| Path | Meaning |
|---|---|
| `tags.<name>` | MLflow tag (e.g. `tags.seed`, `tags.researcher`) |
| `params.<name>` | MLflow param (e.g. `params.lr.base`) |
| `run_name` / `info.run_name` | `mlflow.runName` |
| `run_id` | First 8 chars of run ID |
| anything else | Looked up in tags first, then params; falls back to `run_id` |

### Select modes

| Mode | Behavior | When to use |
|---|---|---|
| `latest` | Most recent N runs that pass filters (start_time DESC) | Default — ablations, recent comparisons |
| `all` | Every run that passes filters | "Show them all" — e.g. seed spread |
| `explicit` | Only the IDs listed in `run_ids` | Pinned pairs like sim-to-real |

---

## CLI usage

```bash
# Render
python -m presets render <preset.yaml> \
    [--tracking_uri http://...] \
    [--output path/to/out.html] \
    [--open]

# List presets under a directory
python -m presets list [<dir>]

# Syntax check only — never contacts MLflow (safe for CI / pre-commit)
python -m presets validate <preset.yaml>
```

The default for `--tracking_uri` is `http://127.0.0.1:5000`. For day-to-day use, wrap it in an alias:

```bash
alias nxs-render='python -m presets render --tracking_uri http://nexus.internal:5000'
nxs-render presets/examples/seed_spread.yaml --open
```

---

## Typical workflows

### 🅰 Ablation report

1. One YAML per ablation under `presets/ablations/`.
2. Snapshot with `--output reports/$(date +%F)_<name>.html`.
3. Attach the HTML to the Confluence experiment page, or link to it.

### 🅱 Weekly static report

Cron a refresh every Monday morning:

```bash
0 9 * * 1 cd /opt/nexus && \
  python -m presets render presets/examples/seed_spread.yaml \
    --output /var/www/reports/seed_spread_weekly.html \
    --tracking_uri http://127.0.0.1:5000
```

### 🅲 Sim-to-Real pair tracking

Copy `sim_to_real_gap.yaml`, swap in the sim + real `run_ids`, render. The pairing itself lives in git, not just in someone's head — useful when you need to revisit an old experiment three months later.

### 🅳 New-researcher onboarding

Day one:

```bash
python -m presets list
python -m presets render presets/examples/ppo_lr_ablation.yaml --open
```

They immediately see which tags and which metrics the team considers important — no tribal knowledge required.

---

## Maintenance

### Where to put presets

| Location | Use |
|---|---|
| `presets/examples/*.yaml` | **Team-standard** presets, committed to the repo |
| `presets/personal/*.yaml` | Personal/experimental (add to `.gitignore`) |
| `~/.nexus/presets/*.yaml` | Global personal collection (just point `--output` wherever) |

> Team-standard presets should change via PR. That way "which comparisons we care about" gets reviewed like code, and `git log presets/` becomes an audit trail of evolving team priorities.

### When to update

| Trigger | Action |
|---|---|
| New metric namespace (`rl/new_metric`) | Add to the `metrics:` of any affected preset |
| Tag rename (e.g. `hardware` → `robot_model`) | Update `filters.tags:` and `group_by:` together |
| An ablation is finished and no longer watched | Move to `presets/archive/` (don't delete — keeps historical reports reproducible) |
| New preset proposal | PR + one-line `description:` stating *why* it exists |

### CI syntax check

Keep `presets/examples/*.yaml` valid at all times via pre-commit or CI:

```bash
find presets -name '*.yaml' -not -path '*/archive/*' \
  -exec python -m presets validate {} \;
```

`validate` never talks to MLflow and doesn't import `plotly`/`mlflow`, so it runs cheaply on any CI runner.

### Handling metric renames

1. Rename the metric in the training code (e.g. `rl/success_rate` → `rl/success`).
2. `grep -rl 'rl/success_rate' presets/` to find affected presets.
3. Bulk-update + open a PR.
4. Run `python -m presets validate presets/**/*.yaml` before merging.

---

## Internal layout (swapping renderers)

```
presets/
├── __init__.py      # Public API
├── __main__.py      # `python -m presets` entry point
├── cli.py           # argparse CLI
├── schema.py        # YAML → dataclass + validation  (no heavy deps)
├── resolver.py      # Preset + MLflow → ResolvedData  ← reused by all renderers
├── renderer.py      # ResolvedData → Plotly HTML
└── examples/
    ├── ppo_lr_ablation.yaml
    ├── seed_spread.yaml
    └── sim_to_real_gap.yaml
```

**The resolver is renderer-agnostic on purpose.** When we later add a Streamlit dashboard or an MLflow-compare-URL generator, the YAML schema and the resolver stay exactly as they are — only a new `renderers/` module is needed.

Adding a Streamlit renderer, for example, means dropping in `presets/renderers/streamlit_app.py` that consumes `ResolvedData`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `experiment '...' not found on http://...` | Check `--tracking_uri`. Confirm the experiment exists under that exact name in the MLflow UI. |
| `0 runs matched the preset filters` | Filters are too narrow. Run `python -m presets validate` first, then try the same tag filter in the MLflow UI by hand to cross-check. |
| `[WARN] empty metrics: [...]` | That metric key isn't actually logged by the matched runs. Check for typos and the correct namespace (`rl/`, `eval/`, etc.). |
| Rendered HTML is empty | Almost always one of the two warnings above — scroll the console output. |
| Too many lines; colors blur together | Reduce `select.limit`, or pick a coarser `group_by` (e.g. `tags.task`). |
| Preset loads fine but the numbers don't look right | Smoothing. Temporarily set `metrics[i].smoothing: 0.0` and re-render to see raw values. |
| `params.lr.base` doesn't work as `group_by` | Any MLflow param key with a dot must be quoted in YAML: `"params.lr.base"`. |

---

## See also

| Document | Description |
|---|---|
| [`README.md`](../README.md) | NEXUS overview |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design |
| [`LOGGER_SETUP.md`](LOGGER_SETUP.md) | Wire the logger into your trainer |
| [`ADVANCED_FEATURES.md`](ADVANCED_FEATURES.md) | SweepLogger, system metrics, and more |

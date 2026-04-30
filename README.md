<div align="center">

# 🔷 NEXUS · Centralized RL Experiment Hub

<img src="docs/LOGO.png" alt="NEXUS Logo" width="600">

**Stop chasing tfevents. Start comparing runs.**

*Unified logging · MLflow + TensorBoard · Air-gapped sync · Team-wide visibility*

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

> ### 📖 New here? Read this first.
>
> Every team-agreed rule and engineering invariant lives on **one page**: [`docs/00_PRINCIPLES.md`](docs/00_PRINCIPLES.md) — *5 min, English*.
> 🇰🇷 한글 온보딩 트랙은 [`docs/ko/`](docs/ko/)에서 시작하세요. 6개 필수 태그, `sim_run_id` 의무, 실패 Run 보존 등 모든 규칙이 정리되어 있습니다.

---

## 📌 Why NEXUS?

Dexterous manipulation demands running hundreds of experiments — reward shaping sweeps, tactile feedback ablations, Sim-to-Real transfer evaluations. Each run produces logs that scatter across individual machines, making team-wide comparison painful.

**NEXUS is the central point where all experiment data converges.**

| Without NEXUS | With NEXUS |
|---|---|
| Logs scattered across personal directories | All runs visible in one MLflow UI |
| "Can you send me your tfevents?" | Filter by experiment, compare curves instantly |
| Hyperparameters lost in commit history | Stored as MLflow params, searchable forever |
| Long training crashes → data gone | Incremental sync preserves intermediate results |
| Team decisions undocumented | Confluence pages linked to every run |

---

## 🏗️ Infrastructure

```
[GPU Server]                               [NEXUS Server]
  No internet — SCP/SSH only                 Internet accessible
  Isaac Lab / your trainer                   MLflow Tracking UI :5000

  ┌────────────────────────────┐             ┌─────────────────────────┐
  │  ╷  ╷  ╷  ╷  ╷             │             │                         │
  │  ●  ●  ●  ●  ●  (GPUs)     │             │  [NXS] MLflow central   │
  │  │  │  │  │  │             │   SCP/SSH   │                         │
  │  ●──●──●──●──●  (bus)      │ ──────────> │  All runs, all teams    │
  │         │                  │             │  Compare · Analyze      │
  │      local MLflow :5100    │             │                         │
  │      (loopback only)       │             │  http://<server>:5000   │
  └────────────────────────────┘             └─────────────────────────┘
```

---

## 🎛️ Logger Modes

Use `make_logger()` with the `mode` argument. Only this argument changes — everything else in your trainer stays exactly the same.

| `mode` | TensorBoard | MLflow | When to use |
|:---:|:---:|:---:|---|
| `"dual"` | ✅ | ✅ | **Recommended** — transition period |
| `"mlflow"` | ❌ | ✅ | After team fully adopts NEXUS |
| `"tensorboard"` | ✅ | ❌ | Rollback / no NEXUS server available |

---

## ⚡ Quick Start

```bash
git clone https://github.com/jonghochoi/nexus.git
cd nexus
bash setup.sh --alias          # installs venv at ~/.nexus/venv + registers `nexus-activate`
source ~/.bashrc               # pick up the alias
nexus-activate                 # works from any directory, any terminal
```

> The venv lives at `~/.nexus/venv` — **outside** the repo — so overwriting or re-cloning the source tree does not destroy the installed packages. Run `bash setup.sh --reinstall` if you ever need to rebuild it from scratch.
>
> Prefer no alias? Drop `--alias` and activate with `source ~/.nexus/activate.sh`.

---

## 📦 Use as a Python Dependency

External projects (a trainer repo, an inference service) can pip-install the logger package directly from this git repo — no need to clone or run `setup.sh`. The default install is **client-only** so it slots into environments that pin transitive deps (e.g. Isaac Lab pinning `prettytable`, `starlette`).

| Install target | `pyproject.toml` entry |
|---|---|
| **Trainer / CI** *(default — client only)* | `"nexus-logger @ git+https://github.com/jonghochoi/nexus.git"` |
| **Central MLflow host** *(adds the server stack)* | `"nexus-logger[server] @ git+https://github.com/jonghochoi/nexus.git"` |

The default pulls **`mlflow-skinny`** — `MlflowClient` and the tracking / entities APIs only — without Flask, SQLAlchemy, alembic, gunicorn. Add `[server]` only on the host that actually runs `mlflow server`. For reproducibility, pin to a tag or commit: `...nexus.git@v0.2.0` or `...nexus.git@<sha>`.

> **Isaac Sim / Isaac Lab**: Omniverse ships a partial `mlflow-skinny` stub (`dist-info` only, no module body), so after the install above also run `pip install --upgrade --force-reinstall --no-deps "mlflow-skinny>=2.0,<3"`. The package's import-time guard will tell you the same if you skip it — bake the line into your Dockerfile to avoid the runtime nag.

> The Quick Start above (`bash setup.sh`) is the **operator** path — for people who clone this repo to run `start_local_mlflow.sh`, the sync cron, or `tests/smoke_test.py`. External consumers don't need it.

---

## 🔀 Two ways to use NEXUS

|   | 🅰️ **Pipeline A** — Live logging | 🅱️ **Pipeline B** — Post-upload |
|---|---|---|
| **When** | New trainer / monitor a live run | Upload a completed tfevents in one shot |
| **Trainer code** | 3-line change: `SummaryWriter` → `make_logger` | Untouched |
| **Cadence** | Every step → local MLflow buffer → cron sync (every 5 min) | Manual, once per run dir |
| **Setup guides** | [`11_LOGGER_SETUP`](docs/11_LOGGER_SETUP.md) (code integration) → [`12_SCHEDULED_SYNC`](docs/12_SCHEDULED_SYNC.md) (cron sync) | [`13_POST_UPLOAD`](docs/13_POST_UPLOAD.md) |

### 30-second preview

**Pipeline A — embed in trainer:**

```python
from nexus.logger import make_logger

logger = make_logger(
    mode="dual",                                  # mlflow + tensorboard
    experiment_name="robot_hand_rl",
    run_name="ppo_baseline_v1",
    tracking_uri="http://127.0.0.1:5100",         # local MLflow on this GPU node
    tags={"researcher": "kim", "task": "in_hand_reorientation", "hardware": "robot_22dof"},
)
logger.add_scalar("train/loss", 0.5, step=100)    # SummaryWriter-compatible
```

Local-MLflow on `127.0.0.1:5100` is started by `bash scheduled_sync/start_local_mlflow.sh`. A cron job (registered via [`12_SCHEDULED_SYNC`](docs/12_SCHEDULED_SYNC.md)) packages new metric points and artifact files (checkpoints, configs, git diffs, eval reports — anything logged via `MLflowLogger`) into a tar.gz delta and ships it to the central server every 5 minutes, so the full run — metrics **and** artifacts — is browsable in the central MLflow UI.

**Pipeline B — upload after training ends:**

```bash
# One-time: set tracking_uri + your fixed tags in ~/.nexus/post_config.json
python post_upload/upload_tb.py --tb_dir /path/to/logs/run_001
# → prompts for missing required tags, uploads, auto-verifies
```

The full flag reference, interactive mode, upload history, `sim_run_id` auto-detection, and troubleshooting live in [`13_POST_UPLOAD`](docs/13_POST_UPLOAD.md).

> ⚠️ **Multi-user GPU server (Pipeline A)** — each user must set their own `researcher` in `~/.nexus/sync_config.json` so cron jobs don't re-export each other's runs. Canonical: [`docs/00_PRINCIPLES.md#multi-user-researcher`](docs/00_PRINCIPLES.md#-multi-user-researcher).

> 💡 **Long-running training** needs Pipeline A (scheduled, incremental). Pipeline B is a one-shot batch upload — use it for back-filling completed runs that were written by an unmodified `SummaryWriter`.
>
> 🎬 **Already have eval artifacts** (rollout mp4, scores, reports) for an existing run? `python post_upload/upload_eval.py --run_name <name> --eval_dir <path>` attaches them under `eval/<id>/` and auto-generates an `index.html` so MLflow's UI plays the mp4 inline. See [`13_POST_UPLOAD`](docs/13_POST_UPLOAD.md) §6.3 for the full flag list and recommended `eval_dir` layout.

---

## 🖥️ What's Running Where

```
🖥️  GPU Server  ───────────────────────────────────────────────
│
├── 🤖  Your Trainer Process
│   └── 🔀  DualLogger
│       ├── 📁  → tfevents/         local disk  (tensorboard --logdir)
│       └── 📡  → 127.0.0.1:5100    local MLflow server
│
├── 🗄️  Local MLflow Server         (start_local_mlflow.sh — always on)
│   └── 💾  all run data stored in mlruns/
│
└── 🔄  sync_mlflow_to_server.sh    (cron, every 5 min)
    └── ⬆️  :5100 ──SCP──► central :5000

🌐  Central Server  ───────────────────────────────────────────
└── 📊  MLflow Server :5000
    └── 🧑‍🤝‍🧑  full team experiment history · UI · run comparison
```

| When | Where to look |
|---|---|
| ⚡ During training | Local server `localhost:5100` — instant, no internet needed |
| 👥 Team review / run comparison | Central server `:5000` |
| 🔌 Network outage | No data loss — local server buffers everything until sync resumes |

---

## 🏷️ Recommended Tags *(reproducibility)*

> Canonical sources: [`docs/00_PRINCIPLES.md#required-tags`](docs/00_PRINCIPLES.md#-required-tags), [`docs/ko/02_EXPERIMENT_STANDARD.md` § 3-1](docs/ko/02_EXPERIMENT_STANDARD.md#-3-tags-규칙), and [`post_upload/config.py::required_tags()`](post_upload/config.py).

| Tag | Example | Required |
|---|---|:---:|
| `experiment` | `robot_hand_rl` | ✅ *(auto from `--experiment`)* |
| `researcher` | `kim` | ✅ |
| `task` | `in_hand_reorientation` | ✅ |
| `hardware` | `robot_22dof` | ✅ |
| `sim_run_id` | `<upstream_run_id>` | ✅ *(real-robot eval only)* |
| `git_commit` | `54696cb326bb...` | auto *(Pipeline A)* |
| `git_dirty` | `false` / `true` | auto *(Pipeline A)* |

> 💡 `sim_run_id` links a real-robot evaluation run back to the exact sim policy deployed — critical for Sim-to-Real failure tracing.

> 💡 `git_commit` and `git_dirty` are set automatically by `MLflowLogger` (Pipeline A). For Pipeline B post-uploads, pass `--git_commit <hash>` manually. When `git_dirty=true`, the full diff is saved as `artifacts/git/git_patch.html` (previewable inline in the MLflow UI). See [`docs/30_ADVANCED_FEATURES.md`](docs/30_ADVANCED_FEATURES.md#-5-git-commit-tracking) for details.

---

## 📚 Further Reading

> Filename prefix conveys reading order. **Everyone reads `00_PRINCIPLES.md` first.** Korean team members continue in [`docs/ko/`](docs/ko/); engineers and operators pick up the relevant track below.

| # | Document | Description |
|:---:|---|---|
| **00** | [`docs/00_PRINCIPLES.md`](docs/00_PRINCIPLES.md) | **Read first.** Team-agreed rules + engineering invariants (single canonical source) |
| **ko** | [`docs/ko/`](docs/ko/) | 🇰🇷 한글 온보딩 트랙 — `01_INTRO.md` (동기/FAQ), `02_EXPERIMENT_STANDARD.md` (운영 표준) |
| **10** | [`docs/10_ARCHITECTURE.md`](docs/10_ARCHITECTURE.md) | Full system design and component map |
| **11** | [`docs/11_LOGGER_SETUP.md`](docs/11_LOGGER_SETUP.md) | Pipeline A — logger integration step-by-step diff |
| **12** | [`docs/12_SCHEDULED_SYNC.md`](docs/12_SCHEDULED_SYNC.md) | Pipeline A — cron sync wiring (config, validate, multi-user, verification checklist) |
| **13** | [`docs/13_POST_UPLOAD.md`](docs/13_POST_UPLOAD.md) | Pipeline B — `upload_tb` / `verify_tb` / `upload_eval` CLIs: config, interactive, history, `sim_run_id`, eval artifact attach |
| **20** | [`docs/20_MLFLOW_SERVER_SETUP.md`](docs/20_MLFLOW_SERVER_SETUP.md) | Operator — central MLflow server install (Step 0 includes local PC verification) |
| **21** | [`docs/21_AIRGAPPED_GPU_SERVER_SETUP.md`](docs/21_AIRGAPPED_GPU_SERVER_SETUP.md) | Operator — GPU node offline bring-up (Step 0 + Step 1 include local + GPU verification) |
| **30** | [`docs/30_ADVANCED_FEATURES.md`](docs/30_ADVANCED_FEATURES.md) | Opt-in — SweepLogger, RL metrics, Model Registry, system metrics, git tracking |
| **31** | [`docs/31_CHART_SETTINGS_GUIDE.md`](docs/31_CHART_SETTINGS_GUIDE.md) | Opt-in — persist MLflow chart/column settings across browser sessions |
| — | [`brand.py`](brand.py) | ASCII art, sigils, and color constants |

---

## 📦 Dependencies

| Package | Version |
|---|---|
| `mlflow` | `2.13.0` |
| `tbparse` | `0.0.8` |
| `tensorboard` | `2.16.2` |
| `tensorboardX` | latest |
| `pandas` | latest |
| `rich` | latest |

```bash
bash setup.sh             # installs all dependencies into ~/.nexus/venv
bash setup.sh --alias     # same, plus register `nexus-activate` in ~/.bashrc
bash setup.sh --reinstall # wipe and recreate ~/.nexus/venv (after source overwrite)
```

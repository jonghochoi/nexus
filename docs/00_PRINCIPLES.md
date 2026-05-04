# 🧭 NEXUS Principles & Assumptions

> **The shortest doc in the repo. Read this first.**
>
> Every team-agreed rule and every cross-cutting technical invariant lives here, with a one-line statement and a link to the canonical source. If a principle conflicts between this doc and a deeper guide, **this doc loses** — fix the conflict at the source and update this doc.

---

## Stable Anchors *(do not rename — external docs link here)*

| Anchor | Title |
|---|---|
| [`#tool-role-separation`](#-tool-role-separation) | MLflow = numbers, Confluence = judgment |
| [`#required-tags`](#-required-tags) | 1 required tag: `experiment` |
| [`#sim-run-id`](#-sim_run_id) | Real-robot eval must carry `sim_run_id` |
| [`#failed-run-preservation`](#-failed-run-preservation) | Never delete failed runs |
| [`#single-cron`](#-single-cron) | Exactly one cron per GPU server — `sync_mlflow_all.sh` syncs all team members' runs |
| [`#checkpoint-policy`](#-checkpoint-policy) | Exactly two artifacts per run: `best.pth`, `last.pth` |
| [`#state-file`](#-state-file) | `~/.nexus/sync_state/` is the source of truth for sync position |
| [`#default-uris`](#-default-uris) | Local 5100, central 5000 — change in lockstep |
| [`#hypothesis-first`](#-hypothesis-first) | Write the hypothesis before starting the run |
| [`#mlflow-skinny-contract`](#-mlflow-skinny-contract) | Runtime depends on `mlflow-skinny` only |

---

## Team Agreements

These are rules every team member must follow. Detailed walkthroughs (Korean) live in [`ko/02_EXPERIMENT_STANDARD.md`](ko/02_EXPERIMENT_STANDARD.md).

### ── Tool role separation

> **MLflow stores numbers. Confluence stores judgment. Never mix the two.**

If you write interpretation in MLflow descriptions, the team won't see it. If you paste numbers into Confluence, they go stale the moment a run is rerun. Detailed rationale and templates: [`ko/02_EXPERIMENT_STANDARD.md` § 0](ko/02_EXPERIMENT_STANDARD.md#0-도구-역할-분리-원칙).

### ── Required tags

Every run must carry this tag:

`experiment`

The **single source of truth** is the code: [`post_upload/config.py::required_tags()`](../post_upload/config.py). Additional tags (e.g. `researcher`, `task`, `hardware`, `seed`) are useful metadata but are not enforced.

### ── `sim_run_id`

Every real-robot eval run **must** carry a `sim_run_id` tag pointing at the upstream sim training run. Without it, Sim-to-Real failure tracing is impossible. Enforcement:
- **Pipeline A** — set the tag in `make_logger(tags={...})`
- **Pipeline B** — drop a `run_meta.json` next to the tfevents, or pass `--tags sim_run_id=...`. Uploads to `--experiment real_robot_eval` are **blocked** if missing.

Detail: [`ko/02_EXPERIMENT_STANDARD.md` § 8](ko/02_EXPERIMENT_STANDARD.md#8-sim-to-real-연결-규칙), [`13_POST_UPLOAD.md` § 5](13_POST_UPLOAD.md).

### ── Failed run preservation

> **Never delete a failed run.**

Stamp `fail_reason` on it, write the failure analysis in Confluence, and leave the run in MLflow forever. Same-mistake-twice prevention is one of NEXUS's core values. Detail: [`ko/02_EXPERIMENT_STANDARD.md` § 7](ko/02_EXPERIMENT_STANDARD.md#7-failed-run-처리-규칙).

### ── Hypothesis first

Write the experiment's hypothesis in Confluence **before** launching the run. Writing the hypothesis after seeing results is post-hoc rationalization, not science. Detail: [`ko/02_EXPERIMENT_STANDARD.md` § 6](ko/02_EXPERIMENT_STANDARD.md#6-실험-생명주기).

---

## Engineering Invariants

These are technical contracts. Violating them silently breaks the system.

### ── Single cron

> ⚠️ On shared GPU servers, **exactly one cron must run on the entire server.**

The operator registers one cron using `sync_mlflow_all.sh`. That wrapper
auto-discovers all experiments from local MLflow each tick and calls
`sync_mlflow_to_server.sh` per experiment. A single state file per experiment
tracks every run_id independently — no per-user filter is needed because there
are no competing crons. Team members need zero sync knowledge.

A second cron from any user on the same server creates a competing state file
and causes duplicate metric points on the central server. `validate_sync.sh`
and `sync_mlflow_all.sh` both warn when a duplicate cron is detected.

Detail: [`12_SCHEDULED_SYNC.md` Step 5](12_SCHEDULED_SYNC.md#step-5--multi-user-servers).

### ── Checkpoint policy

> **Exactly two artifacts per run, ever:** `checkpoints/best.pth`, `checkpoints/last.pth`.

`MLflowLogger.log_checkpoint(path, kind)` enforces `kind ∈ {"best", "last"}` and renames on upload. Both are overwritten in place — intermediate checkpoints (`ep_100_*.pth`) belong on local disk, not in MLflow. Detail: [`10_ARCHITECTURE.md`](10_ARCHITECTURE.md), [`nexus/logger/mlflow_logger.py`](../nexus/logger/mlflow_logger.py).

### ── State file

> The file at `~/.nexus/sync_state/{experiment}.json` is the **source of truth** for "what has been synced."

Deleting it forces a full re-sync on the next cron tick. It used to live in `/tmp` but `/tmp` is wiped on reboot, which silently triggered full re-syncs every cycle. Do not move it back. Detail: `CLAUDE.md`, [`scheduled_sync/export_delta.py`](../scheduled_sync/export_delta.py).

### ── Default URIs

| URI | Role |
|---|---|
| `http://127.0.0.1:5100` | Local MLflow on every GPU server (loopback only) |
| `http://127.0.0.1:5000` / `http://nexus-server:5000` | Central MLflow on the NEXUS server |

These appear hardcoded as defaults across `nexus/logger/`, `scheduled_sync/`, `post_upload/`, `chart_settings/`, and the README diagrams. **Change them in concert** — `grep -rn "5100\|5000"` and update every site. Detail: `CLAUDE.md` § "When adding new features".

### ── mlflow-skinny contract

> Runtime code may only use APIs that exist in `mlflow-skinny`.

The default install (`pip install nexus-logger`) pulls `mlflow-skinny` to slot into pinned environments like Isaac Lab. Allowed APIs: `MlflowClient`, `mlflow.entities.*`, `mlflow.set_tracking_uri/set_experiment/get_experiment_by_name`, `register_model`, `set_experiment_tag`. **Forbidden** (full-mlflow only): `mlflow.pyfunc.load_model`, `mlflow.models.Model` flavor builders, anything under `mlflow.server.*`. The only full-mlflow consumer is `mlflow server` inside `scheduled_sync/start_local_mlflow.sh`. Detail: `CLAUDE.md` § "Things to be careful about", [`pyproject.toml`](../pyproject.toml).

---

## Where to go next

| Audience | Next read |
|---|---|
| 🇰🇷 Korean team members onboarding | [`ko/README.md`](ko/README.md) — Korean track index |
| 🛠️ Engineer integrating Pipeline A | [`10_ARCHITECTURE.md`](10_ARCHITECTURE.md) → [`11_LOGGER_SETUP.md`](11_LOGGER_SETUP.md) → [`12_SCHEDULED_SYNC.md`](12_SCHEDULED_SYNC.md) |
| 📤 Engineer using Pipeline B (post-upload) | [`13_POST_UPLOAD.md`](13_POST_UPLOAD.md) |
| 🖥️ Operator — central server install | [`20_MLFLOW_SERVER_SETUP.md`](20_MLFLOW_SERVER_SETUP.md) (Step 0 covers local-PC verification) |
| 🖥️ Operator — air-gapped GPU node bring-up | [`21_AIRGAPPED_GPU_SERVER_SETUP.md`](21_AIRGAPPED_GPU_SERVER_SETUP.md) (Step 0 + Step 1 cover verification) |
| ⚡ Anyone exploring opt-in features | [`30_ADVANCED_FEATURES.md`](30_ADVANCED_FEATURES.md), [`31_CHART_SETTINGS_GUIDE.md`](31_CHART_SETTINGS_GUIDE.md) |

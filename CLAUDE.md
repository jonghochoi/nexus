# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

NEXUS is a centralized RL experiment hub for an air-gapped GPU-server / internet-accessible-MLflow-server topology. It funnels training metrics, configs, and checkpoints from many GPU machines into one MLflow tracking server so the team can compare runs in one UI. See `README.md` for motivation, the infrastructure diagram, and user-facing walkthroughs; this file focuses on what's needed to **modify the code** safely.

## Environment / common commands

Venv setup and activation are covered in `README.md` → "Quick Start" (`bash setup.sh [--alias|--reinstall]`, `source ~/.nexus/activate.sh`). Two things matter for code changes:

- The venv is at `~/.nexus/venv` — **outside** the source tree — so overwriting the repo does not wipe installed packages. `~/.nexus/` also holds the user's `post_config.json` (Pipeline B), `sync_config.json` (Pipeline A), `sync_state/{exp}.json`, and `history.json`.
- There is no package install (no `setup.py`/`pyproject.toml`); scripts are run directly and `nexus/logger/` is imported via `sys.path.insert(0, ".")` from the repo root (which puts the `nexus` package on `sys.path`).

Smoke / end-to-end tests (require an MLflow server reachable at `--tracking_uri`):

```bash
bash scheduled_sync/start_local_mlflow.sh        # starts local MLflow on 127.0.0.1:5100
python tests/smoke_test.py                       # core: imports, MLflowLogger, DualLogger, factory
python tests/smoke_test.py --advanced            # also: rl_metrics, log_rl_metrics, SweepLogger
python tests/smoke_test.py --tracking_uri http://<host>:5000   # against a different server
```

There is no pytest config and no linter — `smoke_test.py` is a hand-rolled script with sectioned PASS/FAIL output. Run it from the repo root (it does `sys.path.insert(0, ".")` to import the `nexus.logger` package). It writes to a real `nexus_smoke_test` experiment on whatever server you point it at.

Pipeline B CLI smoke (no MLflow upload):

```bash
cd post_upload && python upload_tb.py --tb_dir <path> --dry_run
python upload_tb.py --history                   # show ~/.nexus/history.json
python verify_tb.py --from-last                 # re-verify the last upload
python upload_eval.py --run_name <name> --eval_dir <path>   # attach eval artifacts (mp4, report) to an existing run
```

## High-level architecture

The system has **two independent pipelines** for getting data into central MLflow. Treat them as separate codebases that share only the metric-name sanitizer and tag conventions.

### Pipeline A — `nexus/logger/` + `scheduled_sync/` (live training)

Used when training code is modified to call `make_logger()`. Flow:

1. PPO calls `make_logger(mode="dual"|"mlflow"|"tensorboard", ...)` once at `__init__`. The returned object is `SummaryWriter`-compatible (`add_scalar`, `add_histogram`, `add_image`, `close`), so the rest of the trainer is unchanged.
2. `MLflowLogger` writes to a **local** MLflow server on `127.0.0.1:5100` (started by `scheduled_sync/start_local_mlflow.sh`). It buffers `add_scalar` calls per-step in memory and flushes the whole step's metrics with a single `client.log_batch()` when `global_step` advances. The `_BATCH_SIZE = 1000` constant matches MLflow's hard per-call limit.
3. Run identity is `run_name`, not `run_id` — `_get_or_create_run` searches for an existing run by `tags.mlflow.runName` and resumes it if found, so a crash + restart of the trainer reattaches to the same MLflow run instead of creating a duplicate.
4. `git_utils.get_git_info()` is called automatically (suppress with `track_git=False`) and stamps `git_commit` / `git_dirty` tags. If dirty, the full `git diff HEAD` is uploaded as `artifacts/git/git_patch.diff`.
5. A cron job runs `scheduled_sync/sync_mlflow_to_server.sh` every N minutes. The shell wrapper resolves required values (`experiment`, `remote`, `remote_nexus_dir`, optional `researcher`/`ssh_key`/`ssh_port`/`local_uri`/`remote_uri`/`state_file`) in this order, first non-empty wins per key: CLI flag → `--config <path>` JSON (when given, disables auto-discovery) → `~/.nexus/sync_config.json` (per-user) → `/etc/nexus/sync_config.json` (system / team-wide) → built-in default (only for `local_uri`/`remote_uri`/`ssh_port`). It chains:
   - `export_delta.py` on the GPU server → reads `~/.nexus/sync_state/{experiment}[__{researcher}].json` to find each run's last-synced step per tag, queries the local MLflow, writes only **new** points to a delta JSON. When `--researcher <name>` is set (REQUIRED on shared GPU servers), runs are filtered by `tags.researcher = '<name>'` so each user's cron only exports their own runs — without this filter, parallel cron jobs re-export each other's runs and the central server logs duplicate metric points at identical steps. Exit codes are meaningful: `0` = data to transfer, `1` = configuration error (e.g. unknown experiment), `2` = no new data (wrapper skips SCP).
   - `scp` of the delta JSON to the central server's inbox. The local delta filename is `delta_${USER}_<TS>_<PID>.json` so concurrent cron jobs from different users on the same GPU server cannot collide in `/tmp` or in the remote inbox.
   - `ssh` invocation of `import_delta.py` on the central server, which `get_or_create`s the run by `run_name` and `log_batch`es the new metrics. Params + tags are sent only on first appearance of a run; the run is then stamped with `nexus.lastSyncTime` (UTC ISO) and `nexus.syncedFromHost` (origin GPU server hostname carried in the delta JSON's `source_host` field) so the central UI can flag stale GPU servers.

The state file at `~/.nexus/sync_state/{experiment}[__{researcher}].json` is the source of truth for "what has been synced." Deleting it forces a full re-sync on the next run. (The default location used to be `/tmp/nexus_delta_{experiment}.json`, but `/tmp` is wiped on reboot on most distros and silently triggered a full re-sync every cycle.) When `--researcher` is set, the file is namespaced by researcher so one operator account can host multiple sync identities without their state bleeding into each other.

`scheduled_sync/validate_sync.sh` is a pre-flight checker for cron registration: it loads the same config resolution chain, then verifies SSH reachability, remote inbox writability, presence of `import_delta.py` on the server, remote MLflow `/health`, local MLflow experiment existence (and presence of runs for `researcher` if set), and finally runs `--dry-run` once. Exits 0 only if every step passes; prints a paste-ready cron line on success but never edits crontab itself. Emits a `[WARN]` if `researcher` is unset, since silent multi-user duplication is the most insidious scheduled_sync failure mode.

### Pipeline B — `post_upload/` (one-shot, no trainer changes)

Used to back-fill completed tfevents that were written by an unmodified `SummaryWriter`, and to attach post-hoc evaluation artifacts (mp4 rollouts, reports) to an existing run.

- `upload_tb.py` parses tfevents with `tbparse`, builds `Metric` entities directly from numpy arrays (do not use `iterrows` — the file comments call out that vectorized zip is ~50x faster), and `log_batch`es in chunks of 1000.
- `verify_tb.py` re-fetches the uploaded run and compares tag list / point counts / per-step values against the TB source within `--tolerance`.
- `upload_eval.py` resolves an existing run by `run_name` (via `tags.mlflow.runName` search) and `log_artifacts`es a directory under `eval/<eval_id>/` — never `checkpoints/`, to keep the `best.pth`/`last.pth` policy intact. It also synthesizes an `index.html` viewer next to the mp4 so the MLflow UI can preview the rollout inline (MLflow 2.13's artifact viewer does not natively render mp4, but it does render HTML, and a sibling `<video>` tag works).

Important behaviors (apply primarily to `upload_tb.py`):

- **Multi-run protection**: if `tb_dir.rglob("events.out.tfevents.*")` finds tfevents in more than one parent directory, the script aborts with an error rather than silently merging multiple runs into one MLflow run (which would cause step collisions). Always upload one run dir at a time.
- **Required tags** are decided by `config.required_tags(experiment)`: always `(researcher, seed, task)`, plus `sim_run_id` if the experiment is in `REAL_EVAL_EXPERIMENTS` (currently just `real_robot_eval`). Missing tags drop the CLI into interactive prompts on a TTY; `--force` skips validation; `--dry_run` skips upload entirely.
- **Tag precedence**: 7-level chain (builtin → config → `--repeat-last` → `run_meta.json` → `--tags` → `--git_commit` → `-i`) is documented in `docs/13_POST_UPLOAD.md` §2 — keep that table as the single source of truth if you change the order.
- **Auto-verify**: after upload, `run_verify(run_id, tb_dir, tracking_uri)` is invoked unconditionally unless `--no_verify` is set. If verification fails, the script still records the upload in history but exits `2` (so CI can branch on it).
- **History**: every upload is prepended to `~/.nexus/history.json` (capped at `HISTORY_LIMIT=20`). Records carry a `script` field (`"upload_tb"` / `"upload_eval"`) so the two pipelines coexist; legacy records without `script` are treated as `"upload_tb"`. `--repeat-last` and `verify_tb.py --from-last` filter by `script="upload_tb"` so they never resurrect an eval record.
- The post_upload scripts inject `sys.path.insert(0, str(Path(__file__).resolve().parent))` so they import sibling modules (`config`, `history`, `verify_tb`) whether invoked from the repo root or from `post_upload/`.

### Cross-cutting conventions

- **Default URIs**: GPU-server local MLflow is `http://127.0.0.1:5100`; central MLflow is `http://127.0.0.1:5000` (and `http://nexus-server:5000` from clients). These appear hardcoded as defaults across many files — change them in concert.
- **Metric name sanitization**: `name.replace(" ", "_").replace(":", "-")`. Slashes are preserved so TensorBoard's `losses/actor_loss` hierarchy survives. **Three** copies must stay in lock-step — `MLflowLogger._sanitize` (logger), `upload_tb.sanitize_metric_name` (uploader), and `verify_tb.sanitize_metric_name` (verifier applies it to the TB-side tags before comparing) — or `verify_tb.py` will report tag-list mismatches.
- **Param flattening**: `MLflowLogger._flatten` recursively flattens nested dict params with `.` separator. Lists/tuples are stored via `str(v)` (not flattened).
- **Required reproducibility tags**: the canonical statement is in `docs/00_PRINCIPLES.md` → `#required-tags` (which links the code and the team-facing description). The code-side enforcement lives in `post_upload/config.py::required_tags()` — base is `(researcher, seed, task)`, and experiments in `REAL_EVAL_EXPERIMENTS` additionally require `sim_run_id`. The team-facing description with examples is `docs/ko/02_EXPERIMENT_STANDARD.md` § 3-1, and the user-facing summary is `README.md` → "Recommended Tags". If you add a new required tag, change all four sites.
- **Checkpoint policy**: only two artifacts ever exist under `checkpoints/` in MLflow — `best.pth` (highest score so far) and `last.pth` (most recent epoch). `MLflowLogger.log_checkpoint(path, kind)` enforces `kind in {"best", "last"}` and renames the source file on upload, so the on-disk filename doesn't matter. *(Canonical: `docs/00_PRINCIPLES.md#checkpoint-policy`.)*

### `nexus/logger/` package layout (matters for imports)

`nexus/logger/__init__.py` re-exports **only** the core: `make_logger`, `DualLogger`, `MLflowLogger`, `TBLogger`. Advanced features must be imported by their submodule path:

```python
from nexus.logger.sweep_logger   import SweepLogger          # parent run for HP sweeps; pass parent_run_id to children
from nexus.logger.model_registry import ModelRegistry        # MLflow Model Registry helpers (sim_run_id linkage)
from nexus.logger.system_metrics import SystemMetricsLogger  # background thread, 30s default, optional psutil/pynvml
from nexus.logger                import rl_metrics           # pure-numpy explained_variance, approx_kl, clip_fraction, grad_norm
```

All intra-package imports use the relative form (`from .git_utils import ...`, `from .mlflow_logger import ...`). Do not introduce bare top-level imports between sibling modules — they break when the package is installed via `pip install nexus-logger` because the repo root is not on `sys.path` in that case.

`TBLogger` is **not** interface-equivalent to `MLflowLogger` / `DualLogger` — it implements only the `SummaryWriter` core (`add_scalar`, `add_histogram`, `add_image`, `log_artifact` no-op, `close`). It has **no** `log_checkpoint`, `log_rl_metrics`, `register_checkpoint`, or `promote_model`. So a trainer written against the full logger API will `AttributeError` when `make_logger(mode="tensorboard")` is selected as a rollback path. (`docs/11_LOGGER_SETUP.md` currently says these are "silently ignored" — that's true for `log_artifact` only.) When adding a new method to `MLflowLogger`, decide whether `DualLogger` should forward it (almost always yes) and whether `TBLogger` should stub it (depends on whether you want `mode="tensorboard"` to stay viable).

### `chart_settings/` (separate concern)

`apply_chart_settings.py` persists MLflow column / chart configuration as **experiment tags** (`nexus.chart_settings`, `nexus.chart_settings_version`) so they outlast browser sessions and are shared across the team. The browser-side restoration is a generated JS bookmarklet (printed by `python chart_settings/apply_chart_settings.py bookmarklet`) that fetches the tag and writes the MLflow 2.x localStorage keys. CLI subcommands: `apply`, `show`, `bookmarklet`.

## When adding new features

Several concepts are reflected in multiple places. Change one without auditing the others and the docs will silently rot:

- **New required tag** — add to `post_upload/config.py::required_tags()` (code-side enforcement), the "Recommended Tags" table in `README.md`, the canonical anchor table in `docs/00_PRINCIPLES.md` → `#required-tags`, and the team-facing description in `docs/ko/02_EXPERIMENT_STANDARD.md` § 3-1.
- **New logger mode or core method** — decide whether to re-export from `nexus/logger/__init__.py`, update the `make_logger()` factory in `nexus/logger/dual_logger.py`, extend the "Logger Modes" table in `README.md`, and add a case to `tests/smoke_test.py`. Advanced (opt-in) features are documented in `docs/30_ADVANCED_FEATURES.md`; only the core four (`make_logger`, `DualLogger`, `MLflowLogger`, `TBLogger`) belong in `README.md`.
- **New Pipeline B CLI flag** — `post_upload/upload_tb.py::parse_args()` (or `upload_eval.py::parse_args()` for eval-side flags), plus the flag table in `README.md` "Pipeline B" section and the deeper notes in `docs/13_POST_UPLOAD.md`.
- **New Pipeline A sync option** — add the CLI flag in `scheduled_sync/sync_mlflow_to_server.sh` (argument-parsing case), the matching JSON key in the `KEY_MAP` of the same script's `--config` parser, the example file `scheduled_sync/sync_config.example.json`, the validator's required-key list in `scheduled_sync/validate_sync.sh` if the new key is required, and the operator walkthrough in `docs/12_SCHEDULED_SYNC.md` (both the Step 1 required-keys table and the Verification checklist if the new key is required). Keep CLI-name ↔ JSON-key mapping consistent (`--remote_nexus_dir` ↔ `"remote_nexus_dir"`).
- **Changing the default URIs (`5100`, `5000`)** — defaults are hardcoded across `nexus/logger/`, `scheduled_sync/*`, `post_upload/`, `chart_settings/apply_chart_settings.py`, and the README diagrams. Grep for `5100` and `5000` and change them in concert.
- **New team-wide chart or column** — edit `chart_settings/chart_settings.json`, then run `python chart_settings/apply_chart_settings.py apply` against the central server. The bookmarklet picks up the new payload automatically; no JS edit needed.

## Comment & docstring style (unicode banners)

This repo deliberately uses unicode box-drawing and em-dash characters in comments. The general "default to no comments" guidance does **not** apply here — match the surrounding style and never strip existing comments when editing. New files must follow the same conventions.

| Where | Character | Example |
|------|-----------|---------|
| Module docstring banner | `=` (U+003D) under the path, same length | `nexus/logger/mlflow_logger.py`<br>`=============================` |
| Section divider inside a file | `─` (U+2500 BOX DRAWINGS LIGHT HORIZONTAL) | `# ── Public interface ──────────────────────────` |
| Inline section marker (no trailing rule) | `─` (U+2500), title only | `# ── Step 1: Export delta from local MLflow` |
| Label / description separator, "why" explanations | `—` (U+2014 EM DASH) | `make_logger  — factory function`<br>`# Dirty tree detected — git patch saved` |
| ASCII directory tree in docstrings | `└──` `├──` (U+2514, U+251C) | `└── events.out.tfevents.xxxxx` |

Concrete rules when authoring or editing a file:

1. **Every Python module starts with a docstring** that opens with `module/relative_path.py`, then a line of `=` exactly as long as that path, then a one-paragraph summary. See `nexus/logger/mlflow_logger.py:1`, `post_upload/upload_tb.py:1`, `scheduled_sync/export_delta.py:1`.
2. **Section dividers** use `# ── Title ──...` — pad the trailing `─` run so the comment ends near column 76 (look at neighbouring dividers in the same file and match width). Numbered top-level sections (`# ── 1. Argument parsing ──...`) appear in the larger CLI scripts (`upload_tb.py`, `verify_tb.py`).
3. **Class- or method-internal sections** use the same `# ──` style indented to match the surrounding code (see `MLflowLogger` at `nexus/logger/mlflow_logger.py:93,191`).
4. **Use em dash `—`, not ` - `**, when joining a label to its explanation in docstrings or in "why" comments. Same for prose punctuation inside comments. Hyphen-minus `-` stays for compound words and CLI flags only.
5. **Short "why" comments stay ASCII** (e.g. `# MLflow hard limit per log_batch() call`, `# ~50x faster than iterrows`). Don't rewrite them with unicode.
6. **Shell scripts** follow the same divider style — see `scheduled_sync/sync_mlflow_to_server.sh` for `# ── Step N: ...` markers.
7. When introducing a new file, copy the header of the closest sibling (e.g. a new `nexus/logger/foo.py` should mirror `nexus/logger/rl_metrics.py`'s opening) rather than inventing a new layout.

Audit hint: `grep -nE "^# (-{4,}|={4,})" path/to/file.py` should return nothing. ASCII rule lines mean someone bypassed this convention.

## Things to be careful about

- See the dedicated **Comment & docstring style** section above before editing or creating any source file — the unicode banner / divider conventions are mandatory in this repo.
- **`pyproject.toml` ships `mlflow-skinny` as the default runtime dep**, with full `mlflow` only behind the `[server]` extra. All Python code in `nexus/logger/`, `post_upload/`, `scheduled_sync/`, `chart_settings/`, and `tests/` must stay within client / tracking APIs that exist in skinny — `MlflowClient`, `mlflow.entities.*`, `mlflow.set_tracking_uri/set_experiment/get_experiment_by_name`, `register_model`, `set_experiment_tag`. APIs that require the full distribution — `mlflow.pyfunc.load_model`, `mlflow.models.Model` flavor builders, anything under `mlflow.server.*` — would silently break the default training-node install (`pip install "nexus-logger @ git+..."`). The only full-mlflow consumer in this repo is the `mlflow server` CLI inside `scheduled_sync/start_local_mlflow.sh`, run from the operator venv built by `setup.sh` (which still installs full mlflow), not from `pip install nexus-logger`. *(Canonical: `docs/00_PRINCIPLES.md#mlflow-skinny-contract`.)*
- `setup.sh` pins `mlflow==2.13.0`, `tensorboard==2.16.2`, `tbparse==0.0.8`. The `tbparse` column-name handling in `upload_tb.parse_tfevents` and `verify_tb.fetch_tb_metrics` already has a fallback for older `tbparse` (`tags` → `tag`) — preserve it if upgrading.
- Do not add `git push` / `scp` / cron-installation steps to `setup.sh`. The deployment is intentionally split: setup.sh only builds the venv, and operators wire up cron / SSH keys themselves following `docs/20_MLFLOW_SERVER_SETUP.md`.
- `tests/smoke_test.py` writes real runs to a real MLflow server under the experiment name `nexus_smoke_test`. Don't point it at a production tracking URI without intending to.

## Where to read more

The canonical entry point is `docs/00_PRINCIPLES.md` (team-agreed rules + engineering invariants, single source of truth). `README.md` → "Further Reading" lists every doc with its numeric prefix; the docs are ordered into tracks: `00` = principles, `10–13` = engineer/user (architecture, Pipeline A logger setup, Pipeline A scheduled sync, Pipeline B post-upload), `20–21` = operator infrastructure (central MLflow server install, air-gapped GPU node bring-up — both include their own local-PC + post-install verification steps), `30–31` = opt-in features. Korean onboarding lives at `docs/ko/` (`01_INTRO.md`, `02_EXPERIMENT_STANDARD.md`). `docs/10_ARCHITECTURE.md` is the single best starting point if you need the full data-flow and run-structure picture before touching code.

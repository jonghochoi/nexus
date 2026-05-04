# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

NEXUS is a centralized RL experiment hub for an air-gapped GPU-server / internet-accessible-MLflow-server topology. It funnels training metrics, configs, and checkpoints from many GPU machines into one MLflow tracking server so the team can compare runs in one UI. See `README.md` for motivation, the infrastructure diagram, and user-facing walkthroughs; this file focuses on what's needed to **modify the code** safely.

## Environment / common commands

Venv setup and activation are covered in `README.md` → "Quick Start" (`bash setup.sh [--alias|--reinstall]`, `source ~/.nexus/activate.sh`). Two things matter for code changes:

- The venv is at `~/.nexus/venv` — **outside** the source tree — so overwriting the repo does not wipe installed packages. `~/.nexus/` also holds the user's `post_config.json` (Pipeline B), `sync_state/{exp}.json`, `history.json`, and the local MLflow server's runtime data (`mlruns_training/`, `mlflow_training.log`, `.mlflow_training.pid` — written by `scheduled_sync/start_local_mlflow.sh`).
- There is no mandatory `pip install` step — scripts are run directly and `nexus/logger/` is imported via `sys.path.insert(0, ".")` from the repo root (which puts the `nexus` package on `sys.path`). `pyproject.toml` exists for ruff config and the `nexus-logger` package definition, but is not required for day-to-day development.

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

Used when training code is modified to call `make_logger()`. Full data-flow is in `docs/10_ARCHITECTURE.md`; the operator walkthrough is in `docs/12_SCHEDULED_SYNC.md`. Key facts that affect code changes:

- **Run identity is `run_name`, not `run_id`** — `_get_or_create_run` resumes an existing run on crash+restart; never creates a duplicate.
- **`_BATCH_SIZE = 1000`** matches MLflow's hard per-`log_batch()` limit — do not exceed.
- **Git patch**: if the working tree is dirty, `git diff HEAD` is rendered as a self-contained HTML page and uploaded to `artifacts/git/git_patch.html`. Suppress with `track_git=False`.
- **Config resolution** in `sync_mlflow_to_server.sh`: CLI flag → `--config` JSON → `/etc/nexus/sync_config.json` → built-in default.
- **Exactly one cron per GPU server** — a second cron creates a competing state file and duplicates metric points.
- **State file** at `~/.nexus/sync_state/{experiment}.json` is the sync source of truth. Deleting it triggers a full re-sync. (Old default `/tmp/...` was wiped on reboot — that location is no longer used.)
- **`validate_sync.sh`** is a pre-flight checker — exits 0 only when all checks pass; prints a paste-ready cron line but never edits crontab itself.

### Pipeline B — `post_upload/` (one-shot, no trainer changes)

Back-fills completed tfevents and attaches post-hoc eval artifacts. Full walkthrough in `docs/13_POST_UPLOAD.md`. Key behaviors that affect code changes:

- **Multi-run protection** — aborts if tfevents span more than one parent directory (would cause step collisions). Always upload one run dir at a time.
- **Vectorized metric building** — build `Metric` entities via vectorized zip over numpy arrays, not `iterrows` (~50x faster); the file comments call this out.
- **Tag precedence** (7-level chain) — single source of truth is `docs/13_POST_UPLOAD.md` §2; keep that table authoritative if you change the order.
- **Auto-verify** — `run_verify()` runs unconditionally after upload unless `--no_verify`; exits `2` on failure (so CI can branch on it) but still records the upload in history.
- **Eval artifacts** go under `eval/<eval_id>/` — never `checkpoints/`, to preserve `best.pth`/`last.pth` policy.
- **History** (`~/.nexus/history.json`, capped at `HISTORY_LIMIT=20`) carries a `script` field (`"upload_tb"` / `"upload_eval"`); `--repeat-last` and `--from-last` filter by `script="upload_tb"` so they never resurrect an eval record.
- **`sys.path.insert`** in each script — injects the parent dir so sibling modules (`config`, `history`, `verify_tb`) import correctly from any working directory.

### Cross-cutting conventions

- **Default URIs**: GPU-server local MLflow is `http://127.0.0.1:5100`; central MLflow is `http://127.0.0.1:5000` (and `http://nexus-server:5000` from clients). These appear hardcoded as defaults across many files — change them in concert.
- **Metric name sanitization**: `name.replace(" ", "_").replace(":", "-")`. Slashes are preserved so TensorBoard's `losses/actor_loss` hierarchy survives. **Three** copies must stay in lock-step — `MLflowLogger._sanitize` (logger), `upload_tb.sanitize_metric_name` (uploader), and `verify_tb.sanitize_metric_name` (verifier applies it to the TB-side tags before comparing) — or `verify_tb.py` will report tag-list mismatches.
- **Param flattening**: `MLflowLogger._flatten` recursively flattens nested dict params with `.` separator. Lists/tuples are stored via `str(v)` (not flattened).
- **Required tags**: the canonical statement is in `docs/00_PRINCIPLES.md` → `#required-tags`. The code-side enforcement lives in `post_upload/config.py::required_tags()` — always `(experiment,)`. The user-facing summary is `README.md` → "Recommended Tags". If you change required tags, update both sites.
- **Checkpoint policy**: only two artifacts ever exist under `checkpoints/` in MLflow — `best.pth` (highest score so far) and `last.pth` (most recent epoch). `MLflowLogger.log_checkpoint(path, kind)` enforces `kind in {"best", "last"}` and renames the source file on upload, so the on-disk filename doesn't matter. *(Canonical: `docs/00_PRINCIPLES.md#checkpoint-policy`.)*

### `nexus/logger/` package layout (matters for imports)

`nexus/logger/__init__.py` re-exports **only** the core: `make_logger`, `DualLogger`, `MLflowLogger`, `TBLogger`. Advanced features must be imported by their submodule path:

```python
from nexus.logger.sweep_logger   import SweepLogger          # parent run for HP sweeps; pass parent_run_id to children
from nexus.logger.model_registry import ModelRegistry        # MLflow Model Registry helpers (sim_run_id linkage)
from nexus.logger.system_metrics import SystemMetricsLogger  # background thread, 30s default, optional psutil/nvidia-ml-py
from nexus.logger                import rl_metrics           # pure-numpy explained_variance, approx_kl, clip_fraction, grad_norm
```

All intra-package imports use the relative form (`from .git_utils import ...`, `from .mlflow_logger import ...`). Do not introduce bare top-level imports between sibling modules — they break when the package is installed via `pip install nexus-logger` because the repo root is not on `sys.path` in that case.

`TBLogger` is **not** interface-equivalent to `MLflowLogger` / `DualLogger` — it implements only the `SummaryWriter` core (`add_scalar`, `add_histogram`, `add_image`, `log_artifact` no-op, `close`). It has **no** `log_checkpoint`, `log_rl_metrics`, `register_checkpoint`, or `promote_model`. So a trainer written against the full logger API will `AttributeError` when `make_logger(mode="tensorboard")` is selected as a rollback path. (`docs/11_LOGGER_SETUP.md` currently says these are "silently ignored" — that's true for `log_artifact` only.) When adding a new method to `MLflowLogger`, decide whether `DualLogger` should forward it (almost always yes) and whether `TBLogger` should stub it (depends on whether you want `mode="tensorboard"` to stay viable).

### `chart_settings/` (separate concern)

`apply_chart_settings.py` persists MLflow column / chart configuration as **experiment tags** (`nexus.chart_settings`, `nexus.chart_settings_version`) so they outlast browser sessions and are shared across the team. The browser-side restoration is a generated JS bookmarklet (printed by `python chart_settings/apply_chart_settings.py bookmarklet`) that fetches the tag and writes the MLflow 2.x localStorage keys. CLI subcommands: `apply`, `show`, `bookmarklet`. User-facing guide: `docs/31_CHART_SETTINGS_GUIDE.md`.

### `brand.py` (CLI output utilities)

Shared ANSI styling module imported by CLI scripts across the repo. Exports `SIGIL` (`[NXS]` in bold cyan), `BANNER`, `FLOW`, `VERSION_STRING`, `print_banner()`, `print_flow()`, `rule()`, and `log()`. Import directly from the repo root: `from brand import print_banner, SIGIL, log`. Has no imports from `nexus.*` and no MLflow dependency — safe to import in any context.

## When adding new features

Several concepts are reflected in multiple places. Change one without auditing the others and the docs will silently rot. Use these checklists when making each category of change.

**New required tag**
- [ ] `post_upload/config.py::required_tags()` — code-side enforcement
- [ ] `docs/00_PRINCIPLES.md` → `#required-tags` — canonical anchor table
- [ ] `README.md` → "Recommended Tags" table

**New core logger method** (re-exported from `nexus/logger/__init__.py`)
- [ ] `nexus/logger/__init__.py` — add to re-exports
- [ ] `nexus/logger/dual_logger.py` — add forwarding in `DualLogger`
- [ ] `nexus/logger/tb_logger.py` — decide whether `TBLogger` needs a stub (required if you want `mode="tensorboard"` to stay viable; see package layout note above)
- [ ] `README.md` → "Logger Modes" table
- [ ] `tests/smoke_test.py` — add a core test case

**New opt-in / advanced logger feature** (not re-exported from `__init__`)
- [ ] `docs/30_ADVANCED_FEATURES.md` — document the new feature
- [ ] `tests/smoke_test.py` — add a case under `--advanced`

**New Pipeline B CLI flag**
- [ ] `post_upload/upload_tb.py::parse_args()` (or `upload_eval.py::parse_args()` for eval-side flags)
- [ ] `README.md` → "Pipeline B" flag table
- [ ] `docs/13_POST_UPLOAD.md` — deeper notes

**New Pipeline B script** (new file added to `post_upload/`)
- [ ] `docs/13_POST_UPLOAD.md` — document the new script
- [ ] `README.md` → "Pipeline B" section

**New Pipeline A sync option**
- [ ] `scheduled_sync/sync_mlflow_to_server.sh` — CLI flag (argument-parsing case) + matching JSON key in `KEY_MAP`
- [ ] `scheduled_sync/sync_config.example.json` — add the new key
- [ ] `scheduled_sync/validate_sync.sh` — add to required-key list if the new key is required
- [ ] `docs/12_SCHEDULED_SYNC.md` — Step 1 required-keys table + Verification checklist if required
- [ ] Keep CLI-name ↔ JSON-key mapping consistent (`--remote_nexus_dir` ↔ `"remote_nexus_dir"`)

**New chart setting or column**
- [ ] `chart_settings/chart_settings.json` — add the new entry
- [ ] Run `python chart_settings/apply_chart_settings.py apply` against the central server
- [ ] `docs/31_CHART_SETTINGS_GUIDE.md` — update if the new setting changes user workflow

**Changing default URIs (`5100`, `5000`)**
- [ ] Grep for `5100` and `5000` across `nexus/logger/`, `scheduled_sync/*`, `post_upload/`, `chart_settings/apply_chart_settings.py`, and `README.md` diagrams — change in concert

**Adding a new doc file under `docs/`**
- [ ] `README.md` → "Further Reading" table — add the new entry with its numeric prefix
- [ ] CLAUDE.md → "Where to read more" — update if the doc is relevant to code changes
- [ ] If the doc applies to Korean users, consider adding a corresponding entry in `docs/ko/`

## Docs Markdown style (`docs/`)

All Markdown guide files under `docs/` follow a single header convention established in the standardization pass. When adding or editing docs, match this style exactly.

### Header levels

| Level | Format | Rule |
|:---:|---|---|
| H1 | `# 🔧 Document Title` | One thematic emoji + plain title. Emoji kept — it identifies the doc at a glance in file listings. One H1 per document. |
| H2 | `## Section Title` | **No emoji, ever.** Plain `## Title`. This eliminates the "which emojis are structural?" question entirely. |
| H3 | `### ── Subsection Title` | Prefix `──` (U+2500 × 2 + space). No emoji, no numeric prefix. |
| H4 | `#### ── Sub-subsection Title` | Same `──` prefix pattern as H3. |

### H3 / H4 prefix characters

The `──` characters are U+2500 (BOX DRAWINGS LIGHT HORIZONTAL), matching the source-code section-divider convention used throughout this repo. Copy-paste from an existing header — do not type hyphens or minus signs.

### Anchor IDs for TOC links

GitHub's anchor generator strips the `──` characters but keeps the space that follows them, producing a **leading hyphen** in the anchor:

```
### ── Method selection criteria  →  id="-method-selection-criteria"
```

Consequences:

- **TOC links to H2 sections** — no leading hyphen (H2 headers start with a letter, no leading space after stripping).
  ```markdown
  [Step 1 — Verify](#step-1--verify-install-on-the-gpu-node)
  ```
- **TOC links to H3/H4 sections** — include the leading hyphen:
  ```markdown
  [Method selection criteria](#-method-selection-criteria)
  [Method A — pip wheel offline transfer](#-method-a--pip-wheel-offline-transfer)
  ```
- **Em dash `—` in headers** — the em dash itself is stripped; both surrounding spaces are kept → double hyphen in the anchor:
  ```
  ## Step 0 — Verify  →  #step-0--verify
  ```

### Emojis in body text

Emojis may still appear in **body text** (tables, blockquotes, callout notes such as `⚠️`, `✅`, `💡`). Only headers H2 and below are emoji-free.

---

## Code formatting (ruff)

All Python code in this repository is formatted with **ruff**. The canonical settings live in `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.format]`:

```toml
[tool.ruff]
line-length = 100
indent-width = 4

[tool.ruff.format]
skip-magic-trailing-comma = true
```

Rules:
- **Line length** — 100 characters maximum.
- **Indent** — 4 spaces (never tabs).
- **Trailing commas** — `skip-magic-trailing-comma = true` means ruff will reformat multi-element collections onto one line when they fit within the line-length limit, even if a trailing comma is present. Only use a trailing comma when you genuinely want to force a vertical layout.

When generating new code, always follow these rules. Before committing, run:

```bash
ruff format .
```

Do not introduce `# fmt: off` / `# fmt: skip` blocks without a clear reason.

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

## Commit message style

All merged commits in this repo follow a single, consistent style — derived from the existing history (`git log`), not from a generic "Conventional Commits" template. When generating a commit message for code you authored, match the patterns below exactly. Don't invent a new shape per commit.

### Subject line

```
<type>(<scope>): <verb> <description>
<type>: <verb> <description>           # scope optional when the change is repo-wide
```

Hard rules:

1. **Always start the description with an imperative verb** — this is the most important rule and the one that drifts most easily. Audit your draft: the first word after the colon must be a verb in imperative mood. Past tense (`added`, `fixed`), gerunds (`adding`, `fixing`), and noun-first phrases (`new feature for X`, `support for Y`) are forbidden.
   - Verbs already used in this repo's history (use one of these or a close synonym): `add`, `fix`, `repair`, `remove`, `drop`, `rename`, `move`, `split`, `clean up`, `restructure`, `clarify`, `codify`, `standardize`, `persist`, `generalize`, `support`, `harden`, `consolidate`, `translate`, `pin`, `unify`, `skip`, `treat`, `apply`.
2. **`<type>`** — one of `feat`, `fix`, `refactor`, `docs`, `chore`, `style`, `deps`. The combo form `docs+fix` is acceptable (rare) when a single commit genuinely spans both. Don't invent new types.
3. **`<scope>`** — lowercase, matches a folder or module name in the repo: `logger`, `scheduled_sync`, `post_upload`, `chart_settings`, `validate_sync`, `setup`, `tags`, `sync`, `validation`, `CLAUDE.md`. Omit the scope only for repo-wide changes (e.g. `chore: add ruff formatting config and apply to all Python files`, `docs: ...` touching many tracks).
4. **Description** — lowercase first letter (after the colon), no trailing period, ≲ 72 chars including the type/scope prefix. State *what* the commit does, not why.
5. **Do NOT include `(#NN)` in the local commit subject**. GitHub appends the PR number automatically on squash-merge — adding it manually duplicates it.

Good (from this repo's history):

```
fix(validate_sync): skip dry-run and fix cron output when experiment missing
feat(logger): split params into agent_params and env_params
refactor(tags): clean up required tags to experiment/researcher/task/hardware
docs(scheduled_sync): add Stopping sync section to 12_SCHEDULED_SYNC.md
chore: add ruff formatting config and apply to all Python files
deps: split client (mlflow-skinny) and server (mlflow) installs
```

Bad (don't do this):

```
Added a new logger feature                     # past tense, no type, capitalized
feat: new params for logger.                   # noun-first, trailing period
feat(logger): Splits params into two groups.   # 3rd-person singular, capitalized, period
fix: bug fix for sync                          # noun-first ("bug fix"), vague
update logger                                  # no type, vague verb "update"
```

### Body

Optional for trivial one-liners; required for any commit that touches more than one logical area or needs context to be reviewable. When present:

1. **Blank line** between subject and body.
2. **Wrap at ~72 columns**. Prose paragraphs and bullet lines both wrap; URLs and code blocks may exceed.
3. **Lead with the *why*** — one short paragraph stating the problem or motivation, before listing the *what*. See `fix(validate_sync): treat missing experiment as WARN, not FAIL` for the canonical shape (one paragraph of motivation, then one paragraph describing the fix).
4. **Lists for multiple changes**:
   - Use `-` bullets for parallel small changes (`docs: fix TOC order ...`).
   - Use `1.` `2.` `3.` numbered items when the changes are sequenced or you reference them by number elsewhere (`fix(validate_sync): skip dry-run ...`).
5. **Per-file groupings** for larger commits: file path on its own line ending with `:`, then an indented bullet list of changes for that file. See `refactor(tags): clean up required tags ...` for the canonical layout (`post_upload/config.py:` then `  - ...`).
6. **Unicode dividers** for the largest commits (typically `feat`/`refactor` touching many files, or `docs:` track-wide restructures). Match the source-code divider style:

   ```
   ── 1. Anchor text mismatches ──────────────────────────────────────────
   ```

   Use `─` (U+2500) — never `-` or `=`. Pad the trailing run so the line ends near column 72. Numbered sections (`── 1. ...`, `── 2. ...`) when there are multiple, plain titles (`── Section ──...`) otherwise. See `fix: repair broken TOC anchor links in all guide docs` and `docs: unify style of operator (20/21) and opt-in (30/31) docs` for canonical examples.
7. **Em dash `—` (U+2014)**, not ` - `, when joining a label to its explanation in body prose. Same rule as for source-code comments.
8. **Backticks** around paths (`post_upload/upload_tb.py`), identifiers (`MlflowClient`, `_BASE_REQUIRED`), CLI flags (`--dry_run`), and shell commands.

### Audit checklist before committing

- [ ] First word after `<type>(<scope>):` is an imperative verb (not a noun, not past tense).
- [ ] Description is lowercase, ≲ 72 chars, no trailing period.
- [ ] No `(#NN)` PR-number suffix in the subject.
- [ ] Body (if any) leads with *why*, wraps at ~72, uses `─` dividers (not `-`/`=`) for big commits, and uses em dash `—` for label/explanation joins.

## Things to be careful about

- See the dedicated **Comment & docstring style** section above before editing or creating any source file — the unicode banner / divider conventions are mandatory in this repo.
- **`pyproject.toml` ships `mlflow-skinny` as the default runtime dep**, with full `mlflow` only behind the `[server]` extra. All Python code in `nexus/logger/`, `post_upload/`, `scheduled_sync/`, `chart_settings/`, and `tests/` must stay within client / tracking APIs that exist in skinny — `MlflowClient`, `mlflow.entities.*`, `mlflow.set_tracking_uri/set_experiment/get_experiment_by_name`, `register_model`, `set_experiment_tag`. APIs that require the full distribution — `mlflow.pyfunc.load_model`, `mlflow.models.Model` flavor builders, anything under `mlflow.server.*` — would silently break the default training-node install (`pip install "nexus-logger @ git+..."`). The only full-mlflow consumer in this repo is the `mlflow server` CLI inside `scheduled_sync/start_local_mlflow.sh`, run from the operator venv built by `setup.sh` (which still installs full mlflow), not from `pip install nexus-logger`. *(Canonical: `docs/00_PRINCIPLES.md#mlflow-skinny-contract`.)*
- `setup.sh` pins `mlflow==2.13.0`, `tensorboard==2.16.2`, `tbparse==0.0.8`. The `tbparse` column-name handling in `upload_tb.parse_tfevents` and `verify_tb.fetch_tb_metrics` already has a fallback for older `tbparse` (`tags` → `tag`) — preserve it if upgrading.
- Do not add `git push` / `scp` / cron-installation steps to `setup.sh`. The deployment is intentionally split: setup.sh only builds the venv, and operators wire up cron / SSH keys themselves following `docs/20_MLFLOW_SERVER_SETUP.md`.
- `tests/smoke_test.py` writes real runs to a real MLflow server under the experiment name `nexus_smoke_test`. Don't point it at a production tracking URI without intending to.

## Where to read more

`docs/00_PRINCIPLES.md` is the canonical entry point (team-agreed rules + engineering invariants). `docs/10_ARCHITECTURE.md` is the best starting point for data-flow and run-structure before touching code. All docs are indexed in `README.md` → "Further Reading": tracks `10–13` (engineer/user), `20–21` (operator infrastructure), `30–31` (opt-in features — `30_ADVANCED_FEATURES.md`, `31_CHART_SETTINGS_GUIDE.md`). Korean onboarding: `docs/ko/`.

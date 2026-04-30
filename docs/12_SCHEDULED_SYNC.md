# 🔄 Scheduled Sync — Pipeline A delta upload

> **Purpose:** Wire each GPU server's local MLflow (`127.0.0.1:5100`) to the central NEXUS MLflow server via cron. Each cron tick exports new metric points + new/changed artifact files since the last sync, packages them into a tar.gz delta bundle, and ships it through SCP + remote `import_delta.py` so both metrics and artifacts (checkpoints, configs, git diffs, eval reports) end up browsable in the central UI.
>
> This guide is for **operators** registering or auditing the cron sync. The system architecture is in [`10_ARCHITECTURE.md`](10_ARCHITECTURE.md); the canonical principle for the multi-user invariant is in [`00_PRINCIPLES.md#multi-user-researcher`](00_PRINCIPLES.md#-multi-user-researcher).

---

## 📑 Table of Contents

- [⚡ TL;DR](#-tldr)
- [📋 Prerequisites](#-prerequisites)
- [🚦 Recommended order](#-recommended-order)
- [📌 Step 1 — Sync config file(s)](#-step-1--sync-config-files)
- [📌 Step 2 — Pre-flight check (`validate_sync.sh`)](#-step-2--pre-flight-check-validate_syncsh)
- [📌 Step 3 — Run once manually](#-step-3--run-once-manually)
- [📌 Step 4 — Register cron](#-step-4--register-cron)
- [📌 Step 5 — Multi-user GPU servers](#-step-5--multi-user-gpu-servers)
- [📁 State file & incremental sync](#-state-file--incremental-sync)
- [✅ Verification checklist](#-verification-checklist)
- [⏹️ Stopping sync](#-stopping-sync)
- [🛠️ Troubleshooting](#-troubleshooting)
- [🗺️ Next steps](#-next-steps)

---

## ⚡ TL;DR

```bash
# One-time setup (per user, on the GPU server)
mkdir -p ~/.nexus
cp scheduled_sync/sync_config.example.json ~/.nexus/sync_config.json
$EDITOR ~/.nexus/sync_config.json   # set experiment, researcher, remote, remote_nexus_dir, ssh_key

# Pre-flight check + manual one-shot sync
bash scheduled_sync/validate_sync.sh        # prints a paste-ready cron line on success
bash scheduled_sync/sync_mlflow_to_server.sh

# Register cron — runs every 5 minutes
crontab -e
# */5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh >> $HOME/nexus_sync.log 2>&1
```

> [!IMPORTANT]
> On a **multi-user GPU server**, every user must set their own `researcher` in `~/.nexus/sync_config.json` — otherwise crons cross-contaminate and the central server logs duplicate metric points. Detail: [§ Step 5](#-step-5--multi-user-gpu-servers), canonical: [`00_PRINCIPLES.md#multi-user-researcher`](00_PRINCIPLES.md#-multi-user-researcher).

---

## 📋 Prerequisites

| Requirement | Where to set up |
|---|---|
| GPU server has nexus installed | [`21_AIRGAPPED_GPU_SERVER_SETUP.md`](21_AIRGAPPED_GPU_SERVER_SETUP.md) |
| Local MLflow on GPU server is reachable at `127.0.0.1:5100` | `bash scheduled_sync/start_local_mlflow.sh` |
| Central MLflow server is up | [`20_MLFLOW_SERVER_SETUP.md`](20_MLFLOW_SERVER_SETUP.md) |
| Key-based SSH from GPU server → central server | [`20_MLFLOW_SERVER_SETUP.md` Step 8](20_MLFLOW_SERVER_SETUP.md) |

---

## 🚦 Recommended order

> [!IMPORTANT]
> After writing the config file, proceed in this sequence:
>
> 1. `validate_sync.sh` — pre-flight check (SSH, permissions, dry-run)
> 2. `sync_mlflow_to_server.sh` — manual run to confirm real data transfer end-to-end
> 3. `crontab -e` — register the cron job
> 4. **Start training** — cron must be registered first so sync begins from step 0
>
> Registering cron after training has already started causes no data loss (the state file tracks the full history), but any metrics logged before cron was registered will be uploaded in bulk on the next cron run rather than incrementally.

---

## 📌 Step 1 — Sync config file(s)

The fixed values live in a config file so the cron line is a single bash invocation. Two locations are auto-discovered:

| Path | Owner | Typical contents |
|---|---|---|
| `/etc/nexus/sync_config.json` | Operator (root) | Team-wide values: `remote`, `remote_nexus_dir`, `remote_uri`, `ssh_port` |
| `~/.nexus/sync_config.json`   | Each user       | Per-user overrides: `researcher`, `ssh_key`, optionally a different `experiment` |

Per-key merge: user file overrides system file. CLI flags still win over both. If a single-user team prefers, all values can live in `~/.nexus/sync_config.json` alone.

```bash
# Per-user setup (always applicable):
mkdir -p ~/.nexus
cp scheduled_sync/sync_config.example.json ~/.nexus/sync_config.json
$EDITOR ~/.nexus/sync_config.json
```

**Required keys** (anywhere in the resolution chain): `experiment`, `remote`, `remote_nexus_dir`.
**Optional:** `researcher`, `remote_python`, `ssh_key`, `ssh_port`, `local_uri`, `remote_uri`, `state_file`.

> 💡 **`experiment` vs `researcher` — relationship to logger parameters**
>
> These two keys look similar to the parameters passed when creating a logger in training code, but their roles differ:
>
> - **`experiment`** — same concept in both places: the MLflow experiment name. The value here must exactly match the `experiment_name` passed to `make_logger()` / `MLflowLogger(experiment_name=...)` in the training code. If they differ, `export_delta.py` looks at a different experiment and exports nothing (or the wrong runs).
> - **`researcher`** — different roles, but values must match. In the training code, `tags={"researcher": "kim"}` is metadata attached to the MLflow run. In sync_config, `"researcher": "kim"` is a **filter**: `export_delta.py` only exports runs whose `researcher` tag equals this value. If the two values don't match, that researcher's runs are never exported.

> 💡 **`remote_python`** — Non-interactive SSH does not source `~/.bashrc`, so the MLflow server's venv is never activated and `python3` resolves to the system interpreter (which has no `mlflow`). Set this to the full path of the venv Python on the MLflow server: `"/opt/nexus-mlflow/venv/bin/python3"`.

> ⚠️ **Multi-user GPU servers** — when several researchers share one GPU server (and one local MLflow), each user **MUST** set their own `researcher` in `~/.nexus/sync_config.json`. Without it, every user's cron exports every other user's runs and the central server logs duplicate metric points at identical steps. The validator flags this with a `[WARN]`. Detail: [Step 5](#-step-5--multi-user-gpu-servers) and [`00_PRINCIPLES.md#multi-user-researcher`](00_PRINCIPLES.md#-multi-user-researcher).

---

## 📌 Step 2 — Pre-flight check (`validate_sync.sh`)

`validate_sync.sh` runs the same config resolution as `sync_mlflow_to_server.sh`, then verifies SSH, remote inbox writability, presence of `import_delta.py` on the central server, central MLflow `/health`, local MLflow + experiment existence, and finally executes a `--dry-run`. A clean run prints a paste-ready cron line — it never edits your crontab.

```bash
bash scheduled_sync/validate_sync.sh
# or with a non-default config path:
bash scheduled_sync/validate_sync.sh --config /etc/nexus/sync.json
```

Every failure prints what to fix; the script exits 2 on the first failed step rather than continuing in a broken state.

---

## 📌 Step 3 — Run once manually

```bash
bash scheduled_sync/sync_mlflow_to_server.sh
# or, if your config lives elsewhere:
bash scheduled_sync/sync_mlflow_to_server.sh --config /etc/nexus/sync.json
```

> 💡 `--remote_nexus_dir` is the path where nexus is installed on the NEXUS server (e.g., `/opt/nexus`). Required to locate `import_delta.py` on the server.

> 💡 Add `--dry-run` to exercise the local export step only (state file is updated, no SCP, no remote import). Useful before committing the cron entry.

On success, the run will appear in the NEXUS server's MLflow UI — both the metric curves and the run's artifacts (checkpoints, params, git diff, anything else logged via `MLflowLogger`) are browsable from the same page. Each imported run also gets `nexus.lastSyncTime` and `nexus.syncedFromHost` tags so you can spot stale GPU servers from the central UI.

**On second run:** If there are no new metrics or artifacts, SCP is skipped with the message `[OK] No new data since last sync.`

---

## 📌 Step 4 — Register cron

```bash
crontab -e
# Add the following line (runs every 5 minutes).
# $HOME is set by cron to the crontab owner's home directory, so the same
# line works for any user without editing /home/<name> paths.
*/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh \
    >> $HOME/nexus_sync.log 2>&1
```

Need a per-key override (e.g. running an alternate experiment from one cron line)? Add the matching CLI flag — flags win over the config file:

```cron
*/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh \
    --experiment robot_hand_rl_pilot >> $HOME/nexus_sync.log 2>&1
```

---

## 📌 Step 5 — Multi-user GPU servers

When kim, lee, and park all train on the same GPU server, each user runs their own cron:

1. **Each user sets their own `researcher`** in `~/.nexus/sync_config.json`. This scopes their export to runs tagged with that researcher; otherwise everyone re-exports everyone else's runs and the central server gets duplicate metric points at identical steps.
2. **Operator puts shared values in `/etc/nexus/sync_config.json`** (root-writable, world-readable): `remote`, `remote_nexus_dir`, `remote_uri`, `ssh_port`. Each user's `~/.nexus/sync_config.json` then only carries `researcher` and `ssh_key`.
3. **Stagger cron offsets** so SSH/SCP traffic to the central server is spread out across the interval:

   ```cron
   # kim
   0-59/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh >> $HOME/nexus_sync.log 2>&1
   # lee — offset by 1 minute
   1-59/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh >> $HOME/nexus_sync.log 2>&1
   # park — offset by 2 minutes
   2-59/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh >> $HOME/nexus_sync.log 2>&1
   ```

   The wrapper writes per-user, per-PID delta filenames (`delta_${USER}_<TS>_<PID>.tar.gz`) so concurrent runs don't corrupt each other's `/tmp` files or remote inbox even if you don't stagger — staggering is just polite.

---

## 📁 State file & incremental sync

The local state file (`~/.nexus/sync_state/{experiment}[__{researcher}].json`) records two things per run — the last synced step for each metric tag, and the per-run `__artifacts__` skip set listing artifact paths already shipped. This is the **source of truth** for "what has been synced." On every cron tick, `export_delta.py` reads this file, queries local MLflow for new metric points and new/changed artifact files, packages everything into a tar.gz bundle (`delta.json` + `artifacts/<run_id>/...`), and only ships the delta.

| Behavior | Detail |
|---|---|
| **Survives reboot** | The state file lives in `~/.nexus/`, which persists across reboots (unlike `/tmp`). |
| **Deletion forces full re-sync** | Removing the file makes the next cron run treat every run as "never synced" and re-ship every metric and artifact. Use this if state diverges from what's on the central server. |
| **Per-researcher namespacing** | When `researcher` is set, the file is namespaced — `~/.nexus/sync_state/{exp}__{researcher}.json` — so one operator account can host multiple sync identities. |
| **Artifact sync policy** | Paths under `checkpoints/` are re-synced every cycle (since `best.pth` / `last.pth` are overwritten in place during training). Every other artifact is synced once and recorded in `__artifacts__` so subsequent cycles skip it. The predicate is `export_delta.is_always_sync()`. |
| **Exit codes (cron-friendly)** | `0` = data transferred · `1` = configuration error · `2` = no new data (wrapper skips SCP and exits cleanly). |

---

## ✅ Verification checklist

After `validate_sync.sh` passes, run through this checklist before declaring the sync ready for production:

- [ ] `~/.nexus/sync_config.json` populated with `experiment`, `remote`, `remote_nexus_dir`
- [ ] `bash scheduled_sync/validate_sync.sh` reports "All checks passed"
- [ ] `bash scheduled_sync/sync_mlflow_to_server.sh` manual run succeeded
- [ ] Synced run is visible in the central NEXUS MLflow UI
- [ ] Cron line registered (`crontab -l` shows the entry)
- [ ] Auto-run confirmed after one cron interval (`tail ~/nexus_sync.log` shows non-error output)
- [ ] *(Multi-user GPU server)* every researcher has their own `researcher` set in `~/.nexus/sync_config.json` — `validate_sync.sh` does **not** print `[WARN] researcher unset`
- [ ] Each imported run on the central server carries `nexus.lastSyncTime` and `nexus.syncedFromHost` tags

> ⚠️ **Multi-user invariant** — without per-user `researcher`, every cron exports every other user's runs and the central server logs duplicate metric points at identical steps. Canonical: [`00_PRINCIPLES.md#multi-user-researcher`](00_PRINCIPLES.md#-multi-user-researcher).

---

## ⏹ Stopping sync

### Remove the cron entry (required)

```bash
crontab -e
```

Delete or comment out the `sync_mlflow_to_server.sh` line, then save:

```cron
# */5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh >> ...
```

Confirm it is gone:

```bash
crontab -l   # the sync line must not appear
```

### Kill any in-flight sync process (if needed)

If a cron tick fired just before you removed the entry, a sync may still be running:

```bash
# Check
pgrep -a -f "sync_mlflow_to_server"
pgrep -a -f "export_delta"

# Terminate if found
pkill -f "sync_mlflow_to_server"
pkill -f "export_delta"
```

This is only needed when you must stop immediately (e.g. the central server is being taken down). A mid-flight sync that is killed cleanly causes no data loss — the state file is only updated after a successful export, so the next manual run will re-export any points that were in progress.

### Stop local MLflow (optional)

Only needed if training is also stopping. The local MLflow server is required by the training process (`make_logger`) and by `export_delta.py`, but not by anything else.

```bash
lsof -ti :5100 | xargs kill
```

> 💡 The cron entry is the only thing that drives periodic sync. Once it is removed, no further data is transferred to the central server regardless of whether the local MLflow or any other process is still running.

---

## 🛠 Troubleshooting

### ⚠️ `python3: command not found` on the central server

Non-interactive SSH does not source `~/.bashrc`. The MLflow server's venv `python3` is not on `PATH`. Set `remote_python` to the full path:

```json
{
  "remote_python": "/opt/nexus-mlflow/venv/bin/python3"
}
```

### ⚠️ `Permission denied (publickey)` on cron run, but works manually

Cron runs without your interactive shell's environment. The SSH agent isn't loaded, so password-protected keys won't decrypt. Either:

- Use a dedicated, passphrase-less key for sync (most common — its only privilege is `import_delta.py`)
- Or pre-load `ssh-agent` from cron with `eval $(ssh-agent)` + `ssh-add`

### ⚠️ Central server logs duplicate metric points at the same step

Two GPU server users are running cron without setting their `researcher` tag, so each cron exports the other's runs. Fix: set `researcher` in `~/.nexus/sync_config.json` for every user. Confirm with `bash scheduled_sync/validate_sync.sh` — it warns if `researcher` is unset.

### ⚠️ Sync claims success but central UI shows no new data

Check the state file: `cat ~/.nexus/sync_state/{experiment}__{researcher}.json`. If the recorded steps are ahead of what's actually on the central server (e.g. central was wiped), delete the state file and let the next cron tick re-sync from scratch.

### ⚠️ Wrapper exits 5 with `UnicodeDecodeError: 0x8b` after upgrading

The `0x8b` byte is the gzip magic header — a pre-artifact-sync `import_delta.py` is trying to `json.load()` the new tar.gz delta bundle. The GPU server has the new `export_delta.py` (writes `.tar.gz`) but the central server is still on the old `import_delta.py` (only reads plain JSON). Every cron tick fails this way, so **runs disappear from central until central is upgraded**.

```bash
ssh <central-host> "cd <remote_nexus_dir> && git pull"
```

`validate_sync.sh` now guards this check at step 3, so re-running it after an upgrade catches a half-upgraded fleet before you re-register cron.

### ⚠️ How do I add a new sync option to the wrapper?

See `CLAUDE.md` → "When adding new features" → "New Pipeline A sync option". You need to update: the CLI flag in `sync_mlflow_to_server.sh`, the JSON `KEY_MAP`, `sync_config.example.json`, the validator's required-key list (if required), and this guide.

---

## 🗺 Next steps

- **Architecture detail (export → SCP → import flow)** → [`10_ARCHITECTURE.md`](10_ARCHITECTURE.md)
- **Pipeline B alternative** (one-shot, no cron) → [`13_POST_UPLOAD.md`](13_POST_UPLOAD.md)

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
- [📊 Monitoring & log inspection](#-monitoring--log-inspection)
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

## 📊 Monitoring & log inspection

After cron is registered, sync runs unattended. The three places to check whether it's still healthy are: the **wrapper log** on the GPU server, the **state file** on the GPU server, and the **`nexus.*` tags** on the central MLflow UI. None of them require restarting anything.

### ── Where the log lives

`~/nexus_sync.log` is just the stdout/stderr of `sync_mlflow_to_server.sh`, redirected by the cron line you registered in [§ Step 4](#-step-4--register-cron). Cron itself does not rotate this file — it grows by ~1–2 lines per "no new data" tick and ~10 lines per real upload, so on a typical 5-minute cadence it stays under a few MB per month. Rotate it manually if it ever bothers you (`mv ~/nexus_sync.log ~/nexus_sync.log.old`); the next cron tick will re-create it.

### ── Healthy output — the two normal shapes

**Shape 1 — new data was transferred (most common during active training):**

```
[2026-04-30 14:35:01] MLflow delta sync: robot_hand_rl_pilot (researcher=kim)
  [1/3] Exporting delta from local MLflow (http://127.0.0.1:5100)...
[INFO] Delta: 3 run(s) — 1840 new metric points, 0 new run(s)
[OK] Delta written to: /tmp/delta_kim_20260430_143501_28714.json
  [OK] Delta exported (47 KB)
  [2/3] Transferring delta to nexus@nexus-server...
  [OK] Transfer complete
  [3/3] Importing delta on remote server...
  [OK] my_run_v3: 612 metric points
  [OK] my_run_v4: 612 metric points
  [OK] my_run_v5: 616 metric points
[DONE] Total imported: 1840 metric points.
  [OK] Import complete
  [DONE] Delta sync complete at 2026-04-30 14:35:04
```

**Shape 2 — no new metrics since last tick (most common when nothing is training):**

```
[2026-04-30 14:40:01] MLflow delta sync: robot_hand_rl_pilot (researcher=kim)
  [1/3] Exporting delta from local MLflow (http://127.0.0.1:5100)...
[INFO] No new data since last sync.
  [OK] No new data since last sync. Nothing to transfer.
```

This is **not** an error — `export_delta.py` exits `2`, the wrapper catches it and skips SCP + remote import. Seeing long stretches of Shape 2 just means no training is currently writing to local MLflow.

### ── Quick health-check commands

```bash
# Most recent tick — was it OK or an error?
tail -n 20 ~/nexus_sync.log

# Last successful upload (Shape 1 above)
grep "DONE.*Delta sync complete" ~/nexus_sync.log | tail -n 1

# All errors in the last 24 hours (only timestamp-prefixed header lines reset ts)
awk -v cutoff="$(date -d '24 hours ago' '+%Y-%m-%d %H:%M:%S')" \
    '/^\[20[0-9][0-9]-/{ts=substr($0,2,19)} ts>=cutoff && /\[ERROR\]/' ~/nexus_sync.log

# Is anything stuck in flight right now?
pgrep -a -f "sync_mlflow_to_server|export_delta"
```

> 💡 If `pgrep` shows a process older than ~5 minutes, the sync is wedged (most often on `ssh`/`scp`). Inspect with `ps -o pid,etime,args -p <pid>` and kill if needed — the state file is only updated on a clean export, so killing mid-flight loses no points.

### ── Exit code reference

The wrapper's exit code is what cron sees, and what shows up in `MAILTO=` cron mail subjects if you have one configured. Each code maps to a specific failure point so you can branch monitoring on it.

| Exit | Meaning | What to look at |
|------|---------|-----------------|
| `0` | Data transferred successfully **or** no new data since last tick | Nothing — both are healthy outcomes (Shape 1 / Shape 2) |
| `1` | Configuration error — unknown CLI flag, malformed `--config` JSON, missing required key, or `export_delta.py` couldn't find the experiment | The `[ERROR] ...` line just above; fix the offending key in `~/.nexus/sync_config.json` (most often `experiment`) |
| `3` | Remote inbox not writable (`ssh ... mkdir -p` failed) | SSH key, `remote`, `remote_nexus_dir`, and central server's inbox directory permissions |
| `4` | SCP failed after 3 retries with 5s/10s backoff | Network between GPU server and central server; firewall; key-based auth |
| `5` | Remote `import_delta.py` failed | `remote_python` (most common — venv interpreter not on `PATH`); `remote_nexus_dir`; central MLflow at `remote_uri` |

> 💡 Exit `2` from `export_delta.py` ("no new data") is **swallowed by the wrapper and re-mapped to `0`** — cron and any monitoring layered on top see only `0`. The raw `2` is documented in [§ State file & incremental sync](#-state-file--incremental-sync) for the case where you invoke `export_delta.py` directly.

### ── State file inspection

The state file (`~/.nexus/sync_state/{experiment}[__{researcher}].json`) is human-readable JSON and is the easiest way to confirm "yes, sync is keeping up."

```bash
# Replace path components to match your config
cat ~/.nexus/sync_state/robot_hand_rl_pilot__kim.json
```

```json
{
  "runs": {
    "a3f1...": { "loss": 12480, "reward": 12480, "lr": 12480 },
    "b7c2...": { "loss":  8192, "reward":  8192 }
  },
  "last_sync_time": 1745001304.812
}
```

What to look at:

- **`last_sync_time`** — UNIX seconds of the most recent successful export. Convert with `date -d @1745001304`. If this is more than 2× your cron interval old while training is active, sync is failing silently — go check `~/nexus_sync.log`.
- **Per-run, per-tag steps** — the highest step already shipped for each metric of each run. If these are advancing on every tick, sync is keeping pace with training.
- **Run count** — should match what you see in your local MLflow's experiment (modulo runs filtered out by your `researcher` tag).

> 💡 If the state file's recorded steps are ahead of what's actually on the central server (e.g. central was wiped and restored from an older backup), delete the state file — the next tick re-exports every run from step 0. See also [§ Troubleshooting](#-troubleshooting) → "Sync claims success but central UI shows no new data".

### ── Central-side check — finding stale GPU servers

Every run imported by `import_delta.py` is stamped with two tags on the central MLflow:

| Tag | Value | Use |
|-----|-------|-----|
| `nexus.lastSyncTime` | UTC ISO timestamp of the most recent import for that run | Sort runs by this column to surface ones whose origin GPU server has gone quiet |
| `nexus.syncedFromHost` | Hostname of the GPU server the data came from | Group / filter to see "is host `gpu-03` still syncing?" |

Add them as columns in the MLflow UI and the staleness story becomes visible at a glance. The same data is queryable programmatically — `MlflowClient.search_runs` accepts a `tags.*` filter, so a small Python snippet on the central server (or anywhere with `--tracking_uri` reachable) lists every run whose sync has gone quiet:

```python
from datetime import datetime, timedelta, timezone
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://nexus-server:5000")
exp = client.get_experiment_by_name("robot_hand_rl_pilot")
cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

stale = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string=f"tags.`nexus.lastSyncTime` < '{cutoff}'",
)
for r in stale:
    print(r.data.tags.get("mlflow.runName"), r.data.tags.get("nexus.syncedFromHost"))
```

A run that is supposed to be training but hasn't had its `nexus.lastSyncTime` updated in several cron intervals points at a sync-side problem (cron not firing, log file growing with `[ERROR]`s, or the GPU server itself rebooted without the cron re-arming). Start the diagnosis at `tail ~/nexus_sync.log` on the affected GPU server.

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

# 🔄 Scheduled Sync — Pipeline A delta upload

> **New here? Check your role:**
>
> - 🔧 **Operator** (GPU server admin, owns cron + config)
>   → [Quick Start](#quick-start) · [Config](#step-1--sync-config-file) · [Multi-user](#step-5--multi-user-servers)
>
> - 👤 **Team Member** (runs training, no cron setup required)
>   → [Team Member Checklist](#team-member-checklist)

> **Purpose:** Wire each GPU server's local MLflow (`127.0.0.1:5100`) to the central NEXUS MLflow server via cron. Each cron tick exports new metric points + new/changed artifact files since the last sync, packages them into a tar.gz delta bundle, and ships it through SCP + remote `import_delta.py` so both metrics and artifacts (checkpoints, configs, git diffs, eval reports) end up browsable in the central UI.
>
> This guide is for **operators** registering or auditing the cron sync. The system architecture is in [`10_ARCHITECTURE.md`](10_ARCHITECTURE.md); the canonical principle for the single-cron invariant is in [`00_PRINCIPLES.md#single-cron`](00_PRINCIPLES.md#-single-cron).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Team Member Checklist](#team-member-checklist)
- [Prerequisites](#prerequisites)
- [Recommended order](#recommended-order)
- [Step 1 — Sync config file](#step-1--sync-config-file)
- [Step 2 — Pre-flight check (`validate_sync.sh`)](#step-2--pre-flight-check-validate_syncsh)
- [Step 3 — Run once manually](#step-3--run-once-manually)
- [Step 4 — Register cron](#step-4--register-cron)
- [Step 5 — Multi-user servers](#step-5--multi-user-servers)
- [State file & incremental sync](#state-file--incremental-sync)
- [Verification checklist](#verification-checklist)
- [Monitoring & log inspection](#monitoring--log-inspection)
- [Stopping sync](#stopping-sync)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

---

## Quick Start

> 🔧 **Operator**

```bash
# Local MLflow on the GPU server must be running first — it's the source the
# sync exports from. If you've just set up the node (per doc 21) it's already
# up; otherwise:
bash scheduled_sync/start_local_mlflow.sh   # boots :5100

# One-time setup — the config lives under /etc/nexus/ so it can be read by any
# user, but only sudo can write to it (cron only reads).
sudo mkdir -p /etc/nexus
sudo cp scheduled_sync/sync_config.example.json /etc/nexus/sync_config.json
sudo $EDITOR /etc/nexus/sync_config.json   # set remote, remote_nexus_dir, remote_python, ssh_key

# Pre-flight check — verifies SSH, inbox, dry-run, and prints a cron line
# (validate_sync.sh and the sync scripts self-activate ~/.nexus/venv, so no
# manual `nexus-activate` is required before invoking them)
bash scheduled_sync/validate_sync.sh

# Register ONE cron under the operator's user account (NOT root) — syncs ALL
# team members' experiments automatically. $HOME expands at run time, so the
# same line works regardless of where the operator cloned nexus, as long as
# they followed doc 21's `~/nexus` convention.
crontab -e
# */5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_all.sh >> $HOME/.nexus/sync.log 2>&1
```

`sync_mlflow_all.sh` auto-discovers every non-Default experiment on local MLflow each cron tick. Team members can use any experiment name they like — no coordination required.

> 👤 Team members need no cron setup. See [Team Member Checklist](#team-member-checklist).

---

## Team Member Checklist

> 👤 **Team Member** — once the operator has registered the cron, this is all you need to know.

| Item | Details |
|------|---------|
| ✅ Log to local MLflow | Point training code at `127.0.0.1:5100` — the cron picks it up automatically |
| ✅ Use any experiment name | Any name works — `sync_mlflow_all.sh` auto-discovers all experiments |
| ❌ Do NOT register a sync cron | Never add `sync_mlflow_to_server.sh` or `sync_mlflow_all.sh` to your own crontab |

> ⚠️ Registering a second cron creates a competing state file and causes duplicate metric points on the central server. `validate_sync.sh` and `sync_mlflow_all.sh` both warn if a duplicate is detected.

---

## Prerequisites

> 🔧 **Operator**

| Requirement | Where to set up |
|---|---|
| GPU server has nexus installed | [`21_AIRGAPPED_GPU_SERVER_SETUP.md`](21_AIRGAPPED_GPU_SERVER_SETUP.md) |
| Local MLflow on GPU server is reachable at `127.0.0.1:5100` | `bash scheduled_sync/start_local_mlflow.sh` |
| Central MLflow server is up | [`20_MLFLOW_SERVER_SETUP.md`](20_MLFLOW_SERVER_SETUP.md) |
| Key-based SSH from GPU server → central server | [`20_MLFLOW_SERVER_SETUP.md` Step 8](20_MLFLOW_SERVER_SETUP.md) |

---

## Recommended order

> 🔧 **Operator**

> [!IMPORTANT]
> After writing the config file, proceed in this sequence:
>
> 1. `validate_sync.sh` — pre-flight check (SSH, permissions, dry-run)
> 2. `sync_mlflow_all.sh` — manual run to confirm real data transfer end-to-end
> 3. `crontab -e` (under the operator's user account, not root) — register the cron job
> 4. **Start training** — cron must be registered first so sync begins from step 0
>
> Registering cron after training has already started causes no data loss (the state file tracks the full history), but any metrics logged before cron was registered will be uploaded in bulk on the next cron run rather than incrementally.

---

## Step 1 — Sync config file

> 🔧 **Operator**

The fixed values live in `/etc/nexus/sync_config.json` so the cron line is a single bash invocation with no per-experiment configuration:

```bash
sudo mkdir -p /etc/nexus
sudo cp scheduled_sync/sync_config.example.json /etc/nexus/sync_config.json
sudo $EDITOR /etc/nexus/sync_config.json
```

**Required keys:** `remote`, `remote_nexus_dir`.  
**Optional:** `remote_python`, `ssh_key`, `ssh_port`, `local_uri`, `remote_uri`, `state_file`.

`experiment` is **not** configured here — `sync_mlflow_all.sh` discovers all experiments from local MLflow automatically each cron tick.

> 💡 **`remote_python`** — Non-interactive SSH does not source `~/.bashrc`, so the MLflow server's venv is never activated and `python3` resolves to the system interpreter (which has no `mlflow`). Set this to the full path of the venv Python on the MLflow server: `"/opt/nexus-mlflow/venv/bin/python3"`.

---

## Step 2 — Pre-flight check (`validate_sync.sh`)

> 🔧 **Operator**

`validate_sync.sh` runs the same config resolution as `sync_mlflow_to_server.sh`, then verifies SSH, remote inbox writability, presence of `import_delta.py` on the central server, central MLflow `/health`, local MLflow experiment count, and finally executes a `--dry-run`. A clean run prints a paste-ready cron line — it never edits your crontab.

```bash
bash scheduled_sync/validate_sync.sh
# or with a non-default config path:
bash scheduled_sync/validate_sync.sh --config /etc/nexus/sync_config.json
```

Every failure prints what to fix; the script exits 2 on the first failed step rather than continuing in a broken state.

---

## Step 3 — Run once manually

> 🔧 **Operator**

```bash
bash scheduled_sync/sync_mlflow_all.sh
# or with explicit config:
bash scheduled_sync/sync_mlflow_all.sh --config /etc/nexus/sync_config.json
```

> 💡 `--remote_nexus_dir` is the path where nexus is installed on the NEXUS server (e.g., `/opt/nexus`). Required to locate `import_delta.py` on the server.

> 💡 Add `--dry-run` to exercise the local export step only (state file is updated, no SCP, no remote import). Useful before committing the cron entry.

On success, the run will appear in the NEXUS server's MLflow UI. Each imported run also gets `nexus.lastSyncTime` and `nexus.syncedFromHost` tags so you can spot stale GPU servers from the central UI.

**On second run:** If there are no new metrics or artifacts, SCP is skipped with the message `[OK] No new data since last sync.`

---

## Step 4 — Register cron

> 🔧 **Operator**

Register under the operator's **own user account** — not root. cron sets `HOME` automatically for each user's crontab, so `$HOME` in the line below expands at run time to the operator's home and the same line works regardless of where they cloned nexus (as long as they followed doc 21's `~/nexus` convention). Do NOT use `~` in cron entries — most cron implementations don't perform tilde expansion.

```bash
crontab -e
# Add the following line (runs every 5 minutes):
*/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_all.sh \
    >> $HOME/.nexus/sync.log 2>&1
```

> 💡 If your nexus clone is somewhere other than `~/nexus`, replace `$HOME/nexus` with the absolute path (e.g. `/srv/work/nexus`). `validate_sync.sh` prints the exact path it resolved — copy from there to avoid typos.

Need a per-key override (e.g. a specific `local_uri`)? Add the matching CLI flag — flags win over the config file.

---

## Step 5 — Multi-user servers

> 🔧 **Operator**

`sync_mlflow_all.sh` uses a single state file per experiment (`~/.nexus/sync_state/{experiment}.json`) that tracks each run by its unique `run_id`. Because run_ids are distinct across all team members, there is no filter needed and no risk of one user's runs overwriting another's sync state.

The **only invariant** is: **exactly one cron must exist on the server.**

| Scenario | Safe? | Why |
|----------|:-----:|-----|
| Operator registers one cron | ✅ | Single state file, all runs tracked independently |
| Team member registers a second cron | ❌ | Creates a competing state file → duplicate metric points |
| Operator runs two instances of `sync_mlflow_all.sh` manually at the same time | ⚠️ | Per-experiment lock file prevents actual overlap; only one will proceed |

`sync_mlflow_to_server.sh` warns via `pgrep` if another instance is already running. `validate_sync.sh` checks `/var/spool/cron/crontabs/` (when readable as root) and warns if any other user has a sync cron registered.

---

## State file & incremental sync

> 🔧 **Operator**

The local state file (`~/.nexus/sync_state/{experiment}.json`) records two things per run — the last synced step for each metric tag, and the per-run `__artifacts__` skip set listing artifact paths already shipped. This is the **source of truth** for "what has been synced." On every cron tick, `export_delta.py` reads this file, queries local MLflow for new metric points and new/changed artifact files, packages everything into a tar.gz bundle (`delta.json` + `artifacts/<run_id>/...`), and only ships the delta.

| Behavior | Detail |
|---|---|
| **Survives reboot** | The state file lives in `~/.nexus/`, which persists across reboots (unlike `/tmp`). |
| **Deletion forces full re-sync** | Removing the file makes the next cron run treat every run as "never synced" and re-ship every metric and artifact. Use this if state diverges from what's on the central server. |
| **Artifact sync policy** | Paths under `checkpoints/` are re-synced every cycle (since `best.pth` / `last.pth` are overwritten in place during training). Every other artifact is synced once and recorded in `__artifacts__` so subsequent cycles skip it. The predicate is `export_delta.is_always_sync()`. |
| **Exit codes (cron-friendly)** | `0` = data transferred · `1` = configuration error · `2` = no new data (wrapper skips SCP and exits cleanly). |

---

## Verification checklist

> 🔧 **Operator**

After `validate_sync.sh` passes, run through this checklist before declaring the sync ready for production:

- [ ] `/etc/nexus/sync_config.json` populated with `remote`, `remote_nexus_dir`
- [ ] `bash scheduled_sync/validate_sync.sh` reports "All checks passed"
- [ ] `bash scheduled_sync/sync_mlflow_all.sh` manual run succeeded
- [ ] Synced runs visible in central NEXUS MLflow UI
- [ ] Cron line registered under the operator's user account (`crontab -l` shows the entry — no `sudo`)
- [ ] No other user's crontab contains `sync_mlflow_to_server` or `sync_mlflow_all`
- [ ] Each imported run carries `nexus.lastSyncTime` and `nexus.syncedFromHost` tags
- [ ] Auto-run confirmed after one cron interval (`tail ~/.nexus/sync.log`)

> ⚠️ **Single-cron invariant** — a second cron from any user on the same server creates a competing state file and causes duplicate metric points at identical steps on the central server. Canonical: [`00_PRINCIPLES.md#single-cron`](00_PRINCIPLES.md#-single-cron).

---

## Monitoring & log inspection

> 🔧 **Operator**

After cron is registered, sync runs unattended. The three places to check whether it's still healthy are: the **wrapper log** on the GPU server, the **state file** on the GPU server, and the **`nexus.*` tags** on the central MLflow UI.

### ── Where the log lives

`~/.nexus/sync.log` (the operator's home — cron expands `$HOME` automatically) is the stdout/stderr of the sync script, redirected by the cron line. Cron itself does not rotate this file — it grows by ~1–2 lines per "no new data" tick and ~10 lines per real upload. Rotate it manually if it ever bothers you (`mv ~/.nexus/sync.log ~/.nexus/sync.log.old`); the next cron tick will re-create it.

### ── Healthy output — the two normal shapes

**Shape 1 — new data was transferred (most common during active training):**

```
[2026-04-30 14:35:01] sync_mlflow_all: discovered experiments:
  - robot_hand_rl
  - navigation_rl

── Experiment: robot_hand_rl ──────────────────────────────────────
[2026-04-30 14:35:01] MLflow delta sync: robot_hand_rl
  [1/3] Exporting delta from local MLflow (http://127.0.0.1:5100)...
[INFO] Delta: 3 run(s) — 1840 new metric points, 0 new run(s)
  [OK] Delta exported (47 KB)
  [2/3] Transferring delta to nexus-server...
  [OK] Transfer complete
  [3/3] Importing delta on remote server...
  [DONE] Delta sync complete at 2026-04-30 14:35:04
```

**Shape 2 — no new metrics since last tick:**

```
[2026-04-30 14:40:01] MLflow delta sync: robot_hand_rl
  [1/3] Exporting delta from local MLflow (http://127.0.0.1:5100)...
[INFO] No new data since last sync.
  [OK] No new data since last sync. Nothing to transfer.
```

This is **not** an error — `export_delta.py` exits `2`, the wrapper catches it and skips SCP + remote import.

### ── Quick health-check commands

```bash
# Most recent tick — was it OK or an error?
tail -n 20 ~/.nexus/sync.log

# Last successful upload
grep "DONE.*Delta sync complete" ~/.nexus/sync.log | tail -n 1

# All errors in the last 24 hours
awk -v cutoff="$(date -d '24 hours ago' '+%Y-%m-%d %H:%M:%S')" \
    '/^\[20[0-9][0-9]-/{ts=substr($0,2,19)} ts>=cutoff && /\[ERROR\]/' ~/.nexus/sync.log

# Is anything stuck in flight right now?
pgrep -a -f "sync_mlflow_to_server|sync_mlflow_all|export_delta"
```

> 💡 If `pgrep` shows a process older than ~5 minutes, the sync is wedged (most often on `ssh`/`scp`). Inspect with `ps -o pid,etime,args -p <pid>` and kill if needed — the state file is only updated on a clean export, so killing mid-flight loses no points.

### ── Exit code reference

| Exit | Meaning | What to look at |
|------|---------|-----------------|
| `0` | Data transferred successfully **or** no new data since last tick | Nothing — both are healthy outcomes |
| `1` | Configuration error — unknown CLI flag, malformed config JSON, missing required key, or experiment not found | The `[ERROR] ...` line just above |
| `3` | Remote inbox not writable | SSH key, `remote`, `remote_nexus_dir`, and central server's inbox directory permissions |
| `4` | SCP failed after 3 retries | Network between GPU server and central server; firewall; key-based auth |
| `5` | Remote `import_delta.py` failed | `remote_python` (most common); `remote_nexus_dir`; central MLflow at `remote_uri` |

### ── State file inspection

```bash
cat ~/.nexus/sync_state/robot_hand_rl.json
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

What to look at: **`last_sync_time`** — if this is more than 2× your cron interval old while training is active, sync is failing silently. **Per-run, per-tag steps** — if these are advancing on every tick, sync is keeping pace.

### ── Central-side check — finding stale GPU servers

Every run imported by `import_delta.py` is stamped with two tags:

| Tag | Value | Use |
|-----|-------|-----|
| `nexus.lastSyncTime` | UTC ISO timestamp | Sort runs by this column to surface ones whose origin GPU server has gone quiet |
| `nexus.syncedFromHost` | Hostname of the GPU server | Group / filter to see "is host `gpu-03` still syncing?" |

---

## Stopping sync

> 🔧 **Operator**

### ── Remove the cron entry (required)

```bash
crontab -e   # operator's user crontab — no sudo
```

Delete or comment out the sync line, then confirm:

```bash
crontab -l   # the sync line must not appear
```

### ── Kill any in-flight sync process (if needed)

```bash
pgrep -a -f "sync_mlflow_to_server|sync_mlflow_all"
pkill -f "sync_mlflow_to_server"
pkill -f "sync_mlflow_all"
pkill -f "export_delta"
```

Killing mid-flight causes no data loss — the state file is only updated after a successful export.

### ── Stop local MLflow (optional)

Only needed if training is also stopping.

```bash
lsof -ti :5100 | xargs kill
```

> 💡 The cron entry is the only thing that drives periodic sync. Once it is removed, no further data is transferred regardless of whether the local MLflow or any other process is still running.

---

## Troubleshooting

### ── `python3: command not found` on the central server

Non-interactive SSH does not source `~/.bashrc`. Set `remote_python` to the full path:

```json
{
  "remote_python": "/opt/nexus-mlflow/venv/bin/python3"
}
```

### ── `Permission denied (publickey)` on cron run, but works manually

Cron runs without your interactive shell's environment. Either:

- Use a dedicated, passphrase-less key for sync (most common)
- Or pre-load `ssh-agent` from cron with `eval $(ssh-agent)` + `ssh-add`

### ── Central server logs duplicate metric points

A second cron was registered on this server. Find and remove it:

```bash
# Check all crontabs (requires read access):
sudo grep -r "sync_mlflow_to_server\|sync_mlflow_all" /var/spool/cron/crontabs/
# Or ask each user:
# crontab -l | grep -E "sync_mlflow"
```

Remove the duplicate cron (`crontab -e` as that user), then delete the conflicting state file (`~/.nexus/sync_state/{experiment}.json` under that user's home) to prevent further drift.

### ── Sync claims success but central UI shows no new data

Check the state file: if the recorded steps are ahead of what's actually on the central server (e.g. central was wiped), delete the state file and let the next cron tick re-sync from scratch.

### ── Wrapper exits 5 with `UnicodeDecodeError: 0x8b` after upgrading

The GPU server has the new `export_delta.py` (writes `.tar.gz`) but the central server is still on the old `import_delta.py` (only reads plain JSON).

```bash
ssh <central-host> "cd <remote_nexus_dir> && git pull"
```

`validate_sync.sh` guards this check at step 3, so re-running it after an upgrade catches a half-upgraded fleet before you re-register cron.

### ── How do I add a new sync option to the wrapper?

See `CLAUDE.md` → "When adding new features" → "New Pipeline A sync option". You need to update: the CLI flag in `sync_mlflow_to_server.sh`, the JSON `KEY_MAP` in both shell scripts, `sync_config.example.json`, the validator's required-key list (if required), and this guide.

---

## Next steps

- **Architecture detail (export → SCP → import flow)** → [`10_ARCHITECTURE.md`](10_ARCHITECTURE.md)
- **Pipeline B alternative** (one-shot, no cron) → [`13_POST_UPLOAD.md`](13_POST_UPLOAD.md)

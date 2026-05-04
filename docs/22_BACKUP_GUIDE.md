# 💾 NEXUS MLflow Backup & Restore

> **Purpose:** Protect the central MLflow tree against accidental deletion (`rm -rf`, UI "Permanently delete"), disk loss, and OS reinstall. Snapshots are daily and full-tree; restore is full-tree.
>
> **Audience:** Operators of the central NEXUS MLflow server.
>
> **Environment:** Ubuntu 22.04 LTS / `nexus-mlflow` system account / cron / sqlite3 / rsync 3.x

---

## Table of Contents

- [TL;DR](#tldr)
- [What this protects against](#what-this-protects-against)
- [How it works](#how-it-works)
- [Step 0 — Pick a backup destination](#step-0--pick-a-backup-destination)
- [Step 1 — Verify with one manual run](#step-1--verify-with-one-manual-run)
- [Step 2 — Register the daily cron job](#step-2--register-the-daily-cron-job)
- [Step 3 — Health checks for monitoring](#step-3--health-checks-for-monitoring)
- [Step 4 — Off-site replication *(optional)*](#step-4--off-site-replication-optional)
- [Restore runbook](#restore-runbook)
- [Quarterly drill](#quarterly-drill)
- [Final configuration summary](#final-configuration-summary)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

---

## TL;DR

```bash
# On the central MLflow server, after Step 3 of 20_MLFLOW_SERVER_SETUP.md
sudo apt install -y sqlite3                     # required for the online .backup API
sudo mkdir -p /backup/nexus-mlflow
sudo chown nexus-mlflow:nexus-mlflow /backup/nexus-mlflow
sudo chmod 750 /backup/nexus-mlflow

# Verify with one manual run (Step 1) — leave the MLflow service running
sudo -u nexus-mlflow bash /opt/nexus/scheduled_sync/backup_mlflow.sh \
    --src /opt/nexus-mlflow --dst /backup/nexus-mlflow --keep-daily 14 --verbose

# Then register cron (Step 2) and add health checks (Step 3)
```

> [!IMPORTANT]
> Strongly prefer a backup destination on a **separate disk** from `/opt/nexus-mlflow`. A snapshot on the same disk dies with the source — you would lose live data and backup together.

---

## What this protects against

| Failure | Recoverable? | How |
|---|---|---|
| Operator runs `rm -rf /opt/nexus-mlflow/...` from SSH | ✅ | Full restore from latest snapshot |
| User clicks **Permanently delete** on a run in the MLflow UI | ✅ *(at full-tree granularity)* | Full restore from the most recent snapshot that still contains the run |
| Disk on NEXUS server fails / OS reinstall | ✅ *(if backup is on a different disk or off-site copy exists)* | Full restore from latest snapshot |
| Snapshot itself gets corrupted | ⚠️ Use an older daily snapshot, or off-site copy | `--from-date YYYY-MM-DD` |
| Last few hours of data after the most recent nightly snapshot | ❌ Acceptable trade-off — snapshots are daily |
| Data added to the live tree *after* the chosen snapshot was taken, when restoring an older snapshot to recover a single deleted run | ❌ Restore is full-tree only — see [How it works](#how-it-works) | — |

> [!IMPORTANT]
> Restore is **full-tree only.** Granular per-run / per-experiment restore is not supported because the central backend is sqlite (single `mlflow.db` file) and selective row-level surgery against MLflow's alembic-managed schema is fragile across MLflow versions. To recover a single accidentally-deleted run, you do a *full* restore of the most recent snapshot that still contains the run — accepting that any newer data added after that snapshot is lost.

---

## How it works

Each daily snapshot under `/backup/nexus-mlflow/daily/<date>/` contains:

| Path | Source | Mechanism | Why |
|---|---|---|---|
| `mlruns/mlflow.db` | `/opt/nexus-mlflow/mlruns/mlflow.db` | `sqlite3 ".backup"` (online API) | Transactionally consistent; safe while the MLflow service is running. Plain `cp` / `rsync` of the live `.db` + `.db-wal` + `.db-shm` trio can capture them at inconsistent moments and produce a corrupt copy. |
| `artifacts/` | `/opt/nexus-mlflow/artifacts/` | `rsync -a --link-dest=PREV/artifacts` | Artifacts (checkpoints, eval reports, configs) are write-once at run time. Unchanged files hardlink against yesterday's snapshot, so 14 daily snapshots cost roughly "live tree + daily change delta". |

**Disk cost (typical 5–8 researcher team, 6-month-old server):**

| Component | Per-day cost | 14 daily snapshots |
|---|---|---|
| `mlflow.db` (no rsync dedup — bytes change everywhere on every commit) | ~1 GB | ~14 GB |
| `artifacts/` (rsync hardlink dedup — only new files cost disk) | ~50 GB initial + ~1 GB/day new | ~63 GB |
| **Total** | — | **~77 GB** |

If disk gets tight later, drop `--keep-daily` first (e.g. 14 → 7), and consider mounting `/backup` on a CoW filesystem (XFS / btrfs / ZFS) — block-level reflinks dedup the daily DB across snapshots automatically.

---

## Step 0 — Pick a backup destination

> **Why:** A snapshot that lives on the same disk as the source disappears with the source. Decide where the backup tree will live before you run anything.

### ── Create the destination directory

Strongly prefer a **separate disk** (or at least a separate filesystem) from `/opt/nexus-mlflow`.

```bash
# Example — a second internal HDD mounted at /backup
sudo mkdir -p /backup/nexus-mlflow
sudo chown nexus-mlflow:nexus-mlflow /backup/nexus-mlflow
sudo chmod 750 /backup/nexus-mlflow
```

> 💡 If you have not yet migrated to the dedicated `nexus-mlflow` system account (see [`20_MLFLOW_SERVER_SETUP.md`](20_MLFLOW_SERVER_SETUP.md) Step 3 & 6), substitute your operator account for `nexus-mlflow` above and in every command below — everything else still works.

### ── Estimate disk size

For sizing, see [How it works](#how-it-works). The short rule: budget **2 × your live data size** in free space on the backup filesystem. That comfortably covers `~14 ×` `mlflow.db` plus the artifacts tree with hardlink dedup. `du -sh /opt/nexus-mlflow/{mlruns,artifacts}` gives the live baseline.

✅ **Step 0 done when:** `/backup/nexus-mlflow` exists, is owned by `nexus-mlflow:nexus-mlflow` with mode `750`, and has at least `2 ×` your live data size free on the underlying filesystem.

---

## Step 1 — Verify with one manual run

> **Why:** Don't trust cron with something you haven't seen succeed once. A manual run also seeds the first daily snapshot — subsequent cron runs use it as the `--link-dest` source.

### ── Run the script by hand

The MLflow service can stay running — `sqlite3 .backup` uses sqlite's online backup API and does not block live writes.

```bash
sudo -u nexus-mlflow bash /opt/nexus/scheduled_sync/backup_mlflow.sh \
    --src /opt/nexus-mlflow \
    --dst /backup/nexus-mlflow \
    --keep-daily 14 \
    --verbose
```

**Expected output (tail):**

```
[03:00:00] [4a/6] sqlite3 .backup mlflow.db
[03:00:08] [4b/6] rsync artifacts/
[03:00:43] [5/6] Writing manifest and finalizing
[03:00:43]       Snapshot finalized: /backup/nexus-mlflow/daily/2026-04-27
[03:00:43]       Size: 1.1G mlflow.db + 5.2G artifacts (43s)
[03:00:43] [6/6] Rotating snapshots (keep newest 14)
[03:00:43]       Nothing to rotate (have 1 snapshots, limit 14)
[03:00:43] [DONE] Backup complete at 2026-04-27 03:00:43
```

### ── Inspect the snapshot

```bash
ls -la /backup/nexus-mlflow/daily/
ls -la /backup/nexus-mlflow/current
cat   /backup/nexus-mlflow/daily/$(date +%F)/MANIFEST.txt
```

The `current` symlink should always point at the newest snapshot. The manifest records the source path, the rsync stats, and the duration.

✅ **Step 1 done when:** the script exits `0`, today's snapshot directory contains both `mlruns/` and `artifacts/`, and `current` resolves to it.

---

## Step 2 — Register the daily cron job

> **Why:** Now that the manual run is known-good, schedule it to run at a quiet hour (the example uses 03:00 KST).

### ── Install the cron entry

```bash
sudo tee /etc/cron.d/nexus-mlflow-backup > /dev/null <<'EOF'
# NEXUS MLflow daily backup — see docs/22_BACKUP_GUIDE.md
# Runs as the dedicated nexus-mlflow user so the operator's interactive
# account is not required.
0 3 * * *  nexus-mlflow  bash /opt/nexus/scheduled_sync/backup_mlflow.sh \
    --src /opt/nexus-mlflow \
    --dst /backup/nexus-mlflow \
    --keep-daily 14 \
    >> /var/log/nexus-backup.log 2>&1
EOF
sudo chmod 644 /etc/cron.d/nexus-mlflow-backup
```

If you have not migrated to the `nexus-mlflow` user yet, replace `nexus-mlflow` with your operator account name (e.g. `jonghochoi`).

> ⚠️ **One operator only.** The script takes a `flock` on `<dst>/.lock` so two concurrent runs cannot corrupt each other — but please don't run it manually while the cron is also active.

### ── Confirm cron picks it up

```bash
# cron logs in /var/log/syslog on Ubuntu
sudo grep CRON /var/log/syslog | grep nexus-mlflow-backup | tail -5
```

You should see one entry per scheduled fire. The entry only confirms cron *invoked* the script — actual success is verified through `backup.log` in Step 3.

✅ **Step 2 done when:** `/etc/cron.d/nexus-mlflow-backup` exists, has mode `644`, and the next 03:00 fires a new snapshot directory under `daily/` without your involvement.

---

## Step 3 — Health checks for monitoring

> **Why:** A backup that silently fails for two weeks is worse than no backup at all. Wire the script's log into whatever alerting you already use.

### ── The backup.log line format

The script appends one line per run to `<dst>/backup.log`:

```
[2026-04-27 03:00:43] OK  snapshot=2026-04-27  duration=43s  db=1.1G  artifacts=5.2G
```

### ── Staleness check (recommended alarm)

The most useful single alarm is **"no new `OK` line in the last 26 hours"**.

```bash
tail -1 /backup/nexus-mlflow/backup.log
# If the timestamp is older than yesterday, cron didn't run or the run failed.
```

### ── Snapshot inventory (recurring report)

Drop this into a daily summary email or `~/.bashrc`:

```bash
# How many snapshots are on disk, and how far back do they go?
ls /backup/nexus-mlflow/daily/ | wc -l
ls /backup/nexus-mlflow/daily/ | sort | head -1   # oldest
ls /backup/nexus-mlflow/daily/ | sort | tail -1   # newest
```

✅ **Step 3 done when:** at least one of (a) the staleness check or (b) the inventory report is wired into your existing monitoring or daily-digest pipeline.

---

## Step 4 — Off-site replication *(optional)*

> **Why:** A backup on the same physical host as the live tree does not protect against fire, theft, or whole-host failure. If your team can spare a second host (even a NAS), pull the backup tree to it nightly.

### ── Pull-side cron entry

`backup_mlflow.sh` only writes locally. The recommended pattern is a **pull** from a separate host so the off-site box has its own credentials and the NEXUS server cannot reach into it:

```bash
# /etc/cron.d/nexus-mlflow-offsite  (on the off-site host)
30 3 * * *  backup-user  rsync -a --delete \
    nexus-mlflow@nexus-server:/backup/nexus-mlflow/ \
    /offsite/nexus-mlflow/ \
    >> /var/log/nexus-offsite.log 2>&1
```

The 30-minute offset (`30 3 * * *` vs the on-host `0 3 * * *`) gives the on-host backup time to finish before the pull starts.

### ── Hardlinks across the SSH boundary

`rsync -a` preserves the hardlink structure within each side but cannot preserve hardlinks **across** the SSH boundary, so the off-site copy will use full disk for each snapshot. If disk usage becomes a problem, add `-H`:

```bash
rsync -a -H --delete ...   # re-creates hardlinks on destination
```

`-H` costs significantly more memory during the transfer (rsync builds a full inode map in RAM) — only enable it if you've measured the disk cost first.

✅ **Step 4 done when:** the off-site host shows a fresh `daily/<today>/` directory by 04:00 and `tail /var/log/nexus-offsite.log` ends in an `rsync` exit code `0`.

---

## Restore runbook

`restore_mlflow.sh` has a single mode — `--full` — which replaces the live `mlflow.db` and `artifacts/` tree with the contents of a chosen snapshot. It is **dry-run by default**; pass `--apply` to actually write. A safety snapshot of the previous live tree is written to `/tmp/nexus-mlflow-pre-restore-<timestamp>/` before any overwrite, so a botched restore can be reverted by hand.

### ── Picking which snapshot to restore from

| Scenario | Snapshot to choose |
|---|---|
| Whole tree wiped (disk failure, `rm -rf`, OS reinstall) | `--from latest` (default) |
| Single run / experiment was deleted from the UI today | `--from latest` if the deletion was after last night's backup, otherwise the most recent snapshot that still contains the run |
| Live DB corrupted (sqlite errors in `journalctl`) | `--from latest` if today's backup ran before the corruption, otherwise step back day by day |

> [!IMPORTANT]
> A full restore from snapshot `<date>` discards every run, metric, and artifact that was added to the live tree **after** `<date>`. There is no merge mode. If a single deleted run is the only thing you need, weigh the loss of intervening days' new data against the value of that one run.

### ── The procedure

```bash
# 1. Stop the MLflow service so nothing writes during the restore.
sudo systemctl stop nexus-mlflow

# 2. Preview (dry-run is the default — no files are touched).
sudo -u nexus-mlflow bash /opt/nexus/scheduled_sync/restore_mlflow.sh \
    --full --from latest

# 3. Apply. The script prompts for the literal phrase 'RESTORE FULL'.
sudo bash /opt/nexus/scheduled_sync/restore_mlflow.sh \
    --full --from latest --apply

# 4. Start the service and check it.
sudo systemctl start nexus-mlflow
curl -fs http://127.0.0.1:5000/health && echo OK
```

`sudo` is needed for step 3 because the script needs to `chown` the restored files back to `nexus-mlflow` and overwrite the live DB owned by that user.

For scripted DR with no TTY available, add `--i-am-sure` to step 3 to skip the confirmation prompt.

### ── Picking an older snapshot

By default `--from latest` is used. To restore from a specific date:

```bash
# Pick an exact date
sudo bash restore_mlflow.sh --full --from-date 2026-04-20 --apply

# Or point at any snapshot path explicitly
sudo bash restore_mlflow.sh --full --from /backup/nexus-mlflow/daily/2026-04-20 --apply

# List what's available
ls /backup/nexus-mlflow/daily/
```

### ── What the script actually does on --apply

1. Verifies the snapshot DB with `PRAGMA integrity_check;` — refuses to proceed if it fails.
2. Refuses to proceed if `mlflow server` is still running on port 5000.
3. Copies the live DB and `artifacts/` to `/tmp/nexus-mlflow-pre-restore-<ts>/` as a rollback point.
4. Removes the live `mlflow.db` plus its `-wal` / `-shm` sidecars (the sidecars belong to the OLD database; leaving them next to the new file would let sqlite replay stale transactions and corrupt it).
5. `cp -a` the snapshot's `mlflow.db` into place.
6. `rsync -a --delete` the snapshot's `artifacts/` over the live `artifacts/`.
7. `chown -R nexus-mlflow:nexus-mlflow` if the live tree is owned by that user.
8. Re-runs `PRAGMA integrity_check;` on the restored DB and aborts with a clear pointer to the safety snapshot if the check fails.

---

## Quarterly drill

> 💡 A backup that has never been restored is not a backup. Run this drill once a quarter.

Once a quarter, exercise the restore path end-to-end on a throwaway machine. A live drill on the production server is unsafe because there is only the full-tree mode.

```bash
# On a clean VM that mirrors the central server's directory layout:

# 1. Pull yesterday's snapshot from the production backup tree.
rsync -a backup-user@nexus-server:/backup/nexus-mlflow/daily/$(date -d yesterday +%F)/ \
    /tmp/drill-snapshot/

# 2. Stand up an empty /opt/nexus-mlflow on the VM (no service running).
sudo install -d -o nexus-mlflow -g nexus-mlflow -m 750 \
    /opt/nexus-mlflow/{mlruns,artifacts}

# 3. Restore.
sudo bash /opt/nexus/scheduled_sync/restore_mlflow.sh \
    --full --backup-root /tmp --from /tmp/drill-snapshot --apply --i-am-sure

# 4. Start an MLflow server pointed at the restored tree and confirm
#    a known run from yesterday is browsable in the UI.
```

Record the elapsed wall-clock time — it is your real RTO budget for an actual disaster.

---

## Final configuration summary

After this guide is complete, the on-disk layout looks like this:

```
/opt/nexus-mlflow/                    ← LIVE (read by mlflow server)
├── mlruns/                           ← experiment + run metadata
├── artifacts/                        ← checkpoints, configs
└── sync_inbox/                       ← (excluded from backup)

/backup/nexus-mlflow/                 ← BACKUP root
├── daily/
│   ├── 2026-04-27/                   ← today
│   │   ├── mlruns/
│   │   │   └── mlflow.db                ← consistent sqlite copy (no -wal/-shm)
│   │   ├── artifacts/
│   │   └── MANIFEST.txt
│   ├── 2026-04-26/
│   └── ... (up to --keep-daily, default 14)
├── current → daily/2026-04-27        ← always newest
├── .lock                             ← flock — one backup at a time
└── backup.log                        ← one OK/FAIL line per run

/tmp/nexus-mlflow-pre-restore-<ts>/   ← safety snapshot, written by
                                         restore_mlflow.sh --apply

[Cron jobs]
  /etc/cron.d/nexus-mlflow-backup     ← 0 3 * * * (on the NEXUS server)
  /etc/cron.d/nexus-mlflow-offsite    ← 30 3 * * * (on the off-site host, optional)
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `[ERROR] Source mlflow.db not found: ...` | Source path is wrong, server has never started, or the live tree was actually wiped (in which case the previous snapshot is your last good copy — **do not** rerun backup until restored) | Verify `--src`. If wiped, run a restore first and only then re-enable the cron. |
| `[ERROR] sqlite3 .backup failed` | DB file unreadable (permissions), or the filesystem ran out of inodes mid-backup | Check `sudo -u nexus-mlflow sqlite3 /opt/nexus-mlflow/mlruns/mlflow.db ".tables"` works; check `df -i` on the backup mount. |
| `[ERROR] Snapshot DB failed integrity_check: ...` | The snapshot was written but is corrupt — usually means the source DB itself is corrupt, or the backup mount filled up mid-write | Run the same `PRAGMA integrity_check;` on the source DB. If the source is healthy and the backup mount has space, retry; if the source is bad, restore from the previous good daily snapshot. |
| `[ERROR] Another backup is already running (lock held: ...)` | Previous cron run is still going (very large initial backup, or stuck) | `ps -ef \| grep backup_mlflow`. If hung, kill the rsync/sqlite3 and remove `<dst>/.lock`. |
| Snapshot exists but `current` symlink missing | Backup finished but the symlink update step crashed | `ln -sfn daily/$(ls daily/ \| sort \| tail -1) /backup/nexus-mlflow/current` |
| `rsync: failed to set permissions on ...: Operation not permitted` during off-site rsync | Source files owned by `nexus-mlflow`, off-site account is different | Add `--no-perms --no-owner --no-group` to the off-site rsync, or make the off-site account a member of the same group. |
| Disk fills up faster than expected | Artifacts churn larger than estimated, or hardlinks broken across mount points | Check `du -sh /backup/nexus-mlflow/daily/*`. If each snapshot's `artifacts/` is full-size, `--link-dest` may be crossing a filesystem boundary — confirm dst is on a single mount. The DB itself does not dedup across days; that is expected. |
| Restore aborts with `Snapshot has no mlflow.db` | The selected snapshot path predates the sqlite migration or is from a partial / aborted backup | Pick a different `--from-date`; ensure the directory contains `mlruns/mlflow.db`. |

---

## Next steps

- **Stand up the central server first** → [`20_MLFLOW_SERVER_SETUP.md`](20_MLFLOW_SERVER_SETUP.md) (this guide assumes it is already running)
- **Wire scheduled cron sync from each GPU node** → [`12_SCHEDULED_SYNC.md`](12_SCHEDULED_SYNC.md)
- **Understand the full data flow before customizing** → [`10_ARCHITECTURE.md`](10_ARCHITECTURE.md)

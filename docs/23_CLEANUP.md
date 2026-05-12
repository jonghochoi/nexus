# 🧹 NEXUS MLflow Cleanup & Deletion

> **Purpose:** Permanently remove test runs, abandoned experiments, and the artifacts they leave on disk — from the GPU-node local MLflow and the central tracking server — without breaking the live system.
>
> **Audience:** Engineers cleaning up their own mis-logged runs on a GPU node, and operators reclaiming disk on the central server.
>
> **Environment:** GPU node — `~/.nexus/mlruns_training/`. Central — `/opt/nexus-mlflow/{mlruns,artifacts}/`. Both running `mlflow server --serve-artifacts` (sqlite + proxy artifact store).

---

## Table of Contents

- [TL;DR](#tldr)
- [Why MLflow's "Delete" is not actually delete](#why-mlflows-delete-is-not-actually-delete)
- [Local server cleanup (GPU node)](#local-server-cleanup-gpu-node)
- [Central server cleanup (operator)](#central-server-cleanup-operator)
- [Coordinating local and central](#coordinating-local-and-central)
- [Playbook — wipe a single test experiment end-to-end](#playbook--wipe-a-single-test-experiment-end-to-end)
- [Recovery — I deleted something I shouldn't have](#recovery--i-deleted-something-i-shouldnt-have)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

---

## TL;DR

A test run on the GPU node, never synced to central yet:

```bash
# 1. Soft-delete via the local MLflow UI at http://127.0.0.1:5100 (or via API).
# 2. Permanently reclaim disk:
source ~/.nexus/activate.sh
MLFLOW_TRACKING_URI=http://127.0.0.1:5100 \
    mlflow gc --backend-store-uri "sqlite:///$HOME/.nexus/mlruns_training/mlflow.db"
# 3. If you deleted the entire experiment, also drop its sync state file so
#    the next cron tick doesn't error out on the missing experiment:
rm -f ~/.nexus/sync_state/<experiment_name>.json
```

For the central server, see [Central server cleanup](#central-server-cleanup-operator) — the rule is the same, but **take a backup snapshot first** because central restore is full-tree only.

---

## Why MLflow's "Delete" is not actually delete

The **Delete** button in the MLflow UI (and `MlflowClient.delete_run` / `delete_experiment`) is a **soft delete**. It flips one column in the SQLite database:

```
lifecycle_stage:  active  →  deleted
```

That is the entire effect. Concretely:

| What happens | Soft delete | `mlflow gc` |
|---|---|---|
| Row hidden from UI and from `search_runs()` / `search_experiments()` | ✅ | ✅ (row removed) |
| `mlflow.db` schema integrity | ✅ preserved | ✅ preserved |
| Disk space under `artifacts/<run_id>/` reclaimed | ❌ stays forever | ✅ removed by gc |
| Row physically removed from `mlflow.db` | ❌ flag flipped only | ✅ |
| Reversible from the UI's **Restore** button | ✅ | ❌ — only from a [backup snapshot](22_BACKUP.md) |

> 💡 **Why two steps?** MLflow's design lets you change your mind for the 30 days following a soft delete (`Restore` button in the UI). Operators run `mlflow gc` periodically — or on demand — to make the disk reclaim happen.

This split is what produces the symptom that motivated this doc: an experiment "disappears" from the UI but `~/.nexus/mlruns_training/artifacts/` keeps growing. Both halves of the cleanup are needed.

---

## Local server cleanup (GPU node)

The local server is the GPU-node-local relay at `127.0.0.1:5100`. Its data lives at `~/.nexus/mlruns_training/` — `mlflow.db` (sqlite) plus `artifacts/` (the proxy-served artifact root). See `scheduled_sync/start_local_mlflow.sh` for the canonical startup command.

### ── Step 1 — Soft-delete the run or experiment

Either route works; pick whichever is faster for the number of items.

**UI route** (single run or experiment):

1. Open `http://127.0.0.1:5100` in a browser tunneled through to the GPU node.
2. Tick the run(s) or experiment in the sidebar → **Delete**.
3. Confirm. The row vanishes from the listing.

**API route** (scripted bulk delete):

```python
from mlflow.tracking import MlflowClient

c = MlflowClient(tracking_uri="http://127.0.0.1:5100")

# Delete one experiment by name (all runs inside go with it)
exp = c.get_experiment_by_name("tmp-test")
c.delete_experiment(exp.experiment_id)

# Or delete specific runs only
for r in c.search_runs(experiment_ids=[exp.experiment_id]):
    if r.data.tags.get("mlflow.runName", "").startswith("smoke_"):
        c.delete_run(r.info.run_id)
```

At this point the rows are hidden from the UI but the artifacts on disk are unchanged. Verify with:

```bash
du -sh ~/.nexus/mlruns_training/artifacts/
```

### ── Step 2 — Permanently reclaim disk with `mlflow gc`

This is the step that actually removes the soft-deleted rows from the database **and** deletes the matching artifact files from disk.

```bash
source ~/.nexus/activate.sh
MLFLOW_TRACKING_URI=http://127.0.0.1:5100 \
    mlflow gc --backend-store-uri "sqlite:///$HOME/.nexus/mlruns_training/mlflow.db"
```

> ⚠️ **`MLFLOW_TRACKING_URI` is mandatory here.** `start_local_mlflow.sh` runs the server with `--serve-artifacts`, which means artifact URIs in the DB use the proxy form `mlflow-artifacts://<run_id>/...`. To delete the underlying files, `mlflow gc` has to resolve those URIs by **calling back into the HTTP server** — so the env var must point at the running server, not at `file://`. Without it you'll see `The configured tracking uri scheme: 'file' is invalid for use with the proxy mlflow-artifact scheme`.

The local server can stay running during gc — sqlite's WAL mode handles the concurrent reader cleanly, and the gc transactions are short.

✅ **Step 2 done when:** `du -sh ~/.nexus/mlruns_training/artifacts/` reports a smaller number, and no `<deleted_run_id>` directories remain under `artifacts/`.

### ── Step 3 — Clean the sync state file

Pipeline A's incremental sync stores per-run, per-metric `last_step` cursors in `~/.nexus/sync_state/{experiment}.json`. This file does **not** auto-update when you delete runs.

Two cases:

| You deleted… | sync_state action |
|---|---|
| A few runs inside an otherwise active experiment | **Leave the file as is.** The stale `run_id` keys are harmless — `client.search_runs` skips deleted runs, so the cron tick just ignores those entries. |
| The entire experiment (`delete_experiment`) | **Delete the matching state file.** Otherwise the next cron tick will run `export_delta.py`, fail to resolve the experiment, and exit `1` every 5 minutes (`[ERROR] Experiment not found on local MLflow`). |

```bash
# Only when you deleted the whole experiment
rm -f ~/.nexus/sync_state/<experiment_name>.json
```

> 💡 If the same experiment name is recreated later (e.g. you re-run a smoke test), the next sync starts from scratch — which is the correct behavior, because the central server already has the prior data under the same name.

### ── Step 4 — (optional) Inspect for orphaned artifact directories

`mlflow gc` walks the DB and removes only the artifacts of runs it can find as soft-deleted. Files left over from earlier crashes, killed jobs, or pre-sqlite migration may not be tied to any DB row and won't be touched. To audit:

```bash
ls ~/.nexus/mlruns_training/artifacts/ | head    # one directory per run_id
# Compare against active + deleted runs in the DB
sqlite3 ~/.nexus/mlruns_training/mlflow.db \
    "SELECT run_uuid FROM runs;" | sort > /tmp/db_runs.txt
ls ~/.nexus/mlruns_training/artifacts/ | sort > /tmp/disk_runs.txt
comm -23 /tmp/disk_runs.txt /tmp/db_runs.txt    # on disk but NOT in DB → orphans
```

Confirmed orphans can be removed manually. Don't do this blindly — verify the IDs are not part of a running training job first (`ps aux | grep <run_id>`).

---

## Central server cleanup (operator)

> ⚠️ **Take a backup snapshot first.** Restore on central is full-tree only — see [`22_BACKUP.md`](22_BACKUP.md). A typo in a `delete_experiment` call can lose weeks of team data; a fresh manual snapshot makes it recoverable.

```bash
sudo -u nexus-mlflow bash /opt/nexus/scheduled_sync/backup_mlflow.sh \
    --src /opt/nexus-mlflow --dst /backup/nexus-mlflow --keep-daily 14 --verbose
```

### ── Step 1 — Soft-delete via the UI or API

Same mechanics as local, with the canonical URL the team uses (`http://nexus-server:5000`). UI deletion is fine for one-off cleanup; for scripted bulk operations, `MlflowClient(tracking_uri="http://127.0.0.1:5000")` from the central host avoids the network round-trip.

### ── Step 2 — Permanently reclaim disk with `mlflow gc`

Run as the dedicated service account so file ownership stays consistent with the systemd unit (see `20_MLFLOW_SERVER_SETUP.md` Step 6):

```bash
sudo -u nexus-mlflow bash -c '
  source /opt/nexus-mlflow/venv/bin/activate
  MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
      mlflow gc --backend-store-uri "sqlite:////opt/nexus-mlflow/mlruns/mlflow.db"
'
```

Same `--serve-artifacts` proxy caveat applies — the env var is mandatory.

`mlflow gc` holds short write transactions on `mlflow.db`. With WAL mode (enabled in `20_MLFLOW_SERVER_SETUP.md` Step 4) the live tracking service and a running `import_delta.py` from a GPU node continue to write concurrently without blocking gc, but **avoid the daily backup window** (default `03:00`). The backup's `sqlite3 .backup` API and gc compete for the same write lock — running them simultaneously usually succeeds but can occasionally produce a "database is locked" error in one of the two.

### ── Step 3 — Verify and update the backup baseline

```bash
du -sh /opt/nexus-mlflow/{mlruns,artifacts}
# Trigger an immediate post-cleanup snapshot so the new (smaller) baseline
# is the reference for tomorrow's hardlink-dedup rsync.
sudo -u nexus-mlflow bash /opt/nexus/scheduled_sync/backup_mlflow.sh \
    --src /opt/nexus-mlflow --dst /backup/nexus-mlflow --keep-daily 14
```

The first daily snapshot after a large gc costs full disk for the changed files — but subsequent days hardlink against it as usual.

---

## Coordinating local and central

Once a run has been synced to central, **the local and central copies are independent**. Deleting one does not propagate. The matrix:

| What you want gone | Delete on local | Delete on central |
|---|---|---|
| A run that never synced (still in local DB only) | ✅ required | not present anyway |
| A run that synced once and the GPU node still holds the local copy | ✅ optional (saves GPU-node disk) | ✅ required to remove from the team UI |
| A run that synced and the local copy was already gc'd | not applicable | ✅ |

### ── Recommended order

1. **Delete on central first** if multiple GPU nodes synced into the same experiment, so re-syncs from other nodes don't immediately re-create rows you just removed centrally.
2. **Delete on local** after, to reclaim GPU-node disk.
3. **Clean `sync_state/{experiment}.json` on every GPU node that synced into the deleted experiment** — otherwise their next cron tick errors on the missing-experiment lookup.

For a test experiment that lives on **one** GPU node and was never synced, you can skip central entirely — local cleanup is sufficient.

### ── If you deleted on central while local still has data

Pipeline A's incremental sync won't re-create the deleted central rows automatically — `export_delta.py` reads the local state file, which records `last_step` per metric, and only sends new points. After a fresh `last_step` baseline is established (no new metrics added on local), nothing flows. **But** if training resumes and emits new metric points, the next sync will create a brand-new run on central with the same `run_name` because [`MLflowLogger._get_or_create_run`](../nexus/logger/mlflow_logger.py) resolves runs by name on local, while central sees only the post-resume slice.

If you genuinely want the experiment gone everywhere, also delete it on local (Steps 1–3 above).

---

## Playbook — wipe a single test experiment end-to-end

The most common scenario: a smoke test or feature-test experiment that should disappear without a trace.

```bash
# ── 1. Soft-delete on local
python - <<'PYEOF'
from mlflow.tracking import MlflowClient
c = MlflowClient(tracking_uri="http://127.0.0.1:5100")
exp = c.get_experiment_by_name("tmp-test")
if exp is not None:
    c.delete_experiment(exp.experiment_id)
    print(f"[OK] Soft-deleted local experiment: {exp.name}")
PYEOF

# ── 2. Soft-delete on central (only if it was ever synced there)
python - <<'PYEOF'
from mlflow.tracking import MlflowClient
c = MlflowClient(tracking_uri="http://nexus-server:5000")
exp = c.get_experiment_by_name("tmp-test")
if exp is not None:
    c.delete_experiment(exp.experiment_id)
    print(f"[OK] Soft-deleted central experiment: {exp.name}")
PYEOF

# ── 3. Permanent reclaim on local
MLFLOW_TRACKING_URI=http://127.0.0.1:5100 \
    mlflow gc --backend-store-uri "sqlite:///$HOME/.nexus/mlruns_training/mlflow.db"

# ── 4. Permanent reclaim on central (operator only)
ssh nexus-server '
  sudo -u nexus-mlflow bash -c "
    source /opt/nexus-mlflow/venv/bin/activate
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
        mlflow gc --backend-store-uri sqlite:////opt/nexus-mlflow/mlruns/mlflow.db
  "
'

# ── 5. Clean the sync state file on this GPU node
rm -f ~/.nexus/sync_state/tmp-test.json

# ── 6. Verify the cron isn't still trying to sync the dead experiment
tail -20 ~/.nexus/sync.log
```

Repeat step 5 on every GPU node that ever synced `tmp-test`.

---

## Recovery — I deleted something I shouldn't have

| Stage of the deletion | Can you recover? | How |
|---|---|---|
| Only soft-deleted (UI **Delete** button, `delete_run`/`delete_experiment`) | ✅ Yes, easily | Open the trash-can / **Deleted** filter in the UI → **Restore**. From the API: `MlflowClient.restore_run(run_id)` or `restore_experiment(experiment_id)`. |
| Ran `mlflow gc` on local | ⚠️ Local copy gone. If it had been synced to central, the central copy still has it — re-pull from there. | UI on central, or `mlflow.artifacts.download_artifacts` against central. |
| Ran `mlflow gc` on central | ⚠️ Only via [backup restore](22_BACKUP.md) (full-tree). | `restore_mlflow.sh --full --from <date>` — accepts losing every run added since that snapshot. |
| Ran `mlflow gc` on central **and** no recent backup snapshot | ❌ Data is gone | Treat as a hard lesson — this is exactly why `22_BACKUP.md` Step 1 verifies the snapshot before relying on cron. |

For granular per-run restore on central, see `22_BACKUP.md` → "Picking which snapshot to restore from" — central restore is full-tree only by design, because selective row-level surgery against MLflow's alembic-managed sqlite schema is fragile across MLflow versions.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `mlflow gc` exits with `The configured tracking uri scheme: 'file' is invalid for use with the proxy mlflow-artifact scheme. The allowed tracking schemes are: {'https', 'http'}` | Server is `--serve-artifacts`; artifact URIs are `mlflow-artifacts://...` which need an HTTP tracking URI to resolve | Prepend `MLFLOW_TRACKING_URI=http://127.0.0.1:5100` (or `:5000` for central) before the `mlflow gc` command |
| Soft-deleted an experiment, now every cron tick logs `[ERROR] Experiment not found on local MLflow: 'tmp-test'` and exits 1 | `~/.nexus/sync_state/tmp-test.json` still references the now-missing experiment | `rm -f ~/.nexus/sync_state/tmp-test.json` (the file is rebuilt on next sync if the experiment ever returns) |
| `mlflow gc` fails with `database is locked` | The daily backup window or another long-running write (e.g. `import_delta.py` for a multi-GB artifact bundle) is holding the write lock | Wait a few minutes and retry. If chronic, see `20_MLFLOW_SERVER_SETUP.md` § Troubleshooting → "database is locked" |
| `du -sh ~/.nexus/mlruns_training/artifacts/` did not shrink after gc | (a) Items were not soft-deleted first — gc only removes already-deleted rows. (b) Files are orphans not tracked in the DB. | (a) Re-check the UI **Deleted** filter — items must appear there before gc removes them. (b) Use the audit query in [Step 4](#-step-4--optional-inspect-for-orphaned-artifact-directories) of the local cleanup section. |
| After running gc, the UI shows the experiment is empty but the experiment row itself remains | `delete_experiment` only soft-deletes the experiment container; runs inside are also flipped to `deleted`. gc removes the runs but not the empty experiment row by default. | This is expected — re-running gc with `--older-than 0d 0h 0m 0s` cleans empty experiment containers too. Leave it if the name might be re-used. |
| The local MLflow server log shows `OSError: [Errno 28] No space left on device` after a long-running training job | The artifact tree filled the disk. Soft-deleting runs in the UI alone won't free disk — `mlflow gc` is required. | Run the local cleanup procedure end-to-end; consider raising the disk quota or adding an opt-in artifact retention policy |

---

## Next steps

- **Protect against accidental deletion before it happens** → [`22_BACKUP.md`](22_BACKUP.md)
- **Understand what `sync_state/` records** → [`12_SCHEDULED_SYNC.md`](12_SCHEDULED_SYNC.md) → "State file" section
- **See where artifact URIs come from in the DB** → [`10_ARCHITECTURE.md`](10_ARCHITECTURE.md) → run-structure section

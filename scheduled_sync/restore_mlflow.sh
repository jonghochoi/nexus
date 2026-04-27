#!/bin/bash
# ============================================================
# restore_mlflow.sh  [Run on: NEXUS Server]
#
# Restore the central MLflow tree from a snapshot produced by
# backup_mlflow.sh. Only one mode is supported: --full (replace the
# entire tree). Granular per-run / per-experiment restore is not
# offered because the backend is sqlite (single-file DB) and selective
# row-level surgery against MLflow's alembic-managed schema is fragile
# across MLflow versions. To recover a single accidentally-deleted
# run, do a full restore from the most recent snapshot that still
# contains it (see 22_BACKUP_GUIDE.md for the procedure).
#
# Default mode is dry-run. Pass --apply to actually replace files.
# A safety snapshot of the current live tree is always written under
# /tmp/nexus-mlflow-pre-restore-<timestamp>/ before any overwrite.
#
# Snapshot selection:
#   --from latest                (default — newest daily snapshot)
#   --from-date YYYY-MM-DD       (specific dated snapshot under daily/)
#   --from <path>                (explicit absolute path)
#
# Usage:
#   # Show what would happen (no files touched)
#   bash restore_mlflow.sh --full
#
#   # Disaster recovery (run as root after stopping the service)
#   sudo systemctl stop nexus-mlflow
#   sudo bash restore_mlflow.sh --full --from latest --apply --i-am-sure
#   sudo systemctl start nexus-mlflow
#
# Exit codes:
#   0 — restore succeeded (or dry-run printed plan)
#   1 — argument / pre-flight error / service still running
#   2 — snapshot not found, missing mlflow.db, or failed integrity_check
# ============================================================

set -euo pipefail

# ── Defaults
BACKUP_ROOT="/backup/nexus-mlflow"
LIVE_ROOT="/opt/nexus-mlflow"
FROM="latest"
FROM_DATE=""
MODE=""
APPLY=0
I_AM_SURE=0

usage() {
    cat <<'EOF'
Usage: bash restore_mlflow.sh --full [snapshot] [--apply] [options]

Restore the entire central MLflow tree (mlflow.db + artifacts/) from a
backup snapshot. Granular per-run / per-experiment restore is NOT
supported — see the script header and 22_BACKUP_GUIDE.md.

Snapshot (pick one — default --from latest):
  --from        latest       Newest daily snapshot (default)
  --from-date   YYYY-MM-DD   Snapshot under daily/<date>
  --from        <path>       Explicit snapshot directory

Paths:
  --backup-root <path>       Backup root      (default: /backup/nexus-mlflow)
  --live-root   <path>       Live MLflow data (default: /opt/nexus-mlflow)

Behavior:
  --full                     Required. There is no other mode.
  --apply                    Actually replace files. Without this, prints
                             a plan and exits.
  --i-am-sure                Skip the interactive 'RESTORE FULL' prompt
                             (useful for scripted DR).

  -h, --help                 Show this help

The MLflow service MUST be stopped before --apply runs:
  sudo systemctl stop nexus-mlflow

A safety snapshot of the previous live tree is written under
/tmp/nexus-mlflow-pre-restore-<timestamp>/ before any overwrite.
EOF
}

# ── Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)        MODE="full";                shift ;;
        --from)        FROM="$2";                  shift 2 ;;
        --from-date)   FROM_DATE="$2";             shift 2 ;;
        --backup-root) BACKUP_ROOT="$2";           shift 2 ;;
        --live-root)   LIVE_ROOT="$2";             shift 2 ;;
        --apply)       APPLY=1;                    shift ;;
        --i-am-sure)   I_AM_SURE=1;                shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ "$MODE" != "full" ]]; then
    echo "[ERROR] --full is required (it is the only supported restore mode)" >&2
    usage >&2
    exit 1
fi

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "[ERROR] $*" >&2; exit 1; }

command -v sqlite3 >/dev/null 2>&1 || fail "sqlite3 CLI is not installed (apt install sqlite3)"
command -v rsync   >/dev/null 2>&1 || fail "rsync is not installed"

# ── Step 1: Resolve snapshot path
log "[1/5] Resolving snapshot"

if [[ -n "$FROM_DATE" ]]; then
    SNAPSHOT="$BACKUP_ROOT/daily/$FROM_DATE"
elif [[ "$FROM" == "latest" ]]; then
    LATEST=$(
        find "$BACKUP_ROOT/daily" -mindepth 1 -maxdepth 1 -type d \
             -regextype posix-extended -regex '.*/[0-9]{4}-[0-9]{2}-[0-9]{2}$' \
             -printf '%f\n' 2>/dev/null | sort | tail -n 1
    )
    [[ -n "$LATEST" ]] || { echo "[ERROR] No snapshots found under $BACKUP_ROOT/daily" >&2; exit 2; }
    SNAPSHOT="$BACKUP_ROOT/daily/$LATEST"
else
    SNAPSHOT="$FROM"
fi

DB_SNAPSHOT="$SNAPSHOT/mlruns/mlflow.db"
ART_SNAPSHOT="$SNAPSHOT/artifacts"

[[ -d "$SNAPSHOT" ]]    || { echo "[ERROR] Snapshot not found: $SNAPSHOT" >&2; exit 2; }
[[ -f "$DB_SNAPSHOT" ]] || { echo "[ERROR] Snapshot has no mlflow.db: $DB_SNAPSHOT" >&2; exit 2; }

log "      Snapshot: $SNAPSHOT"
if [[ -f "$SNAPSHOT/MANIFEST.txt" ]]; then
    grep -E '^(Started|Completed|mlflow\.db|artifacts)' "$SNAPSHOT/MANIFEST.txt" \
        | sed 's/^/      /'
fi

# Refuse self-restore.
if [[ "$(realpath "$SNAPSHOT")" == "$(realpath "$LIVE_ROOT")"* ]]; then
    fail "Snapshot path overlaps live root — refusing to restore in-place"
fi

# ── Step 2: Pre-flight integrity check on the snapshot DB
log "[2/5] Verifying snapshot DB integrity"
INTEGRITY=$(sqlite3 "$DB_SNAPSHOT" "PRAGMA integrity_check;" 2>&1)
[[ "$INTEGRITY" == "ok" ]] || { echo "[ERROR] Snapshot DB failed integrity_check: $INTEGRITY" >&2; exit 2; }
log "      DB integrity_check: ok"

# ── Step 3: Plan output and safety gate
log "[3/5] Plan"

DB_LIVE="$LIVE_ROOT/mlruns/mlflow.db"
ART_LIVE="$LIVE_ROOT/artifacts"

echo "      mlflow.db"
echo "        src: $DB_SNAPSHOT  ($(du -h "$DB_SNAPSHOT" 2>/dev/null | cut -f1))"
if [[ -f "$DB_LIVE" ]]; then
    echo "        dst: $DB_LIVE  ($(du -h "$DB_LIVE" 2>/dev/null | cut -f1)) [will be REPLACED]"
else
    echo "        dst: $DB_LIVE  [will be CREATED]"
fi
echo "      artifacts/"
if [[ -d "$ART_SNAPSHOT" ]]; then
    echo "        src: $ART_SNAPSHOT/  ($(du -sh "$ART_SNAPSHOT" 2>/dev/null | cut -f1))"
else
    echo "        src: <none in snapshot>"
fi
echo "        dst: $ART_LIVE/  (rsync --delete to match snapshot)"

if [[ $APPLY -eq 1 ]]; then
    if pgrep -f 'mlflow.*server.*5000' >/dev/null 2>&1; then
        echo "[ERROR] An MLflow server appears to be running on this host." >&2
        echo "        --full restore requires the service to be stopped first:" >&2
        echo "          sudo systemctl stop nexus-mlflow" >&2
        exit 1
    fi
    if [[ $I_AM_SURE -ne 1 ]]; then
        echo ""
        echo "  --apply will REPLACE the entire live MLflow tree with the snapshot."
        echo "  Any data added to the live tree after the snapshot was taken will be LOST."
        read -r -p "  Type 'RESTORE FULL' to continue: " CONFIRM
        [[ "$CONFIRM" == "RESTORE FULL" ]] || { echo "[ABORT] Confirmation phrase did not match." >&2; exit 1; }
    fi
fi

# ── Step 4: Dry-run preview, or apply
if [[ $APPLY -eq 0 ]]; then
    log "[4/5] Dry-run preview (no files written)"
    echo "─── artifacts/ rsync (dry-run) ─────────────────────────────"
    if [[ -d "$ART_SNAPSHOT" ]]; then
        rsync -avn --delete "$ART_SNAPSHOT/" "$ART_LIVE/" || true
    else
        echo "  Snapshot has no artifacts/ — nothing to rsync."
    fi
    echo ""
    log "[DONE] Dry-run complete. Re-run with --apply to actually restore."
    exit 0
fi

log "[4/5] Applying restore"

# Safety snapshot of the live tree before overwriting.
SAFETY="/tmp/nexus-mlflow-pre-restore-$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$SAFETY"
if [[ -f "$DB_LIVE" ]]; then
    cp -a "$DB_LIVE" "$SAFETY/mlflow.db"
fi
# Save WAL/SHM sidecars too — they belong to the live DB at this moment
# and must travel with it if the operator wants to roll back.
for sidecar in "$DB_LIVE-wal" "$DB_LIVE-shm"; do
    [[ -f "$sidecar" ]] && cp -a "$sidecar" "$SAFETY/$(basename "$sidecar")"
done
if [[ -d "$ART_LIVE" ]]; then
    rsync -a "$ART_LIVE/" "$SAFETY/artifacts/" 2>/dev/null || true
fi
log "      Safety snapshot of current state: $SAFETY"

# Replace mlflow.db. The old WAL/SHM sidecars belong to the OLD database
# and must be removed — leaving them next to the new file would let
# sqlite replay stale transactions on the next open and corrupt the DB.
mkdir -p "$LIVE_ROOT/mlruns"
rm -f "$DB_LIVE" "$DB_LIVE-wal" "$DB_LIVE-shm"
cp -a "$DB_SNAPSHOT" "$DB_LIVE"
log "      mlflow.db replaced ($(du -h "$DB_LIVE" | cut -f1))"

# Restore artifacts.
mkdir -p "$ART_LIVE"
if [[ -d "$ART_SNAPSHOT" ]]; then
    rsync -a --delete "$ART_SNAPSHOT/" "$ART_LIVE/"
    log "      artifacts/ rsynced from snapshot"
fi

# Restore ownership if the live root uses a dedicated nexus-mlflow user.
LIVE_OWNER=$(stat -c '%U' "$LIVE_ROOT/mlruns" 2>/dev/null || echo "")
if [[ "$LIVE_OWNER" == "nexus-mlflow" ]] || \
   grep -q '^nexus-mlflow:' /etc/passwd 2>/dev/null; then
    if chown -R nexus-mlflow:nexus-mlflow "$LIVE_ROOT/mlruns" "$LIVE_ROOT/artifacts" 2>/dev/null; then
        log "      chown -R nexus-mlflow:nexus-mlflow on mlruns/ and artifacts/"
    else
        echo "      [WARN] Could not chown — re-run with sudo or fix manually." >&2
    fi
fi

# ── Step 5: Verification
log "[5/5] Verification"

INTEGRITY=$(sqlite3 "$DB_LIVE" "PRAGMA integrity_check;" 2>&1)
if [[ "$INTEGRITY" != "ok" ]]; then
    echo "[ERROR] Restored DB failed integrity_check: $INTEGRITY" >&2
    echo "        Live tree is now in a bad state. The previous live DB is at" >&2
    echo "          $SAFETY/mlflow.db" >&2
    echo "        Restore from there manually before starting the service." >&2
    exit 1
fi
log "      Restored DB integrity_check: ok"

ART_FILES=$(find "$ART_LIVE" -type f 2>/dev/null | wc -l)
log "      Live artifacts/ has $ART_FILES files after restore"

echo ""
log "[DONE] Restore complete. Now restart the service:"
echo "         sudo systemctl start nexus-mlflow"
echo "         curl -fs http://127.0.0.1:5000/health && echo OK"
echo ""
echo "       Safety snapshot of the previous state: $SAFETY"
echo "       (Delete with: rm -rf '$SAFETY' once you've verified the restore.)"
exit 0

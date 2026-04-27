#!/bin/bash
# ============================================================
# backup_mlflow.sh  [Run on: NEXUS Server]
#
# Daily snapshot of the central MLflow data directory. The metadata
# DB and the artifact tree are backed up with different mechanisms
# because they have different consistency requirements:
#
#   mlruns/mlflow.db  — sqlite3 ".backup" online API, runs against the
#                       live server without stopping it and produces a
#                       transactionally consistent single-file copy.
#                       (rsync of the raw .db / -wal / -shm trio while
#                       the server is writing is unsafe.)
#
#   artifacts/        — rsync -a --link-dest=PREV. Artifacts are
#                       write-once at run time, so unchanged files
#                       hardlink against yesterday's snapshot — 14
#                       daily snapshots cost roughly the live tree size
#                       plus the daily change delta.
#
# Layout produced under --dst:
#   <dst>/
#   ├── daily/
#   │   ├── 2026-04-27/        ← today
#   │   │   ├── mlruns/
#   │   │   │   └── mlflow.db  ← consistent sqlite snapshot (no -wal/-shm)
#   │   │   ├── artifacts/
#   │   │   └── MANIFEST.txt
#   │   ├── 2026-04-26/        ← yesterday (this run's --link-dest source)
#   │   └── ... up to --keep-daily entries
#   ├── current → daily/2026-04-27   (atomic symlink)
#   ├── .lock                        (flock — one run at a time)
#   └── backup.log
#
# Usage (cron, daily at 03:00):
#   0 3 * * *  nexus-mlflow  bash /opt/nexus/scheduled_sync/backup_mlflow.sh \
#       --src /opt/nexus-mlflow \
#       --dst /backup/nexus-mlflow \
#       --keep-daily 14 \
#       >> /var/log/nexus-backup.log 2>&1
#
# Exit codes:
#   0 — snapshot written and rotation complete
#   1 — generic failure (sqlite/rsync error, finalize failure, …)
#   2 — another backup is already running (lock held)
#   3 — source mlflow.db missing (server never started, or the live tree
#       was wiped — refuses to overwrite a good snapshot with nothing)
# ============================================================

set -euo pipefail

# ── Defaults
SRC="/opt/nexus-mlflow"
DST="/backup/nexus-mlflow"
KEEP_DAILY=14
DRY_RUN=0
VERBOSE=0
EXTRA_EXCLUDES=()

# ── Argument parsing
usage() {
    cat <<'EOF'
Usage: bash backup_mlflow.sh [options]

  --src         <path>   Source MLflow data dir   (default: /opt/nexus-mlflow)
  --dst         <path>   Backup root              (default: /backup/nexus-mlflow)
  --keep-daily  <n>      Daily snapshots to keep  (default: 14)
  --exclude     <pat>    Additional rsync exclude for artifacts/ (repeatable)
  --dry-run              Show what artifacts rsync would copy; skip DB and rotation
  --verbose              Pass -v to rsync
  -h, --help             Show this help

The source must contain mlruns/mlflow.db; otherwise the script aborts
with exit code 3 to avoid clobbering a good snapshot with nothing.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)         SRC="$2";              shift 2 ;;
        --dst)         DST="$2";              shift 2 ;;
        --keep-daily)  KEEP_DAILY="$2";       shift 2 ;;
        --exclude)     EXTRA_EXCLUDES+=("$2"); shift 2 ;;
        --dry-run)     DRY_RUN=1;             shift ;;
        --verbose)     VERBOSE=1;             shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if ! [[ "$KEEP_DAILY" =~ ^[0-9]+$ ]] || [[ "$KEEP_DAILY" -lt 1 ]]; then
    echo "[ERROR] --keep-daily must be a positive integer (got: $KEEP_DAILY)" >&2
    exit 1
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
TODAY=$(date '+%Y-%m-%d')
DAILY_DIR="$DST/daily"
TARGET="$DAILY_DIR/$TODAY"
WORK="$TARGET.in-progress"
LOCK_FILE="$DST/.lock"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "[ERROR] $*" >&2; exit 1; }

echo "============================================================"
echo "[$TIMESTAMP] NEXUS MLflow backup"
echo "  Source     : $SRC"
echo "  Destination: $DST"
echo "  Today      : $TODAY"
echo "  Keep daily : $KEEP_DAILY"
[[ $DRY_RUN -eq 1 ]] && echo "  Mode       : DRY-RUN (no files written, no rotation)"
echo "============================================================"

# ── Step 1: Pre-flight checks
log "[1/6] Pre-flight checks"

command -v rsync   >/dev/null 2>&1 || fail "rsync is not installed"
command -v sqlite3 >/dev/null 2>&1 || fail "sqlite3 CLI is not installed (apt install sqlite3)"
command -v flock   >/dev/null 2>&1 || fail "flock is not installed (install util-linux)"

[[ -d "$SRC" ]]         || fail "Source directory does not exist: $SRC"
[[ -d "$SRC/mlruns" ]]  || fail "Source has no mlruns/: $SRC/mlruns"

DB_SRC="$SRC/mlruns/mlflow.db"
if [[ ! -f "$DB_SRC" ]]; then
    echo "[ERROR] Source mlflow.db not found: $DB_SRC" >&2
    echo "        Either the server has never started (no DB created yet)," >&2
    echo "        or the live tree was wiped. Do NOT rerun this script —" >&2
    echo "        the previous daily snapshot is your last good copy." >&2
    exit 3
fi

mkdir -p "$DAILY_DIR"
[[ -w "$DST" ]] || fail "Destination is not writable: $DST"

# Same-disk warning (best effort — st_dev comparison)
if [[ "$(stat -c %d "$SRC")" == "$(stat -c %d "$DST")" ]]; then
    echo "[WARN] Source and destination are on the same filesystem." >&2
    echo "       Disk loss would take both. Consider an off-site rsync of $DST." >&2
fi

# ── Step 2: Acquire lock (no concurrent runs)
log "[2/6] Acquiring lock: $LOCK_FILE"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "[ERROR] Another backup is already running (lock held: $LOCK_FILE)" >&2
    exit 2
fi

# ── Step 3: Determine previous snapshot for --link-dest
log "[3/6] Resolving previous snapshot for hardlink dedup"

PREV=""
REPLACING_TODAY=0

if [[ -d "$TARGET" ]]; then
    # Same-day re-run: use today's existing snapshot as the link-dest base
    # so the new snapshot deduplicates against it. The old one is swapped
    # out atomically in Step 5 below.
    PREV="$TARGET"
    REPLACING_TODAY=1
    log "      Today's snapshot already exists — re-running, will replace it"
else
    # Newest dated snapshot that is not today's. Pattern is strict
    # YYYY-MM-DD so .in-progress / stray dirs are ignored.
    while IFS= read -r line; do
        [[ "$line" == "$TODAY" ]] && continue
        PREV="$DAILY_DIR/$line"
        break
    done < <(
        find "$DAILY_DIR" -mindepth 1 -maxdepth 1 -type d \
             -regextype posix-extended -regex '.*/[0-9]{4}-[0-9]{2}-[0-9]{2}$' \
             -printf '%f\n' 2>/dev/null | sort -r
    )
fi

if [[ -n "$PREV" ]]; then
    log "      Link-dest source: $PREV"
else
    log "      No previous snapshot found — this will be a full copy"
fi

# Clean any stale in-progress from a crashed prior run, so rsync starts fresh.
if [[ -d "$WORK" ]]; then
    log "      Removing stale in-progress dir from previous run: $WORK"
    rm -rf "$WORK"
fi

# Clean up .in-progress on failure so the next cron run can retry cleanly.
trap '[[ -d "$WORK" ]] && rm -rf "$WORK"' ERR

# ── Step 4a: sqlite3 .backup of mlflow.db
# Uses sqlite's online backup API: page-by-page copy with proper locking,
# safe to run while the MLflow server is serving requests. Output is a
# single self-contained file; no -wal / -shm sidecars needed in the snapshot.
log "[4a/6] sqlite3 .backup mlflow.db"

mkdir -p "$WORK/mlruns"

if [[ $DRY_RUN -eq 1 ]]; then
    DB_SIZE=$(du -h "$DB_SRC" 2>/dev/null | cut -f1 || echo "?")
    log "      [dry-run] would .backup $DB_SRC (${DB_SIZE}) → $WORK/mlruns/mlflow.db"
    DB_DURATION=0
else
    DB_START=$(date +%s)
    if ! sqlite3 "$DB_SRC" ".backup '$WORK/mlruns/mlflow.db'" 2> "$WORK/.sqlite_err"; then
        cat "$WORK/.sqlite_err" >&2
        fail "sqlite3 .backup failed"
    fi
    rm -f "$WORK/.sqlite_err"
    DB_DURATION=$(( $(date +%s) - DB_START ))

    # Quick integrity check on the snapshot — catches truncated copies that
    # would be useless on restore. integrity_check is local-only, fast.
    INTEGRITY=$(sqlite3 "$WORK/mlruns/mlflow.db" "PRAGMA integrity_check;" 2>&1)
    [[ "$INTEGRITY" == "ok" ]] || fail "Snapshot DB failed integrity_check: $INTEGRITY"
fi

# ── Step 4b: rsync artifacts/ with --link-dest dedup
log "[4b/6] rsync artifacts/"

RSYNC_OPTS=(-a --delete --numeric-ids --stats)
[[ $VERBOSE -eq 1 ]] && RSYNC_OPTS+=(-v)
[[ $DRY_RUN -eq 1 ]] && RSYNC_OPTS+=(-n)
[[ -n "$PREV" && -d "$PREV/artifacts" ]] && RSYNC_OPTS+=(--link-dest="$PREV/artifacts")

RSYNC_OPTS+=(--exclude='*.tmp' --exclude='*.lock')
for pat in "${EXTRA_EXCLUDES[@]}"; do
    RSYNC_OPTS+=(--exclude="$pat")
done

# An empty artifacts/ on a fresh server is fine; rsync handles it as a no-op.
[[ -d "$SRC/artifacts" ]] || mkdir -p "$SRC/artifacts"
mkdir -p "$WORK/artifacts"

ART_START=$(date +%s)
RSYNC_LOG=$(mktemp)
if ! rsync "${RSYNC_OPTS[@]}" "$SRC/artifacts/" "$WORK/artifacts/" > "$RSYNC_LOG" 2>&1; then
    cat "$RSYNC_LOG" >&2
    rm -f "$RSYNC_LOG"
    fail "rsync of artifacts/ failed"
fi
ART_DURATION=$(( $(date +%s) - ART_START ))
DURATION=$((DB_DURATION + ART_DURATION))

if [[ $DRY_RUN -eq 1 ]]; then
    echo "─── rsync (dry-run) output ─────────────────────────────────"
    cat "$RSYNC_LOG"
    rm -f "$RSYNC_LOG"
    rm -rf "$WORK"
    log "[DONE] Dry-run complete. Nothing was written."
    exit 0
fi

# ── Step 5: Manifest + atomic finalize
log "[5/6] Writing manifest and finalizing"

DB_SIZE=$(du -h "$WORK/mlruns/mlflow.db" 2>/dev/null | cut -f1 || echo "?")
ART_SIZE=$(du -sh "$WORK/artifacts" 2>/dev/null | cut -f1 || echo "n/a")
ART_FILES=$(find "$WORK/artifacts" -type f 2>/dev/null | wc -l || echo 0)

{
    echo "NEXUS MLflow backup snapshot"
    echo "============================"
    echo "Source       : $SRC"
    echo "Snapshot     : $TARGET"
    echo "Started      : $TIMESTAMP"
    echo "Completed    : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Duration     : ${DURATION}s (sqlite ${DB_DURATION}s + rsync ${ART_DURATION}s)"
    echo "Hostname     : $(hostname)"
    echo "mlflow.db    : ${DB_SIZE} (sqlite3 .backup, integrity_check=ok)"
    echo "artifacts    : ${ART_SIZE} (${ART_FILES} files)"
    echo "Prev snapshot: ${PREV:-<none>}"
    echo ""
    echo "── rsync --stats (artifacts) ──────────────────────────────"
    cat "$RSYNC_LOG"
} > "$WORK/MANIFEST.txt"
rm -f "$RSYNC_LOG"

# Atomic rename — until this succeeds, "current" still points at the
# previous snapshot. Same-day re-run: swap today's existing snapshot out
# of the way first, then promote the new one, then delete the old.
if [[ $REPLACING_TODAY -eq 1 ]]; then
    OLD_TARGET="${TARGET}.old.$$"
    mv "$TARGET" "$OLD_TARGET"
    mv "$WORK"   "$TARGET"
    rm -rf "$OLD_TARGET"
else
    mv "$WORK" "$TARGET"
fi

# Atomic symlink update via rename.
ln -sfn "daily/$TODAY" "$DST/.current.tmp"
mv -T "$DST/.current.tmp" "$DST/current"

log "      Snapshot finalized: $TARGET"
log "      Size: ${DB_SIZE} mlflow.db + ${ART_SIZE} artifacts (${DURATION}s)"

# ── Step 6: Rotation
log "[6/6] Rotating snapshots (keep newest $KEEP_DAILY)"

# List all dated snapshots, sort oldest-first, drop the newest $KEEP_DAILY,
# rm the rest. Pure-bash to avoid `head -n -N` portability quirks.
mapfile -t ALL_SNAPS < <(
    find "$DAILY_DIR" -mindepth 1 -maxdepth 1 -type d \
         -regextype posix-extended -regex '.*/[0-9]{4}-[0-9]{2}-[0-9]{2}$' \
         -printf '%f\n' | sort
)
TOTAL=${#ALL_SNAPS[@]}
DELETE_COUNT=$((TOTAL - KEEP_DAILY))
if [[ $DELETE_COUNT -gt 0 ]]; then
    for ((i = 0; i < DELETE_COUNT; i++)); do
        log "      Deleting old snapshot: ${ALL_SNAPS[i]}"
        rm -rf "$DAILY_DIR/${ALL_SNAPS[i]}"
    done
else
    log "      Nothing to rotate (have $TOTAL snapshots, limit $KEEP_DAILY)"
fi

# Also clean any leftover .in-progress from old crashed runs.
find "$DAILY_DIR" -mindepth 1 -maxdepth 1 -type d -name '*.in-progress' \
    -mtime +1 -exec rm -rf {} + 2>/dev/null || true

# ── Done
{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK  snapshot=$TODAY  duration=${DURATION}s  db=${DB_SIZE}  artifacts=${ART_SIZE}"
} >> "$DST/backup.log"

log "[DONE] Backup complete at $(date '+%Y-%m-%d %H:%M:%S')"
exit 0

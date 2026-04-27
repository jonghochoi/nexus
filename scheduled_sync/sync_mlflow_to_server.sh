#!/bin/bash
# ============================================================
# sync_mlflow_to_server.sh  [Run on: GPU Server]
#
# Incremental (delta) MLflow sync to central server.
#
#   1. export_delta.py  — queries local MLflow, writes delta JSON
#                         (only metric points with step > last synced step)
#   2. SCP              — transfers delta JSON to MLflow server inbox
#   3. SSH              — runs import_delta.py on server to load delta
#
# State file (GPU server): ~/.nexus/sync_state/{experiment}.json
#   Tracks per-run, per-tag last-synced step.
#   On the first sync all data is transferred; subsequent syncs are incremental.
#
# Config files (optional): two-tier, system + user
#   /etc/nexus/sync_config.json     — operator-provided, team-wide values
#                                     (remote, remote_nexus_dir, ssh_port, ...)
#   ~/.nexus/sync_config.json       — per-user overrides (researcher, ssh_key)
#
#   Both are auto-discovered when --config is not passed. Per-key merge:
#   user file's keys override system file's keys.
#
# Resolution order — first non-empty wins per key:
#   1. CLI flag                       (e.g. --experiment foo)
#   2. --config <path>                (explicit JSON file — replaces auto-discovery)
#   3. ~/.nexus/sync_config.json      (per-user)
#   4. /etc/nexus/sync_config.json    (system / team-wide)
#   5. Built-in default               (only for local_uri / remote_uri / ssh_port)
#
# Multi-user note: when several researchers share one GPU server, each user
# MUST set `researcher` (in their ~/.nexus/sync_config.json or via CLI). Without
# it, every cron job exports every other user's runs and the central server
# logs duplicate metric points at identical steps.
#
# Usage (cron every 5 minutes):
#   bash sync_mlflow_to_server.sh \
#       [--config          ~/.nexus/sync_config.json]   # or set keys directly:
#       [--experiment      robot_hand_rl] \
#       [--researcher      kim] \
#       [--remote          user@mlflow-server:/data/mlflow_delta_inbox] \
#       [--remote_nexus_dir /opt/nexus] \
#       [--local_uri       http://127.0.0.1:5100] \
#       [--remote_uri      http://127.0.0.1:5000] \
#       [--ssh_key         ~/.ssh/id_rsa] \
#       [--ssh_port        22] \
#       [--state_file      ~/.nexus/sync_state/my_state.json] \
#       [--dry-run]                    # export only; skip SCP + remote import
#
# Exit codes:
#   0 — success (or "no new data")
#   1 — bad arguments / configuration (e.g. unknown experiment, bad config JSON)
#   3 — remote inbox path not creatable (SSH connectivity / permissions)
#   4 — SCP transfer failed after retries
#   5 — remote import (ssh import_delta.py) failed
#
# Cron example with a sync_config.json (most common):
#   */5 * * * * bash /path/to/nexus/scheduled_sync/sync_mlflow_to_server.sh \
#       >> /path/to/sync_cron.log 2>&1
# ============================================================

set -euo pipefail

# ── Empty placeholders (defaults are applied AFTER config-file merge)
EXPERIMENT=""
RESEARCHER=""
REMOTE=""
LOCAL_MLFLOW_URI=""
REMOTE_MLFLOW_URI=""
REMOTE_NEXUS_DIR=""
SSH_KEY=""
SSH_PORT=""
STATE_FILE=""
DRY_RUN=0
CONFIG_FILE=""

# ── Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           CONFIG_FILE="$2";       shift 2 ;;
        --experiment)       EXPERIMENT="$2";        shift 2 ;;
        --researcher)       RESEARCHER="$2";        shift 2 ;;
        --remote)           REMOTE="$2";            shift 2 ;;
        --local_uri)        LOCAL_MLFLOW_URI="$2";  shift 2 ;;
        --remote_uri)       REMOTE_MLFLOW_URI="$2"; shift 2 ;;
        --remote_nexus_dir) REMOTE_NEXUS_DIR="$2";  shift 2 ;;
        --ssh_key)          SSH_KEY="$2";           shift 2 ;;
        --ssh_port)         SSH_PORT="$2";          shift 2 ;;
        --state_file)       STATE_FILE="$2";        shift 2 ;;
        --dry-run)          DRY_RUN=1;              shift ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Determine config sources, ordered low-to-high priority
USER_CONFIG="${HOME}/.nexus/sync_config.json"
SYSTEM_CONFIG="/etc/nexus/sync_config.json"
CONFIG_SOURCES=()
if [[ -n "$CONFIG_FILE" ]]; then
    # Explicit --config disables auto-discovery — operator wants exactly this file.
    CONFIG_SOURCES=("$CONFIG_FILE")
else
    [[ -f "$SYSTEM_CONFIG" ]] && CONFIG_SOURCES+=("$SYSTEM_CONFIG")
    [[ -f "$USER_CONFIG"   ]] && CONFIG_SOURCES+=("$USER_CONFIG")
fi

# ── Read each config in order — later files overwrite earlier ones, so user
# overrides system. CLI flags still win because they were assigned first;
# we only ever fill blanks. The shared python script returns shell-quoted
# `CFG_<VAR>=<value>` lines for known keys; unknown keys produce a warning.
parse_config_file() {
    local file="$1"
    local exit_code=0
    local out
    out=$(python - "$file" <<'PYEOF'
import json, shlex, sys
KEY_MAP = {
    "experiment":       "EXPERIMENT",
    "researcher":       "RESEARCHER",
    "remote":           "REMOTE",
    "local_uri":        "LOCAL_MLFLOW_URI",
    "remote_uri":       "REMOTE_MLFLOW_URI",
    "remote_nexus_dir": "REMOTE_NEXUS_DIR",
    "ssh_key":          "SSH_KEY",
    "ssh_port":         "SSH_PORT",
    "state_file":       "STATE_FILE",
}
try:
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
except json.JSONDecodeError as e:
    print(f"[ERROR] {sys.argv[1]} is not valid JSON: {e}", file=sys.stderr)
    sys.exit(2)
if not isinstance(cfg, dict):
    print("[ERROR] sync config must be a JSON object", file=sys.stderr)
    sys.exit(2)
unknown = sorted(set(cfg) - set(KEY_MAP))
if unknown:
    print(f"[WARN] {sys.argv[1]}: ignoring unknown keys: {unknown}", file=sys.stderr)
for k, var in KEY_MAP.items():
    if k in cfg:
        print(f"CFG_{var}={shlex.quote(str(cfg[k]))}")
PYEOF
    ) || exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Failed to parse config file: $file"
        exit 1
    fi
    eval "$out"
}

for src in "${CONFIG_SOURCES[@]}"; do
    if [[ ! -f "$src" ]]; then
        echo "[ERROR] Config file not found: $src"
        exit 1
    fi
    parse_config_file "$src"
done

# CLI > merged config: only fill where CLI didn't already set a value.
EXPERIMENT="${EXPERIMENT:-${CFG_EXPERIMENT:-}}"
RESEARCHER="${RESEARCHER:-${CFG_RESEARCHER:-}}"
REMOTE="${REMOTE:-${CFG_REMOTE:-}}"
LOCAL_MLFLOW_URI="${LOCAL_MLFLOW_URI:-${CFG_LOCAL_MLFLOW_URI:-}}"
REMOTE_MLFLOW_URI="${REMOTE_MLFLOW_URI:-${CFG_REMOTE_MLFLOW_URI:-}}"
REMOTE_NEXUS_DIR="${REMOTE_NEXUS_DIR:-${CFG_REMOTE_NEXUS_DIR:-}}"
SSH_KEY="${SSH_KEY:-${CFG_SSH_KEY:-}}"
SSH_PORT="${SSH_PORT:-${CFG_SSH_PORT:-}}"
STATE_FILE="${STATE_FILE:-${CFG_STATE_FILE:-}}"

# ── Built-in defaults (lowest precedence) for purely-optional knobs
LOCAL_MLFLOW_URI="${LOCAL_MLFLOW_URI:-http://127.0.0.1:5100}"
REMOTE_MLFLOW_URI="${REMOTE_MLFLOW_URI:-http://127.0.0.1:5000}"
SSH_PORT="${SSH_PORT:-22}"

if [[ -z "$EXPERIMENT" || -z "$REMOTE" || -z "$REMOTE_NEXUS_DIR" ]]; then
    echo "Usage: bash sync_mlflow_to_server.sh \\"
    echo "    [--config          <path>]           Path to sync config JSON"
    echo "                                         (auto-discovers /etc/nexus/sync_config.json"
    echo "                                         and ~/.nexus/sync_config.json when omitted)"
    echo "    --experiment       <name>            MLflow experiment name"
    echo "    [--researcher      <name>]           Filter runs by this researcher tag"
    echo "                                         (REQUIRED on shared GPU servers)"
    echo "    --remote           <user@host:/path> SCP destination for delta files"
    echo "    --remote_nexus_dir <path>            nexus installation path on MLflow server"
    echo "    [--local_uri       <uri>]            Local MLflow URI  (default: http://127.0.0.1:5100)"
    echo "    [--remote_uri      <uri>]            Remote MLflow URI (default: http://127.0.0.1:5000)"
    echo "    [--ssh_key         <path>]           SSH private key"
    echo "    [--ssh_port        <port>]           SSH port (default: 22)"
    echo "    [--state_file      <path>]           Override local state file path"
    echo "    [--dry-run]                          Export only; skip SCP + remote import"
    echo ""
    echo "Required values may come from CLI flags or from a config file"
    echo "(experiment, remote, remote_nexus_dir)."
    exit 1
fi

SSH_OPTS="-p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10"
SCP_OPTS="-P $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10"
if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
    SCP_OPTS="$SCP_OPTS -i $SSH_KEY"
fi

REMOTE_HOST="${REMOTE%%:*}"
REMOTE_PATH="${REMOTE##*:}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# `${USER}_..._${PID}` makes the filename unique even when several users on
# the same GPU server fire `*/5 * * * *` cron jobs in the same second. The
# previous `delta_<TS>.json` collided in /tmp and on the remote inbox.
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
DELTA_USER="${USER:-$(id -un)}"
DELTA_FILENAME="delta_${DELTA_USER}_$(date '+%Y%m%d_%H%M%S')_$$.json"
DELTA_FILE="/tmp/${DELTA_FILENAME}"

echo "[$TIMESTAMP] MLflow delta sync: $EXPERIMENT${RESEARCHER:+ (researcher=$RESEARCHER)}"

# ── Activate venv if present (prefer shared ~/.nexus/venv, fall back to ./venv)
if [ -f "${HOME}/.nexus/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/.nexus/venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# ── Step 1: Export delta from local MLflow
echo "  [1/3] Exporting delta from local MLflow ($LOCAL_MLFLOW_URI)..."

EXPORT_ARGS=(--tracking_uri "$LOCAL_MLFLOW_URI"
             --experiment   "$EXPERIMENT"
             --output       "$DELTA_FILE")
[[ -n "$STATE_FILE"  ]] && EXPORT_ARGS+=(--state_file "$STATE_FILE")
[[ -n "$RESEARCHER"  ]] && EXPORT_ARGS+=(--researcher "$RESEARCHER")

# `|| EXPORT_EXIT=$?` is required: under `set -e`, a non-zero python exit
# (including the legitimate exit 2 = "no new data") would abort the script
# before we could inspect the exit code.
EXPORT_EXIT=0
python "${SCRIPT_DIR}/export_delta.py" "${EXPORT_ARGS[@]}" || EXPORT_EXIT=$?

if [[ $EXPORT_EXIT -eq 2 ]]; then
    echo "  [OK] No new data since last sync. Nothing to transfer."
    exit 0
fi
if [[ $EXPORT_EXIT -ne 0 ]]; then
    echo "  [ERROR] export_delta.py failed (exit $EXPORT_EXIT) — see message above."
    exit 1
fi

SIZE_KB=$(du -k "$DELTA_FILE" | cut -f1)
echo "  [OK] Delta exported (${SIZE_KB} KB)"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [DRY-RUN] Skipping SCP + remote import. Delta retained: $DELTA_FILE"
    exit 0
fi

# ── Step 2: SCP delta JSON to MLflow server (with retry)
echo "  [2/3] Transferring delta to $REMOTE_HOST..."
ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH'" || {
    echo "  [ERROR] Cannot prepare remote inbox '$REMOTE_PATH' on $REMOTE_HOST."
    echo "          Check SSH connectivity, key, and directory permissions."
    rm -f "$DELTA_FILE"
    exit 3
}

# 3 attempts with 5s / 10s backoff — handles transient network blips that
# would otherwise drop a 5-minute sync window onto the floor.
SCP_OK=0
for attempt in 1 2 3; do
    if scp $SCP_OPTS "$DELTA_FILE" "${REMOTE_HOST}:${REMOTE_PATH}/${DELTA_FILENAME}"; then
        SCP_OK=1; break
    fi
    if [[ $attempt -lt 3 ]]; then
        sleep_secs=$((attempt * 5))
        echo "  [WARN] SCP attempt $attempt failed — retrying in ${sleep_secs}s..."
        sleep "$sleep_secs"
    fi
done
if [[ $SCP_OK -ne 1 ]]; then
    echo "  [ERROR] SCP to ${REMOTE_HOST}:${REMOTE_PATH} failed after 3 attempts."
    rm -f "$DELTA_FILE"
    exit 4
fi
echo "  [OK] Transfer complete"

# ── Step 3: Import delta on MLflow server via SSH
echo "  [3/3] Importing delta on remote server..."
REMOTE_IMPORT_PY="${REMOTE_NEXUS_DIR}/scheduled_sync/import_delta.py"

ssh $SSH_OPTS "$REMOTE_HOST" \
    "python '$REMOTE_IMPORT_PY' \
        --delta_file   '${REMOTE_PATH}/${DELTA_FILENAME}' \
        --tracking_uri '$REMOTE_MLFLOW_URI' && \
     rm -f '${REMOTE_PATH}/${DELTA_FILENAME}'" || {
    echo "  [ERROR] Remote import_delta.py failed on $REMOTE_HOST."
    echo "          Check --remote_nexus_dir ($REMOTE_NEXUS_DIR) and remote MLflow at $REMOTE_MLFLOW_URI."
    rm -f "$DELTA_FILE"
    exit 5
}

echo "  [OK] Import complete"

# ── Cleanup local delta file
rm -f "$DELTA_FILE"

echo "  [DONE] Delta sync complete at $(date '+%Y-%m-%d %H:%M:%S')"

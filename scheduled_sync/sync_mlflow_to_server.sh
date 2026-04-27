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
# Config file (optional): ~/.nexus/sync_config.json
#   Holds the fixed values (experiment, remote, remote_nexus_dir, ssh_key, ...)
#   so cron lines can be a single bash invocation. Auto-loaded when no
#   --config is passed; an explicit --config <path> wins. CLI flags always
#   override values from the config file.
#
# Resolution order — first non-empty wins per key:
#   1. CLI flag                      (e.g. --experiment foo)
#   2. --config <path>               (explicit JSON file)
#   3. ~/.nexus/sync_config.json     (auto-discovered if it exists)
#   4. Built-in default              (only for local_uri / remote_uri / ssh_port)
#
# Usage (cron every 5 minutes):
#   bash sync_mlflow_to_server.sh \
#       [--config          ~/.nexus/sync_config.json]   # or set keys directly:
#       [--experiment      robot_hand_rl] \
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

# ── Auto-discover ~/.nexus/sync_config.json when --config is not given
DEFAULT_CONFIG="${HOME}/.nexus/sync_config.json"
if [[ -z "$CONFIG_FILE" && -f "$DEFAULT_CONFIG" ]]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
fi

# ── Merge config-file values into any still-empty CLI variables.
# CLI flags win because they were assigned first; we only fill blanks.
if [[ -n "$CONFIG_FILE" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "[ERROR] Config file not found: $CONFIG_FILE"
        exit 1
    fi
    # Emit `CFG_<VAR>=<shell-quoted-value>` for each known key in the JSON.
    # `|| CFG_EXIT=$?` keeps `set -e` from killing us before we can react —
    # under bash, a failing command substitution propagates errexit unless
    # part of a list/conditional, and `inherit_errexit` is off by default.
    CFG_EXIT=0
    CONFIG_VARS=$(python - "$CONFIG_FILE" <<'PYEOF'
import json, shlex, sys, traceback
KEY_MAP = {
    "experiment":       "EXPERIMENT",
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
    print(f"[WARN] Ignoring unknown config keys: {unknown}", file=sys.stderr)
for k, var in KEY_MAP.items():
    if k in cfg:
        print(f"CFG_{var}={shlex.quote(str(cfg[k]))}")
PYEOF
    ) || CFG_EXIT=$?
    if [[ $CFG_EXIT -ne 0 ]]; then
        echo "[ERROR] Failed to parse config file: $CONFIG_FILE"
        exit 1
    fi
    eval "$CONFIG_VARS"
    EXPERIMENT="${EXPERIMENT:-${CFG_EXPERIMENT:-}}"
    REMOTE="${REMOTE:-${CFG_REMOTE:-}}"
    LOCAL_MLFLOW_URI="${LOCAL_MLFLOW_URI:-${CFG_LOCAL_MLFLOW_URI:-}}"
    REMOTE_MLFLOW_URI="${REMOTE_MLFLOW_URI:-${CFG_REMOTE_MLFLOW_URI:-}}"
    REMOTE_NEXUS_DIR="${REMOTE_NEXUS_DIR:-${CFG_REMOTE_NEXUS_DIR:-}}"
    SSH_KEY="${SSH_KEY:-${CFG_SSH_KEY:-}}"
    SSH_PORT="${SSH_PORT:-${CFG_SSH_PORT:-}}"
    STATE_FILE="${STATE_FILE:-${CFG_STATE_FILE:-}}"
fi

# ── Built-in defaults (lowest precedence) for purely-optional knobs
LOCAL_MLFLOW_URI="${LOCAL_MLFLOW_URI:-http://127.0.0.1:5100}"
REMOTE_MLFLOW_URI="${REMOTE_MLFLOW_URI:-http://127.0.0.1:5000}"
SSH_PORT="${SSH_PORT:-22}"

if [[ -z "$EXPERIMENT" || -z "$REMOTE" || -z "$REMOTE_NEXUS_DIR" ]]; then
    echo "Usage: bash sync_mlflow_to_server.sh \\"
    echo "    [--config          <path>]           Path to sync config JSON"
    echo "                                         (default: ~/.nexus/sync_config.json if it exists)"
    echo "    --experiment       <name>            MLflow experiment name"
    echo "    --remote           <user@host:/path> SCP destination for delta files"
    echo "    --remote_nexus_dir <path>            nexus installation path on MLflow server"
    echo "    [--local_uri       <uri>]            Local MLflow URI  (default: http://127.0.0.1:5100)"
    echo "    [--remote_uri      <uri>]            Remote MLflow URI (default: http://127.0.0.1:5000)"
    echo "    [--ssh_key         <path>]           SSH private key"
    echo "    [--ssh_port        <port>]           SSH port (default: 22)"
    echo "    [--state_file      <path>]           Override local state file path"
    echo "    [--dry-run]                          Export only; skip SCP + remote import"
    echo ""
    echo "Required values may come from CLI flags or from the config file"
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

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
DELTA_FILENAME="delta_$(date '+%Y%m%d_%H%M%S').json"
DELTA_FILE="/tmp/${DELTA_FILENAME}"

echo "[$TIMESTAMP] MLflow delta sync: $EXPERIMENT"

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

STATE_ARG=""
[[ -n "$STATE_FILE" ]] && STATE_ARG="--state_file $STATE_FILE"

# `|| EXPORT_EXIT=$?` is required: under `set -e`, a non-zero python exit
# (including the legitimate exit 2 = "no new data") would abort the script
# before we could inspect the exit code.
EXPORT_EXIT=0
python "${SCRIPT_DIR}/export_delta.py" \
    --tracking_uri "$LOCAL_MLFLOW_URI" \
    --experiment   "$EXPERIMENT" \
    --output       "$DELTA_FILE" \
    $STATE_ARG \
    || EXPORT_EXIT=$?

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

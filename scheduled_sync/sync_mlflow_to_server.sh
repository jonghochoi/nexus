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
# State file (GPU server): /tmp/nexus_delta_{experiment}.json
#   Tracks per-run, per-tag last-synced step.
#   On the first sync all data is transferred; subsequent syncs are incremental.
#
# Usage (cron every 5 minutes):
#   bash sync_mlflow_to_server.sh \
#       --experiment       robot_hand_rl \
#       --remote           user@mlflow-server:/data/mlflow_delta_inbox \
#       --remote_nexus_dir /opt/nexus \
#       [--local_uri       http://127.0.0.1:5100] \
#       [--remote_uri      http://127.0.0.1:5000] \
#       [--ssh_key         ~/.ssh/id_rsa] \
#       [--ssh_port        22] \
#       [--state_file      /tmp/my_state.json]
#
# Cron example (every 5 minutes):
#   */5 * * * * bash /path/to/nexus/scheduled_sync/sync_mlflow_to_server.sh \
#       --experiment robot_hand_rl \
#       --remote user@mlflow-server:/data/mlflow_delta_inbox \
#       --remote_nexus_dir /opt/nexus \
#       >> /path/to/sync_cron.log 2>&1
# ============================================================

set -euo pipefail

# ── Defaults
EXPERIMENT=""
REMOTE=""
LOCAL_MLFLOW_URI="http://127.0.0.1:5100"
REMOTE_MLFLOW_URI="http://127.0.0.1:5000"
REMOTE_NEXUS_DIR=""
SSH_KEY=""
SSH_PORT=22
STATE_FILE=""

# ── Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)       EXPERIMENT="$2";        shift 2 ;;
        --remote)           REMOTE="$2";            shift 2 ;;
        --local_uri)        LOCAL_MLFLOW_URI="$2";  shift 2 ;;
        --remote_uri)       REMOTE_MLFLOW_URI="$2"; shift 2 ;;
        --remote_nexus_dir) REMOTE_NEXUS_DIR="$2";  shift 2 ;;
        --ssh_key)          SSH_KEY="$2";           shift 2 ;;
        --ssh_port)         SSH_PORT="$2";          shift 2 ;;
        --state_file)       STATE_FILE="$2";        shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT" || -z "$REMOTE" || -z "$REMOTE_NEXUS_DIR" ]]; then
    echo "Usage: bash sync_mlflow_to_server.sh \\"
    echo "    --experiment       <name>            MLflow experiment name"
    echo "    --remote           <user@host:/path> SCP destination for delta files"
    echo "    --remote_nexus_dir <path>            nexus installation path on MLflow server"
    echo "    [--local_uri       <uri>]            Local MLflow URI  (default: http://127.0.0.1:5100)"
    echo "    [--remote_uri      <uri>]            Remote MLflow URI (default: http://127.0.0.1:5000)"
    echo "    [--ssh_key         <path>]           SSH private key"
    echo "    [--ssh_port        <port>]           SSH port (default: 22)"
    echo "    [--state_file      <path>]           Override local state file path"
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

# ── Activate venv if present
if [ -f "venv/bin/activate" ]; then source venv/bin/activate; fi

# ── Step 1: Export delta from local MLflow
echo "  [1/3] Exporting delta from local MLflow ($LOCAL_MLFLOW_URI)..."

STATE_ARG=""
[[ -n "$STATE_FILE" ]] && STATE_ARG="--state_file $STATE_FILE"

python "${SCRIPT_DIR}/export_delta.py" \
    --tracking_uri "$LOCAL_MLFLOW_URI" \
    --experiment   "$EXPERIMENT" \
    --output       "$DELTA_FILE" \
    $STATE_ARG
EXPORT_EXIT=$?

# Exit code 2 means no new data — skip transfer
if [[ $EXPORT_EXIT -eq 2 ]]; then
    echo "  [OK] No new data since last sync. Nothing to transfer."
    exit 0
fi

SIZE_KB=$(du -k "$DELTA_FILE" | cut -f1)
echo "  [OK] Delta exported (${SIZE_KB} KB)"

# ── Step 2: SCP delta JSON to MLflow server
echo "  [2/3] Transferring delta to $REMOTE_HOST..."
ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH'"
scp $SCP_OPTS "$DELTA_FILE" "${REMOTE_HOST}:${REMOTE_PATH}/${DELTA_FILENAME}"
echo "  [OK] Transfer complete"

# ── Step 3: Import delta on MLflow server via SSH
echo "  [3/3] Importing delta on remote server..."
REMOTE_IMPORT_PY="${REMOTE_NEXUS_DIR}/scheduled_sync/import_delta.py"

ssh $SSH_OPTS "$REMOTE_HOST" \
    "python '$REMOTE_IMPORT_PY' \
        --delta_file   '${REMOTE_PATH}/${DELTA_FILENAME}' \
        --tracking_uri '$REMOTE_MLFLOW_URI' && \
     rm -f '${REMOTE_PATH}/${DELTA_FILENAME}'"

echo "  [OK] Import complete"

# ── Cleanup local delta file
rm -f "$DELTA_FILE"

echo "  [DONE] Delta sync complete at $(date '+%Y-%m-%d %H:%M:%S')"

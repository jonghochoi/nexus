#!/bin/bash
# ============================================================
# start_local_mlflow.sh  [Run on: GPU Server]
#
# Starts a local MLflow tracking server on loopback (127.0.0.1).
# No internet access required — all data is written to local disk.
#
# Run this ONCE before starting any training jobs.
# All PPO instances on all GPUs share this single local server.
#
# Usage:
#   bash start_local_mlflow.sh
#
# Stop (master + workers):
#   lsof -ti :$PORT | xargs kill
# ============================================================

set -e

PORT=5100
# Runtime data lives under ~/.nexus/ — outside the source tree — so it
# survives a `git clean` / repo re-clone and never shows up in `git status`.
# Matches the convention used by venv (~/.nexus/venv), per-user configs
# (sync_config.json, post_config.json), sync state (sync_state/), and
# upload history (history.json).
NEXUS_HOME="${HOME}/.nexus"
MLRUNS_DIR="${NEXUS_HOME}/mlruns_training"
DB_FILE="${MLRUNS_DIR}/mlflow.db"
ARTIFACTS_DIR="${MLRUNS_DIR}/artifacts"
LOG_FILE="${NEXUS_HOME}/mlflow_training.log"
PID_FILE="${NEXUS_HOME}/.mlflow_training.pid"

# Activate venv — prefer the shared ~/.nexus/venv, fall back to a repo-local
# ./venv for legacy installs.
if [ -f "${NEXUS_HOME}/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${NEXUS_HOME}/venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# Check if already running
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[SKIP] Local MLflow server already running on port $PORT"
    exit 0
fi

mkdir -p "$NEXUS_HOME" "$MLRUNS_DIR" "$ARTIFACTS_DIR"

echo "[INFO] Starting local MLflow server on 127.0.0.1:$PORT ..."

mlflow server \
    --host 127.0.0.1 \
    --port $PORT \
    --backend-store-uri "sqlite:///${DB_FILE}" \
    --artifacts-destination "$ARTIFACTS_DIR" \
    --serve-artifacts \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

# Wait for ready
for i in {1..10}; do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "[OK] Local MLflow server ready"
        echo "  PID       : $(cat $PID_FILE)"
        echo "  DB        : $DB_FILE"
        echo "  Artifacts : $ARTIFACTS_DIR"
        echo "  Port      : $PORT"
        echo ""
        echo "Stop with: lsof -ti :$PORT | xargs kill"
        exit 0
    fi
    sleep 1
done

echo "[ERROR] Server failed to start. Check: $LOG_FILE"
exit 1

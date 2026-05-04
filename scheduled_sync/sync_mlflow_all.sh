#!/bin/bash
# scheduled_sync/sync_mlflow_all.sh  [Run on: GPU Server]
# ============================================================
# Cron entry point — discovers ALL experiments on local MLflow and
# syncs each one by delegating to sync_mlflow_to_server.sh.
#
# Auto-discovers experiments each tick so team members can use any
# experiment name freely without operator configuration.
#
# Usage:
#   bash sync_mlflow_all.sh [--config /etc/nexus/sync_config.json] [--dry-run]
#   All flags are forwarded verbatim to sync_mlflow_to_server.sh per experiment.
#   --local_uri <uri>  Override local MLflow URI for experiment discovery
#                      (default: http://127.0.0.1:5100; also read from --config)
#
# Cron example (as root or dedicated sync account):
#   */5 * * * * bash /opt/nexus/scheduled_sync/sync_mlflow_all.sh \
#       >> /var/log/nexus_sync.log 2>&1
#
# Exit codes:
#   0 — all experiments synced (or no experiments found)
#   non-zero — at least one experiment sync failed (exit code of the last failure)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Default local URI — overridable via --local_uri or --config
LOCAL_URI="http://127.0.0.1:5100"

# Scan forwarded args for --local_uri (needed for experiment discovery;
# all args are still forwarded unchanged to sync_mlflow_to_server.sh)
ARGS=("$@")
for ((idx = 0; idx < ${#ARGS[@]}; idx++)); do
    if [[ "${ARGS[$idx]}" == "--local_uri" && $((idx + 1)) -lt ${#ARGS[@]} ]]; then
        LOCAL_URI="${ARGS[$((idx + 1))]}"
        break
    fi
done

# ── Activate venv if present (prefer shared ~/.nexus/venv, fall back to ./venv)
if [[ -f "${HOME}/.nexus/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.nexus/venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# ── Discover all non-Default experiments from local MLflow
EXPERIMENTS=$(python3 - "$LOCAL_URI" <<'PYEOF'
import sys
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri=sys.argv[1])
for exp in client.search_experiments():
    if exp.name != "Default":
        print(exp.name)
PYEOF
) || {
    echo "[ERROR] Cannot reach local MLflow at $LOCAL_URI — is it running?"
    echo "        Start it with: bash scheduled_sync/start_local_mlflow.sh"
    exit 1
}

if [[ -z "$EXPERIMENTS" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] sync_mlflow_all: no experiments on local MLflow ($LOCAL_URI). Nothing to sync."
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] sync_mlflow_all: discovered experiments:"
echo "$EXPERIMENTS" | sed 's/^/  - /'

# ── Sync each experiment, collecting failures
EXIT_CODE=0
while IFS= read -r exp; do
    [[ -z "$exp" ]] && continue
    echo ""
    echo "── Experiment: $exp ──────────────────────────────────────────"
    bash "${SCRIPT_DIR}/sync_mlflow_to_server.sh" --experiment "$exp" "$@" || EXIT_CODE=$?
done <<< "$EXPERIMENTS"

exit "$EXIT_CODE"

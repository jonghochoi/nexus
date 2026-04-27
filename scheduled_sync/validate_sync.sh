#!/bin/bash
# ============================================================
# validate_sync.sh  [Run on: GPU Server]
#
# Pre-flight check for the scheduled MLflow sync. Run this BEFORE registering
# the cron entry — it catches the SSH/key/path/experiment misconfigurations
# that would otherwise silently fail every 5 minutes for days.
#
# Steps (each must pass before the next runs):
#   1. Resolve config (same precedence as sync_mlflow_to_server.sh):
#        --config <path> > ~/.nexus/sync_config.json > exit 1
#      Required keys present: experiment, remote, remote_nexus_dir.
#   2. SSH reachability — non-interactive `ssh true` with 5s timeout.
#   3. Remote inbox writable — `mkdir -p` + write+read a marker file.
#   4. Remote import_delta.py exists at <remote_nexus_dir>/scheduled_sync/.
#   5. Remote MLflow `/health` reachable from the central server.
#   6. Local MLflow `/health` and the configured experiment exists.
#   7. End-to-end dry-run via `sync_mlflow_to_server.sh --dry-run`.
#
# On full success prints a paste-ready cron line. NEVER edits the user's
# crontab — that is a hard-to-reverse action and is left to the operator.
#
# Usage:
#   bash validate_sync.sh                          # uses ~/.nexus/sync_config.json
#   bash validate_sync.sh --config /path/to.json   # explicit config file
#   bash validate_sync.sh --experiment foo --remote ... --remote_nexus_dir ...
# ============================================================

set -euo pipefail

# ── Empty placeholders (filled from CLI / config / defaults below)
EXPERIMENT=""
REMOTE=""
LOCAL_MLFLOW_URI=""
REMOTE_MLFLOW_URI=""
REMOTE_NEXUS_DIR=""
SSH_KEY=""
SSH_PORT=""
CONFIG_FILE=""

# ── Argument parsing — mirrors sync_mlflow_to_server.sh
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
        -h|--help)
            sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

DEFAULT_CONFIG="${HOME}/.nexus/sync_config.json"
if [[ -z "$CONFIG_FILE" && -f "$DEFAULT_CONFIG" ]]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
fi

if [[ -n "$CONFIG_FILE" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "[ERROR] Config file not found: $CONFIG_FILE"
        exit 1
    fi
    # See sync_mlflow_to_server.sh for the rationale on `|| CFG_EXIT=$?`.
    CFG_EXIT=0
    CONFIG_VARS=$(python - "$CONFIG_FILE" <<'PYEOF'
import json, shlex, sys
KEY_MAP = {
    "experiment":       "EXPERIMENT",
    "remote":           "REMOTE",
    "local_uri":        "LOCAL_MLFLOW_URI",
    "remote_uri":       "REMOTE_MLFLOW_URI",
    "remote_nexus_dir": "REMOTE_NEXUS_DIR",
    "ssh_key":          "SSH_KEY",
    "ssh_port":         "SSH_PORT",
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
fi

LOCAL_MLFLOW_URI="${LOCAL_MLFLOW_URI:-http://127.0.0.1:5100}"
REMOTE_MLFLOW_URI="${REMOTE_MLFLOW_URI:-http://127.0.0.1:5000}"
SSH_PORT="${SSH_PORT:-22}"

if [[ -z "$EXPERIMENT" || -z "$REMOTE" || -z "$REMOTE_NEXUS_DIR" ]]; then
    echo "[ERROR] Missing required fields. Need: experiment, remote, remote_nexus_dir."
    echo "        Provide via --experiment / --remote / --remote_nexus_dir or in"
    echo "        ~/.nexus/sync_config.json (see scheduled_sync/sync_config.example.json)."
    exit 1
fi

REMOTE_HOST="${REMOTE%%:*}"
REMOTE_PATH="${REMOTE##*:}"
SSH_OPTS="-p $SSH_PORT -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5"
[[ -n "$SSH_KEY" ]] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step()    { echo ""; echo "── $* ──"; }
ok()      { echo "  [OK]    $*"; }
fail()    { echo "  [FAIL]  $*"; exit 2; }

echo "validate_sync.sh — pre-flight check"
echo "  experiment       : $EXPERIMENT"
echo "  remote           : $REMOTE"
echo "  remote_nexus_dir : $REMOTE_NEXUS_DIR"
echo "  local_uri        : $LOCAL_MLFLOW_URI"
echo "  remote_uri       : $REMOTE_MLFLOW_URI"
[[ -n "$SSH_KEY" ]] && echo "  ssh_key          : $SSH_KEY"
[[ "$SSH_PORT" != "22" ]] && echo "  ssh_port         : $SSH_PORT"
[[ -n "$CONFIG_FILE" ]] && echo "  config source    : $CONFIG_FILE"

# ── 1. SSH reachability
step "1/6  SSH reachability — $REMOTE_HOST"
if ssh $SSH_OPTS "$REMOTE_HOST" true 2>/dev/null; then
    ok "ssh $REMOTE_HOST true succeeded"
else
    fail "Cannot SSH to $REMOTE_HOST. Check key, port, hostname, and that BatchMode auth works."
fi

# ── 2. Remote inbox writable
step "2/6  Remote inbox writable — $REMOTE_PATH"
MARKER=".nexus_validate_$(date +%s)_$$"
if ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH' && \
        echo ok > '$REMOTE_PATH/$MARKER' && \
        cat '$REMOTE_PATH/$MARKER' && \
        rm -f '$REMOTE_PATH/$MARKER'" >/dev/null 2>&1; then
    ok "Created, wrote, read, and cleaned up marker file in $REMOTE_PATH"
else
    fail "Cannot create or write to $REMOTE_PATH on $REMOTE_HOST. Check permissions."
fi

# ── 3. Remote import_delta.py exists
step "3/6  Remote import_delta.py present"
REMOTE_IMPORT_PY="${REMOTE_NEXUS_DIR}/scheduled_sync/import_delta.py"
if ssh $SSH_OPTS "$REMOTE_HOST" "test -f '$REMOTE_IMPORT_PY'" 2>/dev/null; then
    ok "$REMOTE_IMPORT_PY exists"
else
    fail "$REMOTE_IMPORT_PY not found. Verify --remote_nexus_dir points at the nexus checkout."
fi

# ── 4. Remote MLflow /health
step "4/6  Remote MLflow /health — $REMOTE_MLFLOW_URI"
# /health is reached *from the central server* because that's where import_delta.py runs.
if ssh $SSH_OPTS "$REMOTE_HOST" "curl -sS -m 5 '${REMOTE_MLFLOW_URI%/}/health' >/dev/null" 2>/dev/null; then
    ok "Central MLflow responded on $REMOTE_MLFLOW_URI"
else
    fail "Central MLflow at $REMOTE_MLFLOW_URI is not reachable from $REMOTE_HOST."
fi

# ── 5. Local MLflow /health and experiment exists
step "5/6  Local MLflow + experiment '$EXPERIMENT'"
if ! curl -sS -m 5 "${LOCAL_MLFLOW_URI%/}/health" >/dev/null 2>&1; then
    fail "Local MLflow at $LOCAL_MLFLOW_URI not reachable. Run start_local_mlflow.sh first."
fi
ok "Local MLflow is reachable"
EXP_OK=$(python - "$LOCAL_MLFLOW_URI" "$EXPERIMENT" <<'PYEOF'
import sys
from mlflow.tracking import MlflowClient
uri, name = sys.argv[1], sys.argv[2]
client = MlflowClient(tracking_uri=uri)
exp = client.get_experiment_by_name(name)
if exp is None:
    avail = sorted(e.name for e in client.search_experiments())
    print(f"NO|{avail}")
else:
    print("YES")
PYEOF
)
if [[ "$EXP_OK" == YES ]]; then
    ok "Experiment '$EXPERIMENT' exists on local MLflow"
else
    AVAIL="${EXP_OK#NO|}"
    fail "Experiment '$EXPERIMENT' not found on local MLflow. Available: $AVAIL"
fi

# ── 6. End-to-end dry run
step "6/6  Dry-run sync (export only — no SCP, no remote import)"
DRY_ARGS=()
[[ -n "$CONFIG_FILE" ]] && DRY_ARGS+=("--config" "$CONFIG_FILE")
DRY_ARGS+=("--dry-run")
# Pass through any per-key overrides so the dry-run sees the same values we validated.
[[ -n "$EXPERIMENT"        ]] && DRY_ARGS+=("--experiment" "$EXPERIMENT")
[[ -n "$REMOTE"            ]] && DRY_ARGS+=("--remote" "$REMOTE")
[[ -n "$REMOTE_NEXUS_DIR"  ]] && DRY_ARGS+=("--remote_nexus_dir" "$REMOTE_NEXUS_DIR")
if bash "${SCRIPT_DIR}/sync_mlflow_to_server.sh" "${DRY_ARGS[@]}"; then
    ok "Dry-run completed"
else
    fail "Dry-run failed — see output above."
fi

# ── Success — print a paste-ready cron line (we deliberately do NOT touch crontab)
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All checks passed. Suggested cron line (edit interval as needed):"
echo "════════════════════════════════════════════════════════════"
if [[ -n "$CONFIG_FILE" && "$CONFIG_FILE" == "$DEFAULT_CONFIG" ]]; then
    echo "*/5 * * * * bash ${SCRIPT_DIR}/sync_mlflow_to_server.sh \\"
    echo "    >> \$HOME/nexus_sync.log 2>&1"
elif [[ -n "$CONFIG_FILE" ]]; then
    echo "*/5 * * * * bash ${SCRIPT_DIR}/sync_mlflow_to_server.sh \\"
    echo "    --config $CONFIG_FILE \\"
    echo "    >> \$HOME/nexus_sync.log 2>&1"
else
    echo "*/5 * * * * bash ${SCRIPT_DIR}/sync_mlflow_to_server.sh \\"
    echo "    --experiment       $EXPERIMENT \\"
    echo "    --remote           $REMOTE \\"
    echo "    --remote_nexus_dir $REMOTE_NEXUS_DIR \\"
    echo "    >> \$HOME/nexus_sync.log 2>&1"
fi
echo ""
echo "Register with: crontab -e"
exit 0

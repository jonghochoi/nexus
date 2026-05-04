#!/bin/bash
# ============================================================
# validate_sync.sh  [Run on: GPU Server]
#
# Pre-flight check for the scheduled MLflow sync. Run this BEFORE registering
# the cron entry — it catches the SSH/key/path/experiment misconfigurations
# that would otherwise silently fail every 5 minutes for days.
#
# Steps (each must pass before the next runs):
#   0. Crontab conflict check — warns if a sync cron is already registered.
#   1. Resolve config (same precedence as sync_mlflow_to_server.sh):
#        CLI flag > --config <path> > /etc/nexus/sync_config.json > exit 1
#      Required keys present: remote, remote_nexus_dir.
#   2. SSH reachability — non-interactive `ssh true` with 5s timeout.
#   3. Remote inbox writable — `mkdir -p` + write+read a marker file.
#   4. Remote import_delta.py exists at <remote_nexus_dir>/scheduled_sync/.
#   5. Remote MLflow `/health` reachable from the central server.
#   6. Local MLflow `/health` and experiment count check.
#   7. End-to-end dry-run via `sync_mlflow_all.sh --dry-run`.
#
# On full success prints a paste-ready cron line. NEVER edits the user's
# crontab — that is a hard-to-reverse action and is left to the operator.
#
# Usage:
#   bash validate_sync.sh                          # uses /etc/nexus/sync_config.json
#   bash validate_sync.sh --config /path/to.json   # explicit config file
#   bash validate_sync.sh --remote ... --remote_nexus_dir ...
# ============================================================

set -euo pipefail

# ── Empty placeholders (filled from CLI / config / defaults below)
REMOTE=""
LOCAL_MLFLOW_URI=""
REMOTE_MLFLOW_URI=""
REMOTE_NEXUS_DIR=""
REMOTE_PYTHON=""
SSH_KEY=""
SSH_PORT=""
CONFIG_FILE=""

# ── Argument parsing — mirrors sync_mlflow_to_server.sh
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           CONFIG_FILE="$2";       shift 2 ;;
        --remote)           REMOTE="$2";            shift 2 ;;
        --local_uri)        LOCAL_MLFLOW_URI="$2";  shift 2 ;;
        --remote_uri)       REMOTE_MLFLOW_URI="$2"; shift 2 ;;
        --remote_nexus_dir) REMOTE_NEXUS_DIR="$2";  shift 2 ;;
        --remote_python)    REMOTE_PYTHON="$2";     shift 2 ;;
        --ssh_key)          SSH_KEY="$2";           shift 2 ;;
        --ssh_port)         SSH_PORT="$2";          shift 2 ;;
        -h|--help)
            sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# Same config resolution as sync_mlflow_to_server.sh: explicit --config
# disables auto-discovery; otherwise /etc/nexus/sync_config.json is used.
SYSTEM_CONFIG="/etc/nexus/sync_config.json"
CONFIG_SOURCES=()
if [[ -n "$CONFIG_FILE" ]]; then
    CONFIG_SOURCES=("$CONFIG_FILE")
else
    [[ -f "$SYSTEM_CONFIG" ]] && CONFIG_SOURCES+=("$SYSTEM_CONFIG")
fi

parse_config_file() {
    local file="$1"
    local exit_code=0
    local out
    out=$(python - "$file" <<'PYEOF'
import json, shlex, sys
KEY_MAP = {
    "remote":           "REMOTE",
    "local_uri":        "LOCAL_MLFLOW_URI",
    "remote_uri":       "REMOTE_MLFLOW_URI",
    "remote_nexus_dir": "REMOTE_NEXUS_DIR",
    "remote_python":    "REMOTE_PYTHON",
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

REMOTE="${REMOTE:-${CFG_REMOTE:-}}"
LOCAL_MLFLOW_URI="${LOCAL_MLFLOW_URI:-${CFG_LOCAL_MLFLOW_URI:-}}"
REMOTE_MLFLOW_URI="${REMOTE_MLFLOW_URI:-${CFG_REMOTE_MLFLOW_URI:-}}"
REMOTE_NEXUS_DIR="${REMOTE_NEXUS_DIR:-${CFG_REMOTE_NEXUS_DIR:-}}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${CFG_REMOTE_PYTHON:-}}"
SSH_KEY="${SSH_KEY:-${CFG_SSH_KEY:-}}"
SSH_PORT="${SSH_PORT:-${CFG_SSH_PORT:-}}"

LOCAL_MLFLOW_URI="${LOCAL_MLFLOW_URI:-http://127.0.0.1:5100}"
REMOTE_MLFLOW_URI="${REMOTE_MLFLOW_URI:-http://127.0.0.1:5000}"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"

if [[ -z "$REMOTE" || -z "$REMOTE_NEXUS_DIR" ]]; then
    echo "[ERROR] Missing required fields. Need: remote, remote_nexus_dir."
    echo "        Provide via CLI or in /etc/nexus/sync_config.json"
    echo "        (see scheduled_sync/sync_config.example.json)."
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
echo "  remote           : $REMOTE"
echo "  remote_nexus_dir : $REMOTE_NEXUS_DIR"
echo "  remote_python    : $REMOTE_PYTHON"
echo "  local_uri        : $LOCAL_MLFLOW_URI"
echo "  remote_uri       : $REMOTE_MLFLOW_URI"
[[ -n "$SSH_KEY" ]] && echo "  ssh_key          : $SSH_KEY"
[[ "$SSH_PORT" != "22" ]] && echo "  ssh_port         : $SSH_PORT"
if [[ -n "$CONFIG_FILE" ]]; then
    echo "  config source    : $CONFIG_FILE"
else
    [[ ${#CONFIG_SOURCES[@]} -gt 0 ]] && \
        echo "  config source    : ${CONFIG_SOURCES[*]}"
fi

# ── 0. Crontab conflict check ─────────────────────────────────────────────────
step "0/7  Crontab conflict check"
CRON_CONFLICT=0
CURRENT_CRON=$(crontab -l 2>/dev/null || true)
if echo "$CURRENT_CRON" | grep -qE "sync_mlflow_to_server|sync_mlflow_all"; then
    echo "  [WARN] A sync cron is already in your crontab:"
    echo "$CURRENT_CRON" | grep -E "sync_mlflow_to_server|sync_mlflow_all" | sed 's/^/         /'
    echo "         Do NOT add the suggested cron line again — that would create"
    echo "         two competing sync jobs and produce duplicate metric points."
    CRON_CONFLICT=1
else
    ok "No existing sync cron for current user"
fi

# Scan other users' crontabs if accessible (requires root / readable spool dir).
SPOOL_DIR="/var/spool/cron/crontabs"
if [[ -r "$SPOOL_DIR" ]]; then
    OTHER_CONFLICT=0
    for cron_file in "$SPOOL_DIR"/*; do
        [[ -f "$cron_file" ]] || continue
        other_user=$(basename "$cron_file")
        [[ "$other_user" == "$(id -un)" ]] && continue
        if grep -qE "sync_mlflow_to_server|sync_mlflow_all" "$cron_file" 2>/dev/null; then
            echo "  [WARN] User '$other_user' also has a sync cron registered."
            echo "         Only ONE cron must exist on this server."
            echo "         Ask '$other_user' to run: crontab -e (then remove the entry)"
            CRON_CONFLICT=1
            OTHER_CONFLICT=1
        fi
    done
    [[ $OTHER_CONFLICT -eq 0 ]] && ok "No conflicting sync crons found for other users"
else
    echo "  [INFO] Cannot read $SPOOL_DIR (no root access) — verify manually:"
    echo "         Ask each team member: crontab -l | grep -E 'sync_mlflow'"
fi

# ── 1. SSH reachability
step "1/7  SSH reachability — $REMOTE_HOST"
if ssh $SSH_OPTS "$REMOTE_HOST" true 2>/dev/null; then
    ok "ssh $REMOTE_HOST true succeeded"
else
    fail "Cannot SSH to $REMOTE_HOST. Check key, port, hostname, and that BatchMode auth works."
fi

# ── 2. Remote inbox writable
step "2/7  Remote inbox writable — $REMOTE_PATH"
MARKER=".nexus_validate_$(date +%s)_$$"
if ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH' && \
        echo ok > '$REMOTE_PATH/$MARKER' && \
        cat '$REMOTE_PATH/$MARKER' && \
        rm -f '$REMOTE_PATH/$MARKER'" >/dev/null 2>&1; then
    ok "Created, wrote, read, and cleaned up marker file in $REMOTE_PATH"
else
    fail "Cannot create or write to $REMOTE_PATH on $REMOTE_HOST. Check permissions."
fi

# ── 3. Remote import_delta.py exists AND understands tar.gz bundles
#       Pre-feature import_delta.py crashes on gzip magic bytes with
#       `UnicodeDecodeError: 0x8b` and the wrapper exits 5 every cycle —
#       silently losing every run. Catch it here once instead.
step "3/7  Remote import_delta.py present and current"
REMOTE_IMPORT_PY="${REMOTE_NEXUS_DIR}/scheduled_sync/import_delta.py"
if ! ssh $SSH_OPTS "$REMOTE_HOST" "test -f '$REMOTE_IMPORT_PY'" 2>/dev/null; then
    fail "$REMOTE_IMPORT_PY not found. Verify --remote_nexus_dir points at the nexus checkout."
fi
if ! ssh $SSH_OPTS "$REMOTE_HOST" "grep -q 'tarfile.is_tarfile' '$REMOTE_IMPORT_PY'" 2>/dev/null; then
    fail "$REMOTE_IMPORT_PY is from before the artifact-sync feature and cannot
        decode tar.gz delta bundles. Update the central server:
            ssh $REMOTE_HOST 'cd $REMOTE_NEXUS_DIR && git pull'"
fi
ok "$REMOTE_IMPORT_PY exists and supports tar.gz bundles"

# ── 4. Remote Python can import mlflow
step "4/7  Remote Python — $REMOTE_PYTHON"
if ssh $SSH_OPTS "$REMOTE_HOST" "'$REMOTE_PYTHON' -c 'import mlflow'" 2>/dev/null; then
    ok "$REMOTE_PYTHON can import mlflow"
else
    fail "'$REMOTE_PYTHON' cannot import mlflow on $REMOTE_HOST.
        Set remote_python in /etc/nexus/sync_config.json to the venv path,
        e.g. \"remote_python\": \"/opt/nexus-mlflow/venv/bin/python3\""
fi

# ── 5. Remote MLflow /health
step "5/7  Remote MLflow /health — $REMOTE_MLFLOW_URI"
# /health is reached *from the central server* because that's where import_delta.py runs.
if ssh $SSH_OPTS "$REMOTE_HOST" "curl -sS -m 5 '${REMOTE_MLFLOW_URI%/}/health' >/dev/null" 2>/dev/null; then
    ok "Central MLflow responded on $REMOTE_MLFLOW_URI"
else
    fail "Central MLflow at $REMOTE_MLFLOW_URI is not reachable from $REMOTE_HOST."
fi

# ── 6. Local MLflow /health and experiment count check
step "6/7  Local MLflow + experiments"
if ! curl -sS -m 5 "${LOCAL_MLFLOW_URI%/}/health" >/dev/null 2>&1; then
    fail "Local MLflow at $LOCAL_MLFLOW_URI not reachable. Run start_local_mlflow.sh first."
fi
ok "Local MLflow is reachable"

EXP_EXISTS=1
EXP_COUNT=$(python3 - "$LOCAL_MLFLOW_URI" <<'PYEOF'
import sys
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri=sys.argv[1])
print(sum(1 for e in client.search_experiments() if e.name != "Default"))
PYEOF
)
if [[ "$EXP_COUNT" -gt 0 ]]; then
    ok "Local MLflow has $EXP_COUNT experiment(s) — all will be synced by sync_mlflow_all.sh"
else
    echo "  [WARN]  No experiments found yet on local MLflow."
    echo "          sync_mlflow_all.sh will auto-discover them when training starts."
    EXP_EXISTS=0
fi

# ── 7. End-to-end dry run
step "7/7  Dry-run sync (export only — no SCP, no remote import)"
if [[ $EXP_EXISTS -eq 0 ]]; then
    echo "  [SKIP]  No experiments found — skipping dry-run (nothing to export yet)."
else
    DRY_ARGS=()
    [[ -n "$CONFIG_FILE" ]] && DRY_ARGS+=("--config" "$CONFIG_FILE")
    DRY_ARGS+=("--dry-run")
    [[ -n "$REMOTE"           ]] && DRY_ARGS+=("--remote" "$REMOTE")
    [[ -n "$REMOTE_NEXUS_DIR" ]] && DRY_ARGS+=("--remote_nexus_dir" "$REMOTE_NEXUS_DIR")
    [[ -n "$REMOTE_PYTHON"    ]] && DRY_ARGS+=("--remote_python"    "$REMOTE_PYTHON")
    [[ -n "$LOCAL_MLFLOW_URI" ]] && DRY_ARGS+=("--local_uri" "$LOCAL_MLFLOW_URI")
    if bash "${SCRIPT_DIR}/sync_mlflow_all.sh" "${DRY_ARGS[@]}"; then
        ok "Dry-run completed"
    else
        fail "Dry-run failed — see output above."
    fi
fi

# ── Success — print a paste-ready cron line (we deliberately do NOT touch crontab)
echo ""
echo "════════════════════════════════════════════════════════════"
if [[ $CRON_CONFLICT -eq 1 ]]; then
    echo "  All checks passed."
    echo "  [WARN] A sync cron already exists — review before adding another."
    echo "         Reference cron line (do NOT add if already registered):"
else
    echo "  All checks passed. Suggested cron line (edit interval as needed):"
fi
echo "════════════════════════════════════════════════════════════"
echo "*/5 * * * * bash ${SCRIPT_DIR}/sync_mlflow_all.sh >> /var/log/nexus_sync.log 2>&1"
echo ""
echo "  Register as root or under a dedicated sync service account:"
echo "    sudo crontab -e"
exit 0

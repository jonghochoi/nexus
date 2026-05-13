"""
scheduled_sync/_parse_sync_config.py
====================================

Shared parser for the NEXUS sync config JSON — single source of truth for
the CLI-key ↔ shell-variable map used by both `sync_mlflow_to_server.sh`
and `validate_sync.sh`. Emits shell-quoted `CFG_<VAR>=<value>` lines that
the caller `eval`s; unknown keys produce a `[WARN]` on stderr.

Usage (from any scheduled_sync/*.sh):
    out=$(python "$SCRIPT_DIR/_parse_sync_config.py" "$file") || {
        echo "[ERROR] Failed to parse config file: $file"; exit 1;
    }
    eval "$out"

Exit codes — matches the previous inline heredoc contract:
    0 — parsed (any unknown keys reported on stderr)
    2 — JSON invalid or not an object

Callers consume only the CFG_<VAR> names they care about, so emitting
the full map is harmless for scripts that don't accept every key.
"""

import json
import shlex
import sys

# ── Single source of truth — JSON key → shell variable name ─────────────────
KEY_MAP = {
    "experiment": "EXPERIMENT",
    "remote": "REMOTE",
    "local_uri": "LOCAL_MLFLOW_URI",
    "remote_uri": "REMOTE_MLFLOW_URI",
    "remote_nexus_dir": "REMOTE_NEXUS_DIR",
    "remote_python": "REMOTE_PYTHON",
    "ssh_key": "SSH_KEY",
    "ssh_port": "SSH_PORT",
    "state_file": "STATE_FILE",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _parse_sync_config.py <config.json>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    try:
        with open(path) as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] {path} is not valid JSON: {e}", file=sys.stderr)
        return 2
    if not isinstance(cfg, dict):
        print("[ERROR] sync config must be a JSON object", file=sys.stderr)
        return 2
    # Keys starting with `_` are reserved for in-file comments.
    unknown = sorted(k for k in set(cfg) - set(KEY_MAP) if not k.startswith("_"))
    if unknown:
        print(f"[WARN] {path}: ignoring unknown keys: {unknown}", file=sys.stderr)
    for k, var in KEY_MAP.items():
        if k in cfg:
            print(f"CFG_{var}={shlex.quote(str(cfg[k]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

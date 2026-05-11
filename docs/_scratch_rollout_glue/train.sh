#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train.sh — invoke train.py with a pinned RUN_NAME so play.sh can find it.
# ─────────────────────────────────────────────────────────────────────────────
# Required env:
#   EXPERIMENT_NAME — MLflow experiment that owns the run
#   RUN_NAME        — pinned identity. Either set explicitly or let
#                     train_and_eval.sh derive it as $(date +%F_%H-%M-%S).
# Optional env (with defaults):
#   TASK            (default: Sharpa-InHandRotation-Direct-v0)
#   ALGO            (default: PPO)
#   HAND_SIDE       (default: right) — must match between train and play;
#                    a checkpoint trained on one side won't load on the other.
#   PYTHON          interpreter path. Auto-detected: if
#                    /workspace/IsaacLab/_isaac_sim/python.sh or
#                    /isaac-sim/python.sh is executable, it's used;
#                    otherwise falls back to plain `python`. Export
#                    PYTHON yourself to override the autodetection.
#   NEXUS_LOCAL_URI / NEXUS_CENTRAL_URI — passed through to train.py via env;
#                    train.py.snippet.py reads them.
#
# Extra flags after the script name (e.g. --num_envs, --max_agent_steps) are
# forwarded verbatim to train.py.
#
# Examples:
#   EXPERIMENT_NAME=robot_hand_rl RUN_NAME=manual_001 ./train.sh
#   EXPERIMENT_NAME=robot_hand_rl RUN_NAME=smoke_001  ./train.sh --num_envs 64 --max_agent_steps 1000
#   EXPERIMENT_NAME=robot_hand_rl RUN_NAME=v3_seed42  TASK=Sharpa-InHandRotation-Direct-v0 ALGO=PPO ./train.sh --num_envs 256
#   EXPERIMENT_NAME=robot_hand_rl RUN_NAME=v3_seed42  PYTHON=/opt/custom/python ./train.sh --num_envs 64  # override autodetect
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve script dir so `train.py` is found relative to this file, not CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${RUN_NAME:?RUN_NAME is required (set via train_and_eval.sh or export it manually)}"
: "${TASK:=Sharpa-InHandRotation-Direct-v0}"
: "${ALGO:=PPO}"
: "${HAND_SIDE:=right}"

# Auto-detect the Isaac Lab python wrapper — it's a .bashrc alias / function
# in interactive shells, but invisible to non-interactive scripts. Fall back
# to plain `python` outside Isaac containers. Override by exporting PYTHON.
if [[ -z "${PYTHON:-}" ]]; then
    for _cand in /workspace/IsaacLab/_isaac_sim/python.sh /isaac-sim/python.sh; do
        [[ -x "$_cand" ]] && PYTHON="$_cand" && break
    done
    : "${PYTHON:=python}"
fi

"$PYTHON" "${SCRIPT_DIR}/train.py" \
    --task              "$TASK" \
    --algo              "$ALGO" \
    --experiment_name   "$EXPERIMENT_NAME" \
    --run_name          "$RUN_NAME" \
    --hand_side         "$HAND_SIDE" \
    "$@"

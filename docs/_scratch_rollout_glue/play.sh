#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# play.sh — load a trained checkpoint, drop a structured artifact bundle,
#           and (by default) hand it to nexus's upload_eval.py.
# ─────────────────────────────────────────────────────────────────────────────
# Required env:
#   EXPERIMENT_NAME — same as the train run
#   RUN_NAME        — same as the train run (so train_dir resolves correctly)
# Optional env:
#   TASK / ALGO     — must match the training run for the checkpoint to load
#   HAND_SIDE       — must match the training run (right/left); a checkpoint
#                     trained on one side won't load on the other
#   CKPT            — explicit checkpoint path; defaults to <train_dir>/best.pth
#   ARTIFACTS       — "1" (default) creates the structured artifact dir;
#                     "0" reverts to the legacy ./outputs/videos behaviour
#   UPLOAD          — "1" (default) calls upload_eval.py; "0" skips it
#   TARGET          — "central" (default) or "local". 'local' uploads to the
#                     GPU-node-local relay — useful when the run hasn't synced
#                     to central yet. Read from .nexus_run.json.
#   PYTHON          — interpreter path. Auto-detected (Isaac Lab wrappers
#                     under /workspace/IsaacLab/_isaac_sim/python.sh or
#                     /isaac-sim/python.sh); falls back to plain `python`.
#                     Export PYTHON yourself to override.
#   NEXUS_REPO      — path to the nexus checkout (where post_upload/ lives);
#                     play.py.snippet.py reads this when --upload is set
#
# Extra flags after the script name (e.g. --num_envs 4) are forwarded to play.py.
# Tip — RecordVideo follows env_idx=0 only, so for video pass --num_envs 4 (or
# similar small N) even when training used 256.
#
# Layout assumption — train.sh produced:
#   ./logs/<EXPERIMENT_NAME>/<RUN_NAME>/<ALGO>/{best.pth, .nexus_run.json}
# Override LOGS_ROOT if your trainer writes elsewhere.
#
# Examples:
#   EXPERIMENT_NAME=robot_hand_rl RUN_NAME=v3_seed42 ./play.sh                              # default: artifacts on, upload to central
#   EXPERIMENT_NAME=robot_hand_rl RUN_NAME=v3_seed42 UPLOAD=0 ./play.sh --num_envs 4         # local-only smoke, no MLflow call
#   EXPERIMENT_NAME=robot_hand_rl RUN_NAME=v3_seed42 CKPT=./logs/.../last.pth ./play.sh      # eval last.pth instead of best.pth
#   EXPERIMENT_NAME=robot_hand_rl RUN_NAME=v3_seed42 TARGET=local ./play.sh                  # upload to GPU-node-local relay (run not yet synced)
#   EXPERIMENT_NAME=robot_hand_rl RUN_NAME=v3_seed42 LOGS_ROOT=/scratch/team/logs ./play.sh  # custom logs root
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve script dir so `play.py` is found relative to this file, not CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${TASK:=Sharpa-InHandRotation-Direct-v0}"
: "${ALGO:=PPO}"
: "${HAND_SIDE:=right}"
: "${LOGS_ROOT:=./logs}"
: "${ARTIFACTS:=1}"
: "${UPLOAD:=1}"
: "${TARGET:=central}"

# Auto-detect the Isaac Lab python wrapper — see train.sh for rationale.
if [[ -z "${PYTHON:-}" ]]; then
    for _cand in /workspace/IsaacLab/_isaac_sim/python.sh /isaac-sim/python.sh; do
        [[ -x "$_cand" ]] && PYTHON="$_cand" && break
    done
    : "${PYTHON:=python}"
fi

TRAIN_DIR="${LOGS_ROOT}/${EXPERIMENT_NAME}/${RUN_NAME}/${ALGO}"
CKPT="${CKPT:-${TRAIN_DIR}/best.pth}"

if [[ ! -f "$CKPT" ]]; then
    echo "play.sh: checkpoint not found at $CKPT" >&2
    echo "  set CKPT explicitly, or check that train.sh produced best.pth." >&2
    exit 1
fi

EVAL_ID="$(date +%Y%m%d_%H%M%S)"
ARTIFACTS_DIR="${TRAIN_DIR}/eval/${EVAL_ID}"

ARGS=(
    --task       "$TASK"
    --algo       "$ALGO"
    --hand_side  "$HAND_SIDE"
    --checkpoint "$CKPT"
)
# Note — RecordVideo is gated by `--artifacts_dir` inside play.py.snippet.py
# (Block 2/3), not by a separate `--video` flag. Some upstream trainers
# expose their own `--video` toggle; if you keep one, append it via `"$@"`
# at call time rather than hard-coding it here, since argparse prefix
# matching collapses bare `--video` onto `--video_length` when no exact
# `--video` arg is defined.
if [[ "$ARTIFACTS" == "1" ]]; then
    ARGS+=(--artifacts_dir "$ARTIFACTS_DIR")
fi
if [[ "$UPLOAD" == "1" ]]; then
    if [[ "$ARTIFACTS" != "1" ]]; then
        echo "play.sh: UPLOAD=1 requires ARTIFACTS=1 (the artifact dir is what gets uploaded)" >&2
        exit 1
    fi
    ARGS+=(--upload --run_info_dir "$TRAIN_DIR" --target "$TARGET")
fi

echo "play.sh: TRAIN_DIR=$TRAIN_DIR"
echo "play.sh: CKPT=$CKPT"
[[ "$ARTIFACTS" == "1" ]] && echo "play.sh: ARTIFACTS_DIR=$ARTIFACTS_DIR"
[[ "$UPLOAD" == "1" ]] && echo "play.sh: UPLOAD=on (target=$TARGET via .nexus_run.json)"

"$PYTHON" "${SCRIPT_DIR}/play.py" "${ARGS[@]}" "$@"

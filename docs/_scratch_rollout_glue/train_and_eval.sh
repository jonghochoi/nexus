#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_and_eval.sh — chain train.sh and play.sh under a shared RUN_NAME.
# ─────────────────────────────────────────────────────────────────────────────
# The whole point of this wrapper is to *pre-compute* RUN_NAME in the shell so
# both phases write into / read from the same train_dir without needing
# stdout parsing or a `latest` symlink. Once exported here, train.py picks it
# up via --run_name (set inside train.sh) and play.sh reads it from the env.
#
# Required env:
#   EXPERIMENT_NAME — MLflow experiment that owns the run
# Optional env:
#   RUN_NAME        — pin a custom name; otherwise a timestamp is generated
#   PYTHON          — interpreter path; auto-detected by each sub-script
#                     (Isaac Lab wrapper, else plain `python`). Export
#                     PYTHON yourself to override.
#   TASK / ALGO / LOGS_ROOT / NEXUS_REPO / NEXUS_LOCAL_URI / NEXUS_CENTRAL_URI
#                   — forwarded as-is to train.sh / play.sh
#
# Extra flags after the script name are forwarded *only* to train.sh
# (e.g. --num_envs, --max_agent_steps). play.sh runs with no extra flags by
# default — pass them explicitly by editing this script if you need them.
#
# This is a thin convenience wrapper — see README.md "Why three scripts?" for
# why train.sh / play.sh stay separately invocable instead of being collapsed.
#
# Examples:
#   EXPERIMENT_NAME=robot_hand_rl ./train_and_eval.sh --num_envs 64 --max_agent_steps 1000  # smoke E2E
#   EXPERIMENT_NAME=robot_hand_rl RUN_NAME=v3_seed42 ./train_and_eval.sh --num_envs 256     # pinned RUN_NAME
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve sibling sub-scripts relative to this file, not CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
export RUN_NAME="${RUN_NAME:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXPERIMENT_NAME

echo "train_and_eval.sh: shared RUN_NAME=$RUN_NAME"
echo "train_and_eval.sh: EXPERIMENT_NAME=$EXPERIMENT_NAME"

"${SCRIPT_DIR}/train.sh" "$@"
"${SCRIPT_DIR}/play.sh"

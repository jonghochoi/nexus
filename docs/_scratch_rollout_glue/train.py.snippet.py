# ─────────────────────────────────────────────────────────────────────────────
# train.py — patch
# ─────────────────────────────────────────────────────────────────────────────
# Goal: have make_logger() write *both* tracking URIs into the sidecar so
# `play.py --upload` can default to target=central without taking another flag.
#
# Find the existing make_logger(...) call in your train.py — typically in the
# "build logger" block right after the run/output dirs are computed — and
# add the `central_tracking_uri=` keyword.
#
# Nothing else in train.py changes for the rollout-artifact pipeline. The
# sidecar (.nexus_run.json) is written automatically into `tb_dir`.
#
# Required env vars (or hard-coded equivalents):
#   NEXUS_LOCAL_URI    e.g. http://127.0.0.1:5100   (GPU-node-local relay)
#   NEXUS_CENTRAL_URI  e.g. http://nexus-server:5000 (team-shared central)
# ─────────────────────────────────────────────────────────────────────────────

import os

from nexus.logger import make_logger

# train_dir / experiment_name / run_name come from your existing CLI/cfg —
# leave them as-is. The pinned RUN_NAME from train.sh flows in via --run_name.

writer = make_logger(
    mode="dual",
    tb_dir=train_dir,                    # ← unchanged; sidecar lands here
    run_name=run_name,                   # ← unchanged; pinned by train.sh
    experiment_name=experiment_name,     # ← unchanged
    tracking_uri=os.environ.get(
        "NEXUS_LOCAL_URI", "http://127.0.0.1:5100"
    ),
    central_tracking_uri=os.environ.get(  # ← NEW — single line that matters
        "NEXUS_CENTRAL_URI", "http://nexus-server:5000"
    ),
    agent_params=agent_cfg,              # ← unchanged
    env_params=env_cfg,                  # ← unchanged
)

# Result on disk after train.py exits:
#   <train_dir>/
#   ├── events.out.tfevents.*
#   ├── .nexus_run.json   ← contains tracking_uri AND central_tracking_uri
#   └── checkpoints/
#       ├── best.pth
#       └── last.pth

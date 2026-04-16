#!/usr/bin/env python3
"""
export_delta.py  [Run on: GPU Server]
======================================
Exports only NEW metric data from local MLflow since the last sync.
Produces a delta JSON file for SCP transfer to the MLflow server.

Unlike `mlflow experiments export` (which always exports everything),
this script maintains a local state file tracking per-run, per-tag
last-synced step, and only serializes data points beyond that step.

State file (JSON): /tmp/nexus_delta_{experiment}.json
  {
    "runs": {
      "<run_id>": {"reward": 1000, "policy_loss": 500, ...}
    },
    "last_sync_time": 1713456789.0
  }

Delta file (JSON): written to --output, then SCP'd by sync_mlflow_to_server.sh

Exit codes:
  0 — delta written successfully, has data to transfer
  2 — no new data since last sync (caller skips SCP)
"""

import argparse
import json
import sys
import time
from pathlib import Path

from mlflow.tracking import MlflowClient


def parse_args():
    p = argparse.ArgumentParser(
        description="Export MLflow delta (new metrics only) for scheduled sync"
    )
    p.add_argument("--tracking_uri", default="http://127.0.0.1:5100",
                   help="Local MLflow URI (default: http://127.0.0.1:5100)")
    p.add_argument("--experiment",   required=True,
                   help="MLflow experiment name")
    p.add_argument("--output",       required=True,
                   help="Path to write delta JSON")
    p.add_argument("--state_file",   default=None,
                   help="Override state file path (default: /tmp/nexus_delta_{experiment}.json)")
    return p.parse_args()


def load_state(path: str) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {"runs": {}}


def save_state(path: str, state: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def main():
    args = parse_args()
    state_path = args.state_file or f"/tmp/nexus_delta_{args.experiment}.json"

    client = MlflowClient(tracking_uri=args.tracking_uri)

    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        print(f"[WARN] Experiment not found: {args.experiment} — nothing to export.", flush=True)
        sys.exit(2)

    state      = load_state(state_path)
    runs_state = state.get("runs", {})

    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000,
    )

    delta_runs     = []
    new_runs_state = {}

    for run in all_runs:
        run_id    = run.info.run_id
        run_name  = run.data.tags.get("mlflow.runName", run_id)
        run_state = runs_state.get(run_id, {})   # {tag -> last_synced_step}
        new_run_state = dict(run_state)

        is_new_run = run_id not in runs_state

        # ── Collect only new metric points for each tag
        delta_metrics = []
        for key in run.data.metrics:
            last_step = run_state.get(key, -1)
            history   = client.get_metric_history(run_id, key)
            new_pts   = [
                {
                    "key":       key,
                    "value":     m.value,
                    "step":      m.step,
                    "timestamp": m.timestamp,
                }
                for m in history
                if m.step > last_step
            ]
            if new_pts:
                delta_metrics.extend(new_pts)
                new_run_state[key] = max(pt["step"] for pt in new_pts)

        new_runs_state[run_id] = new_run_state

        # Skip runs with no new metrics and already seen before
        if not delta_metrics and not is_new_run:
            continue

        entry = {
            "run_id":   run_id,
            "run_name": run_name,
            "status":   run.info.status,
            "metrics":  delta_metrics,
        }
        # Params and tags only needed on the first sync for a run
        if is_new_run:
            entry["params"] = [{"key": k, "value": v} for k, v in run.data.params.items()]
            entry["tags"]   = dict(run.data.tags)

        delta_runs.append(entry)

    total_metrics  = sum(len(r["metrics"]) for r in delta_runs)
    total_new_runs = sum(1 for r in delta_runs if "params" in r)

    print(
        f"[INFO] Delta: {len(delta_runs)} run(s) — "
        f"{total_metrics} new metric points, {total_new_runs} new run(s)",
        flush=True,
    )

    # Always persist updated state (marks runs as seen even if no metrics yet)
    state["runs"]           = new_runs_state
    state["last_sync_time"] = time.time()
    save_state(state_path, state)

    if total_metrics == 0 and total_new_runs == 0:
        print("[INFO] No new data since last sync.", flush=True)
        sys.exit(2)

    delta = {
        "experiment": args.experiment,
        "sync_time":  time.time(),
        "runs":       delta_runs,
    }
    with open(args.output, "w") as f:
        json.dump(delta, f)

    print(f"[OK] Delta written to: {args.output}", flush=True)


if __name__ == "__main__":
    main()

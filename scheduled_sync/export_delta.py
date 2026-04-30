#!/usr/bin/env python3
"""
scheduled_sync/export_delta.py  [Run on: GPU Server]
======================================================
Exports only NEW data (metrics, params, tags, artifacts) from local MLflow
since the last sync. Produces a tar.gz bundle for SCP transfer to the
MLflow server.

Unlike `mlflow experiments export` (which always exports everything),
this script maintains a local state file tracking per-run last-synced
metric steps and already-synced artifact paths, so each cycle only
transfers what is genuinely new.

When several researchers share one GPU server (and one local MLflow), pass
`--researcher <name>` so each user's cron only exports runs tagged with
that researcher. Without the filter, every cron job re-exports every run
and the central server gets duplicate metric points at the same step.

State file (JSON): ~/.nexus/sync_state/{experiment}[__{researcher}].json
  {
    "runs": {
      "<run_id>": {
        "reward": 1000,             ← metric name → last synced step
        "policy_loss": 500,
        "__artifacts__": {          ← static artifact paths already synced
          "params/agent_params.json": true
        }
      }
    },
    "last_sync_time": 1713456789.0
  }

Bundle (tar.gz): written to --output, then SCP'd by sync_mlflow_to_server.sh
  delta.json                  ← metrics / params / tags
  artifacts/<run_id>/...      ← artifact files (new or changed)

Artifact sync policy:
  checkpoints/*  — always re-synced (best.pth / last.pth change during training)
  everything else — synced once; path recorded in __artifacts__ to skip next time

Exit codes:
  0 — bundle written successfully, has data to transfer
  1 — configuration error (e.g. experiment name not found on local MLflow)
  2 — no new data since last sync (caller skips SCP)
"""

import argparse
import json
import os
import shutil
import socket
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def parse_args():
    p = argparse.ArgumentParser(
        description="Export MLflow delta (metrics + artifacts) for scheduled sync"
    )
    p.add_argument(
        "--tracking_uri",
        default="http://127.0.0.1:5100",
        help="Local MLflow URI (default: http://127.0.0.1:5100)",
    )
    p.add_argument("--experiment", required=True, help="MLflow experiment name")
    p.add_argument("--output", required=True, help="Path to write delta tar.gz bundle")
    p.add_argument(
        "--state_file",
        default=None,
        help="Override state file path "
        "(default: ~/.nexus/sync_state/{experiment}[__{researcher}].json)",
    )
    p.add_argument(
        "--researcher",
        default=None,
        help="Only export runs whose `researcher` tag matches this value. "
        "Required for safe multi-user setups; without it, every user's "
        "cron exports every other user's runs and the central server "
        "gets duplicate points.",
    )
    return p.parse_args()


def default_state_path(experiment: str, researcher: str | None) -> str:
    """
    Default state file location — under ~/.nexus/ so it survives reboots.
    /tmp is wiped on reboot on most distros, which silently triggered a full
    re-sync on the next cron cycle.

    When --researcher is set, the state file is namespaced so the same
    operator account can host multiple sync identities without their state
    bleeding into each other.
    """
    suffix = f"__{researcher}" if researcher else ""
    return str(Path.home() / ".nexus" / "sync_state" / f"{experiment}{suffix}.json")


def load_state(path: str) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {"runs": {}}


def save_state(path: str, state: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def list_artifacts_recursive(client: MlflowClient, run_id: str, path: str = "") -> list:
    """Return all non-directory artifact paths under path for a run."""
    items = client.list_artifacts(run_id, path)
    result = []
    for item in items:
        if item.is_dir:
            result.extend(list_artifacts_recursive(client, run_id, item.path))
        else:
            result.append(item.path)
    return result


def is_always_sync(artifact_path: str) -> bool:
    """Checkpoints change during training and must be re-synced every cycle."""
    return artifact_path.startswith("checkpoints/")


def main():
    args = parse_args()
    state_path = args.state_file or default_state_path(args.experiment, args.researcher)

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        # Configuration error — distinct from "no new data" (exit 2). The wrapper
        # script only skips SCP on exit 2, so a typo'd experiment name now
        # surfaces as a real failure instead of a silent skip on every cron cycle.
        print(f"[ERROR] Experiment not found on local MLflow: {args.experiment!r}", flush=True)
        try:
            available = sorted(e.name for e in client.search_experiments())
            print(f"[ERROR] Available experiments: {available}", flush=True)
        except Exception as e:
            print(f"[ERROR] (could not list experiments: {e})", flush=True)
        sys.exit(1)

    state = load_state(state_path)
    runs_state = state.get("runs", {})

    # `researcher` filter scopes each user's sync to runs they own. Without it,
    # parallel cron jobs on a shared GPU server re-export each other's runs and
    # the central server logs duplicate metric points at identical steps.
    filter_string = f"tags.researcher = '{args.researcher}'" if args.researcher else ""
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=filter_string, max_results=1000
    )

    delta_runs = []
    new_runs_state = {}
    total_artifact_files = 0

    artifact_tmp = tempfile.mkdtemp()
    try:
        for run in all_runs:
            run_id = run.info.run_id
            run_name = run.data.tags.get("mlflow.runName", run_id)
            run_state = runs_state.get(run_id, {})
            is_new_run = run_id not in runs_state

            # ── Metric state: all keys except the reserved __artifacts__ key
            metric_state = {k: v for k, v in run_state.items() if k != "__artifacts__"}
            new_run_state = dict(metric_state)

            # ── Collect only new metric points for each tag
            delta_metrics = []
            for key in run.data.metrics:
                last_step = metric_state.get(key, -1)
                history = client.get_metric_history(run_id, key)
                new_pts = [
                    {"key": key, "value": m.value, "step": m.step, "timestamp": m.timestamp}
                    for m in history
                    if m.step > last_step
                ]
                if new_pts:
                    delta_metrics.extend(new_pts)
                    new_run_state[key] = max(pt["step"] for pt in new_pts)

            # ── Collect artifacts to sync
            synced_artifacts = run_state.get("__artifacts__", {})
            new_run_state["__artifacts__"] = dict(synced_artifacts)
            try:
                all_artifact_paths = list_artifacts_recursive(client, run_id)
            except Exception as e:
                print(f"[WARN] Could not list artifacts for run {run_id}: {e}", flush=True)
                all_artifact_paths = []

            artifacts_to_sync = [
                p for p in all_artifact_paths if p not in synced_artifacts or is_always_sync(p)
            ]

            run_artifact_dir = os.path.join(artifact_tmp, run_id)
            # MlflowClient.download_artifacts requires dst_path to already exist
            os.makedirs(run_artifact_dir, exist_ok=True)
            downloaded = []
            for apath in artifacts_to_sync:
                try:
                    client.download_artifacts(run_id, apath, run_artifact_dir)
                    downloaded.append(apath)
                    if not is_always_sync(apath):
                        new_run_state["__artifacts__"][apath] = True
                except Exception as e:
                    print(
                        f"[WARN] Could not download artifact {apath!r} for run {run_id}: {e}\n"
                        + traceback.format_exc(),
                        flush=True,
                    )

            total_artifact_files += len(downloaded)
            new_runs_state[run_id] = new_run_state

            if not delta_metrics and not is_new_run and not downloaded:
                continue

            entry = {
                "run_id": run_id,
                "run_name": run_name,
                "status": run.info.status,
                "metrics": delta_metrics,
            }
            # Params and tags only needed on the first sync for a run
            if is_new_run:
                entry["params"] = [{"key": k, "value": v} for k, v in run.data.params.items()]
                entry["tags"] = dict(run.data.tags)

            delta_runs.append(entry)

        total_metrics = sum(len(r["metrics"]) for r in delta_runs)
        total_new_runs = sum(1 for r in delta_runs if "params" in r)

        print(
            f"[INFO] Delta: {len(delta_runs)} run(s) — "
            f"{total_metrics} new metric points, {total_new_runs} new run(s), "
            f"{total_artifact_files} artifact file(s)",
            flush=True,
        )

        # Always persist updated state (marks runs as seen even if no metrics yet)
        state["runs"] = new_runs_state
        state["last_sync_time"] = time.time()
        save_state(state_path, state)

        if total_metrics == 0 and total_new_runs == 0 and total_artifact_files == 0:
            print("[INFO] No new data since last sync.", flush=True)
            sys.exit(2)

        delta = {
            "experiment": args.experiment,
            "sync_time": time.time(),
            "source_host": socket.gethostname(),
            "runs": delta_runs,
        }

        # ── Bundle delta.json + artifact files into a single tar.gz
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as jf:
            json.dump(delta, jf)
            json_tmp = jf.name
        try:
            with tarfile.open(args.output, "w:gz") as tar:
                tar.add(json_tmp, arcname="delta.json")
                for run_id_dir in os.listdir(artifact_tmp):
                    run_dir = os.path.join(artifact_tmp, run_id_dir)
                    if os.path.isdir(run_dir) and os.listdir(run_dir):
                        tar.add(run_dir, arcname=f"artifacts/{run_id_dir}")
        finally:
            os.unlink(json_tmp)

        print(f"[OK] Bundle written to: {args.output}", flush=True)

    finally:
        shutil.rmtree(artifact_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

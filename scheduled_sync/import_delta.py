#!/usr/bin/env python3
"""
import_delta.py  [Run on: MLflow Server]
==========================================
Imports a delta bundle (tar.gz produced by export_delta.py) into central MLflow.
Called via SSH by sync_mlflow_to_server.sh after each delta transfer.

Bundle layout:
  delta.json                  ← metrics / params / tags
  artifacts/<run_id>/...      ← artifact files (new or changed)

For each run in the delta:
  - Creates the MLflow run if it does not yet exist (using run_name as key)
  - Logs params and tags only on first appearance
  - Uploads new metric points via log_batch() in chunks of 1000
  - Uploads artifact files into the run's artifact store
  - Stamps `nexus.lastSyncTime` (UTC ISO) and `nexus.syncedFromHost` so the
    central UI can flag stale GPU servers without an external monitor

Usage:
    python import_delta.py \
        --delta_file  /data/mlflow_delta_inbox/delta_user_20240419_143000_1234.tar.gz \
        --tracking_uri http://127.0.0.1:5000
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import tarfile
import tempfile

import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

BATCH_SIZE = 1000


def parse_args():
    p = argparse.ArgumentParser(description="Import MLflow delta bundle into central MLflow")
    p.add_argument(
        "--delta_file", required=True, help="Path to delta tar.gz produced by export_delta.py"
    )
    p.add_argument(
        "--tracking_uri",
        default="http://127.0.0.1:5000",
        help="Central MLflow URI (default: http://127.0.0.1:5000)",
    )
    return p.parse_args()


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_or_create_run(client: MlflowClient, experiment_id: str, run_name: str, tags: dict) -> str:
    """
    Find an existing run by run_name or create a new one.
    run_name is the stable identifier across incremental delta cycles.
    """
    existing = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if existing:
        run_id = existing[0].info.run_id
        if existing[0].info.status != "RUNNING":
            client.update_run(run_id, status="RUNNING")
        return run_id

    run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={**(tags or {}), "mlflow.runName": run_name},
    )
    return run.info.run_id


def main():
    args = parse_args()

    extract_dir = tempfile.mkdtemp()
    try:
        # ── Extract bundle — supports both tar.gz (new) and plain .json (legacy)
        if tarfile.is_tarfile(args.delta_file):
            with tarfile.open(args.delta_file, "r:gz") as tar:
                tar.extractall(extract_dir)
            json_path = os.path.join(extract_dir, "delta.json")
            artifacts_root = os.path.join(extract_dir, "artifacts")
        else:
            # Legacy plain-JSON delta (no artifacts)
            json_path = args.delta_file
            artifacts_root = None

        with open(json_path) as f:
            delta = json.load(f)

        experiment_name = delta["experiment"]
        runs = delta.get("runs", [])
        source_host = delta.get("source_host", "unknown")
        sync_time_iso = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

        if not runs:
            print("[INFO] Empty delta — nothing to import.", flush=True)
            sys.exit(0)

        mlflow.set_tracking_uri(args.tracking_uri)
        client = MlflowClient(tracking_uri=args.tracking_uri)

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            client.create_experiment(experiment_name)
            experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

        total_metrics = 0
        total_artifacts = 0

        for run_data in runs:
            run_name = run_data["run_name"]
            tags = run_data.get("tags", {})
            params_raw = run_data.get("params", [])
            metrics_raw = run_data.get("metrics", [])
            source_run_id = run_data["run_id"]

            run_id = get_or_create_run(client, experiment_id, run_name, tags)

            # Params only present for new runs (first sync)
            if params_raw:
                param_entities = [Param(key=p["key"], value=str(p["value"])) for p in params_raw]
                for batch in chunks(param_entities, 100):
                    client.log_batch(run_id=run_id, params=batch)

            if metrics_raw:
                metric_entities = [
                    Metric(
                        key=m["key"],
                        value=float(m["value"]),
                        timestamp=int(m["timestamp"]),
                        step=int(m["step"]),
                    )
                    for m in metrics_raw
                ]
                for batch in chunks(metric_entities, BATCH_SIZE):
                    client.log_batch(run_id=run_id, metrics=batch)
                total_metrics += len(metric_entities)

            # ── Upload artifact files for this run
            run_artifacts = 0
            if artifacts_root:
                run_artifact_dir = os.path.join(artifacts_root, source_run_id)
                if os.path.isdir(run_artifact_dir):
                    for dirpath, _dirnames, filenames in os.walk(run_artifact_dir):
                        for fname in filenames:
                            local_path = os.path.join(dirpath, fname)
                            # Preserve subdirectory structure relative to the run's artifact root
                            rel = os.path.relpath(local_path, run_artifact_dir)
                            artifact_subdir = os.path.dirname(rel) or None
                            try:
                                client.log_artifact(run_id, local_path, artifact_subdir)
                                run_artifacts += 1
                                total_artifacts += 1
                            except Exception as e:
                                print(
                                    f"  [WARN] Could not upload artifact {rel!r} "
                                    f"for run {run_name}: {e}",
                                    flush=True,
                                )

            # Sync metadata — refreshed every cycle so the central UI can show
            # "last seen N minutes ago" per run, and operators can spot a GPU
            # server that has stopped syncing without SSHing into each node.
            client.set_tag(run_id, "nexus.lastSyncTime", sync_time_iso)
            client.set_tag(run_id, "nexus.syncedFromHost", source_host)

            print(
                f"  [OK] {run_name}: {len(metrics_raw)} metric points, "
                f"{run_artifacts} artifact file(s)",
                flush=True,
            )

        print(
            f"[DONE] Imported: {total_metrics} metric points, {total_artifacts} artifact file(s).",
            flush=True,
        )

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

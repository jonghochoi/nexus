#!/usr/bin/env python3
"""
import_delta.py  [Run on: MLflow Server]
==========================================
Imports a delta JSON file (produced by export_delta.py) into central MLflow.
Called via SSH by sync_mlflow_to_server.sh after each delta transfer.

For each run in the delta:
  - Creates the MLflow run if it does not yet exist (using run_name as key)
  - Logs params and tags only on first appearance
  - Uploads new metric points via log_batch() in chunks of 1000

Usage:
    python import_delta.py \
        --delta_file  /data/mlflow_delta_inbox/delta_20240419_143000.json \
        --tracking_uri http://127.0.0.1:5000
"""

import argparse
import json
import sys

from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

BATCH_SIZE = 1000


def parse_args():
    p = argparse.ArgumentParser(
        description="Import MLflow delta JSON into central MLflow"
    )
    p.add_argument("--delta_file",    required=True,
                   help="Path to delta JSON produced by export_delta.py")
    p.add_argument("--tracking_uri",  default="http://127.0.0.1:5000",
                   help="Central MLflow URI (default: http://127.0.0.1:5000)")
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

    with open(args.delta_file) as f:
        delta = json.load(f)

    experiment_name = delta["experiment"]
    runs            = delta.get("runs", [])

    if not runs:
        print("[INFO] Empty delta — nothing to import.", flush=True)
        sys.exit(0)

    client = MlflowClient(tracking_uri=args.tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name)
        experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    total_uploaded = 0

    for run_data in runs:
        run_name    = run_data["run_name"]
        tags        = run_data.get("tags", {})
        params_raw  = run_data.get("params", [])
        metrics_raw = run_data.get("metrics", [])

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
            total_uploaded += len(metric_entities)

        print(f"  [OK] {run_name}: {len(metrics_raw)} metric points", flush=True)

    print(f"[DONE] Total imported: {total_uploaded} metric points.", flush=True)


if __name__ == "__main__":
    main()

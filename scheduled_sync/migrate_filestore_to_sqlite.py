#!/usr/bin/env python3
"""
scheduled_sync/migrate_filestore_to_sqlite.py  [Run on: MLflow Server]
========================================================================
Migrates all experiments and runs from a legacy file-based MLflow store
(mlruns/ directory tree) into a SQLite-backed MLflow server.

The source is read directly via the `file://` tracking URI — no separate
server process needed for the old data. The destination must be a running
MLflow server with a SQLite backend.

Migration is idempotent: `get_or_create_run` keyed on `run_name` means
re-running the script skips already-migrated runs (metrics are deduplicated
by step). Safe to run while the destination server is live.

Artifacts are optional — pass --no_artifacts to skip them (useful for a
first quick pass to check metric counts before committing disk I/O).

Usage:
    python scheduled_sync/migrate_filestore_to_sqlite.py \\
        --src_store file:///opt/nexus-mlflow/mlruns \\
        --dst_uri   http://127.0.0.1:5000

    # Migrate only specific experiments
    python scheduled_sync/migrate_filestore_to_sqlite.py \\
        --src_store file:///opt/nexus-mlflow/mlruns \\
        --dst_uri   http://127.0.0.1:5000 \\
        --experiments my_exp_1 my_exp_2

    # Dry run — print plan without writing anything
    python scheduled_sync/migrate_filestore_to_sqlite.py \\
        --src_store file:///opt/nexus-mlflow/mlruns \\
        --dst_uri   http://127.0.0.1:5000 \\
        --dry_run
"""

import argparse
import sys
import tempfile
import traceback

import mlflow
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

BATCH_SIZE = 1000  # MLflow hard limit per log_batch() call
PARAM_BATCH = 100  # conservative — params are strings, not typed


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Migrate MLflow file store → SQLite-backed server")
    p.add_argument(
        "--src_store",
        required=True,
        help=(
            "Source MLflow tracking URI. Use file:///absolute/path/to/mlruns "
            "to read the file store directly, or http://host:port for a running server."
        ),
    )
    p.add_argument(
        "--dst_uri",
        required=True,
        help="Destination MLflow server URI (e.g. http://127.0.0.1:5000)",
    )
    p.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Experiment names to migrate. Defaults to all experiments.",
    )
    p.add_argument(
        "--no_artifacts",
        action="store_true",
        help="Skip artifact migration (metrics, params, tags only).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print migration plan without writing anything to the destination.",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_or_create_experiment(dst: MlflowClient, name: str, dry_run: bool) -> str | None:
    if dry_run:
        return None
    exp = dst.get_experiment_by_name(name)
    if exp is None:
        exp_id = dst.create_experiment(name)
        return exp_id
    return exp.experiment_id


def get_or_create_run(
    dst: MlflowClient, experiment_id: str, run_name: str, tags: dict, dry_run: bool
) -> str | None:
    if dry_run:
        return None
    existing = dst.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if existing:
        run_id = existing[0].info.run_id
        if existing[0].info.status != "RUNNING":
            dst.update_run(run_id, status="RUNNING")
        return run_id
    run = dst.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={**(tags or {}), "mlflow.runName": run_name},
    )
    return run.info.run_id


def finalize_run(dst: MlflowClient, run_id: str, src_status: str, dry_run: bool) -> None:
    """Restore the original run status after migrating all data."""
    if dry_run or src_status == "RUNNING":
        return
    dst.update_run(run_id, status=src_status)


# ── Per-run migration ─────────────────────────────────────────────────────────


def migrate_run(
    src: MlflowClient,
    dst: MlflowClient,
    src_run,
    dst_experiment_id: str,
    no_artifacts: bool,
    dry_run: bool,
) -> dict:
    """Migrate a single run. Returns a summary dict."""
    run_id = src_run.info.run_id
    run_name = src_run.data.tags.get("mlflow.runName", run_id)
    src_status = src_run.info.status

    # ── Params and tags
    params = src_run.data.params
    tags = {k: v for k, v in src_run.data.tags.items() if not k.startswith("mlflow.")}
    # Preserve a small set of mlflow.* system tags that carry user intent
    for keep in ("mlflow.source.name", "mlflow.source.type", "mlflow.user"):
        if keep in src_run.data.tags:
            tags[keep] = src_run.data.tags[keep]

    # ── Collect full metric history for every key
    all_metrics = []
    for key in src_run.data.metrics:
        try:
            history = src.get_metric_history(run_id, key)
            all_metrics.extend(
                Metric(key=m.key, value=m.value, timestamp=m.timestamp, step=m.step)
                for m in history
            )
        except Exception as e:
            print(f"    [WARN] Could not fetch metric history for {key!r}: {e}", flush=True)

    summary = {
        "run_name": run_name,
        "params": len(params),
        "tags": len(tags),
        "metrics": len(all_metrics),
        "artifacts": 0,
        "skipped": False,
    }

    if dry_run:
        return summary

    dst_run_id = get_or_create_run(dst, dst_experiment_id, run_name, tags, dry_run=False)

    # Params
    if params:
        param_entities = [Param(key=k, value=str(v)) for k, v in params.items()]
        for batch in chunks(param_entities, PARAM_BATCH):
            try:
                dst.log_batch(run_id=dst_run_id, params=batch)
            except Exception as e:
                print(f"    [WARN] Param batch failed: {e}", flush=True)

    # Tags (set individually — log_batch tags have a 5-tag limit per call)
    for k, v in tags.items():
        try:
            dst.set_tag(dst_run_id, k, v)
        except Exception as e:
            print(f"    [WARN] Tag {k!r} failed: {e}", flush=True)

    # Metrics
    for batch in chunks(all_metrics, BATCH_SIZE):
        try:
            dst.log_batch(run_id=dst_run_id, metrics=batch)
        except Exception as e:
            print(f"    [WARN] Metric batch failed: {e}", flush=True)

    # ── Artifacts
    if not no_artifacts:
        try:
            artifact_paths = _list_artifacts_recursive(src, run_id)
        except Exception as e:
            print(f"    [WARN] Could not list artifacts: {e}", flush=True)
            artifact_paths = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for apath in artifact_paths:
                try:
                    local = src.download_artifacts(run_id, apath, tmpdir)
                    artifact_subdir = apath.rsplit("/", 1)[0] if "/" in apath else None
                    dst.log_artifact(dst_run_id, local, artifact_subdir)
                    summary["artifacts"] += 1
                except Exception as e:
                    print(f"    [WARN] Artifact {apath!r} failed: {e}", flush=True)

    finalize_run(dst, dst_run_id, src_status, dry_run=False)
    return summary


def _list_artifacts_recursive(client: MlflowClient, run_id: str, path: str = "") -> list:
    items = client.list_artifacts(run_id, path)
    result = []
    for item in items:
        if item.is_dir:
            result.extend(_list_artifacts_recursive(client, run_id, item.path))
        else:
            result.append(item.path)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    src = MlflowClient(tracking_uri=args.src_store)
    dst = MlflowClient(tracking_uri=args.dst_uri) if not args.dry_run else None

    if args.dry_run:
        print("[DRY RUN] No data will be written to the destination.", flush=True)

    # Discover experiments from source
    try:
        all_experiments = src.search_experiments()
    except Exception as e:
        print(f"[ERROR] Could not connect to source store: {e}", flush=True)
        sys.exit(1)

    # Filter by name if requested — fail fast on typos
    if args.experiments:
        name_set = set(args.experiments)
        found = {e.name for e in all_experiments}
        missing = name_set - found
        if missing:
            print(f"[ERROR] Experiments not found in source: {sorted(missing)}", flush=True)
            print(f"[ERROR] Available: {sorted(found)}", flush=True)
            sys.exit(1)
        all_experiments = [e for e in all_experiments if e.name in name_set]

    # Exclude the MLflow default experiment (id=0) — it is auto-created on every server
    all_experiments = [e for e in all_experiments if e.experiment_id != "0"]

    print(
        f"[INFO] Migrating {len(all_experiments)} experiment(s) from {args.src_store} "
        f"→ {args.dst_uri}",
        flush=True,
    )

    total_runs = total_metrics = total_artifacts = 0
    errors = 0

    for exp in all_experiments:
        print(f"\n── Experiment: {exp.name!r} ──────────────────────────────────", flush=True)

        dst_exp_id = get_or_create_experiment(dst, exp.name, args.dry_run)

        try:
            runs = src.search_runs(
                experiment_ids=[exp.experiment_id], filter_string="", max_results=50_000
            )
        except Exception as e:
            print(f"  [ERROR] Could not list runs: {e}", flush=True)
            errors += 1
            continue

        print(f"  {len(runs)} run(s) found", flush=True)

        for src_run in runs:
            run_name = src_run.data.tags.get("mlflow.runName", src_run.info.run_id)
            try:
                summary = migrate_run(
                    src=src,
                    dst=dst,
                    src_run=src_run,
                    dst_experiment_id=dst_exp_id,
                    no_artifacts=args.no_artifacts,
                    dry_run=args.dry_run,
                )
                total_runs += 1
                total_metrics += summary["metrics"]
                total_artifacts += summary["artifacts"]
                print(
                    f"  [OK] {summary['run_name']}: "
                    f"{summary['metrics']} metric pts, "
                    f"{summary['params']} params, "
                    f"{summary['artifacts']} artifacts",
                    flush=True,
                )
            except Exception as e:
                print(f"  [ERROR] Run {run_name!r}: {e}", flush=True)
                traceback.print_exc()
                errors += 1

    # ── Summary
    print("\n" + "─" * 60, flush=True)
    label = "[DRY RUN COMPLETE]" if args.dry_run else "[DONE]"
    print(
        f"{label} {total_runs} runs, {total_metrics} metric points, "
        f"{total_artifacts} artifact file(s) migrated.",
        flush=True,
    )
    if errors:
        print(f"[WARN] {errors} error(s) encountered — check output above.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

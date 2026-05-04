"""
nexus/logger/sweep_logger.py
============================
SweepLogger: creates and manages a parent MLflow run for hyperparameter sweeps.

Pass `parent_run_id` to MLflowLogger or make_logger() so child runs appear
under the sweep in the MLflow UI tree.
"""

from __future__ import annotations

import time
from typing import Optional

import mlflow.entities
from mlflow.tracking import MlflowClient


class SweepLogger:
    """Creates and manages a parent run for a hyperparameter sweep.

    Prefer the context manager form — it marks the run FAILED automatically
    if an exception escapes, and FINISHED on clean exit:

        with SweepLogger("ppo_lr_sweep", tracking_uri=..., experiment_name=...) as sweep:
            for trial_cfg in trials:
                logger = make_logger(..., parent_run_id=sweep.parent_run_id, ...)
                run_trial(trial_cfg, logger)
                logger.close()
            sweep.log_summary(best_params={"lr": 3e-4}, best_metrics={"reward": 95.2})

    If you manage the lifecycle manually, call close() in a finally block:

        sweep = SweepLogger(...)
        try:
            ...
            sweep.log_summary(...)
        finally:
            sweep.close()
    """

    def __init__(
        self,
        sweep_name: str,
        tracking_uri: str,
        experiment_name: str,
        sweep_params: Optional[dict] = None,
        tags: Optional[dict] = None,
    ):
        self._sweep_name = sweep_name
        self._closed = False

        # Use MlflowClient directly — avoids polluting the process-global
        # mlflow tracking URI and experiment set by mlflow.set_tracking_uri /
        # mlflow.set_experiment, which would silently affect any concurrent
        # MLflowLogger instances running in the same process.
        self._client = MlflowClient(tracking_uri=tracking_uri)

        exp = self._client.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = self._client.create_experiment(experiment_name)
        else:
            exp_id = exp.experiment_id

        run_tags = {"mlflow.runName": sweep_name, "nexus.sweep": "true", **(tags or {})}
        run = self._client.create_run(experiment_id=exp_id, run_name=sweep_name, tags=run_tags)
        self._run_id = run.info.run_id

        if sweep_params:
            items = [mlflow.entities.Param(k, str(v)) for k, v in sweep_params.items()]
            for i in range(0, len(items), 100):
                self._client.log_batch(run_id=self._run_id, params=items[i : i + 100])

        print(
            f"[SweepLogger] Sweep run created.\n"
            f"  Run ID   : {self._run_id}\n"
            f"  Sweep    : {sweep_name}\n"
            f"  MLflow   : {tracking_uri}"
        )

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> SweepLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.close(status=status)
        return False  # re-raise any exception

    def __del__(self) -> None:
        # Safety net — runs FINISHED if the caller forgot close() entirely.
        if not self._closed:
            try:
                self._client.set_terminated(self._run_id, status="FINISHED")
            except Exception:
                pass

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def parent_run_id(self) -> str:
        """Run ID to pass to child trials as parent_run_id."""
        return self._run_id

    def log_summary(self, best_params: dict, best_metrics: dict) -> None:
        """Record best trial results on the parent run after sweep completes."""
        ts = int(time.time() * 1000)
        metrics = [
            mlflow.entities.Metric(key=f"best/{k}", value=float(v), timestamp=ts, step=0)
            for k, v in best_metrics.items()
        ]
        if metrics:
            self._client.log_batch(run_id=self._run_id, metrics=metrics)

        params = [mlflow.entities.Param(f"best/{k}", str(v)) for k, v in best_params.items()]
        if params:
            self._client.log_batch(run_id=self._run_id, params=params)

    def close(self, status: str = "FINISHED") -> None:
        """Finalize the sweep run.

        Parameters
        ----------
        status : "FINISHED" | "FAILED" | "KILLED"
            MLflow run status to set on the parent run.  The context manager
            passes "FAILED" automatically when an exception escapes the with block.
        """
        if self._closed:
            return
        self._closed = True
        self._client.set_terminated(self._run_id, status=status)
        print(f"[SweepLogger] Sweep finalized ({status}): {self._run_id}")

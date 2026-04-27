"""
logger/mlflow_logger.py
=======================
Drop-in replacement for tensorboardX.SummaryWriter that logs to a local MLflow server.

Key behaviors:
  - Buffers all add_scalar() calls within one training step
  - Flushes as a single log_batch() when the step advances
  - Creates or resumes an MLflow run by run_name (crash-safe)
  - Logs hyperparameters once at run start
  - Logs env_cfg and reward_fn as artifacts at run start
  - Captures git_commit / git_dirty tags at run start (track_git=True by default)
  - Uploads git diff HEAD as artifacts/git/git_patch.diff when working tree is dirty
  - Marks run FINISHED on close() or process exit
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import time
from collections import defaultdict
from typing import Any, Optional

import mlflow
import mlflow.entities
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from .git_utils import get_git_info, get_git_patch

_BATCH_SIZE = 1000   # MLflow hard limit per log_batch() call


class MLflowLogger:
    """
    SummaryWriter-compatible logger that writes to a local MLflow server.

    All add_scalar() calls within the same global_step are buffered and
    flushed together as one HTTP request when the step value changes.
    This keeps HTTP overhead minimal even with many metrics per epoch.
    """

    def __init__(
        self,
        run_name: str,
        tracking_uri: str = "http://127.0.0.1:5100",
        experiment_name: str = "robot_hand_rl",
        params: Optional[dict] = None,
        tags: Optional[dict] = None,
        env_cfg_path: Optional[str] = None,
        reward_fn_path: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        track_git: bool = True,  # set False if not inside a git repo or to suppress git tags
    ):
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self._parent_run_id = parent_run_id

        self._buffer: dict[int, dict[str, float]] = defaultdict(dict)
        self._last_step: int = -1
        self._closed = False

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._client = MlflowClient(tracking_uri=tracking_uri)

        merged_tags = dict(tags or {})
        if track_git:
            merged_tags.update(get_git_info())
        self._run_id = self._get_or_create_run(experiment_name, merged_tags)
        self._track_git = track_git

        if params:
            self._log_params(params)

        self._log_run_artifacts(env_cfg_path, reward_fn_path)

        if track_git:
            self._log_git_patch()

        atexit.register(self.close)

        print(
            f"[MLflowLogger] Run started.\n"
            f"  Run ID     : {self._run_id}\n"
            f"  Experiment : {experiment_name}\n"
            f"  MLflow URI : {tracking_uri}"
        )

    # ── Public interface (SummaryWriter-compatible) ──────────────────────────

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if self._closed:
            return
        if global_step != self._last_step and self._last_step != -1:
            self._flush(self._last_step)
        self._buffer[global_step][self._sanitize(tag)] = float(scalar_value)
        self._last_step = global_step

    def add_histogram(self, *args, **kwargs) -> None:
        pass  # Not supported in MLflow

    def add_image(self, *args, **kwargs) -> None:
        pass  # Not supported

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        if not os.path.exists(local_path):
            print(f"[MLflowLogger] Skipping artifact (not found): {local_path}")
            return
        self._client.log_artifact(self._run_id, local_path, artifact_path)

    def log_checkpoint(self, local_path: str, kind: str) -> None:
        """Upload a checkpoint as checkpoints/best.pth or checkpoints/last.pth.

        kind must be 'best' or 'last'. The source file may have any name;
        it is renamed on upload so only one file per kind is kept.
        """
        if kind not in ("best", "last"):
            raise ValueError(f"kind must be 'best' or 'last', got: {kind!r}")
        if not os.path.exists(local_path):
            print(f"[MLflowLogger] Skipping checkpoint (not found): {local_path}")
            return
        ext = os.path.splitext(local_path)[1] or ".pth"
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, f"{kind}{ext}")
            shutil.copy2(local_path, dst)
            self._client.log_artifact(self._run_id, dst, "checkpoints")

    def register_checkpoint(
        self,
        model_name: str,
        kind: str = "best",
        description: Optional[str] = None,
    ) -> str:
        """Register a checkpoint artifact in the MLflow Model Registry.

        Returns the registered model version string.
        kind must be 'best' or 'last'.
        """
        if kind not in ("best", "last"):
            raise ValueError(f"kind must be 'best' or 'last', got: {kind!r}")
        artifact_uri = f"runs:/{self._run_id}/checkpoints/{kind}.pth"
        mv = mlflow.register_model(artifact_uri, model_name)
        if description:
            self._client.update_model_version(model_name, mv.version, description=description)
        return mv.version

    def promote_model(self, model_name: str, version: str, stage: str) -> None:
        """Transition a model version to a new stage.

        stage: 'Staging' | 'Production' | 'Archived'
        """
        self._client.transition_model_version_stage(model_name, version, stage)

    def log_rl_metrics(
        self,
        step: int,
        *,
        explained_variance: Optional[float] = None,
        approx_kl: Optional[float] = None,
        clip_fraction: Optional[float] = None,
        grad_norm: Optional[float] = None,
        entropy: Optional[float] = None,
        success_rate: Optional[float] = None,
    ) -> None:
        """Log RL diagnostic metrics under the 'rl/' namespace. Skips None values."""
        mapping = {
            "rl/explained_variance": explained_variance,
            "rl/approx_kl": approx_kl,
            "rl/clip_fraction": clip_fraction,
            "rl/grad_norm": grad_norm,
            "rl/entropy": entropy,
            "rl/success_rate": success_rate,
        }
        for tag, value in mapping.items():
            if value is not None:
                self.add_scalar(tag, value, step)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for step in sorted(self._buffer.keys()):
            self._flush(step)
        self._client.set_terminated(self._run_id, status="FINISHED")
        print(f"[MLflowLogger] Run finalized: {self._run_id}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _flush(self, step: int) -> None:
        if step not in self._buffer or not self._buffer[step]:
            return
        ts = int(time.time() * 1000)
        metrics = [
            Metric(key=tag, value=val, timestamp=ts, step=step)
            for tag, val in self._buffer[step].items()
        ]
        for i in range(0, len(metrics), _BATCH_SIZE):
            self._client.log_batch(run_id=self._run_id, metrics=metrics[i:i+_BATCH_SIZE])
        del self._buffer[step]

    def _get_or_create_run(self, experiment_name: str, tags: dict) -> str:
        exp = mlflow.get_experiment_by_name(experiment_name)
        existing = self._client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{self.run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        base_tags = {"source": "mlflow_direct", "mlflow.runName": self.run_name, **tags}
        if self._parent_run_id:
            base_tags["mlflow.parentRunId"] = self._parent_run_id
        if existing:
            run_id = existing[0].info.run_id
            self._client.update_run(run_id, status="RUNNING")
            self._client.set_tags(run_id, base_tags)
            print(f"[MLflowLogger] Resuming run: {run_id}")
            return run_id
        run = self._client.create_run(
            experiment_id=exp.experiment_id,
            run_name=self.run_name,
            tags=base_tags,
        )
        return run.info.run_id

    def _log_run_artifacts(
        self,
        env_cfg_path: Optional[str],
        reward_fn_path: Optional[str],
    ) -> None:
        for path in (env_cfg_path, reward_fn_path):
            if path:
                self.log_artifact(path, artifact_path="configs")

    def _log_git_patch(self) -> None:
        """Upload git diff HEAD as artifacts/git/git_patch.diff when the tree is dirty."""
        patch = get_git_patch()
        if not patch:
            return
        with tempfile.TemporaryDirectory() as tmp:
            patch_path = os.path.join(tmp, "git_patch.diff")
            with open(patch_path, "w") as f:
                f.write(patch)
            self._client.log_artifact(self._run_id, patch_path, "git")
        print("[MLflowLogger] Dirty working tree detected — git patch saved to artifacts/git/git_patch.diff")

    def _log_params(self, params: dict) -> None:
        flat = self._flatten(params)
        items = [mlflow.entities.Param(k, str(v)) for k, v in flat.items()]
        for i in range(0, len(items), 100):
            self._client.log_batch(run_id=self._run_id, params=items[i:i+100])

    @staticmethod
    def _flatten(d: dict, parent: str = "", sep: str = ".") -> dict:
        out: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{parent}{sep}{k}" if parent else k
            # Accept plain dicts and any dict-like object (e.g. OmegaConf DictConfig)
            if isinstance(v, dict) or (hasattr(v, "items") and not isinstance(v, str)):
                out.update(MLflowLogger._flatten(v, key, sep))
            else:
                out[key] = v
        return out

    @staticmethod
    def _sanitize(name: str) -> str:
        return name.replace(" ", "_").replace(":", "-")

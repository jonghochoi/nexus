"""
nexus/logger/dual_logger.py
===========================
DualLogger: writes to TensorBoard AND MLflow simultaneously.

This is the recommended logger for the transition period.
Once the team is comfortable with MLflow, switch to MLflowLogger alone.

Usage in PPO.__init__():

    from nexus.logger import make_logger

    self.writer = make_logger(
        mode="dual",           # "dual" | "mlflow" | "tensorboard"
        tb_dir=output_dir,     # TensorBoard log directory (omit when mode="mlflow")
        run_name=run_name,
        tracking_uri="http://127.0.0.1:5100",
        experiment_name="robot_hand_rl",
        agent_params=agent_cfg,
        env_params=env_cfg,
        tags={...},
    )

No changes needed anywhere else in PPO.
"""

from __future__ import annotations
from typing import Optional

from .tb_logger import TBLogger
from .mlflow_logger import MLflowLogger


class DualLogger:
    """
    Forwards every logging call to both TBLogger and MLflowLogger.

    TensorBoard  → local tfevents files (existing workflow preserved)
    MLflow       → local MLflow server  (new centralized workflow)

    Both are written in the same training loop with zero extra code in PPO.
    """

    def __init__(
        self,
        tb_dir: str,
        run_name: str,
        tracking_uri: str = "http://127.0.0.1:5100",
        experiment_name: str = "robot_hand_rl",
        params: Optional[dict] = None,
        agent_params: Optional[dict] = None,
        env_params: Optional[dict] = None,
        tags: Optional[dict] = None,
        parent_run_id: Optional[str] = None,
    ):
        self._tb = TBLogger(log_dir=tb_dir)
        self._mlflow = MLflowLogger(
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            params=params,
            agent_params=agent_params,
            env_params=env_params,
            tags=tags,
            parent_run_id=parent_run_id,
        )
        print("[DualLogger] Active: TensorBoard + MLflow")

    # ── Public interface (SummaryWriter-compatible) ──────────────────────────

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self._tb.add_scalar(tag, scalar_value, global_step)
        self._mlflow.add_scalar(tag, scalar_value, global_step)

    def add_histogram(self, tag: str, values, global_step: int) -> None:
        self._tb.add_histogram(tag, values, global_step)
        # MLflow does not support histograms — skipped silently

    def add_image(self, tag: str, img_tensor, global_step: int) -> None:
        self._tb.add_image(tag, img_tensor, global_step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        self._mlflow.log_artifact(local_path, artifact_path)
        # TensorBoard does not support artifacts — skipped silently

    def log_checkpoint(self, local_path: str, kind: str) -> None:
        self._mlflow.log_checkpoint(local_path, kind)
        # TensorBoard does not support checkpoints — skipped silently

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
        self._tb.log_rl_metrics(
            step,
            explained_variance=explained_variance,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
            grad_norm=grad_norm,
            entropy=entropy,
            success_rate=success_rate,
        )
        self._mlflow.log_rl_metrics(
            step,
            explained_variance=explained_variance,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
            grad_norm=grad_norm,
            entropy=entropy,
            success_rate=success_rate,
        )

    def close(self) -> None:
        self._tb.close()
        self._mlflow.close()


# ── Factory function ─────────────────────────────────────────────────────────


def make_logger(
    mode: str,
    run_name: str,
    tb_dir: Optional[str] = None,
    tracking_uri: str = "http://127.0.0.1:5100",
    experiment_name: str = "robot_hand_rl",
    params: Optional[dict] = None,
    agent_params: Optional[dict] = None,
    env_params: Optional[dict] = None,
    tags: Optional[dict] = None,
    parent_run_id: Optional[str] = None,
):
    """
    Factory that returns the right logger based on mode.

    mode options:
      "dual"        → TensorBoard + MLflow (recommended for transition)
      "mlflow"      → MLflow only — `tb_dir` is unused and may be omitted
      "tensorboard" → TensorBoard only (legacy, no changes to existing code)

    `tb_dir` is the TensorBoard log directory. It is required for
    `mode="dual"` and `mode="tensorboard"`, and ignored for `mode="mlflow"`.

    `agent_params` and `env_params` are logged as MLflow params with an
    "agent." / "env." prefix respectively, and each is also serialized to
    artifacts/params/agent_params.json and artifacts/params/env_params.json.
    `params` is kept for backward compatibility and is logged without a prefix.

    Example:
        self.writer = make_logger(
            mode="dual",
            tb_dir=...,
            run_name=...,
            agent_params=agent_cfg,
            env_params=env_cfg,
        )
    """
    mode = mode.lower().strip()

    if mode == "dual":
        if tb_dir is None:
            raise ValueError("make_logger(mode='dual') requires tb_dir")
        return DualLogger(
            tb_dir=tb_dir,
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            params=params,
            agent_params=agent_params,
            env_params=env_params,
            tags=tags,
            parent_run_id=parent_run_id,
        )
    elif mode == "mlflow":
        return MLflowLogger(
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            params=params,
            agent_params=agent_params,
            env_params=env_params,
            tags=tags,
            parent_run_id=parent_run_id,
        )
    elif mode == "tensorboard":
        if tb_dir is None:
            raise ValueError("make_logger(mode='tensorboard') requires tb_dir")
        return TBLogger(log_dir=tb_dir)
    else:
        raise ValueError(
            f"Unknown logger mode: '{mode}'. Choose from: 'dual', 'mlflow', 'tensorboard'"
        )

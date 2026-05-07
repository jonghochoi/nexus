"""
nexus/logger/tb_logger.py
=========================
Thin wrapper around tensorboardX.SummaryWriter.
Provides the same interface as MLflowLogger so DualLogger can treat them uniformly.
"""

from __future__ import annotations
from typing import Optional
from tensorboardX import SummaryWriter

from ..brand import log as brand_log


class TBLogger:
    """
    Wraps tensorboardX.SummaryWriter with the same interface as MLflowLogger.
    No behavioral changes — all calls pass through directly to SummaryWriter.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._writer = SummaryWriter(log_dir=log_dir)
        print(brand_log(f"TBLogger writing to {log_dir}", "info"))

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self._writer.add_scalar(tag, scalar_value, global_step)

    def add_histogram(self, tag: str, values, global_step: int) -> None:
        self._writer.add_histogram(tag, values, global_step)

    def add_image(self, tag: str, img_tensor, global_step: int) -> None:
        self._writer.add_image(tag, img_tensor, global_step)

    def set_tag(self, key: str, value: str) -> None:
        pass  # TensorBoard does not support tags

    def log_metrics_now(self, metrics: dict, step: int) -> None:
        pass  # TensorBoard has no direct-flush path

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass  # TensorBoard does not support artifacts

    def log_checkpoint(self, local_path: str, kind: str) -> None:
        pass  # TensorBoard does not support checkpoint artifacts

    # ── Model Registry surface (MLflow-only) ─────────────────────────────────
    # Stubs raise loudly rather than silently no-op'ing — register_checkpoint
    # returns a model version that promote_model consumes, so a fake "" return
    # would silently break the next call. If you hit this in tensorboard mode,
    # switch to mode="dual" or mode="mlflow".

    def register_checkpoint(
        self, model_name: str, kind: str = "best", description: Optional[str] = None
    ) -> str:
        raise NotImplementedError(
            "register_checkpoint requires MLflow — use mode='dual' or mode='mlflow'"
        )

    def promote_model(self, model_name: str, version: str, stage: str) -> None:
        raise NotImplementedError(
            "promote_model requires MLflow — use mode='dual' or mode='mlflow'"
        )

    def close(self) -> None:
        self._writer.close()

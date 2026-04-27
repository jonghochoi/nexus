"""
nexus/logger/tb_logger.py
=========================
Thin wrapper around tensorboardX.SummaryWriter.
Provides the same interface as MLflowLogger so DualLogger can treat them uniformly.
"""

from __future__ import annotations
from typing import Optional
from tensorboardX import SummaryWriter


class TBLogger:
    """
    Wraps tensorboardX.SummaryWriter with the same interface as MLflowLogger.
    No behavioral changes — all calls pass through directly to SummaryWriter.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._writer = SummaryWriter(log_dir=log_dir)
        print(f"[TBLogger] Writing to: {log_dir}")

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self._writer.add_scalar(tag, scalar_value, global_step)

    def add_histogram(self, tag: str, values, global_step: int) -> None:
        self._writer.add_histogram(tag, values, global_step)

    def add_image(self, tag: str, img_tensor, global_step: int) -> None:
        self._writer.add_image(tag, img_tensor, global_step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass  # TensorBoard does not support artifacts

    def log_checkpoint(self, local_path: str, kind: str) -> None:
        pass  # TensorBoard does not support checkpoint artifacts

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
        """Log RL diagnostic metrics under the 'rl/' namespace."""
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
        self._writer.close()

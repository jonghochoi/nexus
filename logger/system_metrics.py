"""
logger/system_metrics.py
=========================
SystemMetricsLogger: background thread that periodically logs system resource metrics.

Requires optional dependencies: psutil (CPU/RAM), pynvml or nvidia-smi (GPU).
Silently skips any metric that cannot be collected.
"""

from __future__ import annotations

import subprocess
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .mlflow_logger import MLflowLogger


def _get_gpu_memory_mb() -> Optional[float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


class SystemMetricsLogger:
    """Background thread that logs CPU/RAM/GPU metrics at a fixed time interval.

    Usage:
        sys_logger = SystemMetricsLogger(mlflow_logger, interval_seconds=30)
        sys_logger.start()
        # ... training loop ...
        sys_logger.stop()

    Metrics logged (when available):
        system/cpu_percent, system/ram_gb, system/gpu_memory_mb
    """

    def __init__(
        self,
        mlflow_logger: "MLflowLogger",
        interval_seconds: float = 30.0,
    ):
        self._logger = mlflow_logger
        self._interval = interval_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background logging thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background logging thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _collect(self) -> dict:
        metrics: dict[str, float] = {}
        try:
            import psutil
            metrics["system/cpu_percent"] = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            metrics["system/ram_gb"] = ram.used / 1024 / 1024 / 1024
        except ImportError:
            pass
        gpu_mb = _get_gpu_memory_mb()
        if gpu_mb is not None:
            metrics["system/gpu_memory_mb"] = gpu_mb
        return metrics

    def _run(self) -> None:
        step = 0
        while not self._stop_event.is_set():
            metrics = self._collect()
            for tag, value in metrics.items():
                self._logger.add_scalar(tag, value, step)
            step += 1
            self._stop_event.wait(timeout=self._interval)

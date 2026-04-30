"""
nexus/logger/system_metrics.py
==============================
SystemMetricsLogger: background thread that periodically logs system resource metrics.

Requires optional dependencies: psutil (CPU/RAM), pynvml or nvidia-smi (GPU).
Silently skips any metric that cannot be collected.
"""

from __future__ import annotations

import os
import subprocess
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .mlflow_logger import MLflowLogger


def _find_gpu_by_pid() -> Optional[int]:
    """Return the physical GPU index used by the current process, or None if not found.

    Scans all GPUs via pynvml and matches by PID. Returns None if pynvml is
    unavailable or the process has not yet allocated GPU memory.
    """
    pid = os.getpid()
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if any(p.pid == pid for p in procs):
                return i
    except Exception:
        pass
    return None


def _get_gpu_stats(gpu_index: int) -> dict[str, float]:
    """Return memory_mb and util_percent for the given physical GPU index."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "system/gpu_memory_mb": mem.used / 1024 / 1024,
            "system/gpu_util_percent": float(util.gpu),
        }
    except Exception:
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"-i={gpu_index}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) == 2:
                return {
                    "system/gpu_memory_mb": float(parts[0].strip()),
                    "system/gpu_util_percent": float(parts[1].strip()),
                }
    except Exception:
        pass
    return {}


class SystemMetricsLogger:
    """Background thread that logs CPU/RAM/GPU metrics at a fixed time interval.

    Usage:
        sys_logger = SystemMetricsLogger(mlflow_logger, interval_seconds=30)
        sys_logger.start()
        # ... training loop ...
        sys_logger.stop()

    Metrics logged (when available):
        system/cpu_percent      — CPU utilisation (%)
        system/ram_gb           — RAM used (GB)
        system/gpu_memory_mb    — GPU memory used (MB)
        system/gpu_util_percent — GPU compute utilisation (%)

    The active GPU is detected by scanning which physical GPU the current process
    has allocated memory on (via pynvml). Detection is lazy — GPU metrics are
    skipped until the process has actually allocated GPU memory (e.g. after
    model.to(device)). Once detected, the index is cached and written to the
    run tag `system.gpu_index`.
    """

    def __init__(self, mlflow_logger: "MLflowLogger", interval_seconds: float = 30.0):
        self._logger = mlflow_logger
        self._interval = interval_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._gpu_index: Optional[int] = None  # cached after first successful PID scan

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

        # ── CPU / RAM ────────────────────────────────────────────────────────
        try:
            import psutil

            metrics["system/cpu_percent"] = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            metrics["system/ram_gb"] = ram.used / 1024 / 1024 / 1024
        except ImportError:
            pass

        # ── GPU — lazy PID scan, cached once found ───────────────────────────
        if self._gpu_index is None:
            self._gpu_index = _find_gpu_by_pid()
            if self._gpu_index is not None:
                self._logger.set_tag("system.gpu_index", str(self._gpu_index))

        if self._gpu_index is not None:
            metrics.update(_get_gpu_stats(self._gpu_index))

        return metrics

    def _run(self) -> None:
        step = 0
        while not self._stop_event.is_set():
            metrics = self._collect()
            if metrics:
                self._logger.log_metrics_now(metrics, step)
            step += 1
            self._stop_event.wait(timeout=self._interval)

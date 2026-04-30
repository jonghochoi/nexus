"""
nexus/logger/system_metrics.py
==============================
SystemMetricsLogger: background thread that periodically logs system resource metrics.

Requires optional dependencies: psutil (CPU/RAM), nvidia-ml-py or nvidia-smi (GPU).
The nvidia-ml-py package exposes the `pynvml` module — install with:
    pip install psutil nvidia-ml-py
Silently skips any metric that cannot be collected.
"""

from __future__ import annotations

import os
import subprocess
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .mlflow_logger import MLflowLogger


def _get_host_pid() -> int:
    """Return the host-namespace PID of the current process.

    Inside a container, os.getpid() returns the container-namespace PID, but
    NVML reports host-namespace PIDs. /proc/self/sched exposes the host PID on
    its first line: 'comm (HOST_PID, ...)'. Falls back to os.getpid() when the
    file is unavailable (non-Linux or restricted kernel config).
    """
    try:
        with open("/proc/self/sched") as f:
            line = f.readline()
            return int(line.split("(")[1].split(",")[0].strip())
    except Exception:
        return os.getpid()


def _find_gpu_by_pid() -> Optional[int]:
    """Return the physical GPU index used by the current process, or None if not found.

    Detection order:
    1. PID scan via pynvml — works on bare metal and most containers.
    2. CUDA_VISIBLE_DEVICES — fallback when only one device is specified.
    3. Visible GPU count == 1 — when the container already limits visibility to
       a single device, that device is unambiguously ours regardless of whether
       a CUDA compute context is registered (e.g. IsaacSim uses a non-compute
       GPU context that nvmlDeviceGetComputeRunningProcesses does not report).
    """
    pid = _get_host_pid()
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()

        # ── 1. PID scan ──────────────────────────────────────────────────────
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if any(p.pid == pid for p in procs):
                return i

        # ── 2. CUDA_VISIBLE_DEVICES (single value only) ──────────────────────
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible and cuda_visible not in ("NoDeviceFiles", "-1"):
            parts = [p.strip() for p in cuda_visible.split(",") if p.strip()]
            if len(parts) == 1:
                try:
                    return int(parts[0])
                except ValueError:
                    pass

        # ── 3. Single visible GPU — container has already isolated the device ─
        if count == 1:
            return 0

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

    Pass gpu_index explicitly to skip auto-detection:
        sys_logger = SystemMetricsLogger(mlflow_logger, gpu_index=2)

    Metrics logged (when available):
        system/cpu_percent      — CPU utilisation (%)
        system/ram_gb           — RAM used (GB)
        system/gpu_memory_mb    — GPU memory used (MB)
        system/gpu_util_percent — GPU compute utilisation (%)

    GPU auto-detection order (skipped when gpu_index is given explicitly):
      1. PID scan via pynvml — works on bare metal and most containers.
      2. CUDA_VISIBLE_DEVICES — used when it specifies exactly one device.
      3. Visible GPU count == 1 — when the container has already isolated a
         single device (e.g. IsaacSim, which opens a non-compute GPU context
         that the PID scan cannot see).
    Once detected, the index is cached and written to the run tag
    `system.gpu_index`.
    """

    def __init__(
        self,
        mlflow_logger: "MLflowLogger",
        interval_seconds: float = 30.0,
        gpu_index: Optional[int] = None,
    ):
        self._logger = mlflow_logger
        self._interval = interval_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Pre-set when caller specifies the index explicitly; otherwise detected lazily.
        self._gpu_index: Optional[int] = gpu_index

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

        # ── GPU — lazy detection, cached once found ──────────────────────────
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
            try:
                metrics = self._collect()
                if metrics:
                    self._logger.log_metrics_now(metrics, step)
            except Exception as e:
                print(f"[SystemMetricsLogger] Collection error (step {step}): {e}", flush=True)
            step += 1
            self._stop_event.wait(timeout=self._interval)

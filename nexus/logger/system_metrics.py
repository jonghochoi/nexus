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

import subprocess
import threading
from typing import TYPE_CHECKING, Optional

from ..brand import log as brand_log

if TYPE_CHECKING:
    from .mlflow_logger import MLflowLogger


def _get_gpu_stats(gpu_index: int) -> dict[str, float]:
    """Return memory_mb and util_percent for the given GPU index.

    The index is interpreted exactly as `pynvml` / `nvidia-smi` interpret it —
    the physical NVML index on bare metal, or the container-relative index when
    the runtime has restricted visibility (e.g. via NVIDIA_VISIBLE_DEVICES).
    Note: CUDA_VISIBLE_DEVICES does NOT remap NVML indices, so an explicit
    `gpu_index` should match what `nvidia-smi` shows, not the cuda:N number a
    framework would use after CUDA_VISIBLE_DEVICES masking.
    """
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
        sys_logger = SystemMetricsLogger(mlflow_logger, interval_seconds=30, gpu_index=3)
        sys_logger.start()
        # ... training loop ...
        sys_logger.stop()

    Metrics logged (when available):
        system/cpu_percent      — CPU utilisation (%)
        system/ram_gb           — RAM used (GB)
        system/gpu_memory_mb    — GPU memory used (MB)        — requires gpu_index
        system/gpu_util_percent — GPU compute utilisation (%) — requires gpu_index

    GPU metrics are opt-in: pass `gpu_index` explicitly to enable them. When
    `gpu_index is None` the GPU collector is skipped entirely (CPU / RAM still
    logged). Auto-detection is intentionally not provided because it is
    unreliable on multi-GPU hosts: frameworks such as PyTorch can establish
    stray CUDA contexts on GPU 0 before tensors move to the real training
    device, which would attribute metrics to the wrong GPU. See
    `docs/30_ADVANCED_FEATURES.md` for the recipe to pick `gpu_index`.

    The chosen index is written once to the run tag `system.gpu_index` when
    `start()` is called, so the active device is visible in the MLflow UI.
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
        # None — GPU metrics disabled. int — collect metrics for that NVML index.
        self._gpu_index: Optional[int] = gpu_index

    def start(self) -> None:
        """Start the background logging thread."""
        if self._gpu_index is not None:
            # Stamp the tag once at start; the value is fixed for the run.
            self._logger.set_tag("system.gpu_index", str(self._gpu_index))
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

        # ── GPU — explicit opt-in only ───────────────────────────────────────
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
                print(
                    brand_log(f"SystemMetricsLogger collection error (step {step}): {e}", "error"),
                    flush=True,
                )
            step += 1
            self._stop_event.wait(timeout=self._interval)

"""
logger/
=======
Unified logging package for RL training.

Core exports:
    make_logger  — factory function (recommended entry point)
    DualLogger   — TensorBoard + MLflow simultaneously
    MLflowLogger — MLflow only
    TBLogger     — TensorBoard only (legacy compatibility)

Advanced features (explicit import required):
    from logger.sweep_logger   import SweepLogger
    from logger.model_registry import ModelRegistry
    from logger.system_metrics import SystemMetricsLogger
    from logger                import rl_metrics

    See docs/ADVANCED_FEATURES.md for usage guide.
"""

from .dual_logger import DualLogger, make_logger
from .mlflow_logger import MLflowLogger
from .tb_logger import TBLogger

__all__ = ["make_logger", "DualLogger", "MLflowLogger", "TBLogger"]

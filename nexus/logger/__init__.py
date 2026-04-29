"""
nexus/logger/
=============
Unified logging package for RL training.

Core exports:
    make_logger  — factory function (recommended entry point)
    DualLogger   — TensorBoard + MLflow simultaneously
    MLflowLogger — MLflow only
    TBLogger     — TensorBoard only (legacy compatibility)

Advanced features (explicit import required):
    from nexus.logger.sweep_logger   import SweepLogger
    from nexus.logger.model_registry import ModelRegistry
    from nexus.logger.system_metrics import SystemMetricsLogger
    from nexus.logger                import rl_metrics

    See docs/30_ADVANCED_FEATURES.md for usage guide.
"""

# ── Sanity check: ensure mlflow is a real package, not a namespace stub ──
# Some environments (notably Isaac Sim / Isaac Lab) ship a partial
# mlflow-skinny: the dist-info is registered but the actual mlflow/ module
# body is missing. The cascading ModuleNotFoundError deep inside
# mlflow_logger ("No module named 'mlflow.entities'") is cryptic — surface
# a single actionable message at the top of the import chain instead.
import mlflow as _mlflow
if _mlflow.__file__ is None:
    raise ImportError(
        "nexus-logger: 'mlflow' is loaded as a namespace package "
        "(mlflow.__file__ is None), which means the mlflow-skinny install "
        "in your environment is incomplete (often the case on Isaac Sim "
        "/ Isaac Lab). Run:\n"
        "    python -m pip install --upgrade --force-reinstall --no-deps "
        "'mlflow-skinny>=2.0,<3'\n"
        "See README.md → 'Use as a Python Dependency' for details."
    )
del _mlflow

from .dual_logger import DualLogger, make_logger
from .mlflow_logger import MLflowLogger
from .tb_logger import TBLogger

__all__ = ["make_logger", "DualLogger", "MLflowLogger", "TBLogger"]

"""
nexus/logger/rl_metrics.py
==========================
Pure NumPy helper functions for common RL diagnostic metrics.
"""

from __future__ import annotations

import numpy as np


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """Fraction of variance in returns explained by value predictions (1.0 = perfect).

    Returns NaN if returns have near-zero variance.
    """
    var_y = np.var(returns)
    if var_y < 1e-8:
        return float("nan")
    return float(1.0 - np.var(returns - values) / var_y)


def approx_kl(log_probs_old: np.ndarray, log_probs_new: np.ndarray) -> float:
    """Approximate KL(old || new) via log ratio: mean((r-1) - log(r))."""
    log_ratio = log_probs_new - log_probs_old
    return float(np.mean((np.exp(log_ratio) - 1) - log_ratio))


def clip_fraction(ratios: np.ndarray, clip_eps: float = 0.2) -> float:
    """Fraction of probability ratios that exceed the PPO clip boundary."""
    return float(np.mean(np.abs(ratios - 1.0) > clip_eps))


def grad_norm(parameters) -> float:
    """L2 norm of gradients. Accepts torch parameter iterator or list of numpy arrays."""
    try:
        import torch  # noqa: F401

        total = 0.0
        for p in parameters:
            if hasattr(p, "grad") and p.grad is not None:
                total += p.grad.detach().norm().item() ** 2
        return float(total**0.5)
    except ImportError:
        total = 0.0
        for arr in parameters:
            total += float(np.sum(np.asarray(arr) ** 2))
        return float(total**0.5)

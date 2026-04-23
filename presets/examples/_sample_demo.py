"""
presets/examples/_sample_demo.py
================================
Generate `sample_output.html` from fabricated training curves — no MLflow
server required. Intended to give new users a preview of what the Plotly
report actually looks like before they point a real preset at their server.

Run from repo root:
    python presets/examples/_sample_demo.py

Output:
    presets/examples/sample_output.html

The curves are synthetic — shapes are plausible (sigmoid success, LR-sized
KL band, EV ramp, entropy decay) but the numbers are fabricated and should
not be read as real results.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as a plain script (`python presets/examples/_sample_demo.py`)
# by making the repo root importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from presets.renderer import render_html
from presets.resolver import ResolvedData
from presets.schema import Chart, Filters, MetricSpec, Preset, Select


# ──────────────────────────────────────────────────────────────────
# Duck-typed stand-ins for mlflow.entities.Run — the renderer only
# touches .info.run_id / .info.run_name and .data.tags / .data.params.
# ──────────────────────────────────────────────────────────────────
@dataclass
class _Info:
    run_id: str
    run_name: str
    status: str = "FINISHED"


@dataclass
class _Data:
    tags: dict
    params: dict


@dataclass
class _Run:
    info: _Info
    data: _Data


# ──────────────────────────────────────────────────────────────────
def _curves(lr: float, seed: int, n: int = 200):
    """Sigmoid success, LR-scaled KL band, EV ramp, entropy decay."""
    rng = random.Random(seed * 7919 + int(lr * 1e6))
    peak = max(0.15, min(0.92, 0.85 - 2.4 * (math.log10(lr) + 3.3) ** 2 * 0.07))
    ts0 = 1_700_000_000_000
    success, kl, ev, ent = [], [], [], []
    for i in range(n):
        x = (i - n * 0.3) / (n * 0.15)
        sigmoid = peak / (1 + math.exp(-x))
        success.append((i, ts0 + i * 1000, max(0.0, sigmoid + rng.gauss(0, 0.03))))
        kl.append     ((i, ts0 + i * 1000, max(0.0, lr * 7 + rng.gauss(0, lr * 2.5))))
        ev.append     ((i, ts0 + i * 1000, min(1.0, 0.05 + 0.75 * i / n + rng.gauss(0, 0.025))))
        ent.append    ((i, ts0 + i * 1000, max(0.2, 1.5 - 1.1 * i / n + rng.gauss(0, 0.04))))
    return success, kl, ev, ent


def _build() -> ResolvedData:
    preset = Preset(
        name="ppo_lr_ablation",
        title="Sample: PPO LR ablation — in_hand_reorientation (synthetic)",
        experiment="robot_hand_rl",
        description=(
            "Fabricated data for documentation purposes.\n"
            "Read the shapes of the curves, not the numbers — nothing here was trained."
        ),
        filters=Filters(tags={"task": "in_hand_reorientation"}, status=["FINISHED"]),
        select=Select(mode="latest", limit=12),
        metrics=[
            MetricSpec(key="rl/success_rate", smoothing=0.6),
            MetricSpec(key="rl/approx_kl", smoothing=0.3),
            MetricSpec(key="rl/explained_variance"),
            MetricSpec(key="rl/entropy"),
        ],
        chart=Chart(x_axis="step", group_by="params.lr.base", height_per_plot=280),
    )

    configs = [(1e-4, 0), (3e-4, 0), (3e-4, 1), (1e-3, 0), (1e-3, 1), (3e-3, 0)]
    runs: list = []
    histories: dict = {m.key: {} for m in preset.metrics}
    for i, (lr, seed) in enumerate(configs):
        run_id = f"{i:02x}" + "deadbeef" * 2
        run = _Run(
            info=_Info(run_id=run_id, run_name=f"ppo_lr_{lr:.0e}_s{seed}"),
            data=_Data(
                tags={
                    "task": "in_hand_reorientation",
                    "seed": str(seed),
                    "researcher": "demo",
                },
                params={"lr.base": f"{lr:.1e}"},
            ),
        )
        runs.append(run)
        s, k, e, ent = _curves(lr, seed)
        histories["rl/success_rate"][run_id] = s
        histories["rl/approx_kl"][run_id] = k
        histories["rl/explained_variance"][run_id] = e
        histories["rl/entropy"][run_id] = ent

    return ResolvedData(preset=preset, runs=runs, histories=histories)


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_output.html"
    render_html(_build(), out)
    print(f"[OK] wrote {out} ({out.stat().st_size / 1024:.1f} KB)")

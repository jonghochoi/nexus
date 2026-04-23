"""
presets/schema.py
=================
Preset YAML schema and loader.

A preset file describes:
  - which MLflow experiment to query
  - filters (tags / params / status) to narrow down runs
  - how to select runs once filtered (latest N, all, or explicit IDs)
  - which metrics to plot and how
  - chart-level options (x-axis, grouping, smoothing default)

The schema is intentionally flat and YAML-friendly. Unknown top-level keys
raise PresetError so typos are caught early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class PresetError(ValueError):
    """Raised when a preset YAML is malformed or logically inconsistent."""


# ──────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────
@dataclass
class Filters:
    tags: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    status: list[str] = field(default_factory=list)


@dataclass
class Select:
    mode: str = "latest"           # latest | all | explicit
    limit: int = 20                # used when mode == latest
    run_ids: list[str] = field(default_factory=list)  # used when mode == explicit

    _VALID_MODES = ("latest", "all", "explicit")


@dataclass
class MetricSpec:
    key: str
    smoothing: float = 0.0         # EMA smoothing factor in [0, 1)
    y_label: str | None = None


@dataclass
class Chart:
    x_axis: str = "step"           # step | timestamp
    group_by: str = "tags.mlflow.runName"  # dotted path into run; colors/legend
    height_per_plot: int = 320
    default_smoothing: float = 0.0

    _VALID_X = ("step", "timestamp")


@dataclass
class Preset:
    name: str
    title: str
    experiment: str
    metrics: list[MetricSpec]
    description: str = ""
    filters: Filters = field(default_factory=Filters)
    select: Select = field(default_factory=Select)
    chart: Chart = field(default_factory=Chart)
    source_path: str | None = None  # filled in by load_preset()


# ──────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────
_TOP_LEVEL_KEYS = {
    "name", "title", "experiment", "description",
    "filters", "select", "metrics", "chart",
}


def load_preset(path: str | Path) -> Preset:
    """Parse and validate a preset YAML file."""
    p = Path(path)
    if not p.exists():
        raise PresetError(f"preset file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise PresetError(f"{p}: top-level YAML must be a mapping")

    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise PresetError(f"{p}: unknown top-level key(s): {sorted(unknown)}")

    for req in ("name", "experiment", "metrics"):
        if req not in raw:
            raise PresetError(f"{p}: missing required key '{req}'")

    metrics_raw = raw["metrics"]
    if not isinstance(metrics_raw, list) or not metrics_raw:
        raise PresetError(f"{p}: 'metrics' must be a non-empty list")

    metrics: list[MetricSpec] = []
    for i, m in enumerate(metrics_raw):
        if isinstance(m, str):
            metrics.append(MetricSpec(key=m))
        elif isinstance(m, dict):
            if "key" not in m:
                raise PresetError(f"{p}: metrics[{i}] missing 'key'")
            metrics.append(MetricSpec(
                key=str(m["key"]),
                smoothing=float(m.get("smoothing", 0.0)),
                y_label=m.get("y_label"),
            ))
        else:
            raise PresetError(f"{p}: metrics[{i}] must be a string or mapping")

    filters_raw = raw.get("filters") or {}
    filters = Filters(
        tags=dict(filters_raw.get("tags") or {}),
        params=dict(filters_raw.get("params") or {}),
        status=list(filters_raw.get("status") or []),
    )

    select_raw = raw.get("select") or {}
    select = Select(
        mode=str(select_raw.get("mode", "latest")),
        limit=int(select_raw.get("limit", 20)),
        run_ids=list(select_raw.get("run_ids") or []),
    )
    if select.mode not in Select._VALID_MODES:
        raise PresetError(
            f"{p}: select.mode must be one of {Select._VALID_MODES}, got '{select.mode}'"
        )
    if select.mode == "explicit" and not select.run_ids:
        raise PresetError(f"{p}: select.mode='explicit' requires non-empty run_ids")

    chart_raw = raw.get("chart") or {}
    chart = Chart(
        x_axis=str(chart_raw.get("x_axis", "step")),
        group_by=str(chart_raw.get("group_by", "tags.mlflow.runName")),
        height_per_plot=int(chart_raw.get("height_per_plot", 320)),
        default_smoothing=float(chart_raw.get("default_smoothing", 0.0)),
    )
    if chart.x_axis not in Chart._VALID_X:
        raise PresetError(
            f"{p}: chart.x_axis must be one of {Chart._VALID_X}, got '{chart.x_axis}'"
        )
    if not (0.0 <= chart.default_smoothing < 1.0):
        raise PresetError(f"{p}: chart.default_smoothing must be in [0, 1)")

    for m in metrics:
        if m.smoothing == 0.0 and chart.default_smoothing > 0.0:
            m.smoothing = chart.default_smoothing
        if not (0.0 <= m.smoothing < 1.0):
            raise PresetError(f"{p}: metric '{m.key}' smoothing must be in [0, 1)")

    return Preset(
        name=str(raw["name"]),
        title=str(raw.get("title") or raw["name"]),
        experiment=str(raw["experiment"]),
        description=str(raw.get("description") or ""),
        filters=filters,
        select=select,
        metrics=metrics,
        chart=chart,
        source_path=str(p),
    )

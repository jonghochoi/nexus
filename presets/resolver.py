"""
presets/resolver.py
===================
Turn a Preset into concrete data by querying MLflow.

Given a preset, resolve():
  1. Looks up the experiment by name
  2. Builds an MLflow filter_string from preset.filters
  3. Searches runs, applies post-filters (status, OR-over-list values)
  4. Narrows down by preset.select (latest N / all / explicit IDs)
  5. Fetches metric histories for every (run, metric) pair

Returns a ResolvedData bundle the renderer consumes — no Plotly imports here,
so this layer stays reusable for future renderers (Streamlit, URL generator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from presets.schema import Preset


@dataclass
class ResolvedData:
    preset: Preset
    runs: list[Run]
    # metric_key -> run_id -> list[(step, timestamp_ms, value)]
    histories: dict[str, dict[str, list[tuple[int, int, float]]]] = field(default_factory=dict)
    # metric keys that had zero points across all runs (useful for warnings)
    empty_metrics: list[str] = field(default_factory=list)

    @property
    def run_count(self) -> int:
        return len(self.runs)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def _as_list(v: Any) -> list[str]:
    """Normalize a filter value to a list of strings. None/empty -> []."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def _build_filter_string(preset: Preset) -> str:
    """
    Build an MLflow search filter_string for the parts we can push down.

    MLflow search_runs supports AND'd clauses like:
        tags.task = 'in_hand_reorientation' AND params.lr = '0.0003'

    For list values (OR semantics), we push down only single-value filters
    and handle multi-value ones client-side in _post_filter().
    """
    clauses: list[str] = []
    for k, v in preset.filters.tags.items():
        values = _as_list(v)
        if len(values) == 1:
            clauses.append(f"tags.`{k}` = '{values[0]}'")
    for k, v in preset.filters.params.items():
        values = _as_list(v)
        if len(values) == 1:
            clauses.append(f"params.`{k}` = '{values[0]}'")
    return " AND ".join(clauses)


def _post_filter(runs: list[Run], preset: Preset) -> list[Run]:
    """Apply filters MLflow's filter_string can't express (OR-over-list, status)."""
    out = []
    tag_multi = {k: _as_list(v) for k, v in preset.filters.tags.items()
                 if len(_as_list(v)) > 1}
    param_multi = {k: _as_list(v) for k, v in preset.filters.params.items()
                   if len(_as_list(v)) > 1}
    status_allow = [s.upper() for s in preset.filters.status]

    for r in runs:
        ok = True
        for k, allowed in tag_multi.items():
            if r.data.tags.get(k) not in allowed:
                ok = False
                break
        if not ok:
            continue
        for k, allowed in param_multi.items():
            if r.data.params.get(k) not in allowed:
                ok = False
                break
        if not ok:
            continue
        if status_allow and r.info.status.upper() not in status_allow:
            continue
        out.append(r)
    return out


def _select_runs(runs: list[Run], preset: Preset, client: MlflowClient) -> list[Run]:
    mode = preset.select.mode
    if mode == "all":
        return runs
    if mode == "latest":
        # runs come back ordered by start_time DESC from search_runs by default
        return runs[: preset.select.limit]
    if mode == "explicit":
        by_id = {r.info.run_id: r for r in runs}
        resolved = []
        for rid in preset.select.run_ids:
            if rid in by_id:
                resolved.append(by_id[rid])
                continue
            # Not in the filtered set — fetch directly (filters don't apply to explicit IDs)
            try:
                resolved.append(client.get_run(rid))
            except Exception:
                pass  # skip missing/unreadable runs silently — renderer will note run_count
        return resolved
    raise ValueError(f"unknown select.mode: {mode}")


# ──────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────
def resolve(preset: Preset, tracking_uri: str) -> ResolvedData:
    """Execute a preset against an MLflow server and return the data bundle."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    exp = client.get_experiment_by_name(preset.experiment)
    if exp is None:
        raise RuntimeError(
            f"experiment '{preset.experiment}' not found on {tracking_uri}"
        )

    filter_string = _build_filter_string(preset)
    # search_runs returns in start_time DESC by default
    raw_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        max_results=5000,
    )
    filtered = _post_filter(raw_runs, preset)
    selected = _select_runs(filtered, preset, client)

    histories: dict[str, dict[str, list[tuple[int, int, float]]]] = {}
    empty: list[str] = []
    for m in preset.metrics:
        per_run: dict[str, list[tuple[int, int, float]]] = {}
        total = 0
        for r in selected:
            points = client.get_metric_history(r.info.run_id, m.key)
            # MLflow returns a list[Metric] with .step, .timestamp, .value
            series = [(p.step, p.timestamp, p.value) for p in points]
            series.sort(key=lambda x: x[0])
            per_run[r.info.run_id] = series
            total += len(series)
        histories[m.key] = per_run
        if total == 0:
            empty.append(m.key)

    return ResolvedData(
        preset=preset,
        runs=selected,
        histories=histories,
        empty_metrics=empty,
    )

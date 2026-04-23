"""
presets/renderer.py
===================
Render ResolvedData as a standalone, self-contained HTML report.

One Plotly subplot per metric. Each run is a differently-colored line; legend
entries and hover labels use preset.chart.group_by to pull a friendly name out
of the run (tag, param, or run_name).

Output is a single .html file with Plotly bundled inline so it can be opened
directly in a browser, attached to Confluence, or emailed around. No server
required.
"""

from __future__ import annotations

import datetime as _dt
import html
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from mlflow.entities import Run
from plotly.subplots import make_subplots

from presets.resolver import ResolvedData
from presets.schema import MetricSpec, Preset


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def _run_label(run: Run, group_by: str) -> str:
    """
    Resolve a dotted path like 'tags.seed' / 'params.lr.base' / 'info.run_name'
    against a Run. Falls back to a short run_id if the key is missing.
    """
    if group_by.startswith("tags."):
        return run.data.tags.get(group_by[len("tags."):], run.info.run_id[:8])
    if group_by.startswith("params."):
        return run.data.params.get(group_by[len("params."):], run.info.run_id[:8])
    if group_by == "info.run_name" or group_by == "run_name":
        return run.info.run_name or run.info.run_id[:8]
    if group_by == "run_id":
        return run.info.run_id[:8]
    # catch-all: look in tags then params
    return (run.data.tags.get(group_by)
            or run.data.params.get(group_by)
            or run.info.run_id[:8])


def _ema(values: list[float], alpha: float) -> list[float]:
    """Standard EMA smoothing used by TensorBoard: s_t = a*s_{t-1} + (1-a)*v_t."""
    if alpha <= 0.0 or not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * out[-1] + (1.0 - alpha) * v)
    return out


def _x_values(points: list[tuple[int, int, float]], x_axis: str) -> list[Any]:
    if x_axis == "timestamp":
        return [_dt.datetime.fromtimestamp(ts / 1000.0) for _, ts, _ in points]
    return [s for s, _, _ in points]


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────
def render_html(data: ResolvedData, output_path: str | Path) -> Path:
    """Render a preset's resolved data to a self-contained HTML file."""
    preset = data.preset
    metrics = preset.metrics
    n = len(metrics)

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[m.key for m in metrics],
        vertical_spacing=max(0.04, 0.15 / max(n, 1)),
        shared_xaxes=False,
    )

    palette = _palette(len(data.runs))
    run_colors: dict[str, str] = {
        r.info.run_id: palette[i] for i, r in enumerate(data.runs)
    }

    seen_in_legend: set[str] = set()
    for row_idx, m in enumerate(metrics, start=1):
        per_run = data.histories.get(m.key, {})
        for run in data.runs:
            points = per_run.get(run.info.run_id, [])
            if not points:
                continue
            xs = _x_values(points, preset.chart.x_axis)
            ys_raw = [v for _, _, v in points]
            ys = _ema(ys_raw, m.smoothing)

            label = _run_label(run, preset.chart.group_by)
            show_legend = label not in seen_in_legend
            seen_in_legend.add(label)

            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=show_legend,
                    line=dict(color=run_colors[run.info.run_id], width=1.5),
                    hovertemplate=(
                        f"<b>{html.escape(label)}</b><br>"
                        f"{html.escape(m.key)}<br>"
                        f"x=%{{x}}<br>y=%{{y:.4g}}<extra></extra>"
                    ),
                ),
                row=row_idx, col=1,
            )
        fig.update_yaxes(
            title_text=m.y_label or "", row=row_idx, col=1,
        )
    fig.update_xaxes(title_text=preset.chart.x_axis, row=n, col=1)

    fig.update_layout(
        title=dict(text=preset.title, x=0.01, xanchor="left"),
        height=preset.chart.height_per_plot * n + 120,
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.6)",
        ),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build a small header with the preset metadata so the HTML is self-describing
    header = _header_html(data)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    page = _page_template(preset.title, header, fig_html)
    out.write_text(page, encoding="utf-8")
    return out


# ──────────────────────────────────────────────────────────────────
# Presentation helpers
# ──────────────────────────────────────────────────────────────────
# Palette: 12 distinct lines cycles gracefully; more than that still works (mod).
_PALETTE = [
    "#0194E2", "#FF6F00", "#2CA02C", "#D62728", "#9467BD", "#8C564B",
    "#E377C2", "#17BECF", "#BCBD22", "#393B79", "#7F7F7F", "#637939",
]


def _palette(n: int) -> list[str]:
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def _header_html(data: ResolvedData) -> str:
    p = data.preset
    filters_bits: list[str] = []
    for k, v in p.filters.tags.items():
        filters_bits.append(f"tags.{html.escape(k)}={html.escape(str(v))}")
    for k, v in p.filters.params.items():
        filters_bits.append(f"params.{html.escape(k)}={html.escape(str(v))}")
    if p.filters.status:
        filters_bits.append(f"status={html.escape(','.join(p.filters.status))}")
    filters_text = " · ".join(filters_bits) or "—"

    warn = ""
    if data.empty_metrics:
        warn = (
            "<p class='warn'>⚠ No data for metric(s): "
            + html.escape(", ".join(data.empty_metrics))
            + "</p>"
        )

    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    desc = html.escape(p.description or "").replace("\n", "<br>")
    return f"""
<div class="meta">
  <h1>{html.escape(p.title)}</h1>
  <p class="sub">
    <b>preset:</b> {html.escape(p.name)} ·
    <b>experiment:</b> {html.escape(p.experiment)} ·
    <b>runs:</b> {data.run_count} ·
    <b>select:</b> {html.escape(p.select.mode)}
    {'(limit=' + str(p.select.limit) + ')' if p.select.mode == 'latest' else ''}
  </p>
  <p class="sub"><b>filters:</b> {filters_text}</p>
  {f'<p class="desc">{desc}</p>' if desc else ''}
  {warn}
  <p class="sub dim">generated {generated}</p>
</div>
"""


def _page_template(title: str, header: str, fig_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)} — NEXUS preset</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 24px; color: #222; background: #fafafa; }}
  .meta {{ max-width: 1200px; margin: 0 auto 16px; }}
  .meta h1 {{ font-size: 20px; margin: 0 0 6px; }}
  .meta .sub {{ font-size: 13px; color: #555; margin: 2px 0; }}
  .meta .desc {{ font-size: 13px; color: #333; margin: 8px 0;
                 padding: 8px 12px; background: #fff; border-left: 3px solid #0194E2; }}
  .meta .dim {{ color: #999; }}
  .warn {{ color: #b54708; background: #fffaeb; padding: 6px 10px;
           border-left: 3px solid #f79009; font-size: 13px; }}
  .chart {{ max-width: 1400px; margin: 0 auto; background: #fff;
            border: 1px solid #eee; border-radius: 6px; padding: 8px; }}
</style>
</head>
<body>
  {header}
  <div class="chart">{fig_html}</div>
</body>
</html>
"""

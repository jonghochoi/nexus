"""
presets/
========
YAML-defined visualization presets for the NEXUS MLflow tracking server.

A preset bundles together a run-selection query (experiment + tag/param filters)
with a list of metrics and chart options. Rendering produces a standalone HTML
report that overlays every matching run on one Plotly chart per metric — useful
for routine comparisons (ablations, seed spreads, Sim-to-Real gaps, etc.)
without having to rebuild the MLflow UI filter state by hand each time.

Public API (import lazily — keeps `validate` usable without mlflow/plotly):
    from presets.schema   import load_preset, Preset, PresetError
    from presets.resolver import resolve, ResolvedData   # requires mlflow
    from presets.renderer import render_html             # requires plotly

CLI:
    python -m presets render   <preset.yaml>  [--output <file>]
    python -m presets list     [<dir>]
    python -m presets validate <preset.yaml>
"""

# Only the lightweight schema layer is re-exported eagerly so that
# `python -m presets validate` works in CI without mlflow/plotly installed.
from presets.schema import Preset, PresetError, load_preset

__all__ = ["Preset", "PresetError", "load_preset"]

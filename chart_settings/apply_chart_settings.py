"""
chart_settings/apply_chart_settings.py
=======================================
Persist MLflow chart/column settings across browser sessions.

Settings are stored as MLflow experiment tags, so they:
  - Survive server restarts
  - Are shared across all team members
  - Persist regardless of browser or machine

Usage:
  # Save settings to the MLflow server
  python chart_settings/apply_chart_settings.py apply

  # Apply to a specific experiment only
  python chart_settings/apply_chart_settings.py apply --experiment real_robot_eval

  # Print a browser bookmarklet to restore localStorage column settings
  python chart_settings/apply_chart_settings.py bookmarklet

  # Show settings currently stored on the server
  python chart_settings/apply_chart_settings.py show

Options:
  --config PATH           Path to chart_settings.json (default: file next to this script)
  --tracking-uri URI      MLflow server address (default: ~/.nexus/config.json or http://127.0.0.1:5000)
  --experiment NAME       Experiment name (default: all experiments in the config file)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

_HERE = Path(__file__).parent
_DEFAULT_SETTINGS = _HERE / "chart_settings.json"
_NEXUS_CONFIG = Path.home() / ".nexus" / "config.json"

TAG_SETTINGS = "nexus.chart_settings"
TAG_VERSION = "nexus.chart_settings_version"


# ── Config helpers ────────────────────────────────────────────────────────────


def _load_nexus_config() -> dict:
    if not _NEXUS_CONFIG.exists():
        return {}
    with open(_NEXUS_CONFIG) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _resolve_tracking_uri(cli_uri: str | None) -> str:
    if cli_uri:
        return cli_uri
    nexus = _load_nexus_config()
    return nexus.get("tracking_uri", "http://127.0.0.1:5000")


def _load_settings(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"[ERROR] Settings file not found: {path}")
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(f"[ERROR] Failed to parse JSON ({path}): {e}")


# ── Core operations ───────────────────────────────────────────────────────────


def cmd_apply(args: argparse.Namespace) -> None:
    settings = _load_settings(Path(args.config))
    tracking_uri = _resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiments_cfg = settings.get("experiments", {})
    if not experiments_cfg:
        sys.exit("[ERROR] 'experiments' key missing from chart_settings.json.")

    targets = [args.experiment] if args.experiment else list(experiments_cfg.keys())

    print(f"MLflow server: {tracking_uri}")
    print()

    for exp_name in targets:
        if exp_name not in experiments_cfg:
            print(f"  [SKIP] '{exp_name}' — not found in config file")
            continue

        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            print(f"  [SKIP] '{exp_name}' — experiment does not exist on the MLflow server")
            continue

        exp_cfg = experiments_cfg[exp_name]
        payload = json.dumps(exp_cfg, ensure_ascii=False, separators=(",", ":"))
        client.set_experiment_tag(exp.experiment_id, TAG_SETTINGS, payload)
        client.set_experiment_tag(
            exp.experiment_id, TAG_VERSION, str(settings.get("version", "1.0"))
        )

        col_tags = exp_cfg.get("visible_columns", {}).get("tags", [])
        col_metrics = exp_cfg.get("visible_columns", {}).get("metrics", [])
        n_charts = len(exp_cfg.get("charts", []))
        print(f"  [OK] {exp_name}")
        print(f"       tag columns   : {', '.join(col_tags) or '(none)'}")
        print(f"       metric columns: {', '.join(col_metrics) or '(none)'}")
        print(f"       charts        : {n_charts}")

    print()
    print("Settings saved to the MLflow server.")
    print("To restore in browser: python chart_settings/apply_chart_settings.py bookmarklet")


def cmd_show(args: argparse.Namespace) -> None:
    tracking_uri = _resolve_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    settings = _load_settings(Path(args.config))
    experiments_cfg = settings.get("experiments", {})
    targets = [args.experiment] if args.experiment else list(experiments_cfg.keys())

    print(f"MLflow server: {tracking_uri}")
    print()

    for exp_name in targets:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            print(f"  [missing] {exp_name} — experiment not found on server")
            continue

        tags = {t.key: t.value for t in exp.tags} if exp.tags else {}
        raw = tags.get(TAG_SETTINGS)
        version = tags.get(TAG_VERSION, "(none)")

        if raw is None:
            print(f"  [not set] {exp_name} — no chart_settings stored")
            print(f"            Run the 'apply' command to save settings.")
            continue

        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  [corrupt] {exp_name} — stored settings are not valid JSON")
            continue

        col = cfg.get("visible_columns", {})
        print(f"  experiment: {exp_name}  (v{version})")
        print(f"    tag columns   : {', '.join(col.get('tags', [])) or '(none)'}")
        print(f"    params        : {', '.join(col.get('params', [])) or '(none)'}")
        print(f"    metric columns: {', '.join(col.get('metrics', [])) or '(none)'}")
        for i, chart in enumerate(cfg.get("charts", []), 1):
            print(
                f"    chart {i}: {chart.get('title', '(untitled)')} — {', '.join(chart.get('metrics', []))}"
            )
        print()


def cmd_bookmarklet(args: argparse.Namespace) -> None:
    tracking_uri = _resolve_tracking_uri(args.tracking_uri)
    settings = _load_settings(Path(args.config))
    experiments_cfg = settings.get("experiments", {})
    targets = [args.experiment] if args.experiment else list(experiments_cfg.keys())

    for exp_name in targets:
        if exp_name not in experiments_cfg:
            continue
        js = _build_bookmarklet_js(tracking_uri, exp_name)
        print(f"=== [{exp_name}] Browser Bookmarklet ===")
        print()
        print("Paste the JavaScript below into the browser console (F12 > Console)")
        print("while on the MLflow page, or save it as a browser bookmark to apply with one click.")
        print()
        print(js)
        print()


def _build_bookmarklet_js(tracking_uri: str, experiment_name: str) -> str:
    # Uses a relative path so the script works in any browser without knowing
    # the tracking URI — the fetch goes to the same origin the browser is on.
    return f"""\
javascript:(function(){{
  const expName = {json.dumps(experiment_name)};
  const TAG_KEY = {json.dumps(TAG_SETTINGS)};

  fetch('/ajax-api/2.0/mlflow/experiments/get-by-name?experiment_name=' + encodeURIComponent(expName))
    .then(function(r){{ return r.json(); }})
    .then(function(data){{
      var exp = data.experiment;
      if (!exp) {{ alert('[NEXUS] Experiment not found: ' + expName); return; }}

      var tags = exp.tags || [];
      var settingsTag = tags.find(function(t){{ return t.key === TAG_KEY; }});
      if (!settingsTag) {{
        alert('[NEXUS] No settings found on server.\\nRun the apply command first.');
        return;
      }}

      var cfg = JSON.parse(settingsTag.value);
      var cols = cfg.visible_columns || {{}};
      var expId = exp.experiment_id;

      // Write to all known MLflow 2.x localStorage key patterns for column visibility
      var colKeys = [
        'mlflow.run.columnDefs.' + expId,
        'mlflow.experimentRunsView.selectedColumns.' + expId,
        'mlflow.runs.' + expId + '.selectedColumns',
      ];
      var colValue = JSON.stringify({{
        tags: (cols.tags || []),
        params: (cols.params || []),
        metrics: (cols.metrics || [])
      }});
      colKeys.forEach(function(k){{ localStorage.setItem(k, colValue); }});

      // Cache the full settings payload for this session
      sessionStorage.setItem('nexus.chart_settings.' + expId, settingsTag.value);

      console.log('[NEXUS] Settings restored for:', expName, '(experiment_id=' + expId + ')');
      console.log('[NEXUS] Tag columns:', cols.tags);
      console.log('[NEXUS] Metric columns:', cols.metrics);
      alert('[NEXUS] Settings restored. Reloading page.');
      location.reload();
    }})
    .catch(function(e){{ console.error('[NEXUS] Error:', e); alert('[NEXUS] Error: ' + e.message); }});
}})();\
"""


# ── CLI entry point ───────────────────────────────────────────────────────────


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Persist MLflow chart/column settings across browser sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=str(_DEFAULT_SETTINGS), help="path to chart_settings.json")
    p.add_argument(
        "--tracking-uri", dest="tracking_uri", default=None, help="MLflow server address"
    )
    p.add_argument(
        "--experiment", default=None, help="experiment name (default: all in config file)"
    )

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("apply", help="save settings to the MLflow server")
    sub.add_parser("show", help="display settings currently stored on the server")
    sub.add_parser("bookmarklet", help="print a JS bookmarklet to restore browser localStorage")

    return p


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    dispatch = {"apply": cmd_apply, "show": cmd_show, "bookmarklet": cmd_bookmarklet}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

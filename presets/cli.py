"""
presets/cli.py
==============
Command-line entry point for the preset system.

    python -m presets render   <preset.yaml> [--output out.html] [--tracking_uri ...] [--open]
    python -m presets list     [<dir>]
    python -m presets validate <preset.yaml>

Kept deliberately thin — all real work lives in schema/resolver/renderer.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from presets.schema import PresetError, load_preset

# resolver/renderer imports are deferred to the render subcommand so that
# `validate` and `list` stay usable without mlflow/plotly installed.

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_OUTPUT_DIR = Path("preset_outputs")


def _cmd_render(args: argparse.Namespace) -> int:
    try:
        preset = load_preset(args.preset)
    except PresetError as e:
        print(f"[ERR] invalid preset: {e}", file=sys.stderr)
        return 2

    # Heavy deps only needed when we actually render
    from presets.renderer import render_html
    from presets.resolver import resolve

    output = Path(args.output) if args.output else (
        DEFAULT_OUTPUT_DIR / f"{preset.name}.html"
    )

    print(f"[NXS] resolving preset '{preset.name}' against {args.tracking_uri}")
    try:
        data = resolve(preset, tracking_uri=args.tracking_uri)
    except Exception as e:
        print(f"[ERR] resolve failed: {e}", file=sys.stderr)
        return 3

    if data.run_count == 0:
        print("[WARN] 0 runs matched the preset filters — nothing to plot")
        # Still render the empty page so the user sees the header + filters
    if data.empty_metrics:
        print(f"[WARN] empty metrics: {data.empty_metrics}")

    print(f"[NXS] rendering {data.run_count} run(s) × {len(preset.metrics)} metric(s)")
    path = render_html(data, output)
    print(f"[OK]  wrote {path}")

    if args.open:
        webbrowser.open(path.resolve().as_uri())
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    if not root.exists():
        print(f"[ERR] no such directory: {root}", file=sys.stderr)
        return 2
    files = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    if not files:
        print(f"(no preset files under {root})")
        return 0
    for p in files:
        try:
            preset = load_preset(p)
            print(f"  {preset.name:30s}  {preset.experiment:20s}  {p}")
        except PresetError as e:
            print(f"  [INVALID] {p}  ({e})")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        preset = load_preset(args.preset)
    except PresetError as e:
        print(f"[ERR] {e}", file=sys.stderr)
        return 2
    print(f"[OK]  '{preset.name}' is valid ({len(preset.metrics)} metric(s))")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="presets",
        description="NEXUS — YAML-defined MLflow visualization presets",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render a preset to a standalone HTML file")
    r.add_argument("preset", help="path to preset YAML")
    r.add_argument("--tracking_uri", default=DEFAULT_TRACKING_URI,
                   help=f"MLflow server URI (default: {DEFAULT_TRACKING_URI})")
    r.add_argument("--output", default=None,
                   help="output HTML path (default: preset_outputs/<name>.html)")
    r.add_argument("--open", action="store_true",
                   help="open the rendered HTML in the default browser")
    r.set_defaults(func=_cmd_render)

    ls = sub.add_parser("list", help="list preset YAMLs under a directory")
    ls.add_argument("dir", nargs="?", default="presets/examples",
                    help="directory to scan (default: presets/examples)")
    ls.set_defaults(func=_cmd_list)

    v = sub.add_parser("validate", help="parse a preset and report errors")
    v.add_argument("preset", help="path to preset YAML")
    v.set_defaults(func=_cmd_validate)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

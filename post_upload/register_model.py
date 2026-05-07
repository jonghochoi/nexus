#!/usr/bin/env python3
"""
post_upload/register_model.py
=============================
Post-hoc Model Registry registration (Pipeline B)

Usage:
    python register_model.py \
        --tracking_uri http://nexus-server:5000 \
        --experiment shadow_hand_rl \
        --run_name exp_v3_seed42 \
        --kind best \
        --model_name shadow_hand_ppo \
        --description "PPO v3 — success rate 87% on real hand" \
        --stage Staging

Workflow context:
  Run *after* training has finished and `scheduled_sync` has propagated
  the run + checkpoints/<kind>.pth artifact to the central MLflow server.
  Typically invoked from an inference / evaluation host (or operator
  desktop) — i.e. anywhere that can reach the central server.

  Run identity is `run_name`, consistent with MLflowLogger and Pipeline B.
  A new model version is created on every successful invocation; idempotency
  is the caller's responsibility.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure sibling modules resolve whether invoked from repo root or post_upload/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Add repo root so `nexus.logger.model_registry` resolves without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from config import DEFAULT_CONFIG_PATH, load_config
from history import print_history, save_upload
from nexus.logger.model_registry import ModelRegistry

console = Console()


# ── 1. Argument parsing ──────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Register a checkpoint from an existing MLflow run as a Model Registry version"
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=None,
        help="MLflow server URI — typically central. Required unless set in "
        "~/.nexus/post_config.json.",
    )
    parser.add_argument(
        "--experiment", type=str, default=None, help="Experiment name the source run lives in"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Source run name (run identity). Required for registration.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model Registry name to register against. Required for registration.",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="best",
        choices=("best", "last"),
        help="Which checkpoint to register: 'best' or 'last' (default: best)",
    )
    parser.add_argument(
        "--description", type=str, default=None, help="Free-text description (strongly recommended)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=("Staging", "Production", "Archived"),
        help="Optional stage to transition the new version to",
    )
    parser.add_argument(
        "--archive_existing_production",
        action="store_true",
        help="With --stage Production, archive existing Production versions first "
        "(keeps exactly one Production version)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to JSON config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve the run + artifact and print what would be registered, without registering",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print recent register_model invocations (~/.nexus/history.json) and exit",
    )
    return parser.parse_args()


# ── 2. Pre-flight summary ────────────────────────────────────────────────────
def print_preflight(args, source_run_id: str, source: str) -> None:
    table = Table(title="[bold]Register Model[/bold]", header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Tracking URI", args.tracking_uri)
    table.add_row("Experiment", args.experiment)
    table.add_row("Run name", args.run_name)
    table.add_row("Source run_id", source_run_id)
    table.add_row("Source artifact", source)
    table.add_row("Model name", args.model_name)
    table.add_row("Kind", args.kind)
    table.add_row("Stage", args.stage or "(unchanged)")
    table.add_row("Description", args.description or "(none)")
    if args.archive_existing_production:
        table.add_row("Archive existing Prod", "yes")
    console.print(table)


# ── 3. Result summary ────────────────────────────────────────────────────────
def print_result(args, result: dict) -> None:
    table = Table(title="[bold green]Registered[/bold green]", header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Model name", args.model_name)
    table.add_row("Version", str(result["version"]))
    table.add_row("Stage", result["stage"] or "None")
    table.add_row("Source", result["source"])
    table.add_row("Run ID", result["run_id"])
    console.print(table)


def _preparse_config_path() -> Optional[str]:
    """Scan sys.argv for --config so we can load the config before argparse
    builds its defaults. Mirrors upload_tb._preparse_config_path."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


# ── 4. Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()

    # --history is a local-file lookup; skip everything else.
    if args.history:
        print_history(script="register_model")
        return 0

    missing = [
        name for name in ("experiment", "run_name", "model_name") if getattr(args, name) is None
    ]
    if missing:
        console.print(f"[red]Missing required:[/red] {', '.join('--' + m for m in missing)}")
        return 1

    # Resolve tracking_uri with strict precedence:
    #   CLI flag (non-None)  >  config file (when present)  >  fail
    # Builtin default is *not* used silently — it would point at
    # localhost which is rarely correct for register_model invocations
    # from a desktop or inference node.
    config = load_config(_preparse_config_path())
    if args.tracking_uri is None:
        if config["source"] == "<builtin>":
            console.print(
                "[red]Missing required:[/red] --tracking_uri "
                "(no ~/.nexus/post_config.json found, so it can't fall back to a default)"
            )
            return 1
        args.tracking_uri = config["tracking_uri"]

    registry = ModelRegistry(tracking_uri=args.tracking_uri)

    # ── Pre-flight: resolve run + artifact (raises with actionable hints on miss)
    try:
        resolved = registry.resolve_checkpoint_source(
            experiment=args.experiment, run_name=args.run_name, kind=args.kind
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]Pre-flight failed:[/red] {e}")
        return 1

    print_preflight(args, resolved["run_id"], resolved["source"])

    if args.dry_run:
        console.print("[yellow]Dry run — no registration performed.[/yellow]")
        return 0

    # ── Register
    try:
        result = registry.register_from_run_name(
            experiment=args.experiment,
            run_name=args.run_name,
            model_name=args.model_name,
            kind=args.kind,
            description=args.description,
            stage=args.stage,
            archive_existing_production=args.archive_existing_production,
        )
    except Exception as e:
        console.print(f"[red]Registration failed:[/red] {e}")
        return 1

    print_result(args, result)

    # ── History
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "register_model",
        "run_id": result["run_id"],
        "tb_dir": "",
        "experiment": args.experiment,
        "run_name": args.run_name,
        "tracking_uri": args.tracking_uri,
        "tags": {
            "model_name": args.model_name,
            "kind": args.kind,
            "version": str(result["version"]),
            "stage": result["stage"] or "None",
        },
        "verify_ok": None,
    }
    save_upload(record)

    return 0


if __name__ == "__main__":
    sys.exit(main())

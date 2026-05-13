#!/usr/bin/env python3
"""
post_upload/upload_eval.py
==========================
Post-hoc rollout artifact upload (Pipeline B)

Wraps ``nexus.logger.eval_logger.EvalLogger`` with a CLI so eval bundles
(rollout videos, metrics.json, coverage plots, …) can be attached to an
existing MLflow run from any shell — symmetric with ``upload_tb.py`` and
``register_model.py``.

Usage:
    # Sidecar-driven — the sidecar wins.
    python upload_eval.py \\
        --eval-dir   ./eval_results/exp1__ckpt_5000__20260510 \\
        --run-info   ./logs/exp1/run_001/PPO

    # Explicit — when no sidecar is available (e.g. uploading to a
    # different server than the trainer logged to).
    python upload_eval.py \\
        --eval-dir              ./eval_results/exp1__ckpt_5000__20260510 \\
        --central-tracking-uri  http://nexus-server:5000 \\
        --experiment            robot_hand_rl \\
        --run-name              exp1_run_001 \\
        --metrics-from          ./eval_results/exp1__ckpt_5000__20260510/metrics.json \\
        --tag observer.commit=abc123 \\
        --tag observer.preset=hardware_safe

Workflow context:
  Run *after* observer (or any eval driver) has produced a local result
  directory, and *after* ``scheduled_sync`` has propagated the source run
  to the central MLflow server. ``EvalLogger`` resolves the run by
  ``run_name`` — the same identity used throughout Pipeline A / B.

  The CLI defaults to ``target=central`` when ``--run-info`` is provided
  so eval mp4s land directly on the team-shared central server rather
  than going through the local-relay sync queue.
"""

import argparse
import sys
from pathlib import Path

# Ensure sibling modules resolve whether invoked from repo root or post_upload/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Add repo root so `nexus.logger.eval_logger` resolves without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from config import load_config
from history import make_record, print_history, save_upload
from nexus.logger.eval_logger import EvalLogger

console = Console()


# ── 1. Argument parsing ──────────────────────────────────────────────────────
def parse_args():
    defaults = load_config()
    parser = argparse.ArgumentParser(
        description="Attach a local eval directory to an existing MLflow run as artifacts"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Local directory of eval outputs (mp4, metrics.json, plots, …). "
        "Walked recursively; everything inside is uploaded under eval/<eval_id>/.",
    )

    # Run identity — sidecar OR the explicit triple.
    parser.add_argument(
        "--run-info",
        type=str,
        default=None,
        help="Path to .nexus_run.json or its parent directory (the trainer's "
        "tb_dir). When set, run_name / experiment / tracking_uri are read "
        "from the sidecar — wins over the explicit flags below.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="central",
        choices=("central", "local"),
        help="Sidecar URI to use when --run-info is provided (default: central). "
        "'local' targets the GPU-node-local relay — useful for in-progress runs "
        "before scheduled_sync has propagated them.",
    )
    parser.add_argument(
        "--central-tracking-uri",
        type=str,
        default=defaults["central_tracking_uri"],
        help=f"Central MLflow server URI (default: {defaults['central_tracking_uri']}). "
        "Used when --run-info is not provided.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name owning the run. Required when --run-info is not provided.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name to attach artifacts to. Required when --run-info is not provided.",
    )

    # Bundle shape.
    parser.add_argument(
        "--eval-id",
        type=str,
        default=None,
        help="Subdirectory name under eval/ (default: timestamp YYYYmmdd_HHMMSS). "
        "Use a stable name to overwrite a previous bundle.",
    )
    parser.add_argument(
        "--metrics-from",
        type=str,
        default=None,
        help="Path to a JSON file whose numeric scalars are auto-promoted as "
        "eval/<key> metrics on the run (nested dicts are dotted-key flattened).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Extra tag to set on the run (repeatable). Bare keys are prefixed "
        "with 'eval.' — pass a dotted key (e.g. observer.commit=...) to "
        "bypass the prefix.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Suppress the auto-generated index.html — the eval_dir presumably ships its own.",
    )

    # Modes.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the run and list what would be uploaded; touch nothing on MLflow.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print recent upload_eval invocations (~/.nexus/history.json) and exit",
    )
    return parser.parse_args()


# ── 2. Tag list parser ───────────────────────────────────────────────────────
def parse_tag_args(raw: list) -> dict:
    """Parse repeated ``--tag KEY=VAL`` flags into a dict.

    Bare keys (no '.') are passed through as-is — ``EvalLogger`` will prefix
    them with ``eval.``. Keys that already contain '.' (e.g. ``observer.commit``)
    are passed through unchanged so callers can stamp non-eval namespaces.
    """
    out: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            console.print(f"[red]Invalid --tag (missing '='):[/red] {item!r}")
            sys.exit(1)
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            console.print(f"[red]Invalid --tag (empty key):[/red] {item!r}")
            sys.exit(1)
        out[k] = v
    return out


# ── 3. Pre-flight summary ────────────────────────────────────────────────────
def print_preflight(ev: EvalLogger, args, tags: dict) -> None:
    table = Table(title="[bold]Upload Eval[/bold]", header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Tracking URI", ev.tracking_uri)
    table.add_row("Experiment", ev.experiment)
    table.add_row("Run name", ev.run_name)
    table.add_row("Eval dir", str(Path(args.eval_dir).resolve()))
    table.add_row("Eval id", args.eval_id or "(timestamp)")
    table.add_row("Auto index.html", "no" if args.no_index else "yes")
    table.add_row("metrics_from", args.metrics_from or "(none)")
    if tags:
        table.add_row("Extra tags", ", ".join(f"{k}={v}" for k, v in tags.items()))
    console.print(table)


# ── 4. Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()

    # --history is a local-file lookup; skip everything else.
    if args.history:
        print_history(script="upload_eval")
        return 0

    if args.eval_dir is None:
        console.print("[red]Missing required:[/red] --eval-dir")
        return 1

    # ── Resolve run identity — sidecar OR explicit triple.
    if args.run_info is not None:
        try:
            ev = EvalLogger.from_run_info(args.run_info, target=args.target)
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]--run-info resolution failed:[/red] {e}")
            return 1
    else:
        missing = [name for name in ("experiment", "run_name") if getattr(args, name) is None]
        if missing:
            flags = ", ".join("--" + m.replace("_", "-") for m in missing)
            console.print(
                f"[red]Missing required:[/red] {flags} "
                "(or pass --run-info to read identity from .nexus_run.json)"
            )
            return 1
        ev = EvalLogger(
            run_name=args.run_name,
            tracking_uri=args.central_tracking_uri,
            experiment=args.experiment,
        )

    tags = parse_tag_args(args.tag)
    print_preflight(ev, args, tags)

    # ── Upload — EvalLogger handles dry_run, index.html, metrics_from, etc.
    try:
        eval_id = ev.upload(
            eval_dir=args.eval_dir,
            eval_id=args.eval_id,
            metrics_from=args.metrics_from,
            tags=tags or None,
            generate_index=not args.no_index,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Upload failed:[/red] {e}")
        return 1
    except Exception as e:  # pragma: no cover — surface unexpected MLflow errors
        console.print(f"[red]Upload failed:[/red] {e}")
        return 1

    if args.dry_run:
        console.print("[yellow]Dry run — no artifacts uploaded.[/yellow]")
        return 0

    # ── History — record alongside upload_tb / register_model invocations.
    save_upload(
        make_record(
            run_id="",
            tb_dir=args.eval_dir,
            experiment=ev.experiment,
            run_name=ev.run_name,
            central_tracking_uri=ev.tracking_uri,
            tags={"eval_id": eval_id, **{k: str(v) for k, v in tags.items()}},
            verify_ok=None,
            script="upload_eval",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
post_upload/upload_tb.py
========================
TensorBoard tfevents -> MLflow conversion uploader (Pipeline B)

Usage:
    python upload_tb.py --tb_dir ./logs/run_001 --experiment robot_hand_grasp --run_name ppo_v1

Expected tfevents directory structure:
    logs/
    └── run_001/
        └── events.out.tfevents.xxxxx
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure sibling modules resolve whether invoked from repo root or post_upload/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import pandas as pd
from tbparse import SummaryReader
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich import print as rprint

from config import DEFAULT_CONFIG_PATH, load_config, required_tags
from history import last_upload, make_record, print_history, save_upload
from verify_tb import run_verify

console = Console()

# MLflow log_batch() hard limit: 1000 metrics per call
BATCH_SIZE = 1000


# ── 1. Argument parsing ──────────────────────────────────────────────────────
def parse_args(defaults: dict):
    """Build the argument parser, using `defaults` (from config) as fallbacks."""
    parser = argparse.ArgumentParser(
        description="TensorBoard -> MLflow conversion uploader"
    )
    parser.add_argument(
        "--tb_dir",
        type=str,
        default=None,
        help="Path to directory containing tfevents files "
             "(required for uploads; optional with --history)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=defaults["experiment"],
        help=f"MLflow experiment name (default: {defaults['experiment']})",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name (default: dirname_timestamp)",
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=defaults["tracking_uri"],
        help=f"MLflow server URI (default: {defaults['tracking_uri']})",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=[],
        help="Additional tags (e.g. researcher=kim seed=42 task=grasp); "
             "merged on top of config-file tags",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to JSON config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Prompt for researcher/seed/task interactively, "
             "even if already supplied",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip required-tag validation (researcher, seed, task)",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip the automatic post-upload verification step",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print parsed results only, without uploading",
    )
    parser.add_argument(
        "--upload_artifacts",
        action="store_true",
        help="Upload files in tb_dir as MLflow artifacts",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print recent uploads (~/.nexus/history.json) and exit",
    )
    parser.add_argument(
        "--repeat-last",
        dest="repeat_last",
        action="store_true",
        help="Inherit experiment/run_name/tags from the most recent upload "
             "(CLI flags and -i still override)",
    )
    parser.add_argument(
        "--git_commit",
        type=str,
        default=None,
        metavar="HASH",
        help="Git commit hash of the training code (e.g. abc1234); "
             "stored as git_commit tag",
    )
    return parser.parse_args()


# ── 2. tfevents parsing ──────────────────────────────────────────────────────
def parse_tfevents(tb_dir: str) -> pd.DataFrame:
    """
    Parse tfevents -> DataFrame using tbparse.
    Returned columns: tag, step, value
    """
    tb_path = Path(tb_dir)
    if not tb_path.exists():
        console.print(f"[red][ERROR] Directory not found: {tb_dir}[/red]")
        sys.exit(1)

    # Recursively search for tfevents files
    tfevents_files = list(tb_path.rglob("events.out.tfevents.*"))
    if not tfevents_files:
        console.print(f"[red][ERROR] No tfevents files found in: {tb_dir}[/red]")
        sys.exit(1)

    # Detect multiple run directories — each unique parent dir is one run.
    # Multiple tfevents in the same directory is fine (resumed run), but
    # tfevents spread across different subdirectories means multiple runs
    # were passed, which would silently merge their data into one MLflow run.
    run_dirs = sorted(set(f.parent for f in tfevents_files))
    if len(run_dirs) > 1:
        console.print(f"[red][ERROR] Multiple run directories detected under: {tb_dir}[/red]")
        console.print(f"  Found {len(run_dirs)} separate run directories:")
        for d in run_dirs:
            console.print(f"    • {d}")
        console.print(
            "\n  Uploading a parent directory merges all runs into one MLflow run,\n"
            "  causing step collisions and making data uninterpretable.\n"
            "\n  Upload each run directory individually instead:\n"
        )
        console.print(
            f"  [yellow]for run_dir in {tb_dir}/*/; do\n"
            f'      python upload_tb.py --tb_dir "$run_dir" \\\n'
            f"          --experiment <experiment> --run_name $(basename \"$run_dir\") ...\n"
            f"  done[/yellow]"
        )
        sys.exit(1)

    console.print(f"\n[cyan]Discovered tfevents files:[/cyan]")
    for f in tfevents_files:
        size_kb = f.stat().st_size / 1024
        console.print(f"  • {f}  ({size_kb:.1f} KB)")

    console.print("\n[yellow]Parsing...[/yellow]")
    try:
        reader = SummaryReader(str(tb_path), pivot=False)
        df = reader.scalars
    except Exception as e:
        console.print(f"[red][ERROR] Parsing failed: {e}[/red]")
        sys.exit(1)

    if df.empty:
        console.print("[red][ERROR] Scalar data is empty.[/red]")
        console.print("  -> The log may contain only non-scalar data (histogram, image, etc.)")
        sys.exit(1)

    # Normalize column names to handle tbparse version differences
    df.columns = [c.lower() for c in df.columns]
    if "tag" not in df.columns:
        # Older tbparse versions use 'tags' instead of 'tag'
        df = df.rename(columns={"tags": "tag"})

    return df


# ── 3. Preview parsed results ────────────────────────────────────────────────
def preview_dataframe(df: pd.DataFrame):
    """Print a summary table of parsed metrics"""
    summary = (
        df.groupby("tag")
        .agg(
            steps=("step", "count"),
            step_min=("step", "min"),
            step_max=("step", "max"),
            val_min=("value", "min"),
            val_max=("value", "max"),
            val_last=("value", "last"),
        )
        .reset_index()
    )

    table = Table(
        title="[bold]Parsed TensorBoard Metrics Summary[/bold]",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Tag (Metric)", style="cyan", min_width=30)
    table.add_column("Steps", justify="right")
    table.add_column("Step Range", justify="center")
    table.add_column("Val Min", justify="right")
    table.add_column("Val Max", justify="right")
    table.add_column("Val Last", justify="right")

    for _, row in summary.iterrows():
        table.add_row(
            str(row["tag"]),
            str(int(row["steps"])),
            f"{int(row['step_min'])}~{int(row['step_max'])}",
            f"{row['val_min']:.4f}",
            f"{row['val_max']:.4f}",
            f"{row['val_last']:.4f}",
        )

    console.print(table)
    console.print(
        f"\n[green]Total: {len(summary)} tags, {len(df):,} data points[/green]\n"
    )


# ── 4. Tag parsing utility ───────────────────────────────────────────────────
def parse_extra_tags(tag_list: list) -> dict:
    """Convert a list of 'key=value' strings to a dict"""
    tags = {}
    for item in tag_list:
        if "=" in item:
            k, v = item.split("=", 1)
            tags[k.strip()] = v.strip()
        else:
            console.print(f"[yellow][WARN] Ignoring malformed tag: '{item}' (expected key=value format)[/yellow]")
    return tags


def prompt_for_tags(tags: dict, required: tuple, force_all: bool) -> dict:
    """Interactively prompt for `required` tag keys.

    If `force_all` is True, prompt for every required tag (showing current
    values as defaults). Otherwise prompt only for the ones that are missing.
    Aborts cleanly if stdin is not a TTY.
    """
    if not sys.stdin.isatty():
        return tags

    console.print("\n[bold cyan]Interactive tag entry[/bold cyan] "
                  "(press Enter to accept default)")
    for key in required:
        current = tags.get(key)
        if current is not None and not force_all:
            continue
        answer = Prompt.ask(f"  {key}", default=current or None)
        if answer:
            tags[key] = str(answer).strip()
    return tags


def validate_required_tags(tags: dict, required: tuple) -> list:
    """Return a list of required tag keys that are missing or empty."""
    return [k for k in required if not tags.get(k)]


def detect_sim_run_id(tb_dir: str) -> Optional[str]:
    """Look for run_meta.json alongside the tfevents and extract sim_run_id.

    Returns None if the file is absent, unreadable, or lacks the key.
    Convention: real-robot eval scripts drop run_meta.json next to the
    tfevents file with {"sim_run_id": "<upstream sim run_id>", ...}.
    """
    import json
    meta = Path(tb_dir) / "run_meta.json"
    if not meta.exists():
        return None
    try:
        with open(meta) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    value = data.get("sim_run_id") if isinstance(data, dict) else None
    return str(value) if value else None


# ── 5. MLflow upload ─────────────────────────────────────────────────────────
def upload_to_mlflow(
    df: pd.DataFrame,
    tb_dir: str,
    experiment_name: str,
    run_name: Optional[str],
    tracking_uri: str,
    extra_tags: dict,
    upload_artifacts: bool,
):
    # Verify MLflow server connection
    mlflow.set_tracking_uri(tracking_uri)
    console.print(f"[cyan]MLflow URI:[/cyan] {tracking_uri}")

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        console.print(f"[red][ERROR] Failed to connect to MLflow server: {e}[/red]")
        console.print("  -> Check that the server is running: bash start_all.sh")
        sys.exit(1)

    # Auto-generate run name if not provided
    if run_name is None:
        run_name = f"{Path(tb_dir).name}_{int(time.time())}"

    # Base tags applied to every run
    base_tags = {
        "source": "tensorboard_import",
        "tb_dir": str(Path(tb_dir).resolve()),
        "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra_tags,
    }

    console.print(f"\n[cyan]Experiment:[/cyan] {experiment_name}")
    console.print(f"[cyan]Run Name  :[/cyan] {run_name}")
    console.print(f"[cyan]Tags      :[/cyan] {base_tags}\n")

    total_rows = len(df)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(base_tags)
        run_id = run.info.run_id

        # ── Build Metric objects upfront (vectorized, no iterrows)
        # Converts the entire DataFrame to a list of MLflow Metric entities.
        # Using zip over numpy arrays is ~50x faster than iterrows() for large logs.
        timestamp_ms = int(time.time() * 1000)
        all_metrics = [
            Metric(
                key=sanitize_metric_name(tag),
                value=float(value),
                timestamp=timestamp_ms,
                step=int(step),
            )
            for tag, value, step in zip(
                df["tag"].values,
                df["value"].values,
                df["step"].values,
            )
        ]

        # ── Upload via log_batch() — max 1000 metrics per call (MLflow hard limit)
        client = MlflowClient(tracking_uri=tracking_uri)
        total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} batches"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Uploading {total_rows:,} data points in batches of {BATCH_SIZE}...",
                total=total_batches,
            )

            for i in range(0, total_rows, BATCH_SIZE):
                batch = all_metrics[i : i + BATCH_SIZE]
                client.log_batch(run_id=run_id, metrics=batch)
                progress.advance(task)

        # ── Artifact upload (optional)
        if upload_artifacts:
            console.print("\n[yellow]Uploading artifacts...[/yellow]")
            mlflow.log_artifacts(tb_dir, artifact_path="tensorboard_logs")
            console.print("[green][OK] tfevents artifact upload complete[/green]")

    console.print(f"\n[bold green]✓ Upload complete![/bold green]")
    console.print(f"  Run ID      : [yellow]{run_id}[/yellow]")
    console.print(f"  Data points : [green]{total_rows:,}[/green]  ({total_batches} batches)")
    console.print(f"  UI URL      : [blue]{tracking_uri}[/blue]")
    console.print(f"\n  -> Open the URL above in your browser to verify.\n")

    return run_id


# ── 6. Metric name sanitization ──────────────────────────────────────────────
def sanitize_metric_name(name: str) -> str:
    """
    Apply MLflow metric naming rules:
    Slashes (/) are allowed; replace only spaces and select special characters.
    """
    # Preserve TensorBoard hierarchy by keeping '/' (allowed by MLflow)
    # Only replace whitespace and a few problematic special characters
    return name.replace(" ", "_").replace(":", "-")


# ── 7. Main ──────────────────────────────────────────────────────────────────
def _preparse_config_path() -> Optional[str]:
    """Scan sys.argv for --config so we can load the config before argparse
    builds its defaults. Returns the path if present, else None."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def main():
    # Load config first so its values can be used as argparse defaults.
    config = load_config(_preparse_config_path())
    args = parse_args(defaults=config)

    # --history: print recent uploads and exit (no tb_dir required).
    # Restrict to upload_tb records — eval uploads have their own --history.
    if args.history:
        print_history(script="upload_tb")
        return

    if not args.tb_dir:
        console.print("[red][ERROR] --tb_dir is required for uploads.[/red]")
        sys.exit(1)

    console.rule("[bold blue]TensorBoard -> MLflow Uploader[/bold blue]")
    console.print(f"[dim]Config source: {config['source']}[/dim]")

    # --repeat-last: inherit tags/experiment/run_name from the most recent
    # upload. Precedence overall: builtin < config < history < CLI < interactive.
    tags = dict(config["tags"])
    experiment = args.experiment
    run_name = args.run_name

    if args.repeat_last:
        last = last_upload(script="upload_tb")
        if last is None:
            console.print("[yellow][WARN] --repeat-last: no previous upload in history.[/yellow]")
        else:
            console.print(
                f"[cyan]Reusing previous upload:[/cyan] "
                f"{last.get('run_name')} ({last.get('ts')})"
            )
            tags.update(last.get("tags", {}))
            # Only override experiment/run_name if user didn't pass them explicitly.
            # argparse fills experiment from config defaults, so compare to that.
            if args.experiment == config["experiment"] and last.get("experiment"):
                experiment = last["experiment"]
            if args.run_name is None:
                # Don't reuse the exact run_name — it would collide; auto-regenerate below.
                run_name = None

    # Auto-detect sim_run_id from run_meta.json next to the tfevents dir.
    # Overrides any value carried over by --repeat-last (run_meta.json is
    # ground truth for *this* tb_dir); CLI --tags still takes final precedence.
    sim = detect_sim_run_id(args.tb_dir)
    if sim:
        previous = tags.get("sim_run_id")
        if previous and previous != sim:
            console.print(
                f"[yellow]run_meta.json sim_run_id ({sim}) overrides "
                f"carried-over value ({previous})[/yellow]"
            )
        else:
            console.print(f"[cyan]Detected sim_run_id from run_meta.json:[/cyan] {sim}")
        tags["sim_run_id"] = sim

    tags.update(parse_extra_tags(args.tags))

    if args.git_commit:
        tags["git_commit"] = args.git_commit

    required = required_tags(experiment)

    if args.interactive:
        tags = prompt_for_tags(tags, required, force_all=True)
    else:
        missing = validate_required_tags(tags, required)
        if missing and sys.stdin.isatty() and not args.force and not args.dry_run:
            console.print(
                f"[yellow]Missing required tags: {', '.join(missing)} — "
                f"entering interactive mode.[/yellow]"
            )
            tags = prompt_for_tags(tags, required, force_all=False)

    missing = validate_required_tags(tags, required)
    if missing and not args.force and not args.dry_run:
        console.print(
            f"[red][ERROR] Required tags missing: {', '.join(missing)}.[/red]\n"
            f"  Supply them via --tags, ~/.nexus/post_config.json, or -i "
            f"(or re-run with --force to skip this check)."
        )
        sys.exit(1)

    # Parse tfevents and always show preview before uploading.
    df = parse_tfevents(args.tb_dir)
    preview_dataframe(df)

    if args.dry_run:
        console.print("[bold yellow]--dry_run mode: skipping upload.[/bold yellow]")
        console.print(f"[cyan]Would upload with tags:[/cyan] {tags}")
        return

    console.print(f"[cyan]Tags to upload:[/cyan] {tags}")
    console.print("Upload the above data to MLflow? [bold](y/n)[/bold]: ", end="")
    answer = input().strip().lower()
    if answer != "y":
        console.print("[yellow]Upload cancelled.[/yellow]")
        return

    # Resolve run_name now so the same value ends up in MLflow and in history.
    if run_name is None:
        run_name = f"{Path(args.tb_dir).name}_{int(time.time())}"

    run_id = upload_to_mlflow(
        df=df,
        tb_dir=args.tb_dir,
        experiment_name=experiment,
        run_name=run_name,
        tracking_uri=args.tracking_uri,
        extra_tags=tags,
        upload_artifacts=args.upload_artifacts,
    )

    # Auto-verify — removes the manual run_id copy/paste step.
    verify_ok: Optional[bool] = None
    if args.no_verify:
        console.print("[dim]Skipping verification (--no_verify).[/dim]")
    else:
        console.print("\n[bold cyan]Running automatic verification...[/bold cyan]")
        verify_ok = run_verify(
            run_id=run_id,
            tb_dir=args.tb_dir,
            tracking_uri=args.tracking_uri,
        )

    # Record for --history / --repeat-last / --from-last. Persist even if
    # verification failed, so the user can retry or replay.
    save_upload(make_record(
        run_id=run_id,
        tb_dir=args.tb_dir,
        experiment=experiment,
        run_name=run_name,
        tracking_uri=args.tracking_uri,
        tags=tags,
        verify_ok=verify_ok,
        script="upload_tb",
    ))

    if verify_ok is False:
        sys.exit(2)


if __name__ == "__main__":
    main()

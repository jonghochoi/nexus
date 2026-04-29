#!/usr/bin/env python3
"""
post_upload/verify_tb.py
========================
Validates the uploaded MLflow run against the original TensorBoard data.

Usage:
    python verify_tb.py --run_id <run_id> --tb_dir ./logs/run_001
"""

import argparse
import sys
from pathlib import Path

# Ensure sibling modules resolve whether invoked from repo root or post_upload/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import pandas as pd
from tbparse import SummaryReader
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate MLflow upload against TensorBoard source"
    )
    parser.add_argument("--run_id", type=str, default=None, help="MLflow Run ID to validate")
    parser.add_argument("--tb_dir", type=str, default=None, help="Original tfevents directory")
    parser.add_argument(
        "--tracking_uri", type=str, default="http://127.0.0.1:5000", help="MLflow server URI"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, help="Numeric comparison tolerance (default: 1e-6)"
    )
    parser.add_argument(
        "--from-last",
        dest="from_last",
        action="store_true",
        help="Re-verify the most recent upload from ~/.nexus/history.json "
        "(fills run_id/tb_dir/tracking_uri automatically)",
    )
    args = parser.parse_args()

    if args.from_last:
        from history import last_upload

        last = last_upload(script="upload_tb")
        if last is None:
            parser.error("--from-last: no previous upload in history.")
        args.run_id = args.run_id or last["run_id"]
        args.tb_dir = args.tb_dir or last["tb_dir"]
        # Only inherit tracking_uri if user didn't override it from the default.
        if args.tracking_uri == "http://127.0.0.1:5000":
            args.tracking_uri = last["tracking_uri"]

    if not args.run_id or not args.tb_dir:
        parser.error("--run_id and --tb_dir are required (or pass --from-last).")

    return args


def fetch_mlflow_metrics(run_id: str, tracking_uri: str) -> pd.DataFrame:
    """Fetch all metric histories from MLflow for a given run"""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception as e:
        console.print(f"[red][ERROR] Run ID not found: {e}[/red]")
        sys.exit(1)

    # Print run metadata
    console.print(f"\n[cyan]Run Info[/cyan]")
    console.print(f"  Run ID   : {run.info.run_id}")
    console.print(f"  Run Name : {run.info.run_name}")
    console.print(f"  Status   : {run.info.status}")
    console.print(f"  Tags     : {run.data.tags}")

    # Collect all metric keys
    metric_keys = list(run.data.metrics.keys())
    console.print(f"\n  Found {len(metric_keys)} metrics\n")

    # Fetch full history for each metric
    rows = []
    for key in metric_keys:
        history = client.get_metric_history(run_id, key)
        for h in history:
            rows.append({"tag": key, "step": h.step, "value": h.value})

    return pd.DataFrame(rows)


def fetch_tb_metrics(tb_dir: str) -> pd.DataFrame:
    """Parse and return scalar metrics from a TensorBoard log directory"""
    reader = SummaryReader(tb_dir, pivot=False)
    df = reader.scalars
    df.columns = [c.lower() for c in df.columns]
    if "tag" not in df.columns:
        df = df.rename(columns={"tags": "tag"})
    return df


def sanitize_metric_name(name: str) -> str:
    return name.replace(" ", "_").replace(":", "-")


def verify(tb_df: pd.DataFrame, mlflow_df: pd.DataFrame, tolerance: float):
    """Compare TensorBoard source data against MLflow uploaded data"""

    console.rule("[bold blue]Verification Start[/bold blue]")

    # Normalize tag names using the same logic as the uploader
    tb_df = tb_df.copy()
    tb_df["tag"] = tb_df["tag"].apply(sanitize_metric_name)

    tb_tags = set(tb_df["tag"].unique())
    mlflow_tags = set(mlflow_df["tag"].unique())

    # ── 1. Compare tag lists
    missing_in_mlflow = tb_tags - mlflow_tags
    extra_in_mlflow = mlflow_tags - tb_tags
    matched_tags = tb_tags & mlflow_tags

    tag_table = Table(
        title="[bold]Tag (Metric) List Comparison[/bold]", header_style="bold magenta"
    )
    tag_table.add_column("Status", style="bold", width=12)
    tag_table.add_column("Tag Name", style="cyan")

    for tag in sorted(matched_tags):
        tag_table.add_row("[green]✓ Match[/green]", tag)
    for tag in sorted(missing_in_mlflow):
        tag_table.add_row("[red]✗ Missing[/red]", tag)
    for tag in sorted(extra_in_mlflow):
        tag_table.add_row("[yellow]+ Extra[/yellow]", tag)

    console.print(tag_table)

    # ── 2. Compare data point counts
    count_table = Table(
        title="[bold]Data Point Count Comparison[/bold]", header_style="bold magenta"
    )
    count_table.add_column("Tag", style="cyan", min_width=30)
    count_table.add_column("TB Source", justify="right")
    count_table.add_column("MLflow", justify="right")
    count_table.add_column("Diff", justify="right")
    count_table.add_column("Status", justify="center")

    all_count_ok = True
    for tag in sorted(matched_tags):
        tb_count = len(tb_df[tb_df["tag"] == tag])
        mlf_count = len(mlflow_df[mlflow_df["tag"] == tag])
        diff = mlf_count - tb_count
        ok = diff == 0
        if not ok:
            all_count_ok = False
        status = "[green]✓[/green]" if ok else f"[red]✗ ({diff:+d})[/red]"
        count_table.add_row(tag, str(tb_count), str(mlf_count), str(diff), status)

    console.print(count_table)

    # ── 3. Numeric accuracy check (matched tags only)
    value_errors = []
    for tag in sorted(matched_tags):
        tb_sub = tb_df[tb_df["tag"] == tag].sort_values("step").reset_index(drop=True)
        mlf_sub = mlflow_df[mlflow_df["tag"] == tag].sort_values("step").reset_index(drop=True)

        # Skip if step counts differ
        if len(tb_sub) != len(mlf_sub):
            continue

        for i, (tb_row, mlf_row) in enumerate(zip(tb_sub.itertuples(), mlf_sub.itertuples())):
            if abs(tb_row.value - mlf_row.value) > tolerance:
                value_errors.append(
                    {
                        "tag": tag,
                        "step": tb_row.step,
                        "tb_value": tb_row.value,
                        "mlflow_value": mlf_row.value,
                        "diff": abs(tb_row.value - mlf_row.value),
                    }
                )

    # ── 4. Final result summary
    console.rule("[bold]Verification Summary[/bold]")

    checks = [
        ("Tag list fully matched", len(missing_in_mlflow) == 0 and len(extra_in_mlflow) == 0),
        ("Data point counts matched", all_count_ok),
        (f"Values within tolerance (tolerance={tolerance})", len(value_errors) == 0),
    ]

    all_pass = True
    for check_name, passed in checks:
        icon = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"  {icon}  {check_name}")
        if not passed:
            all_pass = False

    if value_errors:
        console.print(f"\n  [red]{len(value_errors)} value error(s) detected:[/red]")
        for err in value_errors[:5]:  # Show at most 5 errors
            console.print(
                f"    tag={err['tag']} step={err['step']} "
                f"TB={err['tb_value']:.6f} MLflow={err['mlflow_value']:.6f} "
                f"diff={err['diff']:.2e}"
            )

    console.print()
    if all_pass:
        console.print(
            "[bold green]✓ All checks passed! TB -> MLflow porting is accurate.[/bold green]"
        )
    else:
        console.print("[bold red]✗ Verification failed. Review the items above.[/bold red]")

    return all_pass


def run_verify(run_id: str, tb_dir: str, tracking_uri: str, tolerance: float = 1e-6) -> bool:
    """Programmatic entry point — callable from upload_tb.py for auto-verify.

    Returns True if all checks pass, False otherwise.
    """
    console.rule("[bold blue]MLflow Upload Verifier[/bold blue]")

    console.print("[yellow]Fetching MLflow metrics...[/yellow]")
    mlflow_df = fetch_mlflow_metrics(run_id, tracking_uri)

    console.print("[yellow]Parsing TensorBoard source...[/yellow]")
    tb_df = fetch_tb_metrics(tb_dir)

    return verify(tb_df, mlflow_df, tolerance)


def main():
    args = parse_args()
    run_verify(args.run_id, args.tb_dir, args.tracking_uri, args.tolerance)


if __name__ == "__main__":
    main()

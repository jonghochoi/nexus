#!/usr/bin/env python3
"""
nexus/post_upload/upload_eval.py
================================
Attaches post-hoc evaluation artifacts (mp4 rollouts, GIF previews, reports,
score JSONs) to an existing MLflow run that was created by upload_tb.py or by
Pipeline A's MLflowLogger. The run is resolved by --run_name; the directory
passed via --eval_dir lands under eval/<eval_id>/ — never under checkpoints/.

Two entry points:
    1. CLI — ``nexus-upload-eval --run_name X --eval_dir Y``
            (or ``python -m nexus.post_upload.upload_eval ...``)
    2. Python — ``from nexus.post_upload.upload_eval import upload_eval``
       Used by glue scripts that drive an external eval tool (e.g. observer)
       and want to upload its outputs in-process. The Python API is
       non-interactive (no y/n prompt) and accepts a ``metrics`` dict
       directly so the caller can flatten domain-specific metrics.json files
       however it likes — upload_eval.py itself stays generic.

Optional convenience flags:
    --run-info <path>     Read run_name / experiment / tracking_uri from a
                          ``.nexus_run.json`` sidecar (written by make_logger).
                          Explicit CLI args take precedence.
    --metrics-from <path> Auto-promote numeric scalars from a JSON file
                          (dotted-key flatten) into ``eval/<key>`` metrics.

MLflow 2.13's artifact viewer renders HTML inline but not mp4. To make
rollouts playable in the UI, an index.html is auto-generated next to the
mp4 with a <video> tag pointing at the local filename. Disable with
--no-index when the observer ships its own index.html.

Expected eval_dir layout:
    eval_outputs/<run_name>/
    ├── rollout.mp4
    ├── rollout_preview.gif
    ├── report.md
    ├── metrics.json
    └── success_rate.png
"""

import argparse
import html
import json
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from ..logger.run_info import read_run_info
from .config import DEFAULT_CONFIG_PATH, load_config
from .history import make_eval_record, print_history, save_upload

console = Console()

# Extensions that the auto-generated index.html knows how to embed.
VIDEO_EXTS = (".mp4", ".webm", ".mov")
IMAGE_EXTS = (".gif", ".png", ".jpg", ".jpeg", ".svg")
TEXT_EXTS = (".md", ".txt", ".json", ".yaml", ".yml", ".log")


# ── 1. Argument parsing ──────────────────────────────────────────────────────
def parse_args(defaults: dict):
    parser = argparse.ArgumentParser(
        description="Attach evaluation artifacts (mp4/report/scores) to an existing MLflow run"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Target MLflow run name — searched via tags.mlflow.runName "
        "(required for uploads; optional with --history)",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        help="Local directory whose contents will be attached as eval artifacts",
    )
    parser.add_argument(
        "--eval_id",
        type=str,
        default=None,
        help="Subdirectory name under eval/ (default: timestamp YYYYmmdd_HHMMSS)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=defaults["experiment"],
        help=f"Restrict the run search to this experiment (default: {defaults['experiment']})",
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=defaults["tracking_uri"],
        help=f"MLflow server URI (default: {defaults['tracking_uri']})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to JSON config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=[],
        help="Scalar eval metrics to log (e.g. success_rate=0.87 mean_return=132.4); "
        "each becomes 'eval/<key>' on the run",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=[],
        help="Tags to set on the run (e.g. observer_commit=abc123 evaluator=lee); "
        "auto-prefixed with 'eval.' if no namespace is present",
    )
    parser.add_argument(
        "--no-index",
        dest="no_index",
        action="store_true",
        help="Don't auto-generate index.html (use when the observer ships its own)",
    )
    parser.add_argument(
        "--run-info",
        dest="run_info",
        type=str,
        default=None,
        help="Path to a .nexus_run.json sidecar (or its parent directory). "
        "Fills run_name / experiment / tracking_uri when those flags are absent — "
        "explicit flags take precedence.",
    )
    parser.add_argument(
        "--metrics-from",
        dest="metrics_from",
        type=str,
        default=None,
        help="Path to a JSON file. Numeric scalars (with dotted-key flattening "
        "for nested dicts) are auto-promoted to 'eval/<key>' metrics on the run. "
        "Merged with --metrics; --metrics wins on conflict.",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="List what would be uploaded, then exit"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print recent eval uploads (~/.nexus/history.json) and exit",
    )
    return parser.parse_args()


# ── 2. Run resolution ────────────────────────────────────────────────────────
def resolve_run(client: MlflowClient, experiment: str, run_name: str) -> dict:
    """Find an existing run by run_name. Aborts if 0 or >1 matches.

    Returns a small dict with run_id and resolved experiment name — we
    re-read the experiment so eval records track the canonical value
    rather than what the user passed (the search may have spanned a
    different experiment than the default).
    """
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        console.print(
            f"[red][ERROR] Experiment not found: '{experiment}'[/red]\n"
            f"  -> Create it with the upstream upload_tb.py first, "
            f"or pass --experiment <name>."
        )
        sys.exit(1)

    # Filter on the canonical mlflow.runName tag so resumed/renamed runs
    # are matched the same way Pipeline A's _get_or_create_run does.
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=2,
    )
    if not runs:
        console.print(
            f"[red][ERROR] No run found in experiment '{experiment}' "
            f"with run_name='{run_name}'.[/red]\n"
            f"  -> Run upload_tb.py for this run first, or check the spelling."
        )
        sys.exit(1)
    if len(runs) > 1:
        console.print(
            f"[red][ERROR] Multiple runs share run_name='{run_name}' in "
            f"'{experiment}'. Disambiguate by deleting duplicates or "
            f"renaming.[/red]"
        )
        for r in runs:
            console.print(f"    • run_id={r.info.run_id}  status={r.info.status}")
        sys.exit(1)

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "experiment_id": exp.experiment_id,
        "experiment_name": exp.name,
    }


# ── 3. eval_dir scan ─────────────────────────────────────────────────────────
def scan_eval_dir(eval_dir: Path) -> list:
    """Return a sorted list of (relative_path, size_bytes) for files to upload.

    Walks eval_dir recursively and includes everything. If the user's bundle
    already contains index.html, the auto-generation step is skipped (see
    main()) so the user-shipped page wins.
    """
    if not eval_dir.exists():
        console.print(f"[red][ERROR] eval_dir not found: {eval_dir}[/red]")
        sys.exit(1)
    if not eval_dir.is_dir():
        console.print(f"[red][ERROR] eval_dir is not a directory: {eval_dir}[/red]")
        sys.exit(1)

    files = []
    for p in sorted(eval_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(eval_dir)
        files.append((rel, p.stat().st_size))

    if not files:
        console.print(f"[red][ERROR] eval_dir is empty: {eval_dir}[/red]")
        sys.exit(1)
    return files


def preview_files(eval_dir: Path, files: list):
    table = Table(title=f"[bold]Files in {eval_dir}[/bold]", header_style="bold magenta")
    table.add_column("Relative Path", style="cyan", min_width=30)
    table.add_column("Size", justify="right")
    table.add_column("Embed?", justify="center", style="dim")

    for rel, size in files:
        ext = rel.suffix.lower()
        if ext in VIDEO_EXTS:
            embed = "[green]video[/green]"
        elif ext in IMAGE_EXTS:
            embed = "[green]image[/green]"
        elif ext in TEXT_EXTS:
            embed = "link"
        else:
            embed = "link"
        table.add_row(str(rel), _fmt_size(size), embed)

    console.print(table)
    total = sum(s for _, s in files)
    console.print(f"\n[green]Total: {len(files)} files, {_fmt_size(total)}[/green]\n")


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


# ── 4. index.html synthesis ──────────────────────────────────────────────────
def _url_attr(rel_path: str) -> str:
    """URL-encode a relative artifact path for use in an HTML attribute.

    Spaces, em-dashes, and other non-ASCII filename chars must be percent-
    encoded so the browser can fetch the artifact through MLflow's
    /get-artifact endpoint. '/' is left alone to preserve subdir paths.
    The result is also HTML-attribute-safe (no quotes / angle brackets).
    """
    return html.escape(urllib.parse.quote(rel_path, safe="/"), quote=True)


def build_index_html(run_name: str, eval_id: str, files: list) -> str:
    """Render a minimal index.html that embeds videos/images and links files.

    Paths are written as relative URLs, so MLflow's HTML preview can resolve
    them as siblings within the same artifact directory.
    """
    videos = [str(rel) for rel, _ in files if rel.suffix.lower() in VIDEO_EXTS]
    images = [str(rel) for rel, _ in files if rel.suffix.lower() in IMAGE_EXTS]
    others = [str(rel) for rel, _ in files if rel.suffix.lower() not in VIDEO_EXTS + IMAGE_EXTS]

    title = html.escape(f"{run_name} — eval {eval_id}")
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{title}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "max-width:960px;margin:2em auto;padding:0 1em;color:#222}",
        "h1{font-size:1.3em;margin-bottom:0}",
        "h2{font-size:1.05em;margin-top:1.6em;border-bottom:1px solid #ddd;padding-bottom:.2em}",
        "video,img{max-width:100%;height:auto;border-radius:6px;"
        "box-shadow:0 1px 4px rgba(0,0,0,.1)}",
        "ul{padding-left:1.2em}",
        "code{background:#f4f4f4;padding:1px 4px;border-radius:3px}",
        ".meta{color:#666;font-size:.9em}",
        "</style></head><body>",
        f"<h1>{title}</h1>",
        f'<p class="meta">Generated by <code>upload_eval.py</code> '
        f"on {time.strftime('%Y-%m-%d %H:%M:%S')}.</p>",
    ]

    if videos:
        parts.append("<h2>Rollouts</h2>")
        for v in videos:
            src, label = _url_attr(v), html.escape(v)
            parts.append(
                f"<p><strong>{label}</strong></p>"
                f'<video controls preload="metadata" src="{src}"></video>'
            )

    if images:
        parts.append("<h2>Plots / Previews</h2>")
        for im in images:
            src, label = _url_attr(im), html.escape(im)
            parts.append(f'<p><strong>{label}</strong></p><img alt="{label}" src="{src}">')

    if others:
        parts.append("<h2>Reports & Files</h2><ul>")
        for o in others:
            src, label = _url_attr(o), html.escape(o)
            parts.append(f'<li><a href="{src}">{label}</a></li>')
        parts.append("</ul>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ── 5. Upload ────────────────────────────────────────────────────────────────
def parse_kv_list(items: list, label: str) -> dict:
    """Convert ['k=v', ...] to dict; warn on malformed entries."""
    out = {}
    for item in items:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            console.print(
                f"[yellow][WARN] Ignoring malformed {label}: '{item}' (expected key=value)[/yellow]"
            )
    return out


def coerce_metric(value: str) -> Optional[float]:
    """Best-effort string -> float. Returns None if not numeric."""
    try:
        return float(value)
    except ValueError:
        return None


def flatten_metrics_json(path: Path) -> dict:
    """Flatten a metrics JSON into a {dotted_key: float} dict.

    Walks nested dicts with '.' separators (e.g. ``{"a": {"b": 1}}`` →
    ``{"a.b": 1.0}``). Non-numeric leaves and lists are skipped silently —
    the upstream metrics.json may carry strings (checkpoint name, dominant
    failure mode) that aren't meaningful as MLflow scalars.
    """
    if not path.exists():
        console.print(f"[red][ERROR] --metrics-from file not found: {path}[/red]")
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red][ERROR] {path} is not valid JSON: {e}[/red]")
        sys.exit(1)
    if not isinstance(data, dict):
        console.print(f"[red][ERROR] {path} must contain a JSON object at top level[/red]")
        sys.exit(1)

    out: dict[str, float] = {}

    def walk(obj, prefix: str):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                walk(v, key)
            elif isinstance(v, bool):
                # bool is a subclass of int — keep as 0/1 metric.
                out[key] = float(v)
            elif isinstance(v, (int, float)):
                out[key] = float(v)
            # Strings, lists, None — silently skipped.

    walk(data, "")
    return out


def namespace_tags(tags: dict) -> dict:
    """Prefix bare tag keys with 'eval.' so they don't collide with run-level tags.

    Keys that already contain a '.' are left alone (e.g. mlflow.* or
    user-supplied namespaces like observer.*).
    """
    out = {}
    for k, v in tags.items():
        out[k if "." in k else f"eval.{k}"] = v
    return out


def upload_artifacts(
    client: MlflowClient,
    run_id: str,
    eval_dir: Path,
    artifact_path: str,
    files: list,
    index_html: Optional[str],
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        console=console,
    ) as progress:
        total = len(files) + (1 if index_html else 0)
        task = progress.add_task(f"Uploading {total} file(s) to {artifact_path}/", total=total)

        for rel, _ in files:
            local = eval_dir / rel
            # Preserve subdirectories under artifact_path/.
            sub = artifact_path
            if rel.parent != Path("."):
                sub = f"{artifact_path}/{rel.parent.as_posix()}"
            client.log_artifact(run_id, str(local), sub)
            progress.advance(task)

        if index_html is not None:
            # Write index.html into a tempdir so MLflow uses that exact
            # filename on the server, without mutating the user's eval_dir.
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp) / "index.html"
                tmp_path.write_text(index_html, encoding="utf-8")
                client.log_artifact(run_id, str(tmp_path), artifact_path)
            progress.advance(task)


# ── 6. Python API ────────────────────────────────────────────────────────────
def upload_eval(
    *,
    run_name: str,
    eval_dir,
    experiment: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    eval_id: Optional[str] = None,
    metrics: Optional[dict] = None,
    metrics_from=None,
    tags: Optional[dict] = None,
    generate_index: bool = True,
    dry_run: bool = False,
    confirm: bool = False,
    record_history: bool = True,
    config_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Programmatic entry point — used by glue scripts that drive an external
    eval tool (observer, custom evaluators, etc.) and want to upload its
    outputs in-process.

    ``metrics`` is a {key: numeric} dict; ``metrics_from`` is the path to a
    JSON file whose numeric scalars are auto-promoted (dotted-key flatten).
    The two are merged — explicit ``metrics`` wins on key conflict.

    ``experiment`` / ``tracking_uri`` fall back to ``~/.nexus/post_config.json``
    (or ``config_path``) when omitted, matching the CLI's behavior.

    ``confirm=False`` (default) skips the interactive y/n prompt — programmatic
    callers should leave it off; the CLI passes ``confirm=True``.

    Returns the ``eval_id`` (subdirectory under ``eval/``).
    """
    config = load_config(config_path)
    experiment = experiment or config["experiment"]
    tracking_uri = tracking_uri or config["tracking_uri"]

    if verbose:
        console.rule("[bold blue]Eval Artifact Uploader[/bold blue]")
        console.print(f"[dim]Config source: {config['source']}[/dim]")
        console.print(f"[cyan]MLflow URI:[/cyan] {tracking_uri}")
        console.print(f"[cyan]Experiment:[/cyan] {experiment}")
        console.print(f"[cyan]Run Name  :[/cyan] {run_name}\n")

    eval_dir_path = Path(eval_dir).expanduser().resolve()
    files = scan_eval_dir(eval_dir_path)
    if verbose:
        preview_files(eval_dir_path, files)

    resolved_id = eval_id or time.strftime("%Y%m%d_%H%M%S")
    artifact_path = f"eval/{resolved_id}"
    if verbose:
        console.print(f"[cyan]Will upload to:[/cyan] artifacts/{artifact_path}/\n")

    # Merge: --metrics-from JSON is the floor; explicit metrics dict wins.
    merged_metrics: dict = {}
    if metrics_from is not None:
        merged_metrics.update(flatten_metrics_json(Path(metrics_from)))
    if metrics:
        merged_metrics.update(metrics)

    namespaced = namespace_tags(dict(tags or {}))

    # Resolve the target run *before* the dry-run early-exit so misspelled
    # run_names are caught up front rather than after a real upload attempt.
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    resolved = resolve_run(client, experiment, run_name)
    run_id = resolved["run_id"]
    if verbose:
        console.print(f"[cyan]Resolved run_id:[/cyan] {run_id}\n")

    if dry_run:
        if verbose:
            console.print("[bold yellow]--dry_run mode: skipping upload.[/bold yellow]")
            if merged_metrics:
                preview = {f"eval/{k}": v for k, v in merged_metrics.items()}
                console.print(f"[cyan]Would log metrics:[/cyan] {preview}")
            if namespaced:
                console.print(f"[cyan]Would set tags:[/cyan] {namespaced}")
        return resolved_id

    if confirm:
        console.print("Upload to MLflow? [bold](y/n)[/bold]: ", end="")
        answer = input().strip().lower()
        if answer != "y":
            console.print("[yellow]Upload cancelled.[/yellow]")
            return resolved_id

    # Build index.html unless suppressed or one already exists in eval_dir.
    has_user_index = any(rel.name == "index.html" for rel, _ in files)
    index_html: Optional[str] = None
    if generate_index and not has_user_index:
        index_html = build_index_html(run_name, resolved_id, files)
        if verbose:
            console.print("[dim]Auto-generating index.html for in-UI playback.[/dim]")
    elif has_user_index and verbose:
        console.print("[dim]eval_dir already contains index.html — leaving it alone.[/dim]")

    upload_artifacts(client, run_id, eval_dir_path, artifact_path, files, index_html)

    # Log scalar eval metrics on the parent run, namespaced under eval/.
    # Step is the unix timestamp so multiple eval bundles plot in order.
    logged_metrics: dict = {}
    if merged_metrics:
        ts_step = int(time.time())
        for k, v in merged_metrics.items():
            f = v if isinstance(v, (int, float)) else coerce_metric(str(v))
            if f is None:
                console.print(f"[yellow][WARN] Skipping non-numeric metric: {k}={v}[/yellow]")
                continue
            client.log_metric(run_id, f"eval/{k}", float(f), step=ts_step)
            logged_metrics[k] = float(f)

    # Always stamp eval.last_id so consumers can find the latest bundle.
    persistent_tags = {"eval.last_id": resolved_id, **namespaced}
    for k, v in persistent_tags.items():
        client.set_tag(run_id, k, v)

    if verbose:
        console.print("\n[bold green]✓ Eval upload complete![/bold green]")
        console.print(f"  Run ID        : [yellow]{run_id}[/yellow]")
        console.print(f"  Artifact path : artifacts/{artifact_path}/")
        if index_html is not None:
            console.print(
                f"  UI playback   : open the run in MLflow → Artifacts → {artifact_path}/index.html"
            )

    if record_history:
        save_upload(
            make_eval_record(
                run_id=run_id,
                eval_dir=str(eval_dir_path),
                eval_id=resolved_id,
                experiment=resolved["experiment_name"],
                run_name=run_name,
                tracking_uri=tracking_uri,
                artifact_path=artifact_path,
                files=[str(rel) for rel, _ in files],
                metrics=logged_metrics,
                tags=namespaced,
            )
        )
    return resolved_id


# ── 7. Main (CLI shell) ──────────────────────────────────────────────────────
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
    config = load_config(_preparse_config_path())
    args = parse_args(defaults=config)

    if args.history:
        print_history(script="upload_eval")
        return

    # --run-info — fill in missing run_name / experiment / tracking_uri from
    # the .nexus_run.json sidecar. Explicit CLI args always win.
    if args.run_info:
        info = read_run_info(args.run_info)
        if not args.run_name:
            args.run_name = info["run_name"]
        # The argparse defaults below come from load_config(), so we can't
        # tell "user passed it" from "config supplied it". Fall back to
        # run_info only when it'd otherwise differ from the post_config
        # default — anything else risks silently overriding the user.
        if args.experiment == config["experiment"] and info.get("experiment"):
            args.experiment = info["experiment"]
        if args.tracking_uri == config["tracking_uri"] and info.get("tracking_uri"):
            args.tracking_uri = info["tracking_uri"]

    if not args.run_name or not args.eval_dir:
        console.print(
            "[red][ERROR] --run_name and --eval_dir are required for uploads.[/red]\n"
            "[dim]  -> Pass --run-info <path/to/.nexus_run.json> to fill run_name automatically.[/dim]"
        )
        sys.exit(1)

    upload_eval(
        run_name=args.run_name,
        eval_dir=args.eval_dir,
        experiment=args.experiment,
        tracking_uri=args.tracking_uri,
        eval_id=args.eval_id,
        metrics=parse_kv_list(args.metrics, "metric"),
        metrics_from=args.metrics_from,
        tags=parse_kv_list(args.tags, "tag"),
        generate_index=not args.no_index,
        dry_run=args.dry_run,
        confirm=True,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()

"""Upload history persistence for the nexus post-upload CLI.

Records each upload in ~/.nexus/history.json (newest first, capped at
HISTORY_LIMIT) so users can replay tag sets, re-verify a past upload,
or audit recent activity without MLflow UI round-trips.

Each record carries a `script` field — currently `"upload_tb"` or
`"upload_eval"` — so the two pipelines coexist in the same history
file. Records written before the field was introduced are treated
as `"upload_tb"` (the only producer at the time).
"""

import json
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from config import HISTORY_LIMIT, HISTORY_PATH

console = Console()

# Default script-tag for legacy records that pre-date the `script` field.
LEGACY_SCRIPT = "upload_tb"


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_history(script: Optional[str] = None) -> list:
    """Return the list of upload records (newest first). Empty on any error.

    If `script` is given, only records whose `script` field matches are
    returned. Legacy records without the field are treated as `LEGACY_SCRIPT`,
    so passing `script="upload_tb"` keeps them visible.
    """
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Best-effort: corrupt history shouldn't block uploads.
        return []
    if not isinstance(data, list):
        return []
    if script is None:
        return data
    return [r for r in data if r.get("script", LEGACY_SCRIPT) == script]


def save_upload(record: dict) -> None:
    """Prepend a record to history and truncate to HISTORY_LIMIT."""
    # Always read the *full* file so eval and tb records share one cap.
    records = load_history()
    records.insert(0, record)
    records = records[:HISTORY_LIMIT]
    _ensure_parent(HISTORY_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=2)


def last_upload(script: Optional[str] = None) -> Optional[dict]:
    """Return the most recent record (optionally restricted to one script)."""
    records = load_history(script=script)
    return records[0] if records else None


def make_record(
    run_id: str,
    tb_dir: str,
    experiment: str,
    run_name: str,
    tracking_uri: str,
    tags: dict,
    verify_ok: Optional[bool],
    script: str = LEGACY_SCRIPT,
) -> dict:
    """Construct a history record for a completed upload."""
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": script,
        "run_id": run_id,
        "tb_dir": str(Path(tb_dir).resolve()),
        "experiment": experiment,
        "run_name": run_name,
        "tracking_uri": tracking_uri,
        "tags": dict(tags),
        "verify_ok": verify_ok,
    }


def make_eval_record(
    run_id: str,
    eval_dir: str,
    eval_id: str,
    experiment: str,
    run_name: str,
    tracking_uri: str,
    artifact_path: str,
    files: list,
    metrics: dict,
    tags: dict,
) -> dict:
    """Construct a history record for an `upload_eval.py` invocation.

    Distinct from `make_record` — eval uploads attach files to an existing
    run rather than creating one, so verify_ok / tb_dir don't apply. The
    record keeps `run_name` / `run_id` / `experiment` for cross-reference.
    """
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "upload_eval",
        "run_id": run_id,
        "eval_dir": str(Path(eval_dir).resolve()),
        "eval_id": eval_id,
        "artifact_path": artifact_path,
        "files": list(files),
        "experiment": experiment,
        "run_name": run_name,
        "tracking_uri": tracking_uri,
        "metrics": dict(metrics),
        "tags": dict(tags),
    }


def print_history(script: Optional[str] = None) -> None:
    """Render recent uploads as a rich table.

    Pass `script="upload_tb"` or `"upload_eval"` to filter; default shows
    both, with a `Kind` column distinguishing them.
    """
    records = load_history(script=script)
    if not records:
        console.print("[yellow]No uploads recorded yet.[/yellow]")
        console.print(f"  History file: {HISTORY_PATH}")
        return

    table = Table(
        title=f"[bold]Recent uploads (last {len(records)})[/bold]", header_style="bold magenta"
    )
    table.add_column("When", style="cyan")
    table.add_column("Kind")
    table.add_column("Experiment")
    table.add_column("Run Name")
    table.add_column("Run ID", style="yellow")
    table.add_column("Verify / Files", justify="center")
    table.add_column("Key Tags / Metrics", style="dim")

    for r in records:
        kind = r.get("script", LEGACY_SCRIPT)
        if kind == "upload_eval":
            kind_cell = "[magenta]eval[/magenta]"
            files = r.get("files", [])
            status_cell = f"{len(files)} file(s)"
            metrics = r.get("metrics", {})
            extra = ", ".join(f"{k}={v}" for k, v in metrics.items())
        else:
            kind_cell = "[blue]tb[/blue]"
            verify = r.get("verify_ok")
            status_cell = (
                "[green]✓[/green]" if verify is True else "[red]✗[/red]" if verify is False else "-"
            )
            extra = ", ".join(
                f"{k}={r['tags'][k]}"
                for k in ("task", "researcher", "hardware")
                if k in r.get("tags", {})
            )
        table.add_row(
            r.get("ts", "?"),
            kind_cell,
            r.get("experiment", "?"),
            r.get("run_name", "?"),
            (r.get("run_id") or "")[:12],
            status_cell,
            extra,
        )

    console.print(table)
    console.print(f"\n[dim]History file: {HISTORY_PATH}[/dim]")

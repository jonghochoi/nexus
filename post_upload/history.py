"""Upload history persistence for the nexus post-upload CLI.

Records each upload in ~/.nexus/history.json (newest first, capped at
HISTORY_LIMIT) so users can replay tag sets, re-verify a past upload,
or audit recent activity without MLflow UI round-trips.
"""

import json
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from config import HISTORY_LIMIT, HISTORY_PATH

console = Console()


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_history() -> list:
    """Return the list of upload records (newest first). Empty on any error."""
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Best-effort: corrupt history shouldn't block uploads.
        return []
    return data if isinstance(data, list) else []


def save_upload(record: dict) -> None:
    """Prepend a record to history and truncate to HISTORY_LIMIT."""
    records = load_history()
    records.insert(0, record)
    records = records[:HISTORY_LIMIT]
    _ensure_parent(HISTORY_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=2)


def last_upload() -> Optional[dict]:
    """Return the most recent record, or None if history is empty."""
    records = load_history()
    return records[0] if records else None


def make_record(
    run_id: str,
    tb_dir: str,
    experiment: str,
    run_name: str,
    tracking_uri: str,
    tags: dict,
    verify_ok: Optional[bool],
) -> dict:
    """Construct a history record for a completed upload."""
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "tb_dir": str(Path(tb_dir).resolve()),
        "experiment": experiment,
        "run_name": run_name,
        "tracking_uri": tracking_uri,
        "tags": dict(tags),
        "verify_ok": verify_ok,
    }


def print_history() -> None:
    """Render recent uploads as a rich table."""
    records = load_history()
    if not records:
        console.print("[yellow]No uploads recorded yet.[/yellow]")
        console.print(f"  History file: {HISTORY_PATH}")
        return

    table = Table(
        title=f"[bold]Recent uploads (last {len(records)})[/bold]",
        header_style="bold magenta",
    )
    table.add_column("When", style="cyan")
    table.add_column("Experiment")
    table.add_column("Run Name")
    table.add_column("Run ID", style="yellow")
    table.add_column("Verify", justify="center")
    table.add_column("Key Tags", style="dim")

    for r in records:
        verify = r.get("verify_ok")
        verify_cell = (
            "[green]✓[/green]" if verify is True
            else "[red]✗[/red]" if verify is False
            else "-"
        )
        key_tags = ", ".join(
            f"{k}={r['tags'][k]}"
            for k in ("seed", "task", "researcher")
            if k in r.get("tags", {})
        )
        table.add_row(
            r.get("ts", "?"),
            r.get("experiment", "?"),
            r.get("run_name", "?"),
            (r.get("run_id") or "")[:12],
            verify_cell,
            key_tags,
        )

    console.print(table)
    console.print(f"\n[dim]History file: {HISTORY_PATH}[/dim]")

"""
nexus/logger/run_info.py
========================
Lightweight read/write helpers for ``.nexus_run.json`` — a sidecar file
written by ``make_logger()`` into the trainer's output directory so a
downstream eval / upload step can recover the run identity (``run_name``,
``experiment``, ``tracking_uri``) without re-deriving it from training
configs.

Schema (versioned):

    {
      "schema_version":       1,
      "run_name":             "ppo_v17_seed3",
      "run_id":               "<local mlflow uuid>",
      "experiment":           "robot_hand_rl",
      "tracking_uri":         "http://127.0.0.1:5100",
      "central_tracking_uri": "http://nexus-server:5000",
      "created_at":           "2026-05-06T12:34:56Z"
    }

``central_tracking_uri`` is optional — null/absent when ``make_logger()`` was
called without a ``central_tracking_uri`` argument. Readers must handle both
cases. The required keys are: ``run_name``, ``run_id``, ``experiment``,
``tracking_uri`` — adding more optional fields does not require a version
bump; reserve a bump for a genuinely incompatible change (rename, type change,
removed required field).

Design notes:
  - Atomic write — temp file + ``os.replace`` so a concurrent reader never
    sees a half-written JSON document.
  - Read accepts either the file path or the directory containing it, so
    glue scripts can be written as ``read_run_info(output_dir)`` without
    knowing the filename.
  - No mlflow import here — keep this importable from the user's eval
    glue even when mlflow itself is not installed in that environment.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Union

from ..brand import log as brand_log

RUN_INFO_FILENAME = ".nexus_run.json"
SCHEMA_VERSION = 1


# ── Public interface ─────────────────────────────────────────────────────────


def write_run_info(
    out_dir: Union[str, Path],
    *,
    run_name: str,
    run_id: str,
    experiment: str,
    tracking_uri: str,
    central_tracking_uri: Optional[str] = None,
) -> Path:
    """Write ``out_dir/.nexus_run.json`` atomically.

    ``tracking_uri`` is the server the trainer is logging to (typically the
    GPU-node-local ``http://127.0.0.1:5100``). ``central_tracking_uri`` is the
    NEXUS central MLflow that ``scheduled_sync`` ships data to — recording it
    in the sidecar lets downstream eval / upload glue land artifacts directly
    on central without reading an external config. Pass ``None`` (or omit)
    when the trainer doesn't know the central URI; readers will handle the
    absence explicitly.

    Returns the path that was written. Silently no-ops (returns the would-be
    path) when ``out_dir`` is falsy — callers don't need to guard the call.
    """
    if not out_dir:
        return Path()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / RUN_INFO_FILENAME

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "run_id": run_id,
        "experiment": experiment,
        "tracking_uri": tracking_uri,
        "central_tracking_uri": central_tracking_uri,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Atomic write — temp file in the same directory, then os.replace.
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, target)
    return target


def read_run_info(path_or_dir: Union[str, Path]) -> dict:
    """Read a ``.nexus_run.json`` file given its path or its parent directory.

    Raises ``FileNotFoundError`` if no file is found, ``ValueError`` if the
    payload is unparseable or missing required keys. Forward-compatible: an
    unknown ``schema_version`` is accepted with a warning print so future
    additions don't break old readers.
    """
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / RUN_INFO_FILENAME
    if not p.exists():
        raise FileNotFoundError(
            f"{RUN_INFO_FILENAME} not found at {p} — "
            f"was the run started with make_logger(tb_dir=...)?"
        )

    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{p} is not valid JSON: {e}") from e

    required = ("run_name", "run_id", "experiment", "tracking_uri")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{p} is missing required keys: {missing}")

    version = data.get("schema_version")
    if version is not None and version > SCHEMA_VERSION:
        # Forward-compat — newer writer, older reader. Don't fail; the
        # required keys above are the contract.
        print(
            brand_log(
                f"run_info {p} has schema_version={version}, "
                f"this reader supports {SCHEMA_VERSION} — extra keys ignored.",
                "warn",
            )
        )
    return data

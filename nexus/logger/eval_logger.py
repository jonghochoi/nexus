"""
nexus/logger/eval_logger.py
===========================
EvalLogger: attaches post-training evaluation artifacts to an existing MLflow run.

Uploads a local directory of eval outputs under ``eval/<eval_id>/`` on a run
that was previously created by ``MLflowLogger`` / ``make_logger()`` or by
Pipeline B's ``upload_tb.py``. The run is resolved by ``run_name`` via
``tags.mlflow.runName`` — the same identity key used throughout the rest of
nexus.

Scope — the anchor feature is **inline rollout video playback**. MLflow 2.13's
artifact viewer renders HTML inline but not ``.mp4``, so an auto-generated
``index.html`` next to any video file embeds it in a ``<video controls>`` tag.
Other files (images, reports, JSON, …) are uploaded verbatim and previewed /
downloaded one-by-one from the MLflow Artifacts pane — they are not embedded
in the auto index. Future inline rendering for additional file types may be
added later; until then, ship your own ``index.html`` if you need it.

Typical usage from an external training / eval repo:

    # Training produces .nexus_run.json via make_logger(tb_dir=output_dir).
    # Eval step picks it up and uploads results in one call:

    from nexus.logger.eval_logger import EvalLogger

    ev = EvalLogger.from_run_info(output_dir)           # reads .nexus_run.json
    eval_id = ev.upload(
        eval_dir=output_dir / "eval",
        metrics={"success_rate": 0.87, "mean_return": 132.4},
        tags={"observer_commit": "abc123"},
    )

Or with explicit params when the sidecar is not available:

    ev = EvalLogger(
        run_name="ppo_v3_seed0",
        tracking_uri="http://nexus-server:5000",
        experiment="robot_hand_rl",
    )
    ev.upload(eval_dir="./eval_out/ppo_v3_seed0/")

Expected eval_dir layout (flat or nested — everything is walked recursively):

    eval_outputs/<run_name>/
    ├── rollout.mp4            ← embedded in auto-generated index.html
    ├── rollout_preview.gif    ← uploaded as-is, previewable from Artifacts pane
    ├── report.md              ← uploaded as-is, downloadable
    ├── metrics.json           ← uploaded as-is; pass via metrics_from= for scalars
    └── success_rate.png       ← uploaded as-is, previewable from Artifacts pane
"""

from __future__ import annotations

import base64
import html
import json
import logging
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Optional, Union

import mlflow
from mlflow.tracking import MlflowClient

from ..brand import CYAN, DIM, RESET
from ..brand import log as brand_log
from ..brand import rule as brand_rule
from .run_info import read_run_info

_log = logging.getLogger(__name__)

# Extensions that the auto-generated index.html embeds inline as <video>.
# Other file types are uploaded as-is — no inline embedding.
_VIDEO_EXTS = (".mp4", ".webm", ".mov")

# MIME types matching what ffmpeg's libx264 default writes for each container.
_VIDEO_MIME = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}

# Cap on raw video size for inline base64 embedding. MLflow's HTML preview
# iframe breaks relative URLs to sibling artifacts, so videos are embedded
# directly into `<video src="data:...">` — at the cost of ~1.37× base64
# overhead and the whole HTML having to download before playback. Above
# this ceiling the index falls back to a plain download link; the mp4 is
# uploaded as a sibling artifact regardless.
_MAX_DATA_URI_BYTES = 30 * 1024 * 1024


# ── Module-level helpers ──────────────────────────────────────────────────────


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def _url_attr(rel_path: str) -> str:
    """URL-encode a relative artifact path for use in an HTML attribute.

    Spaces and non-ASCII filename characters are percent-encoded so the browser
    can fetch the artifact through MLflow's /get-artifact endpoint. '/' is left
    alone to preserve subdir paths. The result is also HTML-attribute-safe.
    """
    return html.escape(urllib.parse.quote(rel_path, safe="/"), quote=True)


def flatten_metrics_json(path: Union[str, Path]) -> dict:
    """Flatten a metrics JSON file into a ``{dotted_key: float}`` dict.

    Walks nested dicts with '.' separators — e.g. ``{"a": {"b": 1}}`` becomes
    ``{"a.b": 1.0}``. Non-numeric leaves and lists are skipped silently;
    booleans are kept as 0/1.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"metrics_from file not found: {p}")
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{p} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a JSON object at top level")

    out: dict[str, float] = {}

    def _walk(obj: dict, prefix: str) -> None:
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                _walk(v, key)
            elif isinstance(v, bool):
                out[key] = float(v)
            elif isinstance(v, (int, float)):
                out[key] = float(v)

    _walk(data, "")
    return out


def namespace_tags(tags: dict) -> dict:
    """Prefix bare tag keys with ``'eval.'`` to avoid collisions with run-level tags.

    Keys that already contain a '.' (e.g. ``mlflow.*``, ``observer.*``) are
    passed through unchanged.
    """
    return {k if "." in k else f"eval.{k}": v for k, v in tags.items()}


# ── EvalLogger ────────────────────────────────────────────────────────────────


class EvalLogger:
    """Attaches post-training evaluation artifacts to an existing MLflow run.

    Construct directly with explicit params or via ``from_run_info()`` to read
    run identity from the ``.nexus_run.json`` sidecar written by ``make_logger()``.

    The ``upload()`` method is the main entry point — it can be called multiple
    times on the same instance to attach successive eval bundles; each bundle
    lands in its own ``eval/<eval_id>/`` subdir so they never collide.
    """

    def __init__(
        self, run_name: str, tracking_uri: str, experiment: str, *, verbose: bool = True
    ) -> None:
        """
        Parameters
        ----------
        run_name      — MLflow run name (matched via ``tags.mlflow.runName``).
        tracking_uri  — MLflow server URI (e.g. ``http://nexus-server:5000``).
        experiment    — Experiment name that owns the run.
        verbose       — Print brand-styled progress output (default True). Set
                        False for silent in-process use.
        """
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self._verbose = verbose

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_run_info(
        cls,
        path_or_dir: Union[str, Path],
        *,
        tracking_uri: Optional[str] = None,
        target: str = "central",
        verbose: bool = True,
    ) -> "EvalLogger":
        """Construct from a ``.nexus_run.json`` sidecar (or its parent directory).

        Resolution order for the MLflow URI eval artifacts will land on:

        1. Explicit ``tracking_uri`` argument — wins unconditionally. Use this
           when the sidecar's URIs are stale or you want a one-off override.
        2. ``target="central"`` (default) → sidecar's ``central_tracking_uri``.
           This is the recommended path in a NEXUS deployment — eval results
           land on the team-shared central MLflow with no waiting for the next
           ``scheduled_sync`` cycle, and large mp4/gif bundles don't bloat the
           sync queue. Raises ``ValueError`` if the sidecar has no
           ``central_tracking_uri`` (i.e. ``make_logger()`` was called without
           ``central_tracking_uri=``); the error message points to the
           migration path.
        3. ``target="local"`` → sidecar's ``tracking_uri`` (the GPU-node-local
           relay the trainer logged to). Use this for debugging an in-progress
           training run before it has been synced to central.

        Run identity is always resolved by ``run_name`` via
        ``tags.mlflow.runName`` — the local and central run UUIDs differ but
        the run name is identical on both servers, so the local-only ``run_id``
        in the sidecar is not consulted by this code path.
        """
        info = read_run_info(path_or_dir)

        if tracking_uri is not None:
            resolved_uri = tracking_uri
        elif target == "central":
            central = info.get("central_tracking_uri")
            if not central:
                raise ValueError(
                    f"{path_or_dir}/.nexus_run.json has no central_tracking_uri — "
                    "either upgrade the trainer to pass make_logger(central_tracking_uri=...), "
                    'pass tracking_uri="http://<central>:5000" explicitly to from_run_info(), '
                    'or call from_run_info(..., target="local") to use the sidecar\'s '
                    "local tracking_uri."
                )
            resolved_uri = central
        elif target == "local":
            resolved_uri = info["tracking_uri"]
        else:
            raise ValueError(f"target must be 'central' or 'local', got: {target!r}")

        return cls(
            run_name=info["run_name"],
            tracking_uri=resolved_uri,
            experiment=info["experiment"],
            verbose=verbose,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def upload(
        self,
        eval_dir: Union[str, Path],
        *,
        eval_id: Optional[str] = None,
        metrics: Optional[dict] = None,
        metrics_from: Optional[Union[str, Path]] = None,
        tags: Optional[dict] = None,
        generate_index: bool = True,
        dry_run: bool = False,
    ) -> str:
        """Upload the contents of ``eval_dir`` as artifacts on the target run.

        Parameters
        ----------
        eval_dir        — Local directory to upload recursively.
        eval_id         — Subdirectory name under ``eval/`` (default: timestamp
                          ``YYYYmmdd_HHMMSS``). Use a stable name to overwrite a
                          previous bundle or leave as default to append a new one.
        metrics         — ``{key: numeric}`` dict of scalar eval metrics. Each
                          key is logged as ``eval/<key>`` on the run; step is the
                          Unix timestamp so multiple bundles plot in order.
        metrics_from    — Path to a JSON file whose numeric scalars are auto-
                          promoted (dotted-key flatten for nested dicts). Merged
                          with ``metrics``; the explicit dict wins on conflict.
        tags            — Extra tags to set on the run. Bare keys are prefixed
                          with ``'eval.'`` automatically.
        generate_index  — Auto-generate ``index.html`` embedding any video file
                          (``.mp4`` / ``.webm`` / ``.mov``) found in ``eval_dir``
                          so rollouts play in-browser from the Artifacts pane
                          (default True). Other file types are uploaded as-is
                          and not embedded; if no video is present, a short
                          placeholder page is generated instead. Set False if
                          your eval tool already ships its own index page.
        dry_run         — List what would be uploaded and return without touching
                          MLflow.

        Returns
        -------
        str — the ``eval_id`` used (useful when letting it default to a timestamp).
        """
        if self._verbose:
            print(brand_rule("Eval Artifact Upload"))
            print(f"{CYAN}MLflow URI :{RESET} {self.tracking_uri}")
            print(f"{CYAN}Experiment :{RESET} {self.experiment}")
            print(f"{CYAN}Run name   :{RESET} {self.run_name}\n")

        eval_dir_path = Path(eval_dir).expanduser().resolve()
        files = self._scan_dir(eval_dir_path)
        if self._verbose:
            self._preview_files(eval_dir_path, files)

        resolved_id = eval_id or time.strftime("%Y%m%d_%H%M%S")
        artifact_path = f"eval/{resolved_id}"
        if self._verbose:
            print(f"{CYAN}Artifact path:{RESET} {artifact_path}/\n")

        # Merge metrics_from (floor) with explicit metrics dict (wins on conflict).
        merged_metrics: dict = {}
        if metrics_from is not None:
            merged_metrics.update(flatten_metrics_json(metrics_from))
        if metrics:
            merged_metrics.update(metrics)

        namespaced_tags = namespace_tags(dict(tags or {}))

        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        run_id = self._resolve_run(client)
        if self._verbose:
            print(f"{CYAN}Resolved run_id:{RESET} {run_id}\n")

        if dry_run:
            if self._verbose:
                print(brand_log("dry_run=True — skipping upload.", "warn"))
                if merged_metrics:
                    preview = {f"eval/{k}": v for k, v in merged_metrics.items()}
                    print(f"{CYAN}Would log metrics:{RESET} {preview}")
                if namespaced_tags:
                    print(f"{CYAN}Would set tags:{RESET} {namespaced_tags}")
            return resolved_id

        # Build index.html unless suppressed or the eval_dir already ships one.
        has_user_index = any(rel.name == "index.html" for rel, _ in files)
        index_html: Optional[str] = None
        if generate_index and not has_user_index:
            index_html = self._build_index_html(resolved_id, eval_dir_path, files)
            if self._verbose:
                print(f"{DIM}Auto-generating index.html for in-UI playback.{RESET}")
        elif has_user_index and self._verbose:
            print(f"{DIM}eval_dir already contains index.html — skipping auto-gen.{RESET}")

        self._upload_artifacts(client, run_id, eval_dir_path, artifact_path, files, index_html)

        # Log scalar metrics namespaced under eval/<key>; step = unix timestamp
        # so multiple bundles on the same run appear in chronological order.
        if merged_metrics:
            ts_step = int(time.time())
            for k, v in merged_metrics.items():
                try:
                    f_val = float(v)
                except (TypeError, ValueError):
                    _log.warning("EvalLogger: skipping non-numeric metric %s=%r", k, v)
                    continue
                client.log_metric(run_id, f"eval/{k}", f_val, step=ts_step)

        # Always stamp eval.last_id so consumers can find the latest bundle.
        persistent_tags = {"eval.last_id": resolved_id, **namespaced_tags}
        for k, v in persistent_tags.items():
            client.set_tag(run_id, k, v)

        if self._verbose:
            print()
            print(brand_log("Eval upload complete!", "ok"))
            print(f"  Run ID        : {run_id}")
            print(f"  Artifact path : artifacts/{artifact_path}/")
            if index_html is not None:
                print(f"  UI playback   : open run → Artifacts → {artifact_path}/index.html")

        return resolved_id

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_run(self, client: MlflowClient) -> str:
        """Return the run_id for ``self.run_name`` within ``self.experiment``.

        Raises ``ValueError`` if the experiment is missing, the run is not
        found, or more than one run shares the name — so callers can catch and
        handle the error rather than the process being hard-killed by sys.exit().
        """
        exp = client.get_experiment_by_name(self.experiment)
        if exp is None:
            raise ValueError(
                f"Experiment not found: '{self.experiment}'. "
                f"Create it via upload_tb.py or pass the correct experiment name."
            )

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{self.run_name}'",
            max_results=2,
        )
        if not runs:
            raise ValueError(
                f"No run found in '{self.experiment}' with run_name='{self.run_name}'."
            )
        if len(runs) > 1:
            ids = ", ".join(r.info.run_id for r in runs)
            raise ValueError(
                f"Multiple runs share run_name='{self.run_name}' in '{self.experiment}'. "
                f"Disambiguate by deleting duplicates. run_ids: {ids}"
            )

        return runs[0].info.run_id

    def _scan_dir(self, eval_dir: Path) -> list:
        """Return sorted ``[(relative_path, size_bytes), ...]`` for all files."""
        if not eval_dir.exists():
            raise FileNotFoundError(f"eval_dir not found: {eval_dir}")
        if not eval_dir.is_dir():
            raise ValueError(f"eval_dir is not a directory: {eval_dir}")

        files = [
            (p.relative_to(eval_dir), p.stat().st_size)
            for p in sorted(eval_dir.rglob("*"))
            if p.is_file()
        ]
        if not files:
            raise ValueError(f"eval_dir is empty: {eval_dir}")
        return files

    def _preview_files(self, eval_dir: Path, files: list) -> None:
        print(f"\n{CYAN}Files in {eval_dir}{RESET}")
        print(f"  {'Relative Path':<40} {'Size':>10}  Inline?")
        print(f"  {'-' * 40} {'-' * 10}  {'-' * 7}")
        for rel, size in files:
            inline = "video" if rel.suffix.lower() in _VIDEO_EXTS else "-"
            print(f"  {str(rel):<40} {_fmt_size(size):>10}  {inline}")

        total = sum(s for _, s in files)
        print(brand_log(f"Total: {len(files)} files, {_fmt_size(total)}", "ok"))
        print()

    def _build_index_html(self, eval_id: str, eval_dir: Path, files: list) -> str:
        """Render an index.html that embeds rollout videos inline as base64 data URIs.

        Only ``.mp4`` / ``.webm`` / ``.mov`` files are embedded — relative URLs
        to sibling artifacts don't resolve in MLflow's HTML preview iframe, so
        the bytes are inlined into ``<video src="data:...">``. Videos over
        ``_MAX_DATA_URI_BYTES`` fall back to a download link. Other files in
        the bundle are uploaded as-is and viewed individually from the
        Artifacts pane — they are not referenced here.
        """
        videos = [(rel, size) for rel, size in files if rel.suffix.lower() in _VIDEO_EXTS]

        title = html.escape(f"{self.run_name} — eval {eval_id}")
        parts = [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>{title}</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
            "max-width:960px;margin:2em auto;padding:0 1em;color:#222}",
            "h1{font-size:1.3em;margin-bottom:0}",
            "h2{font-size:1.05em;margin-top:1.6em;border-bottom:1px solid #ddd;"
            "padding-bottom:.2em}",
            "video{max-width:100%;height:auto;border-radius:6px;"
            "box-shadow:0 1px 4px rgba(0,0,0,.1)}",
            "code{background:#f4f4f4;padding:1px 4px;border-radius:3px}",
            ".meta{color:#666;font-size:.9em}",
            "</style></head><body>",
            f"<h1>{title}</h1>",
            f'<p class="meta">Generated by <code>nexus.logger.EvalLogger</code> '
            f"on {time.strftime('%Y-%m-%d %H:%M:%S')}.</p>",
        ]

        if videos:
            parts.append("<h2>Rollouts</h2>")
            for rel, size in videos:
                label = html.escape(str(rel))
                if size > _MAX_DATA_URI_BYTES:
                    download_href = _url_attr(str(rel))
                    parts.append(
                        f"<p><strong>{label}</strong> "
                        f'<span class="meta">({_fmt_size(size)} — too large to inline, '
                        f'<a href="{download_href}" download>download</a> and play locally)</span></p>'
                    )
                    continue
                mime = _VIDEO_MIME.get(rel.suffix.lower(), "video/mp4")
                b64 = base64.b64encode((eval_dir / rel).read_bytes()).decode("ascii")
                parts.append(
                    f"<p><strong>{label}</strong> "
                    f'<span class="meta">({_fmt_size(size)})</span></p>'
                    f'<video controls preload="metadata" '
                    f'src="data:{mime};base64,{b64}"></video>'
                )
        else:
            parts.append(
                '<p class="meta">No video artifacts in this bundle — '
                "browse the Artifacts pane to view individual files.</p>"
            )

        parts.append("</body></html>")
        return "\n".join(parts)

    def _upload_artifacts(
        self,
        client: MlflowClient,
        run_id: str,
        eval_dir: Path,
        artifact_path: str,
        files: list,
        index_html: Optional[str],
    ) -> None:
        total = len(files) + (1 if index_html else 0)

        if self._verbose:
            print(f"{CYAN}Uploading {total} file(s) to {artifact_path}/{RESET}")

        i = 0
        for rel, _ in files:
            local = eval_dir / rel
            # Preserve subdirectories within artifact_path/.
            sub = (
                artifact_path
                if rel.parent == Path(".")
                else f"{artifact_path}/{rel.parent.as_posix()}"
            )
            client.log_artifact(run_id, str(local), sub)
            i += 1
            if self._verbose:
                print(f"  [{i}/{total}] {rel}")

        if index_html is not None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp) / "index.html"
                tmp_path.write_text(index_html, encoding="utf-8")
                client.log_artifact(run_id, str(tmp_path), artifact_path)
            i += 1
            if self._verbose:
                print(f"  [{i}/{total}] index.html (auto-generated)")

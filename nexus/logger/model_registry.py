"""
nexus/logger/model_registry.py
==============================
ModelRegistry: utilities for MLflow Model Registry operations.

Provides sim-to-real version tracking, stage management, and
production model queries for robot policy deployment workflows.
"""

from __future__ import annotations

from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


class ModelRegistry:
    """Utilities for querying and managing MLflow Model Registry entries.

    Usage:
        registry = ModelRegistry(tracking_uri=...)
        prod = registry.get_production_model("shadow_hand_ppo")
        registry.set_sim_to_real_link("shadow_hand_ppo", version="3", sim_run_id="abc123")
    """

    def __init__(self, tracking_uri: str = "http://127.0.0.1:5100"):
        self._tracking_uri = tracking_uri
        # Pin the *global* tracking URI as well as the client's, so that
        # MLflow operations that resolve proxy artifact URIs (e.g.
        # `list_artifacts`, `register_model`) — which consult
        # `mlflow.get_tracking_uri()` rather than the client — do not fall
        # back to the default `file://` scheme.
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)

    def get_production_model(self, model_name: str) -> Optional[dict]:
        """Return info about the current Production version, or None if none exists."""
        versions = self._client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        mv = versions[0]
        return {
            "version": mv.version,
            "run_id": mv.run_id,
            "description": mv.description,
            "sim_run_id": mv.tags.get("nexus.sim_run_id"),
            "creation_timestamp": mv.creation_timestamp,
        }

    def list_versions(self, model_name: str) -> list[dict]:
        """Return all versions with stage, run_id, and registration timestamp."""
        versions = self._client.search_model_versions(f"name='{model_name}'")
        return [
            {
                "version": mv.version,
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "description": mv.description,
                "creation_timestamp": mv.creation_timestamp,
                "sim_run_id": mv.tags.get("nexus.sim_run_id"),
            }
            for mv in versions
        ]

    def set_sim_to_real_link(self, model_name: str, version: str, sim_run_id: str) -> None:
        """Tag a model version with the sim run ID for traceability."""
        self._client.set_model_version_tag(model_name, version, "nexus.sim_run_id", sim_run_id)

    def archive_old_production(self, model_name: str) -> None:
        """Move all current Production versions to Archived stage."""
        versions = self._client.get_latest_versions(model_name, stages=["Production"])
        for mv in versions:
            self._client.transition_model_version_stage(model_name, mv.version, stage="Archived")

    # ── Post-hoc registration ────────────────────────────────────────────────

    def resolve_checkpoint_source(self, experiment: str, run_name: str, kind: str = "best") -> dict:
        """Resolve a (experiment, run_name, kind) triple to a Model Registry
        source URI. Used as the pre-flight step for register_from_run_name
        and exposed publicly so callers (e.g. CLI dry-run) can validate
        without registering.

        Returns: {"run_id", "source"} on success. Raises ValueError /
        FileNotFoundError with actionable messages on miss.
        """
        if kind not in ("best", "last"):
            raise ValueError(f"kind must be 'best' or 'last', got: {kind!r}")

        exp = self._client.get_experiment_by_name(experiment)
        if exp is None:
            raise ValueError(f"Experiment not found on {self._tracking_uri}: {experiment!r}")

        runs = self._client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(
                f"Run not found on {self._tracking_uri}: experiment={experiment!r}, "
                f"run_name={run_name!r}. If this is a GPU-side run, verify scheduled_sync "
                f"has uploaded it to central."
            )
        run_id = runs[0].info.run_id

        artifact_path = f"checkpoints/{kind}.pth"
        existing = {a.path for a in self._client.list_artifacts(run_id, "checkpoints")}
        if artifact_path not in existing:
            raise FileNotFoundError(
                f"Run {run_name!r} has no {artifact_path!r} artifact. "
                f"Confirm the trainer called log_checkpoint(kind={kind!r}) and that "
                f"sync has propagated the artifact."
            )

        return {"run_id": run_id, "source": f"runs:/{run_id}/{artifact_path}"}

    def register_from_run_name(
        self,
        experiment: str,
        run_name: str,
        model_name: str,
        kind: str = "best",
        description: Optional[str] = None,
        stage: Optional[str] = None,
        archive_existing_production: bool = False,
    ) -> dict:
        """Register a checkpoint artifact from an existing run, identified by run_name.

        Intended for the post-training workflow: a user evaluates several
        runs, picks the one(s) worth promoting, and registers the chosen
        run's `checkpoints/<kind>.pth` against the *configured* tracking
        server (typically central MLflow). Run identity is `run_name`,
        consistent with the rest of NEXUS.

        kind must be 'best' or 'last'. stage, when given, transitions the
        new version. archive_existing_production only applies when
        stage='Production' — moves any existing Production versions to
        Archived first so there is exactly one Production version.

        Returns: {"version", "run_id", "source", "stage"}.
        """
        resolved = self.resolve_checkpoint_source(experiment, run_name, kind)
        run_id = resolved["run_id"]
        source = resolved["source"]

        if stage == "Production" and archive_existing_production:
            self.archive_old_production(model_name)

        mv = mlflow.register_model(source, model_name)

        if description:
            self._client.update_model_version(model_name, mv.version, description=description)

        # Stamp source run_name on the version so operators can correlate the
        # registry entry back to the run without dereferencing run_id.
        self._client.set_model_version_tag(model_name, mv.version, "nexus.sourceRunName", run_name)

        if stage is not None:
            self._client.transition_model_version_stage(model_name, mv.version, stage)
            current_stage = stage
        else:
            current_stage = mv.current_stage

        return {"version": mv.version, "run_id": run_id, "source": source, "stage": current_stage}

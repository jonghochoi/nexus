"""
logger/model_registry.py
=========================
ModelRegistry: utilities for MLflow Model Registry operations.

Provides sim-to-real version tracking, stage management, and
production model queries for robot policy deployment workflows.
"""

from __future__ import annotations

from typing import Optional

from mlflow.tracking import MlflowClient


class ModelRegistry:
    """Utilities for querying and managing MLflow Model Registry entries.

    Usage:
        registry = ModelRegistry(tracking_uri=...)
        prod = registry.get_production_model("shadow_hand_ppo")
        registry.set_sim_to_real_link("shadow_hand_ppo", version="3", sim_run_id="abc123")
    """

    def __init__(self, tracking_uri: str = "http://127.0.0.1:5100"):
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
            self._client.transition_model_version_stage(
                model_name, mv.version, stage="Archived"
            )

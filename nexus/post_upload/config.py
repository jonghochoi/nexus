"""Config loader for the nexus post-upload CLI.

Reads ~/.nexus/post_config.json (or a custom path) and returns a merged dict
with defaults for tracking_uri, experiment, and tags.
Command-line flags override these defaults.
"""

import json
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_PATH = Path.home() / ".nexus" / "post_config.json"
LEGACY_CONFIG_PATH = Path.home() / ".nexus" / "config.json"
HISTORY_PATH = Path.home() / ".nexus" / "history.json"
HISTORY_LIMIT = 20

# Team-wide fixed values. These mirror the current NEXUS deployment so the
# CLI works with zero setup; override per user via ~/.nexus/post_config.json.
BUILTIN_DEFAULTS = {
    "tracking_uri": "http://127.0.0.1:5000",
    "experiment": "robot_hand_rl",
    "tags": {"hardware": "robot_22dof"},
}

# experiment is auto-injected from the --experiment argument in upload_tb.py.
_BASE_REQUIRED = ("experiment",)


def required_tags(experiment: str) -> tuple:
    """Return the tuple of required tags for a given experiment."""
    return _BASE_REQUIRED


# Back-compat constant for callers that don't care about experiment context.
REQUIRED_TAGS = _BASE_REQUIRED


def load_config(path: Optional[str] = None) -> dict:
    """Load config from JSON, merged on top of BUILTIN_DEFAULTS.

    Returns a dict with keys: tracking_uri (str), experiment (str),
    tags (dict[str, str]), source (str — path or '<builtin>').
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    merged = {
        "tracking_uri": BUILTIN_DEFAULTS["tracking_uri"],
        "experiment": BUILTIN_DEFAULTS["experiment"],
        "tags": dict(BUILTIN_DEFAULTS["tags"]),
        "source": "<builtin>",
    }

    if not config_path.exists():
        # One-time migration nudge — the file was renamed from the legacy path.
        # Print once and fall through to defaults so the user can act on the message.
        if path is None and LEGACY_CONFIG_PATH.exists():
            print(
                f"[WARN] Found legacy config at {LEGACY_CONFIG_PATH}. "
                f"Rename it to {DEFAULT_CONFIG_PATH} (one-time):\n"
                f"       mv {LEGACY_CONFIG_PATH} {DEFAULT_CONFIG_PATH}",
                flush=True,
            )
        return merged

    with open(config_path) as f:
        try:
            user = json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(f"[ERROR] {config_path} is not valid JSON: {e}")

    if not isinstance(user, dict):
        raise SystemExit(f"[ERROR] {config_path} must contain a JSON object")

    if "tracking_uri" in user:
        merged["tracking_uri"] = str(user["tracking_uri"])
    if "experiment" in user:
        merged["experiment"] = str(user["experiment"])
    if "tags" in user:
        if not isinstance(user["tags"], dict):
            raise SystemExit(f"[ERROR] 'tags' in {config_path} must be an object")
        merged["tags"].update({str(k): str(v) for k, v in user["tags"].items()})

    merged["source"] = str(config_path)
    return merged

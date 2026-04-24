"""
logger/git_utils.py
===================
Utilities for capturing git repository state at training time.

Called automatically by MLflowLogger.__init__() when track_git=True (default).
Produces two MLflow tags on every run:

  git_commit   — full SHA of HEAD (e.g. "54696cb326bb0aedc7b1d51e766a1036a227e568")
  git_dirty    — "true" if the working tree has uncommitted changes, else "false"

When git_dirty is "true", the full `git diff HEAD` output is also uploaded as
an artifact at artifacts/git/git_patch.diff so the exact state can be restored
with `git apply git_patch.diff` on top of the recorded commit.

All functions degrade gracefully — if git is not installed or the training
directory is not inside a git repo, they return empty dicts / None without
raising. Disable entirely with track_git=False on MLflowLogger.
"""
from __future__ import annotations

import subprocess
from typing import Optional


def _run_git(cmd: list[str], cwd: Optional[str] = None) -> str:
    """Run a git command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def get_git_info(repo_path: Optional[str] = None) -> dict:
    """Return git_commit and git_dirty tags for the current repo.

    Returns an empty dict if the directory is not a git repo or git is unavailable.
    """
    commit = _run_git(["git", "rev-parse", "HEAD"], cwd=repo_path)
    if not commit:
        return {}

    status = _run_git(["git", "status", "--porcelain"], cwd=repo_path)
    dirty = bool(status)

    return {
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
    }


def get_git_patch(repo_path: Optional[str] = None) -> Optional[str]:
    """Return `git diff HEAD` output if the working tree is dirty, else None."""
    patch = _run_git(["git", "diff", "HEAD"], cwd=repo_path)
    return patch if patch else None

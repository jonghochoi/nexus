# ─────────────────────────────────────────────────────────────────────────────
# play.py — patch
# ─────────────────────────────────────────────────────────────────────────────
# Goal: turn play.py into something that, given a checkpoint, drops a
# structured artifact directory and (optionally) hands it to nexus's
# upload_eval.py so the bundle lands on the same MLflow run that owns
# the checkpoint.
#
# Three things change:
#   1. New CLI args:  --artifacts_dir, --run_info_dir, --upload, --video_length
#   2. The RecordVideo wrapper writes into <artifacts_dir>/videos/ when
#      --artifacts_dir is set; otherwise falls back to ./outputs/videos
#      (preserving current behaviour for callers that haven't migrated).
#   3. After agent.test() returns and *before* simulation_app.close(),
#      we (a) write metrics.json + stub READMEs into the artifact tree,
#      and (b) optionally shell out to nexus's upload_eval.py.
#
# The existing _play_rsl_rl path is left untouched — only _play_custom_algo
# (or whichever branch runs your in-house agent) needs the new wiring.
#
# This is not a runnable file by itself; copy each block into the matching
# location in your play.py.
# ─────────────────────────────────────────────────────────────────────────────

# ── Block 1: argparse additions ──────────────────────────────────────────────
# Add inside the existing argparse setup, alongside --task / --algo / --checkpoint.

parser.add_argument(
    "--artifacts_dir",
    type=str,
    default=None,
    help="Structured output dir for rollout artifacts. When set, videos land "
    "in <artifacts_dir>/videos/ and stub subdirs are created for the future "
    "trajectory/tactile/object/embedding hooks. When unset, falls back to "
    "the legacy ./outputs/videos path.",
)
parser.add_argument(
    "--run_info_dir",
    type=str,
    default=None,
    help="Path to the trainer's tb_dir (the directory containing .nexus_run.json). "
    "Required when --upload is set; ignored otherwise.",
)
parser.add_argument(
    "--upload",
    action="store_true",
    help="After rollout, hand <artifacts_dir> to nexus's upload_eval.py so "
    "the bundle attaches to the MLflow run identified by --run_info_dir.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=1000,
    help="Frames per recorded rollout (default: 1000). Note — fps and "
    "resolution come from the env (Isaac Lab's --camera_resolution / env cfg's "
    "render_fps), not from this flag.",
)
parser.add_argument(
    "--target",
    type=str,
    default="central",
    choices=("central", "local"),
    help="When --upload is set, which sidecar URI EvalLogger should target "
    "(central | local). Default: central — bypasses scheduled_sync. Switch "
    "to 'local' for in-progress runs that haven't synced yet.",
)
# If your trainer's play.py already has Isaac Lab's native --video /
# --video_length / --video_interval flags, prefer those — drop the duplicates
# above and forward existing flags to the RecordVideo wrapper below.


# ── Block 2: artifact dir setup — runs *before* env construction ─────────────
# Place this right after `args_cli = parser.parse_args()`, before the env is
# built. It chooses the video folder and seeds the stub layout.

import json
import os
from pathlib import Path

if args_cli.artifacts_dir:
    artifacts_dir = Path(args_cli.artifacts_dir).expanduser().resolve()
    video_folder = artifacts_dir / "videos"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    video_folder.mkdir(parents=True, exist_ok=True)

    # Stub categories — folders + README so the layout is self-documenting.
    # The Python hooks below are no-ops in MVP; later PRs will fill them in
    # without changing the directory contract.
    stubs = {
        "trajectories": (
            "Joint trajectory artifacts (planned).\n"
            "Format: HDF5 with datasets {qpos: (T, dof), qvel: (T, dof), "
            "ee_pose: (T, 7)}. Filename: rollout_<ep>.h5\n"
        ),
        "tactile": (
            "Tactile contact artifacts (planned).\n"
            "Format: NPZ with arrays {contact_pos, contact_force, contact_normal} "
            "indexed (T, N_contacts, 3). Filename: rollout_<ep>.npz\n"
        ),
        "objects": (
            "Object pose / velocity artifacts (planned).\n"
            "Format: NPZ with {obj_pose: (T, 7), obj_vel: (T, 6)} per object id.\n"
        ),
        "embeddings": (
            "Encoder activation artifacts (planned).\n"
            "Format: .npy per layer name; shape (T, D).\n"
        ),
    }
    for name, body in stubs.items():
        d = artifacts_dir / name
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.md"
        if not readme.exists():
            readme.write_text(body, encoding="utf-8")
else:
    artifacts_dir = None
    video_folder = Path("outputs") / "videos"
    video_folder.mkdir(parents=True, exist_ok=True)


# ── Block 2.5: Isaac Lab integration — auto enable_cameras + render_mode ─────
# Two glue points that must run *between* `parser.parse_known_args()` and
# `AppLauncher(args_cli)` / `gym.make(...)`:
#
#   1. `--enable_cameras` is required for Isaac Lab to spin up the RTX
#      renderer when there's no GUI viewport. We treat `--artifacts_dir`
#      as the single source of truth — if you ask for an artifact bundle,
#      we know you want video, so we flip it on for you.
#   2. `gym.make(render_mode=...)` must be `"rgb_array"` for `RecordVideo`
#      to receive frames. Gating on `--artifacts_dir` (same signal as
#      Block 2's video_folder choice) keeps the on/off contract single.
#
# `AppLauncher.__init__` consumes `enable_cameras` at construction time and
# boots Kit accordingly — mutating `args_cli.enable_cameras` afterwards is a
# silent no-op (renderer stays without cameras, `env.render()` returns None,
# RecordVideo flushes "zero frames to save"). So we set it on `args_cli`
# before constructing AppLauncher.
#
# Paste this immediately after `args_cli, hydra_args = parser.parse_known_args()`
# and before `app_launcher = AppLauncher(args_cli)`:
#
#     if args_cli.artifacts_dir is not None:
#         args_cli.enable_cameras = True
#
# And change the existing `gym.make(...)` call to derive render_mode from
# the same gate (drop any `--video` arg + its conditional):
#
#     render_mode = "rgb_array" if args_cli.artifacts_dir else None
#     env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)


# ── Block 3: RecordVideo wrapper + auto-stop + flush guard ───────────────────
# Design notes that diverge from a textbook gym RecordVideo:
#
#   - **No `episode_trigger`** — Isaac Lab vec env auto-resets internally
#     (per-env, async), and those resets don't propagate to the gym
#     RecordVideo wrapper, so `episode_trigger=...` only fires once at
#     the very first reset. We use `step_trigger=lambda s: s == 0` to
#     record one continuous mp4 of `--video_length` frames per
#     invocation, which matches what the setup can actually deliver.
#   - `name_prefix` includes ckpt stem + timestamp so re-runs don't collide.
#   - fps / resolution come from the env, not from RecordVideo — tune at
#     the env level (Isaac Lab: --camera_resolution, env cfg render_fps).
#   - **StopAfterFrames wrapper** — RecordVideo flushes mp4 to disk only
#     when `video_length` is reached *or* `env.close()` is called. Without
#     it the rollout loop would keep stepping forever and the artifact
#     bundle would never be ready for upload. We raise a sentinel
#     exception after `video_length + flush buffer` steps; play.py catches
#     it as planned termination.
#   - **try / finally around agent.test()** — guarantees `env.close()`
#     runs even on Ctrl+C, which is the only way to flush an in-progress
#     mp4 segment (the encoder accumulates frames in RAM, not streaming).

import time
import gymnasium as gym

ckpt_stem = Path(args_cli.checkpoint).stem if args_cli.checkpoint else "rollout"
stamp = time.strftime("%Y%m%d_%H%M%S")
name_prefix = f"{ckpt_stem}_{stamp}"

env = gym.wrappers.RecordVideo(
    env,
    video_folder=str(video_folder),
    step_trigger=lambda step: step == 0,
    video_length=args_cli.video_length,
    name_prefix=name_prefix,
    disable_logger=True,
)


class _PlayDone(Exception):
    """Sentinel — raised after the artifact bundle is complete."""


class StopAfterFrames:
    """Raise `_PlayDone` after `target_frames` steps so agent.test() exits.

    The +5 buffer gives RecordVideo a couple of extra `step()` calls past
    `video_length` to finish its flush before we tear down — without it
    the last few frames can be lost when the wrapper's internal counter
    and our counter race.

    Plain object (not `gymnasium.Wrapper`) because the inner env here is
    typically a duck-typed project wrapper that doesn't inherit from
    `gymnasium.Env` — gymnasium 1.0+'s strict `isinstance` check in
    `Wrapper.__init__` would reject it. `__getattr__` forwards everything
    we don't define to the inner wrapper; `step` stays a pass-through so
    the inner wrapper's tuple shape is preserved.
    """

    def __init__(self, env, target_frames: int):
        self.env = env
        self._target = target_frames + 5
        self._steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        result = self.env.step(action)
        self._steps += 1
        if self._steps >= self._target:
            raise _PlayDone(f"reached {self._steps} steps; mp4 is flushed")
        return result


# Only install the auto-stop when we're actually recording — without
# artifacts_dir the script preserves its legacy "run until killed" behavior.
if artifacts_dir is not None:
    env = StopAfterFrames(env, target_frames=args_cli.video_length)


# Wrap the existing agent.test() call in a *nested* try so:
#
#   - inner try catches `_PlayDone` and resumes normal flow
#     (otherwise post-processing in Block 5/6 is jumped over),
#   - outer try/finally guarantees `env.close()` + `simulation_app.close()`
#     run on success, on `_PlayDone`, *and* on Ctrl+C (`KeyboardInterrupt`
#     bypasses the inner `except _PlayDone` and bubbles to the outer
#     `finally`).
#
# Block 5/6 *must* live in the outer try body (between the inner
# try/except and the finally) — that's the only spot where:
#   1. the rollout is complete (so metrics.json is meaningful),
#   2. the env is still open (so the hook stubs can read env state),
#   3. exceptions raised by Block 5/6 still hit the outer finally
#      (so cleanup runs even if metrics extraction throws).
#
# Skeleton (drop Block 5/6 verbatim into the marked region):
#
#     try:
#         try:
#             agent.test()
#         except _PlayDone as e:
#             print(f"[play.py] {e}")
#
#         # ── Block 5 + Block 6 go here ─────────────────────────────
#
#     finally:
#         env.close()
#         simulation_app.close()
#         print("\n[INFO] Simulation closed")
#
# Why a single finally for both close() calls: `env.close()` cascades
# through StopAfterFrames → RecordVideo, flushing the in-RAM frame buffer
# to mp4; `simulation_app.close()` tears down Kit. Both must run before
# the process exits or you'll see "no mp4 on disk" (SIGINT-killed before
# flush) or zombie Kit processes (Python exited before Kit teardown).


# ── Block 4: hook stubs (module-level, alongside other helpers) ──────────────
# These exist now so the call sites can be wired in MVP without runtime
# behaviour changes. Each is a no-op until a follow-up PR fills it in.

def dump_trajectories(env, episode_idx: int, artifacts_dir: Path) -> None:
    """Dump per-step joint qpos/qvel/ee_pose for one episode. (stub)"""
    return None


def dump_tactile(env, episode_idx: int, artifacts_dir: Path) -> None:
    """Dump per-step contact positions/forces/normals for one episode. (stub)"""
    return None


def dump_objects(env, episode_idx: int, artifacts_dir: Path) -> None:
    """Dump per-step object pose/velocity for one episode. (stub)"""
    return None


def dump_embeddings(agent, obs, artifacts_dir: Path) -> None:
    """Dump encoder activations for the current observation. (stub)"""
    return None


# ── Block 5: post-rollout — runs after agent.test() returns ──────────────────
# Place this in the outer try body of the Block 3 skeleton — after the
# inner `try / except _PlayDone` and *before* the outer `finally` that
# closes env + simulation_app. That's the only spot where the env is
# still alive (hook stubs need it) and exceptions raised here still hit
# the same finally (cleanup runs even if metrics extraction throws).
# The hook calls are guarded by `if artifacts_dir is not None` so the
# legacy path stays a no-op.

if artifacts_dir is not None:
    # Always write a metrics.json — even if minimal — so EvalLogger can
    # promote scalars via --metrics_from. Replace this stub with the real
    # success_rate / mean_return extraction once it lands in agent.test().
    metrics = {
        "video_length": int(args_cli.video_length),
        # "success_rate": ...,
        # "mean_return": ...,
    }
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Human-readable sidecar — what conditions produced these mp4s? Lives
    # next to the videos so a 6-month-old artifact is self-explanatory.
    info = {
        "checkpoint": str(args_cli.checkpoint) if args_cli.checkpoint else None,
        "task": args_cli.task,
        "algo": args_cli.algo,
        "video_length": int(args_cli.video_length),
        "name_prefix": name_prefix,
        "produced_at": stamp,
    }
    (video_folder / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Hooks are still no-ops in MVP — single ep_idx=0 invocation since
    # vec env auto-reset prevents reliable episode counting at this layer
    # (see Block 3 notes). Real per-episode wiring will go *inside*
    # agent.test() in a follow-up, not here.
    dump_trajectories(env, 0, artifacts_dir)
    dump_tactile(env, 0, artifacts_dir)
    dump_objects(env, 0, artifacts_dir)
    dump_embeddings(agent=None, obs=None, artifacts_dir=artifacts_dir)


# ── Block 6: optional MLflow upload — runs only with --upload ────────────────
# subprocess-based to avoid pulling mlflow / rich into play.py's import graph.
# Adjust NEXUS_REPO to wherever the team has nexus checked out, or expose it
# as an env var.

import shlex
import subprocess
import sys

if args_cli.upload:
    if artifacts_dir is None:
        raise SystemExit("--upload requires --artifacts_dir to be set")
    if args_cli.run_info_dir is None:
        raise SystemExit("--upload requires --run_info_dir (the trainer's tb_dir)")

    nexus_repo = os.environ.get("NEXUS_REPO")
    if not nexus_repo:
        raise SystemExit(
            "--upload requires the NEXUS_REPO env var (path to the nexus "
            "checkout root, e.g. /workspace/nexus). Export it before running play.py."
        )
    upload_cli = Path(nexus_repo) / "post_upload" / "upload_eval.py"
    if not upload_cli.exists():
        raise SystemExit(
            f"NEXUS_REPO={nexus_repo} has no post_upload/upload_eval.py — "
            "check the path points at the nexus checkout root."
        )

    cmd = [
        sys.executable,
        str(upload_cli),
        "--run_info", str(args_cli.run_info_dir),
        "--target", args_cli.target,
        "--eval_dir", str(artifacts_dir),
        "--metrics_from", str(artifacts_dir / "metrics.json"),
        "--tag", f"observer.task={args_cli.task}",
        "--tag", f"observer.algo={args_cli.algo}",
    ]
    print("[play.py] handing off to upload_eval:", " ".join(shlex.quote(c) for c in cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        # Don't fail the whole script if the upload fails — the artifacts
        # are still on disk and can be retried via upload_eval.py directly.
        print(f"[play.py] WARN: upload_eval exited with code {rc}")


# simulation_app.close() comes here — unchanged.

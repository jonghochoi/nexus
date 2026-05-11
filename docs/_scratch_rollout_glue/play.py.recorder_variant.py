# ─────────────────────────────────────────────────────────────────────────────
# play.py — patch (observer recorder variant)
# ─────────────────────────────────────────────────────────────────────────────
# Block-by-block parallel of play.py.snippet.py. Blocks 1, 2, 4, 5, 6 are
# byte-identical to that file — copy them from there. THIS file only ships
# the differences:
#
#   - Block 1: + a few recorder-specific argparse args (camera config, fps,
#              resolution, codec).
#   - Block 3: gym.wrappers.RecordVideo  →  observer.isaac.VideoRecorder +
#              CameraController, driven by observer's record_all_views(...).
#
# Use this variant when you need:
#   - Multi-camera output (one mp4 per pose, sequential sweep)
#   - Codec / CRF / pix_fmt / fps / resolution control at the recorder level
#   - Cinematic orbit sweeps via CameraController.generate_orbit_poses(...)
#   - JSON-driven camera pose configs reused across runs
#
# Prerequisites (HARD — variant won't run without these):
#   - Run under isaaclab.sh -p (Replicator + viewport APIs are Isaac-process
#     bound; standalone python entrypoint can't see them).
#   - GUI viewport mandatory — Replicator's RenderProduct needs a live
#     viewport. If you only have headless, use play.py.snippet.py instead.
#   - observer installed in the trainer env: pip install -e /path/to/observer
#
# Output contract — same as play.py.snippet.py. Videos land at
# <artifacts_dir>/videos/<pose_name>.mp4, the four stub category dirs and
# metrics.json are unchanged, and Block 6's upload_eval.py subprocess call
# is byte-identical. EvalLogger doesn't care which variant produced the
# bundle.
# ─────────────────────────────────────────────────────────────────────────────


# ── Block 1 — diffs against play.py.snippet.py ─────────────────────────────
# Add these *in addition to* the args from play.py.snippet.py Block 1.

parser.add_argument(
    "--camera_config",
    type=str,
    default=None,
    help="Path to a JSON file of camera poses "
    "([{name, eye, target, record_steps}, ...]). When unset, an 8-pose orbit "
    "is generated programmatically via CameraController.generate_orbit_poses.",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=30,
    help="Output video frame rate, written into the ffmpeg invocation.",
)
parser.add_argument(
    "--video_resolution",
    type=str,
    default="1920x1080",
    help="Output video resolution as WxH (e.g. 1280x720, 1920x1080).",
)
parser.add_argument(
    "--video_codec",
    type=str,
    default="libx264",
    help="ffmpeg codec for output mp4 (e.g. libx264, libx265).",
)
parser.add_argument(
    "--video_crf",
    type=int,
    default=18,
    help="ffmpeg CRF (constant rate factor) — lower is higher quality.",
)
parser.add_argument(
    "--orbit_target",
    type=str,
    default="0,0,0.4",
    help="When --camera_config is unset, fallback orbit target as 'x,y,z'.",
)
parser.add_argument(
    "--orbit_radius",
    type=float,
    default=0.6,
    help="Fallback orbit radius (m).",
)
parser.add_argument(
    "--orbit_steps",
    type=int,
    default=8,
    help="Fallback orbit pose count (45° apart by default at 8).",
)


# ── Block 3 (REPLACEMENT) — observer.isaac instead of gym.RecordVideo ──────
# Drop this in place of the gym.wrappers.RecordVideo call from
# play.py.snippet.py. Requires `video_folder` / `artifacts_dir` from Block 2
# (unchanged from play.py.snippet.py) to already exist.

from observer.isaac import VideoRecorder, CameraController

# Recorder — Replicator-backed, ffmpeg-encoded. One instance, reused across
# all camera poses; start()/stop() bracket each pose's mp4.
recorder = VideoRecorder(
    output_dir=str(video_folder),
    fps=args_cli.video_fps,
    resolution=tuple(int(x) for x in args_cli.video_resolution.split("x")),
    codec=args_cli.video_codec,
    crf=args_cli.video_crf,
    pix_fmt="yuv420p",
)

# Camera poses — load from JSON if provided, else synthesize an orbit.
camera_controller = CameraController()
if args_cli.camera_config:
    camera_controller.load_config(args_cli.camera_config)
else:
    target_xyz = tuple(float(x) for x in args_cli.orbit_target.split(","))
    for pose in CameraController.generate_orbit_poses(
        target=target_xyz,
        radius=args_cli.orbit_radius,
        n_steps=args_cli.orbit_steps,
        record_steps_each=args_cli.video_length,
    ):
        camera_controller.add_pose(**pose)


# Recording loop — observer's helper drives the per-pose start/capture/stop
# cycle, calling our `step_fn` between frames. Replaces the implicit
# gym.RecordVideo wrap-and-forget pattern. NOTE — `agent.test()` no longer
# runs the rollout here; this loop *is* the rollout. If your existing
# play.py drives the env step loop differently, adapt accordingly.

def _step(observation):
    # Wire to your existing agent — single-step, deterministic eval.
    action = agent.act(observation, deterministic=True)
    obs, _reward, terminated, truncated, _info = env.step(action)
    done = bool(terminated) or bool(truncated)
    return obs, done

# observer's record_all_views drives sim + camera + recorder together.
# Signature (recorder.py:178-205):
#   record_all_views(sim, policy, camera_controller, recorder, step_fn=None)
from observer.isaac.recorder import record_all_views

# `sim` here is your Isaac Lab SimulationContext / app handle; substitute
# whatever your trainer exposes. `policy` is unused when we pass step_fn.
record_all_views(
    sim=simulation_app,
    policy=None,
    camera_controller=camera_controller,
    recorder=recorder,
    step_fn=_step,
)

# After this call returns, <video_folder>/<pose_name>.mp4 exists for each
# pose in camera_controller. Block 5 (post-rollout — unchanged) takes over.

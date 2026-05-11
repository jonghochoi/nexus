# `_scratch_rollout_glue/` — copy-paste templates for the external trainer

> ⚠️ **Temporary scratch area.** This folder is a working draft of the
> trainer-side glue (`train.py` / `play.py` snippets, shell scripts) that pairs
> with `post_upload/upload_eval.py`. It ships with the nexus PR for review only —
> once the templates have been verified against the actual external trainer,
> they will be promoted into the appropriate guide doc (likely a new section
> inside `docs/32_EVAL_ARTIFACT_INGESTION.md`) and this folder will be removed.

## What's here

| File | What it is | Where it ends up |
|---|---|---|
| `train.py.snippet.py` | Minimal patch to the trainer's existing `train.py` — make `make_logger()` write `central_tracking_uri` into the sidecar so eval can target central without extra flags. | Apply to the trainer's `train.py`. |
| `play.py.snippet.py` | Full diff to add `--artifacts_dir` / `--run_info_dir` / `--upload` / `--target` / `--gif_preview` to the trainer's `play.py`, plus stub hook signatures for the not-yet-implemented artifact types. **Uses `gym.wrappers.RecordVideo` — works headless.** | Apply to the trainer's `play.py`. |
| `play.py.recorder_variant.py` | Same blocks as `play.py.snippet.py`, but Block 3 swaps `gym.wrappers.RecordVideo` for `observer.isaac.VideoRecorder` + `CameraController`. **Requires `isaaclab.sh` + GUI viewport + observer pip-installed.** Pick this when you need multi-pose / cinematic / codec-controlled output. | Apply to the trainer's `play.py` — see "Video recorder choice" below. |
| `train.sh` | Thin wrapper that pins `RUN_NAME` so train + play share an identity. | Drop into the trainer repo root. |
| `play.sh` | Wrapper around `play.py` that derives `train_dir` from the shared `RUN_NAME` and passes `--artifacts_dir / --upload / --target` per env vars. | Drop into the trainer repo root. |
| `train_and_eval.sh` | Top-level chain: pre-computes `RUN_NAME`, runs `train.sh` then `play.sh`. | Drop into the trainer repo root. |
| `plans/*.html` | Phase 1 + Phase 2 design documents that produced this folder, kept in-tree for review. | Reference only — not copied to the trainer repo. |

## Usage cookbook

Six representative invocations covering the env-var surface. All shells assume
`cd` to the trainer repo root after `train.sh` / `play.sh` / `train_and_eval.sh`
have been dropped in.

| # | Scenario | Invocation |
|---|---|---|
| 1 | Smoke E2E — train + eval + upload, all defaults | `EXPERIMENT_NAME=robot_hand_rl ./train_and_eval.sh --num_envs 64 --max_agent_steps 1000` |
| 2 | Train only (no eval) with a manually pinned name | `EXPERIMENT_NAME=robot_hand_rl RUN_NAME=manual_001 ./train.sh --num_envs 256` |
| 3 | Eval an existing run, **no MLflow upload** (local-only smoke) | `EXPERIMENT_NAME=robot_hand_rl RUN_NAME=run_2026_05_10 UPLOAD=0 ./play.sh --num_envs 4` |
| 4 | Eval an explicit checkpoint (`last.pth` instead of `best.pth`) | `EXPERIMENT_NAME=robot_hand_rl RUN_NAME=run_2026_05_10 CKPT=./logs/.../last.pth ./play.sh` |
| 5 | Upload to the **GPU-node-local relay** (run not yet synced to central) | `EXPERIMENT_NAME=robot_hand_rl RUN_NAME=run_2026_05_10 TARGET=local ./play.sh` |
| 6 | Custom logs root (training output writes outside `./logs`) | `EXPERIMENT_NAME=robot_hand_rl RUN_NAME=run_2026_05_10 LOGS_ROOT=/scratch/team/logs ./play.sh` |
| 7 | Left hand instead of right (must match between train and play) | `EXPERIMENT_NAME=robot_hand_rl HAND_SIDE=left ./train_and_eval.sh --num_envs 64 --max_agent_steps 1000` |

> 💡 `--num_envs 4` for play (vs 256 for train) is intentional — `gym.wrappers.RecordVideo` only follows `env_idx=0`, so a large vec env wastes simulation cycles on envs that won't appear in the mp4. The `play.py.recorder_variant.py` (observer recorder) is agnostic to `num_envs`.

> 💡 **Python interpreter** — both scripts auto-detect the Isaac Lab wrapper at `/workspace/IsaacLab/_isaac_sim/python.sh` or `/isaac-sim/python.sh` (the `.bashrc` alias inside Isaac Lab containers is invisible to non-interactive scripts, so the wrapper has to be invoked by path). Outside Isaac containers, plain `python` is used. Export `PYTHON=/path/to/interpreter` to override the autodetection.

## Why three scripts?

`train_and_eval.sh` is genuinely thin (~5 lines once `RUN_NAME` is exported),
and on first read the trio looks like over-engineering — couldn't we just
collapse everything into one entrypoint? In practice no, and here's why:

- **Eval-without-train is a daily workflow.** Researchers re-run play against
  an existing checkpoint many times per training run — different camera
  angles, different `--video_length`, different `UPLOAD` toggles. Collapsing
  forces re-training or a `--skip_train` flag, neither of which is cleaner.
- **Failure isolation.** Training takes hours; play takes seconds-to-minutes.
  A play crash should not waste a successful train, and a train crash should
  not leave the artifact dir half-populated.
- **Different launch contexts.** Train runs under `sbatch` / `nohup` / a
  long-lived tmux. Play runs interactively from a laptop SSH'd to the GPU
  box. Different supervisors, different env-var defaults.
- **Different resource shapes.** Train wants `--num_envs 256` for throughput;
  play wants `--num_envs 4` for video. Hard to express both naturally in one
  entrypoint without flag duplication.
- **CI / smoke testability.** `train.sh --max_agent_steps 100` proves the
  trainer end-to-end without invoking eval/MLflow. `play.sh UPLOAD=0` proves
  the artifact pipeline without invoking the trainer. Combining the trio
  forces fragile flag combinations to retain that surface.
- **`train_and_eval.sh` is genuinely thin.** Collapsing the three would
  *grow* the entrypoint, not shrink it, and obscure the per-phase contracts.

## Video recorder choice — gym RecordVideo vs observer recorder

Two variants ship side-by-side. They share the same artifact directory
contract (`<artifacts_dir>/videos/*.mp4`) so `upload_eval.py` is unchanged
either way — the choice is purely about how the mp4s get produced.

| Axis | `play.py.snippet.py` (gym `RecordVideo`) | `play.py.recorder_variant.py` (observer) |
|---|---|---|
| Backend | env `render(mode="rgb_array")` → moviepy / imageio mp4 | `omni.replicator.core` `RenderProduct` → ffmpeg subprocess |
| Headless support | ✅ works with `--headless --enable_cameras --video` | ❌ requires GUI viewport (Replicator + viewport API) |
| Cameras | 1 (env render view) | 1 viewport, swept across N JSON-defined poses sequentially |
| Codec / quality knobs | none (env-level only) | `codec` / `crf` / `pix_fmt` / `resolution` / `fps` in ctor |
| Triggers | gym `episode_trigger` (one mp4 per episode) | manual `recorder.capture_frame()` per sim step |
| Multi-camera output | none | one `.mp4` per pose, via `record_all_views(...)` |
| Cinematic sweeps | none | `CameraController.generate_orbit_poses(target, radius, n_steps)` |
| Vec env (256 envs) | follows `env_idx=0` only | one camera; agnostic to env count |
| Integration cost | drop-in replacement of one `gym.wrappers.RecordVideo(...)` line | rewrites the rollout loop + must run under `isaaclab.sh` + observer pip-installed |

**Verdict:**
- Pick `play.py.snippet.py` (gym) for **headless / CI / smoke runs / single-view rollouts**. Lower friction and works in a sandbox.
- Pick `play.py.recorder_variant.py` (observer) for **demo / poster videos with multi-pose orbital sweeps, explicit codec control, and a GUI session**.

The two variants are byte-identical in Blocks 1, 2, 4, 5, 6 — only Block 3
(the recorder wrapper) differs. The variant file documents the prerequisites
inline.

## Improvements baked into the gym variant

`play.py.snippet.py` doesn't just call `RecordVideo(...)`; it applies the
following so the gym path is operationally usable, not just minimally
correct:

- **`episode_trigger` (not `step_trigger`).** One mp4 per episode, with proper
  episode boundaries — easier to compare cases in the MLflow Artifacts pane.
- **`name_prefix=f"{ckpt_stem}_{stamp}"`.** Re-runs don't collide; each file
  is self-identifying.
- **`videos/info.json` sidecar.** Records `{checkpoint, task, algo, video_length, name_prefix, produced_at}` so a 6-month-old artifact bundle is still readable.
- **fps / resolution control deferred to env.** `RecordVideo` derives fps
  from `env.metadata["render_fps"]` and resolution from the env's camera
  config; the snippet's comment makes this explicit so you tune at the right
  layer (Isaac Lab's `--camera_resolution`, env cfg's `render_fps`).
- **Optional GIF preview (`--gif_preview`).** When `moviepy` is available,
  drops a 3s 320px-wide GIF next to each mp4. MLflow's artifact viewer
  previews GIFs inline (mp4 needs the auto `index.html`); useful as a
  thumbnail. Off by default — no hard moviepy dependency.
- **vec-env-idx-0 caveat documented inline.** RecordVideo only sees
  `env_idx=0`, so for video runs `--num_envs 4` is enough.
- **Isaac Lab native `--video` integration note.** If your trainer's
  `play.py` already exposes `--video --video_length --video_interval` from
  Isaac Lab, prefer those and drop the snippet's duplicate flags.

## Test plan (what to walk through together)

1. Apply `train.py.snippet.py`, run `./train.sh` headless — verify
   `<train_dir>/.nexus_run.json` contains both `tracking_uri` and
   `central_tracking_uri`.
2. Apply `play.py.snippet.py`, run `UPLOAD=0 ./play.sh --video_length 60` —
   verify `<train_dir>/eval/<eval_id>/videos/<stem>_<stamp>-step-0.mp4`
   exists, plus `videos/info.json`, plus the four stub subdirs
   (`trajectories/`, `tactile/`, `objects/`, `embeddings/`) each contain a
   `README.md`.
3. Run `./play.sh` (with `UPLOAD=1` default) — verify the MLflow run gains an
   `eval/<eval_id>/` artifact dir with the videos and an auto `index.html`.
4. Run `TARGET=local ./play.sh` — verify the same bundle appears on the
   GPU-node-local relay (`:5100`) instead of central.
5. End-to-end: `./train_and_eval.sh --num_envs 64 --max_agent_steps 1000` —
   both phases share the same `RUN_NAME`; the central run accumulates
   training metrics and eval artifacts under one identity.
6. (If GUI / Isaac Lab available) repeat step 2 with `play.py.recorder_variant.py`
   in place — confirm one `<pose_name>.mp4` per pose in the orbit.

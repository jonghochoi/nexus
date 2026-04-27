# Post-Upload Guide (Pipeline B)

This guide covers the ergonomic features of the post-upload CLI (`post_upload/tb_to_mlflow.py`) beyond the basic invocation in the README: the config file, interactive tag entry, automatic verification, upload history, and Sim-to-Real `sim_run_id` auto-detection.

> For the high-level "what is Pipeline B" overview, see [`README.md`](../README.md#-pipeline-b--tensorboard-post-upload-one-shot-no-code-changes) and [`VALIDATION_GUIDE.md`](./VALIDATION_GUIDE.md) Phase 1-B.

---

## TL;DR

```bash
# One-time setup
cp post_upload/post_config.example.json ~/.nexus/post_config.json
$EDITOR ~/.nexus/post_config.json   # set tracking_uri + your fixed tags

# Every upload
python post_upload/tb_to_mlflow.py --tb_dir /path/to/run_001
# → prompts for seed/task if missing, uploads, auto-verifies
```

Repeat the same config for another seed?

```bash
python post_upload/tb_to_mlflow.py --tb_dir /path/to/run_002 --repeat-last --tags seed=2
```

List recent uploads / re-verify the last one:

```bash
python post_upload/tb_to_mlflow.py --history
python post_upload/verify_upload.py --from-last
```

---

## 1. `~/.nexus/post_config.json` — team-fixed values

The CLI reads defaults from `~/.nexus/post_config.json`. Ship the example file to every team member:

```bash
mkdir -p ~/.nexus
cp post_upload/post_config.example.json ~/.nexus/post_config.json
```

Example:

```json
{
  "tracking_uri": "http://nexus-server:5000",
  "experiment": "robot_hand_rl",
  "tags": {
    "researcher": "kim",
    "isaac_lab_version": "1.2.0",
    "physx_solver": "TGS",
    "hardware": "robot_22dof"
  }
}
```

| Field | Purpose |
|---|---|
| `tracking_uri` | Central MLflow server URL (local testing: `http://127.0.0.1:5100`) |
| `experiment` | Default experiment name |
| `tags.researcher` | Your name — per-user, set once and forget |
| `tags.isaac_lab_version`, `physx_solver`, `hardware` | Team-fixed reproducibility tags |

Override the config path with `--config /path/to/other.json` (useful for CI or second-machine setups).

---

## 2. Required tags & interactive mode

Every uploaded run must carry these tags — the CLI blocks the upload otherwise (pass `--force` to bypass):

| Tag | Source | Notes |
|---|---|---|
| `researcher` | `~/.nexus/post_config.json` | Set once per user |
| `seed` | `--tags` or interactive prompt | Per-run |
| `task` | `--tags` or interactive prompt | Per-run |
| `sim_run_id` | run_meta.json or `--tags` | **Required** when `experiment=real_robot_eval` (see §5) |

### Interactive prompting

- **Automatic** — if any required tag is missing and `stdin` is a TTY, the CLI enters interactive mode and prompts only for the missing values, showing config values as defaults:

  ```
  Missing required tags: seed, task — entering interactive mode.

  Interactive tag entry (press Enter to accept default)
    seed: 42
    task: in_hand_reorientation
  ```

- **Explicit** — `-i` / `--interactive` prompts for every required tag regardless, which is handy for editing a repeated set (see §4):

  ```bash
  python post_upload/tb_to_mlflow.py --tb_dir /path/to/run_003 --repeat-last -i
  ```

### Precedence (low → high)

| Priority | Source | Scope |
|:---:|---|---|
| 1 | Builtin defaults | all tags |
| 2 | `~/.nexus/post_config.json` | all tags |
| 3 | `--repeat-last` (history) | all tags |
| 4 | `run_meta.json` | `sim_run_id` only |
| 5 | `--tags k=v ...` | all tags |
| 6 | `--git_commit HASH` | `git_commit` only |
| 7 | `-i` interactive input | required tags |

> **Note:** `--git_commit` is a convenience shorthand for `--tags git_commit=<hash>`. Use it for post-hoc uploads where the training commit is known but the working tree is no longer in that state. For scheduled-sync runs (Pipeline A), the commit is captured automatically — no flag needed.

---

## 3. Automatic verification

After every successful upload, `tb_to_mlflow.py` runs `verify_upload.py` against the returned `run_id` automatically. You don't need to copy the run_id anywhere.

```
✓ Upload complete!
  Run ID      : a1b2c3d4e5f6...
  Data points : 2,100  (3 batches)

Running automatic verification...
✓ All checks passed! TB -> MLflow porting is accurate.
```

Pass `--no_verify` to skip (e.g. in CI, where a separate job verifies).

Re-verify a past upload:

```bash
# Most recent
python post_upload/verify_upload.py --from-last

# A specific run
python post_upload/verify_upload.py \
    --run_id a1b2c3d4e5f6... \
    --tb_dir /path/to/run_001
```

---

## 4. Upload history — `~/.nexus/history.json`

Every upload is recorded in `~/.nexus/history.json` (newest first, capped at 20 entries) with its run_id, tags, and verification result.

### `--history` — list recent uploads

```
$ python post_upload/tb_to_mlflow.py --history

        Recent uploads (last 3)
  When                  Experiment         Run Name          Run ID          Verify  Key Tags
  2026-04-23T22:30:41   robot_hand_rl      ppo_v17_seed3     abc123def456      ✓     seed=3, task=in_hand_reorientation, researcher=kim
  2026-04-23T21:58:02   robot_hand_rl      ppo_v17_seed2     def456abc789      ✓     seed=2, ...
  2026-04-23T21:12:03   real_robot_eval    real_v3           789xyz...         ✗     seed=7, ..., sim_run_id=abc123
```

### `--repeat-last` — reuse the last tag set

Inherit `experiment`, `run_name`, and `tags` from the most recent upload — perfect for seed sweeps where only one value changes:

```bash
# Upload 10 seeds of the same task without re-typing everything:
python post_upload/tb_to_mlflow.py --tb_dir ./runs/seed1 --tags seed=1 task=in_hand_reorientation
python post_upload/tb_to_mlflow.py --tb_dir ./runs/seed2 --repeat-last --tags seed=2
python post_upload/tb_to_mlflow.py --tb_dir ./runs/seed3 --repeat-last --tags seed=3
...
```

Combine with `-i` to edit a tag interactively with the previous value prefilled:

```bash
python post_upload/tb_to_mlflow.py --tb_dir ./runs/seed2 --repeat-last -i
#   seed [1]: 2
#   task [in_hand_reorientation]: ↵  (accept)
```

### `--from-last` (verify_upload.py)

Re-run verification on the most recent upload without copy/paste:

```bash
python post_upload/verify_upload.py --from-last
```

---

## 5. Real-robot evaluation — `sim_run_id` auto-detection

Sim-to-Real traceability requires every real-robot eval run to carry a `sim_run_id` tag pointing at the sim training run whose policy was deployed (see [`EXPERIMENT_STANDARD_KO.md`](./EXPERIMENT_STANDARD_KO.md) § "Sim-to-Real 연결").

When `--experiment real_robot_eval`, `sim_run_id` is **required** — the CLI will prompt for it just like `seed` and `task`.

To avoid manual lookups, drop a `run_meta.json` file next to the tfevents:

```
logs/real_eval_2026-04-23/
├── events.out.tfevents.xxx
└── run_meta.json    ← {"sim_run_id": "abc123def456"}
```

On upload, the CLI detects and prefills:

```
Detected sim_run_id from run_meta.json: abc123def456
```

`run_meta.json` is treated as ground truth for that tb_dir — it overrides any value carried over by `--repeat-last`, but can still be overridden by explicit `--tags sim_run_id=...`.

### Recommended real-eval pipeline integration

Have your real-robot eval launcher script drop `run_meta.json` at the same moment it starts writing tfevents. Minimal Python:

```python
import json, pathlib, mlflow

policy = mlflow.pyfunc.load_model("models:/ppo_policy/Production")
sim_run_id = policy.metadata.run_id   # or read from your registry metadata

log_dir = pathlib.Path("logs/real_eval_2026-04-23")
log_dir.mkdir(parents=True, exist_ok=True)
with open(log_dir / "run_meta.json", "w") as f:
    json.dump({"sim_run_id": sim_run_id}, f)
```

---

## 6. All CLI flags at a glance

### `tb_to_mlflow.py`

| Flag | Purpose |
|---|---|
| `--tb_dir PATH` | tfevents directory (required for uploads) |
| `--experiment NAME` | MLflow experiment (default: from config) |
| `--run_name NAME` | MLflow run name (default: `{dirname}_{timestamp}`) |
| `--tracking_uri URL` | MLflow server (default: from config) |
| `--tags k=v ...` | Per-run tags, highest-priority source after `-i` |
| `--git_commit HASH` | Git commit hash of the training code; stored as `git_commit` tag |
| `--config PATH` | Alternate config file path |
| `-i`, `--interactive` | Prompt for every required tag |
| `--repeat-last` | Inherit experiment/run_name/tags from last history entry |
| `--force` | Skip required-tag validation |
| `--no_verify` | Skip automatic post-upload verification |
| `--dry_run` | Parse & preview only (skips validation and upload) |
| `--upload_artifacts` | Also attach tfevents files as MLflow artifacts |
| `--history` | Print recent uploads and exit |

### `verify_upload.py`

| Flag | Purpose |
|---|---|
| `--run_id ID` | MLflow run ID to verify |
| `--tb_dir PATH` | Source tfevents directory to compare against |
| `--tracking_uri URL` | MLflow server (default: `http://127.0.0.1:5000`) |
| `--tolerance F` | Numeric match tolerance (default: `1e-6`) |
| `--from-last` | Fill run_id/tb_dir/tracking_uri from last history entry |

---

## 7. Workflows

Concrete end-to-end walkthroughs of common situations. Each section shows the setup, the exact commands, and what you'd see in the terminal.

### 7.1 Day 1 — a new researcher onboards

You've just joined the team and cloned the repo. Get set up and do your first upload:

```bash
# Install — venv is created at ~/.nexus/venv (outside the repo)
bash setup.sh --alias && source ~/.bashrc
nexus-activate   # or: source ~/.nexus/activate.sh

# One-time config — fill in your name and the central server
cp post_upload/post_config.example.json ~/.nexus/post_config.json
$EDITOR ~/.nexus/post_config.json
```

```json
{
  "tracking_uri": "http://nexus-server:5000",
  "experiment": "robot_hand_rl",
  "tags": {
    "researcher": "lee",
    "isaac_lab_version": "1.2.0",
    "physx_solver": "TGS",
    "hardware": "robot_22dof"
  }
}
```

First upload — the CLI handles the rest:

```
$ python post_upload/tb_to_mlflow.py --tb_dir ./logs/ppo_first_try

──────── TensorBoard -> MLflow Uploader ────────
Config source: /home/lee/.nexus/post_config.json

Missing required tags: seed, task — entering interactive mode.

Interactive tag entry (press Enter to accept default)
  seed: 42
  task: in_hand_reorientation

Discovered tfevents files:
  • ./logs/ppo_first_try/events.out.tfevents.1730000000  (234.5 KB)

[metric summary table ...]

Tags to upload: {researcher: lee, seed: 42, task: in_hand_reorientation, ...}
Upload the above data to MLflow? (y/n): y

[upload progress ...]

✓ Upload complete!
  Run ID      : 7f3a9c8d2e1b4f6a...
  UI URL      : http://nexus-server:5000

Running automatic verification...
✓ All checks passed! TB -> MLflow porting is accurate.
```

That's it — from here on, just `--tb_dir` + answer two prompts.

---

### 7.2 Seed sweep — 5 runs of the same PPO config

You trained 5 seeds overnight:

```
~/runs/ppo_v17_seed1/events.out.tfevents.xxx
~/runs/ppo_v17_seed2/events.out.tfevents.xxx
~/runs/ppo_v17_seed3/events.out.tfevents.xxx
~/runs/ppo_v17_seed4/events.out.tfevents.xxx
~/runs/ppo_v17_seed5/events.out.tfevents.xxx
```

Upload the first with full metadata:

```bash
python post_upload/tb_to_mlflow.py \
    --tb_dir   ~/runs/ppo_v17_seed1 \
    --run_name ppo_v17_seed1 \
    --tags     seed=1 task=in_hand_reorientation
```

For seeds 2–5, `--repeat-last` inherits experiment/tags from history — just override the seed and run_name:

```bash
for S in 2 3 4 5; do
    python post_upload/tb_to_mlflow.py \
        --tb_dir   ~/runs/ppo_v17_seed${S} \
        --run_name ppo_v17_seed${S} \
        --repeat-last --tags seed=${S}
done
```

Confirm all five landed:

```
$ python post_upload/tb_to_mlflow.py --history

  Recent uploads (last 5)
  When                Experiment     Run Name         Run ID     Verify  Key Tags
  2026-04-23T18:45    robot_hand_rl  ppo_v17_seed5    abc...      ✓      seed=5, task=in_hand_...
  2026-04-23T18:42    robot_hand_rl  ppo_v17_seed4    def...      ✓      seed=4, task=in_hand_...
  2026-04-23T18:40    robot_hand_rl  ppo_v17_seed3    ghi...      ✓      seed=3, task=in_hand_...
  2026-04-23T18:37    robot_hand_rl  ppo_v17_seed2    jkl...      ✓      seed=2, task=in_hand_...
  2026-04-23T18:35    robot_hand_rl  ppo_v17_seed1    mno...      ✓      seed=1, task=in_hand_...
```

> 💡 Prefer editing interactively? `--repeat-last -i` prefills each tag with the previous value so you can step through and accept or change:
>
> ```
> seed [1]: 2              ← type new value
> task [in_hand_reorientation]: ↵  (accept)
> ```

---

### 7.3 Real-robot evaluation linked to a Production policy

Your eval script writes `run_meta.json` next to the tfevents:

```python
# real_robot_eval.py (excerpt)
import json, pathlib, mlflow
policy = mlflow.pyfunc.load_model("models:/ppo_policy/Production")

log_dir = pathlib.Path(f"logs/real_eval_{date}")
log_dir.mkdir(parents=True)
(log_dir / "run_meta.json").write_text(
    json.dumps({"sim_run_id": policy.metadata.run_id})
)
# ... run eval, write tfevents to log_dir ...
```

Upload after eval completes:

```
$ python post_upload/tb_to_mlflow.py \
    --tb_dir     ./logs/real_eval_2026-04-23 \
    --experiment real_robot_eval \
    --tags       seed=42 task=in_hand_reorientation

──────── TensorBoard -> MLflow Uploader ────────
Config source: /home/kim/.nexus/post_config.json
Detected sim_run_id from run_meta.json: 7f3a9c8d2e1b4f6a

[upload + auto-verify ...]
✓ All checks passed!
```

The MLflow run now carries `sim_run_id=7f3a9c...`; click through and you land on the exact sim training run whose policy was deployed.

If `run_meta.json` is missing, you can't forget `sim_run_id` silently — `--experiment real_robot_eval` promotes it to a required tag:

```
Missing required tags: sim_run_id — entering interactive mode.

Interactive tag entry (press Enter to accept default)
  sim_run_id: ▁      ← you must supply it
```

---

### 7.4 MLflow server blipped mid-upload

Upload failed:

```
$ python post_upload/tb_to_mlflow.py --tb_dir ~/runs/ppo_v19_seed1 \
      --tags seed=1 task=in_hand_reorientation
...
[ERROR] Failed to connect to MLflow server: HTTPConnectionPool(...)
```

Nothing gets saved to history on connect failure, so the command works to replay as-is once the server is back. If you just want to retry with the same tags you used 10 minutes ago on a previous run:

```bash
# Confirm what the last successful upload looked like
python post_upload/tb_to_mlflow.py --history

# Replay tags + change only the seed and run_name
python post_upload/tb_to_mlflow.py \
    --tb_dir   ~/runs/ppo_v19_seed1 \
    --run_name ppo_v19_seed1 \
    --repeat-last --tags seed=1
```

**Upload succeeded, verification failed** (network hiccup during the verify step): the record IS saved with `verify_ok=False`. Re-verify later without copy-paste:

```bash
python post_upload/verify_upload.py --from-last
```

---

### 7.5 "Did I already upload this?"

You come back from lunch and can't remember if you uploaded `ppo_v17_seed3`:

```
$ python post_upload/tb_to_mlflow.py --history

  Recent uploads (last 3)
  2026-04-23T12:18    robot_hand_rl   ppo_v17_seed3   abc...   ✓   seed=3, task=in_hand_...
  2026-04-23T12:15    robot_hand_rl   ppo_v17_seed2   def...   ✓   seed=2, task=in_hand_...
  2026-04-23T12:12    robot_hand_rl   ppo_v17_seed1   ghi...   ✓   seed=1, task=in_hand_...
```

Yes — 12:18, verified. The Run ID is shown; click through in MLflow UI at `http://nexus-server:5000/#/experiments/.../runs/abc...` to inspect.

History persists across shells and terminals, so you can check from a different tmux pane or a freshly-opened session.

---

### 7.6 CI / non-interactive usage

In CI (no TTY) every required tag must be explicit; there's no interactive fallback. A typical GitHub Actions step:

```yaml
- name: Upload run to MLflow
  env:
    MLFLOW_URI: ${{ secrets.MLFLOW_URI }}
  run: |
    python post_upload/tb_to_mlflow.py \
        --tb_dir       ./logs/${RUN_NAME} \
        --experiment   robot_hand_rl \
        --run_name     ${RUN_NAME} \
        --tracking_uri ${MLFLOW_URI} \
        --tags         researcher=ci seed=${SEED} task=${TASK} \
                       isaac_lab_version=${ISAAC_VER} \
                       physx_solver=TGS hardware=robot_22dof
```

Exit codes you can branch on:

| Code | Meaning |
|---|---|
| `0` | Upload and auto-verify both succeeded |
| `1` | Connection error, missing tfevents, missing required tags, etc. |
| `2` | Upload succeeded but auto-verify failed |

If a separate CI job handles verification, add `--no_verify` and fail the pipeline later on the verify step instead. For leaner configuration on a long-lived runner, keep `~/.nexus/post_config.json` populated and drop the fixed flags:

```bash
python post_upload/tb_to_mlflow.py \
    --tb_dir ./logs/${RUN_NAME} \
    --tags   seed=${SEED} task=${TASK}
```

---

## 8. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `[ERROR] Required tags missing: researcher, seed, task.` | No config file and not a TTY. Either populate `~/.nexus/post_config.json`, pass `--tags`, or re-run in a TTY to get the interactive prompt. |
| `[ERROR] Multiple run directories detected under: <dir>` | `--tb_dir` pointed at a parent containing multiple runs. Upload each run dir individually (use the shell loop suggested in the error). |
| `[ERROR] Failed to connect to MLflow server` | Wrong `tracking_uri`. Check `~/.nexus/post_config.json`; for local testing use `http://127.0.0.1:5100`, for central use `http://<server>:5000`. |
| `[WARN] --repeat-last: no previous upload in history.` | Empty `~/.nexus/history.json`. Do one manual upload first, then `--repeat-last` works. |
| Auto-verify prints `✗ Verification failed` | Tag list, counts, or values diverge. Compare via MLflow UI + `verify_upload.py --from-last` to inspect which tags/steps differ. |
| `[yellow]run_meta.json sim_run_id (X) overrides carried-over value (Y)` | `--repeat-last` had a different sim_run_id than the tb_dir's run_meta.json. The file's value wins (ground truth for this dir). If the file is wrong, delete it or override with `--tags sim_run_id=...`. |

# 🧪 NEXUS Step-by-Step Validation Guide

> Verify functionality locally first, then proceed to the GPU server. Check the checklist after completing each Phase.

---

## 📑 Table of Contents

- [💻 Phase 1-A — Local PC Validation](#phase-1-a--local-pc-validation)
- [📤 Phase 1-B — Actual Upload Test with Existing TensorBoard Files](#phase-1-b--actual-upload-test-with-existing-tensorboard-files)
- [📦 Phase 2 — GPU Server Dependency Installation (Offline Environment)](#phase-2--gpu-server-dependency-installation-offline-environment)
- [🖥️ Phase 3 — GPU Server Validation](#phase-3--gpu-server-validation)
- [🔄 Phase 4 — Cross-Server Sync Validation](#phase-4--cross-server-sync-validation)
- [🛠️ Troubleshooting](#troubleshooting)

---

## 💻 Phase 1-A — Local PC Validation

> **Purpose:** Verify that all features work correctly locally before deploying to the GPU server.

### A-1. Clone the repository

```bash
git clone https://github.com/jonghochoi/nexus.git
cd nexus
```

### A-2. Verify Python version

```bash
python3 --version
```

Must be `Python 3.8` or higher. 3.10 or 3.11 is recommended.

### A-3. Install environment

```bash
bash setup.sh
```

After installation, activate the virtual environment:

```bash
source ~/.nexus/activate.sh
# or, if you ran `bash setup.sh --alias`:
nexus-activate
```

> When `(venv)` appears at the start of the terminal prompt, the environment is activated. The venv itself lives at `~/.nexus/venv` — outside the repo — so replacing or re-cloning nexus sources does not wipe it.

### A-4. Verify installation

```bash
python -c "import mlflow; print('mlflow:', mlflow.__version__)"
python -c "import tbparse; print('tbparse OK')"
python -c "from nexus.logger import make_logger; print('logger OK')"
```

If all three lines output without errors, the installation is successful.

### A-5. Start local MLflow server

```bash
bash scheduled_sync/start_local_mlflow.sh
```

Expected output:

```
[NXS] Starting local MLflow server on 127.0.0.1:5100...
[NXS] MLflow ready at http://127.0.0.1:5100
```

Open `http://localhost:5100` in a browser and verify the MLflow UI appears.

> It is fine if a message appears saying the server is already running.

### A-6. Run smoke test

The smoke test automatically verifies the core functionality of NEXUS (package installation, MLflow connection, logging, validation).

```bash
# Run from the root of the nexus directory
python tests/smoke_test.py
```

All items must show `[PASS]`:

```
  [PASS]  Package imports
  [PASS]  MLflow server connection
  [PASS]  MLflowLogger logging
  [PASS]  make_logger factory
  [PASS]  DualLogger (Dual)

  All tests passed! NEXUS is working correctly.
```

After the test completes, the MLflow UI (`http://localhost:5100`) should show newly created runs under the `nexus_smoke_test` experiment.

### ✅ Phase 1-A Checklist

- [ ] `bash setup.sh` completed without errors
- [ ] `(venv)` prompt appears after `source ~/.nexus/activate.sh` (or `nexus-activate`)
- [ ] All three import commands succeed
- [ ] MLflow UI accessible at `http://localhost:5100` in browser
- [ ] All `[PASS]` in `python tests/smoke_test.py`
- [ ] `nexus_smoke_test` experiment and runs confirmed in MLflow UI

---

## 📤 Phase 1-B — Actual Upload Test with Existing TensorBoard Files

> **Purpose:** Upload existing tfevents files to MLflow and validate that data was transferred accurately. Complete all Phase 1 checklist items before proceeding (MLflow server must be running).

---

### B-1. Locate tfevents files

First, confirm where the tfevents files to be uploaded are located.

```bash
# Verify tfevents files exist in the directory
ls /path/to/your/logs/run_001/
```

Expected output:
```
events.out.tfevents.1700000000.hostname.12345.0
```

> tfevents files typically start with `events.out.tfevents.`. Having multiple files is fine — the script recursively searches the folder.

---

### B-2. Dry Run — Preview parsing results without uploading

Before the actual upload, use the `--dry_run` option to preview which metrics will be parsed and how many there are.

```bash
cd nexus/
source ~/.nexus/activate.sh   # or: nexus-activate

python post_upload/tb_to_mlflow.py \
    --tb_dir    /path/to/your/logs/run_001 \
    --dry_run
```

Expected output (metric summary table):

```
 Parsed TensorBoard Metrics Summary
┌──────────────────────────────────┬───────┬────────────┬──────────┬──────────┬──────────┐
│ Tag (Metric)                     │ Steps │ Step Range │  Val Min │  Val Max │ Val Last │
├──────────────────────────────────┼───────┼────────────┼──────────┼──────────┼──────────┤
│ train/episode_reward             │  1000 │ 0~999      │   0.0000 │  98.3200 │  87.5100 │
│ train/loss                       │  1000 │ 0~999      │   0.0021 │   1.2300 │   0.0412 │
│ eval/success_rate                │   100 │ 0~99       │   0.0000 │   0.8700 │   0.8200 │
└──────────────────────────────────┴───────┴────────────┴──────────┴──────────┴──────────┘

Total: 3 tags, 2,100 data points

--dry_run mode: skipping upload.
```

If the table appears, parsing is working correctly. Verify that metric names and data counts match your expectations.

---

### B-3. Run actual upload

After reviewing the contents in the dry run, proceed with the actual upload.

#### One-time setup — `~/.nexus/post_config.json`

Team-fixed values (`tracking_uri`, `isaac_lab_version`, `physx_solver`, `hardware`) and your personal `researcher` tag live in a config file, so you don't retype them for every run:

```bash
mkdir -p ~/.nexus
cp post_upload/post_config.example.json ~/.nexus/post_config.json
$EDITOR ~/.nexus/post_config.json
```

For local-testing MLflow (`:5100`), set `"tracking_uri": "http://127.0.0.1:5100"` in the config. The built-in default is `:5000` (central server).

#### Upload

```bash
# Short form — config supplies tracking_uri, researcher, hardware, etc.
# Missing required tags (researcher/seed/task) are prompted interactively.
python post_upload/tb_to_mlflow.py --tb_dir /path/to/your/logs/run_001

# Or fully explicit (for CI or ad-hoc overrides)
python post_upload/tb_to_mlflow.py \
    --tb_dir       /path/to/your/logs/run_001 \
    --experiment   robot_hand_rl \
    --run_name     ppo_baseline_v1 \
    --tracking_uri http://127.0.0.1:5100 \
    --tags         researcher=kim seed=42 task=in_hand_reorientation
```

The script shows a metric summary, asks for confirmation, uploads, and then **automatically runs verification** against the returned run_id:

```
Upload the above data to MLflow? (y/n): y
...
✓ Upload complete!
  Run ID      : a1b2c3d4e5f6...
  Data points : 2,100  (3 batches)
  UI URL      : http://127.0.0.1:5100

Running automatic verification...
✓ All checks passed! TB -> MLflow porting is accurate.
```

Pass `--no_verify` to skip the automatic check (e.g. in CI where you verify later).

#### Available options

| Option | Description | Example |
|---|---|---|
| `--tb_dir` | Path to folder containing tfevents files (required) | `--tb_dir ./logs/run_001` |
| `--experiment` | MLflow experiment name (default: from config) | `--experiment my_exp` |
| `--run_name` | MLflow run name (default: folder name + timestamp) | `--run_name ppo_v1` |
| `--tracking_uri` | MLflow server address (default: from config) | `--tracking_uri http://127.0.0.1:5100` |
| `--tags` | Per-run tags (`key=value`, space-separated); overrides config | `--tags seed=42 task=grasp` |
| `--config` | Path to JSON config file (default: `~/.nexus/post_config.json`) | `--config ./ci-config.json` |
| `-i`, `--interactive` | Prompt for researcher/seed/task, with config values as defaults | `-i` |
| `--force` | Skip required-tag validation (researcher, seed, task) | `--force` |
| `--no_verify` | Skip automatic post-upload verification | `--no_verify` |
| `--dry_run` | Print parsing results only without uploading (also skips validation) | `--dry_run` |
| `--upload_artifacts` | Also attach tfevents files as MLflow artifacts | `--upload_artifacts` |

---

### B-4. Validate upload accuracy

Automatic verification runs at the end of every upload — no manual step needed. If you want to re-verify a previous upload, or validated an upload that was run with `--no_verify`:

```bash
python post_upload/verify_upload.py \
    --run_id       a1b2c3d4e5f6...   \
    --tb_dir       /path/to/your/logs/run_001 \
    --tracking_uri http://127.0.0.1:5100
```

Three items are checked:

| Check Item | Meaning |
|---|---|
| Tag list fully matched | All metric names from TB also exist in MLflow |
| Data point counts matched | Data point count for each metric is identical |
| Values within tolerance | All values match within tolerance (default `1e-6`) |

If all pass:

```
✓ All checks passed! TB -> MLflow porting is accurate.
```

---

### B-5. Verify results in MLflow UI

Open `http://localhost:5100` in a browser and verify the uploaded run.

1. Click the experiment name (`robot_hand_rl`) in the left sidebar.
2. Click the run you just uploaded (`ppo_baseline_v1`) in the run list.
3. **Metrics** tab → Click a metric name to see the training curve graph.
4. **Parameters** tab → The tags specified with `--tags` should appear here.

Select multiple runs and click the **Compare** button to compare curves side by side.

---

### ✅ Phase 1-B Checklist

- [ ] `~/.nexus/post_config.json` populated with tracking_uri, researcher, team-fixed tags
- [ ] Confirmed tfevents file existence with `ls`
- [ ] Reviewed metric parsing results after `--dry_run`
- [ ] Ran actual upload; automatic verification printed `✓ All checks passed!`
- [ ] Training curve graphs confirmed in MLflow UI

---

## 📦 Phase 2 — GPU Server Dependency Installation (Offline Environment)

> **Problem:** The GPU server has no internet access, so `pip install` does not work.
>
> **Solution:** Download package files (.whl) in advance on an internet-connected machine and transfer them via SCP.

Choose the appropriate method for your situation from the two options below.

---

### Method A — pip wheel offline transfer *(No Docker required, recommended)*

This method works without Docker and transfers only Python packages, so the file size is small.

> **Critical — wheels must match the GPU server's Python, not your local Python.** The OS/Python version on your local PC and the GPU server almost always differ. The `--python-version` flag below refers to the **target (GPU server) Python**, not your local one. Your local Python only needs to be new enough to run `pip download` (any 3.8+ is fine).

#### A-1. On local machine — Download wheel files

**Step 1 — Check the GPU server's Python version and architecture first:**

```bash
# SSH into the GPU server and run:
python3 --version                                          # e.g., Python 3.12.3
python3 -c "import platform; print(platform.machine())"    # e.g., x86_64
```

**Step 2 — Export those values as shell variables on your local machine** so every command below stays in sync:

```bash
# Example values — replace with whatever the GPU server reported in Step 1.
export GPU_PY=3.12                    # major.minor only (no patch)
export GPU_PLATFORM=manylinux2014_x86_64   # x86_64 → manylinux2014_x86_64
```

**Step 3 — Download the wheels for that target:**

```bash
# Run inside the nexus folder
mkdir nexus_wheels

pip download \
    --platform "${GPU_PLATFORM}" \
    --python-version "${GPU_PY}" \
    --only-binary=:all: \
    -d ./nexus_wheels \
    virtualenv \
    mlflow==2.13.0 \
    tbparse==0.0.8 \
    tensorboard==2.16.2 \
    tensorboardX \
    pandas \
    rich
```

#### A-2. On local machine — Transfer nexus code + wheel files to GPU server

```bash
# Transfer nexus_wheels folder and nexus code.
# Replace `user` with your GPU server login name. The remote `~/` expands to
# that user's home directory on the server — no need to hardcode `/home/<name>`.
scp -r nexus_wheels user@gpu-server:~/
scp -r nexus        user@gpu-server:~/
```

> If SSH uses a non-standard port, add the `-P port_number` option. Example: `scp -P 22222 -r nexus_wheels user@gpu-server:~/`

#### A-3. On GPU server — Offline installation

After SSH-ing into the GPU server. `~` below automatically expands to the current login user's home directory, so these commands work regardless of your username.

```bash
cd ~/nexus

# Step 1: Install virtualenv with system pip first (alternative to venv module)
pip install --no-index --find-links ~/nexus_wheels --break-system-packages virtualenv

# Step 2: Create virtual environment at ~/.nexus/venv (outside the repo so
#         replacing/updating the nexus source tree does not wipe it). Use the
#         same Python version you passed to `--python-version` when downloading
#         the wheels. `python3 -m virtualenv` uses the default python3 — if the
#         server has multiple versions, call it explicitly (e.g.
#         `python3.12 -m virtualenv`).
mkdir -p ~/.nexus
python3 -m virtualenv ~/.nexus/venv
source ~/.nexus/venv/bin/activate

# Step 3: Install pinned setuptools version (versions 70+ exclude pkg_resources)
pip install --force-reinstall --no-index --find-links ~/nexus_wheels "setuptools==69.5.1"

# Upgrade pip itself (works offline)
pip install --upgrade pip

# Offline install remaining packages from wheel files
pip install \
    --no-index \
    --find-links ~/nexus_wheels \
    mlflow==2.13.0 \
    tbparse==0.0.8 \
    tensorboard==2.16.2 \
    tensorboardX \
    pandas \
    rich
```

> **Why `virtualenv`:** On Ubuntu/Debian, the `pythonX.Y-venv` package (e.g. `python3.12-venv`) ships via apt and cannot be transferred as a pip wheel. `virtualenv` is a pip package that can be transferred offline as a wheel, and its usage is identical to venv.

---

### Method B — Docker image transfer *(Fully isolated environment, when Docker is available)*

If Docker is installed on the GPU server, this method is the most reliable.

#### B-1. On local machine — Write Dockerfile

Create the following `Dockerfile` in the `nexus/` folder. The base image's Python version is self-contained inside the container — it does **not** need to match the GPU server's system Python. Pick any 3.10+ tag that your code is tested on (`python:3.10-slim`, `python:3.11-slim`, `python:3.12-slim`, etc.).

```dockerfile
FROM python:3.10-slim

WORKDIR /nexus

# Install packages (uses internet at build time)
RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    tbparse==0.0.8 \
    tensorboard==2.16.2 \
    tensorboardX \
    pandas \
    rich

# Copy nexus code
COPY . /nexus/

CMD ["bash"]
```

#### B-2. On local machine — Build and save image

```bash
cd nexus/

# Build image
docker build -t nexus-env:latest .

# Save image to file (compressed)
docker save nexus-env:latest | gzip > nexus-env.tar.gz

# Check file size
ls -lh nexus-env.tar.gz
```

> Image size is typically 500MB–1GB.

#### B-3. Transfer image to GPU server

```bash
scp nexus-env.tar.gz user@gpu-server:~/
```

#### B-4. On GPU server — Load and run image

```bash
# Load image (from the SSH-logged-in user's home directory)
docker load < ~/nexus-env.tar.gz

# Run container (mount nexus folder). $HOME expands to the current user's home
# before docker sees it — no need to hardcode /home/<name>.
docker run --rm -it \
    -v "$HOME/nexus":/nexus \
    -p 5100:5100 \
    nexus-env:latest bash
```

Run all subsequent commands inside the container.

---

### Method selection criteria

| Situation | Recommended Method |
|---|---|
| No Docker on GPU server, local machine is Linux | Method A (matching platform) |
| No Docker on GPU server, local machine is macOS/Windows | Method A (platform must be specified) |
| Docker is installed on GPU server | Method B |
| Entire team needs to share the same environment | Method B |

---

## 🖥️ Phase 3 — GPU Server Validation

> Proceed while SSH-connected to the GPU server.

### 3-1. Verify installation

```bash
cd ~/nexus
source ~/.nexus/activate.sh  # For Method A (or `nexus-activate`)

python -c "import mlflow; print('mlflow:', mlflow.__version__)"
python -c "from nexus.logger import make_logger; print('logger OK')"
```

### 3-2. Start local MLflow server (inside GPU server)

Start a local MLflow on the GPU server (no internet required, loopback only):

```bash
bash scheduled_sync/start_local_mlflow.sh
```

> This MLflow is only accessible from within the GPU server (`127.0.0.1:5100`). To access it externally, use SSH tunneling:
>
> ```bash
> # On local PC terminal (tunnel GPU server MLflow to local)
> ssh -L 5100:127.0.0.1:5100 user@gpu-server
> # Then access http://localhost:5100 in the local browser
> ```

### 3-3. Run smoke test

```bash
python tests/smoke_test.py
```

All items must show `[PASS]`, same as on the local PC.

### 3-4. Integrated test with actual training code (optional)

If you have modified the PPO code, run a short training session (e.g., 100 steps) to verify that actual metrics are being recorded.

```python
# In the training code initialization section (refer to docs/LOGGER_SETUP.md)
from nexus.logger import make_logger
import os

self.writer = make_logger(
    mode="dual",
    tb_dir=output_dir,
    run_name=os.path.basename(output_dir),
    tracking_uri="http://127.0.0.1:5100",
    experiment_name="robot_hand_rl",
    params=agent_cfg,
    tags={
        "researcher": os.environ.get("USER", "unknown"),
        "seed":       str(agent_cfg.get("seed", -1)),
        "task":       agent_cfg.get("task", "unknown"),
        "hardware":   "robot_22dof",
    },
)
```

After training starts, verify that metrics are accumulating in real time in the MLflow UI (`http://localhost:5100`) via SSH tunneling.

### ✅ Phase 3 Checklist

- [ ] `import mlflow` succeeds on GPU server
- [ ] `bash scheduled_sync/start_local_mlflow.sh` runs without errors
- [ ] All `[PASS]` in `python tests/smoke_test.py`
- [ ] (Optional) Metrics confirmed in MLflow UI after PPO run

---

## 🔄 Phase 4 — Cross-Server Sync Validation

> This step synchronizes experiment data from the GPU server to the central MLflow server (NEXUS server). Proceed after the NEXUS server is ready.

### 4-1. Pipeline A — Delta Sync (MLflow incremental)

> [!IMPORTANT]
> **Recommended order** — after writing the config file, proceed in this sequence:
>
> 1. `validate_sync.sh` — pre-flight check (SSH, permissions, dry-run)
> 2. `sync_mlflow_to_server.sh` — manual run to confirm real data transfer end-to-end
> 3. `crontab -e` — register the cron job
> 4. **Start training** — cron must be registered first so sync begins from step 0
>
> Registering cron after training has already started causes no data loss (`~/.nexus/sync_state/` tracks the full history), but any metrics logged before cron was registered will be uploaded in bulk on the next cron run rather than incrementally.

#### Step 1 — One-time setup: sync config file(s)

The fixed values live in a config file so the cron line is a single bash invocation. Two locations are auto-discovered:

| Path | Owner | Typical contents |
|---|---|---|
| `/etc/nexus/sync_config.json` | Operator (root) | Team-wide values: `remote`, `remote_nexus_dir`, `remote_uri`, `ssh_port` |
| `~/.nexus/sync_config.json`   | Each user       | Per-user overrides: `researcher`, `ssh_key`, optionally a different `experiment` |

Per-key merge: user file overrides system file. CLI flags still win over both. If a single-user team prefers, all values can live in `~/.nexus/sync_config.json` alone.

```bash
# Per-user setup (always applicable):
mkdir -p ~/.nexus
cp scheduled_sync/sync_config.example.json ~/.nexus/sync_config.json
$EDITOR ~/.nexus/sync_config.json
```

Required keys (anywhere in the resolution chain): `experiment`, `remote`, `remote_nexus_dir`. Optional: `researcher`, `remote_python`, `ssh_key`, `ssh_port`, `local_uri`, `remote_uri`, `state_file`.

> 💡 **`remote_python`** — Non-interactive SSH does not source `~/.bashrc`, so the MLflow server's venv is never activated and `python3` resolves to the system interpreter (which has no `mlflow`). Set this to the full path of the venv Python on the MLflow server: `"/opt/nexus-mlflow/venv/bin/python3"`.

> ⚠️ **Multi-user GPU servers**: when several researchers share one GPU server (and one local MLflow), each user **MUST** set their own `researcher` in `~/.nexus/sync_config.json`. Without it, every user's cron exports every other user's runs and the central server logs duplicate metric points at identical steps. The validator flags this with a `[WARN]`.

#### Step 2 — Pre-flight check

`validate_sync.sh` runs the same config resolution as `sync_mlflow_to_server.sh`, then verifies SSH, remote inbox writability, presence of `import_delta.py` on the central server, central MLflow `/health`, local MLflow + experiment existence, and finally executes a `--dry-run`. A clean run prints a paste-ready cron line — it never edits your crontab.

```bash
bash scheduled_sync/validate_sync.sh
# or with a non-default config path:
bash scheduled_sync/validate_sync.sh --config /etc/nexus/sync.json
```

Every failure prints what to fix; the script exits 2 on the first failed step rather than continuing in a broken state.

#### Step 3 — Run once manually

```bash
bash scheduled_sync/sync_mlflow_to_server.sh
# or, if your config lives elsewhere:
bash scheduled_sync/sync_mlflow_to_server.sh --config /etc/nexus/sync.json
```

> 💡 `--remote_nexus_dir` is the path where nexus is installed on the NEXUS server (e.g., `/opt/nexus`). Required to locate `import_delta.py` on the server.

> 💡 Add `--dry-run` to exercise the local export step only (state file is updated, no SCP, no remote import). Useful before committing the cron entry.

On success, the run will appear in the NEXUS server's MLflow UI. The local state file (`~/.nexus/sync_state/{experiment}.json`) records the last synced step for each run and tag. Each imported run also gets `nexus.lastSyncTime` and `nexus.syncedFromHost` tags so you can spot stale GPU servers from the central UI.

**On second run:** If there are no new metrics, SCP is skipped with the message `[OK] No new data since last sync.`

#### Step 4 — Register cron

```bash
crontab -e
# Add the following line (runs every 5 minutes).
# $HOME is set by cron to the crontab owner's home directory, so the same
# line works for any user without editing /home/<name> paths.
*/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh \
    >> $HOME/nexus_sync.log 2>&1
```

Need a per-key override (e.g. running an alternate experiment from one cron line)? Add the matching CLI flag — flags win over the config file:

```cron
*/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh \
    --experiment robot_hand_rl_pilot >> $HOME/nexus_sync.log 2>&1
```

#### Step 5 — Multi-user GPU servers

When kim, lee, and park all train on the same GPU server, each user runs their own cron:

1. **Each user sets their own `researcher`** in `~/.nexus/sync_config.json`. This scopes their export to runs tagged with that researcher; otherwise everyone re-exports everyone else's runs and the central server gets duplicate metric points.
2. **Operator puts shared values in `/etc/nexus/sync_config.json`** (root-writable, world-readable): `remote`, `remote_nexus_dir`, `remote_uri`, `ssh_port`. Each user's `~/.nexus/sync_config.json` then only carries `researcher` and `ssh_key`.
3. **Stagger cron offsets** so SSH/SCP traffic to the central server is spread out across the interval:

   ```cron
   # kim
   0-59/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh >> $HOME/nexus_sync.log 2>&1
   # lee — offset by 1 minute
   1-59/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh >> $HOME/nexus_sync.log 2>&1
   # park — offset by 2 minutes
   2-59/5 * * * * bash $HOME/nexus/scheduled_sync/sync_mlflow_to_server.sh >> $HOME/nexus_sync.log 2>&1
   ```

   The wrapper writes per-user, per-PID delta filenames (`delta_${USER}_<TS>_<PID>.json`) so concurrent runs don't corrupt each other's `/tmp` files or remote inbox even if you don't stagger — staggering is just polite.

### 4-2. Pipeline B — One-time batch upload (post_upload validation)

If you have existing tfevents files, you can upload them one time after training completes. Use Pipeline A for scheduled sync.

With `~/.nexus/post_config.json` pointing at the central server (`tracking_uri: http://nexus-server:5000`), the upload is one line and verification is automatic:

```bash
python post_upload/tb_to_mlflow.py \
    --tb_dir   /path/to/logs/run_001 \
    --run_name ppo_baseline_v1 \
    --tags     seed=42 task=in_hand_reorientation
```

See Phase 1-B (B-3, B-4) above for the full options reference and for re-running `verify_upload.py` standalone.

### ✅ Phase 4 Checklist

- [ ] `~/.nexus/sync_config.json` populated with experiment, remote, remote_nexus_dir
- [ ] `bash scheduled_sync/validate_sync.sh` reports "All checks passed"
- [ ] `sync_mlflow_to_server.sh` manual run successful
- [ ] Synced run confirmed in NEXUS server MLflow UI
- [ ] Auto-run confirmed after cron registration (`cat ~/nexus_sync.log`)
- [ ] (Optional) Pipeline B one-time upload succeeded with automatic verification

---

## 🛠️ Troubleshooting

### ⚠️ When MLflow server won't start

```bash
# Check existing processes
lsof -i :5100

# Kill all master + worker processes, then restart
lsof -ti :5100 | xargs kill
bash scheduled_sync/start_local_mlflow.sh
```

> **Why `kill $(cat .mlflow_local.pid)` doesn't work:** MLflow uses gunicorn internally to spawn multiple worker processes. The PID file only stores the master PID, so killing only the master leaves workers as orphan processes. Terminating by port kills both master and all workers at once.

### ⚠️ `externally-managed-environment` error during `pip install`

This error (PEP 668) is raised by distros that mark the system Python as externally managed — typically Ubuntu 23.04+ / Debian 12+ or any Python 3.11+ system install. Add the `--break-system-packages` flag when bootstrapping `virtualenv`:

```bash
pip install --no-index --find-links ~/nexus_wheels --break-system-packages virtualenv
```

After creating and activating a virtual environment (e.g. `python3 -m virtualenv venv`, or `python3.X -m virtualenv venv` to pin a specific version), the venv's internal pip is used and this error will not occur again.

### ⚠️ `ModuleNotFoundError: No module named 'pkg_resources'`

Starting from setuptools version 70+, `pkg_resources` has been removed from wheels. Switch to the last stable version that includes `pkg_resources` (69.5.1):

```bash
# On local machine: download pinned version (uses GPU_PY / GPU_PLATFORM from A-1)
pip download --platform "${GPU_PLATFORM}" --python-version "${GPU_PY}" \
    --only-binary=:all: -d ./nexus_wheels "setuptools==69.5.1"

scp nexus_wheels/setuptools-69.5.1*.whl user@gpu-server:~/nexus_wheels/

# On GPU server: force reinstall
pip install --force-reinstall --no-index --find-links ~/nexus_wheels "setuptools==69.5.1"

# Verify
python -c "import pkg_resources; print('OK')"
```

### ⚠️ Some packages fail with `pip download --only-binary`

Some packages don't have binary wheels and require source compilation. Separate individual packages instead of using `--only-binary=:all:`:

```bash
# Packages with binary wheels (uses GPU_PY / GPU_PLATFORM from A-1)
pip download --platform "${GPU_PLATFORM}" --python-version "${GPU_PY}" \
    --only-binary=:all: -d ./nexus_wheels \
    mlflow==2.13.0 pandas rich virtualenv

# Packages without binary wheels, download separately
pip download -d ./nexus_wheels \
    tbparse==0.0.8 tensorboardX
```

The `--no-build-isolation` option may also be required when installing on the GPU server:

```bash
pip install --no-index --find-links ./nexus_wheels --no-build-isolation \
    tbparse==0.0.8
```

### ⚠️ SSH connection keeps dropping (long-running sessions)

```bash
# Run in background with nohup + log file
nohup bash scheduled_sync/start_local_mlflow.sh > mlflow_local.log 2>&1 &
```

Or use `tmux`/`screen`:

```bash
tmux new -s nexus
bash scheduled_sync/start_local_mlflow.sh
# Ctrl+B, D to detach
```

### ⚠️ `tbparse` import error (`protobuf` version conflict)

A `protobuf` version conflict may occur between MLflow and TensorBoard:

```bash
pip install "protobuf>=3.20,<5.0"
```

### ⚠️ `MLflow server connection failed` in smoke_test.py

1. Verify MLflow server is running: `lsof -i :5100`
2. Verify correct URI is being used: `http://127.0.0.1:5100` (use IP directly instead of localhost)
3. Verify firewall is not blocking the port: `curl http://127.0.0.1:5100/health`

---

## ⚡ Quick Reference — Key Commands

```bash
# Activate environment
source ~/.nexus/activate.sh   # or: nexus-activate

# Start local MLflow
bash scheduled_sync/start_local_mlflow.sh

# Smoke test
python tests/smoke_test.py

# Smoke test (different server URI)
python tests/smoke_test.py --tracking_uri http://nexus-server:5000

# Stop MLflow server (master + all workers)
lsof -ti :5100 | xargs kill

# Access GPU server MLflow locally (SSH tunnel)
ssh -L 5100:127.0.0.1:5100 user@gpu-server
```

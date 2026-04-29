# 📦 Air-gapped GPU Node Setup

> **Purpose:** Bring up a GPU server that has no internet access. Download Python packages on an internet-connected machine, transfer them via SCP, and install offline.
>
> This guide is for **operators** standing up a new GPU node. Once the GPU node smoke test passes (Step C below), wire the cron sync via [`12_SCHEDULED_SYNC.md`](12_SCHEDULED_SYNC.md).

---

## 📑 Table of Contents

- [⚡ TL;DR](#-tldr)
- [📋 Prerequisites](#-prerequisites)
- [📌 Step 0 — Verify install on local PC first](#-step-0--verify-install-on-local-pc-first-recommended)
- [🅰️ Method A — pip wheel offline transfer](#-method-a--pip-wheel-offline-transfer-no-docker-required-recommended)
- [🅱️ Method B — Docker image transfer](#-method-b--docker-image-transfer-fully-isolated-environment-when-docker-is-available)
- [🧭 Method selection criteria](#-method-selection-criteria)
- [📌 Step C — Verify install on the GPU node](#-step-c--verify-install-on-the-gpu-node)
- [🛠️ Troubleshooting](#-troubleshooting)
- [🗺️ Next steps](#-next-steps)

---

## ⚡ TL;DR

```bash
# On your internet-connected local PC — pin to the GPU server's Python
export GPU_PY=3.12 GPU_PLATFORM=manylinux2014_x86_64
pip download --platform "$GPU_PLATFORM" --python-version "$GPU_PY" \
    --only-binary=:all: -d ./nexus_wheels \
    virtualenv mlflow==2.13.0 tbparse==0.0.8 tensorboard==2.16.2 tensorboardX pandas rich

# Transfer to the GPU server
scp -r nexus_wheels nexus user@gpu-server:~/

# On the GPU server — offline install via virtualenv
cd ~/nexus
pip install --no-index --find-links ~/nexus_wheels --break-system-packages virtualenv
python3 -m virtualenv ~/.nexus/venv && source ~/.nexus/venv/bin/activate
pip install --no-index --find-links ~/nexus_wheels mlflow==2.13.0 tbparse==0.0.8 \
    tensorboard==2.16.2 tensorboardX pandas rich

# Verify (Step C)
python tests/smoke_test.py
```

> [!IMPORTANT]
> **Wheels must match the GPU server's Python**, not your local Python. Always pin `--python-version` and `--platform` to the values reported by the GPU server. Defaults below assume Method A; choose Method B (Docker) if your GPU server has Docker.

---

## 📋 Prerequisites

| What | Where |
|---|---|
| Internet-connected machine (your local PC) | Used to download wheels / build Docker image |
| GPU server SSH access (key-based) | Target node — receives the transferred bundle |
| GPU server's Python version + arch | You will pin downloads against this — see Method A Step 1 |

The GPU server needs **outgoing SSH** to your local PC only for the initial transfer. After that it operates fully air-gapped except for cron-triggered SCP/SSH to the central NEXUS server (covered in [`12_SCHEDULED_SYNC.md`](12_SCHEDULED_SYNC.md)).

---

## 📌 Step 0 — Verify install on local PC first *(recommended)*

> **Why:** Confirm that NEXUS itself works on a familiar machine before pinning wheels for an offline target. If the smoke test fails locally, it will fail on the GPU node too — debug it once on your laptop.

### 0.1 Clone the repository

```bash
git clone https://github.com/jonghochoi/nexus.git
cd nexus
```

### 0.2 Verify Python version

```bash
python3 --version
```

Must be `Python 3.8` or higher. 3.10 or 3.11 is recommended.

### 0.3 Install environment

```bash
bash setup.sh
source ~/.nexus/activate.sh   # or: nexus-activate (if you ran setup.sh --alias)
```

The venv lives at `~/.nexus/venv` — **outside** the repo — so replacing or re-cloning nexus sources does not wipe it.

### 0.4 Verify installation

```bash
python -c "import mlflow; print('mlflow:', mlflow.__version__)"
python -c "import tbparse; print('tbparse OK')"
python -c "from nexus.logger import make_logger; print('logger OK')"
```

If all three lines output without errors, the package install is sound.

### 0.5 Start local MLflow + run smoke test

```bash
bash scheduled_sync/start_local_mlflow.sh   # boots :5100
python tests/smoke_test.py                  # all items must report [PASS]
```

The smoke test creates real runs under the `nexus_smoke_test` experiment on `http://localhost:5100`. Open the UI in a browser to confirm the runs are visible.

✅ **Step 0 done when:** `python tests/smoke_test.py` shows `All tests passed!` and `nexus_smoke_test` runs are visible in the local MLflow UI. Now you know the package itself works — proceed to one of the offline methods below.

---

## 🅰️ Method A — pip wheel offline transfer *(no Docker required, recommended)*

This method works without Docker and transfers only Python packages, so the bundle size is small (~100MB).

> ⚠️ **Wheels must match the GPU server's Python, not your local Python.** The OS / Python version on your local PC and the GPU server almost always differ. The `--python-version` flag below refers to the **target (GPU server) Python**, not your local one. Your local Python only needs to be new enough to run `pip download` (any 3.8+ is fine).

### A.1 On local machine — Download wheel files

**Step 1 — Check the GPU server's Python version and architecture first:**

```bash
# SSH into the GPU server and run:
python3 --version                                          # e.g., Python 3.12.3
python3 -c "import platform; print(platform.machine())"    # e.g., x86_64
```

**Step 2 — Export those values as shell variables on your local machine** so every command below stays in sync:

```bash
# Example values — replace with whatever the GPU server reported in Step 1.
export GPU_PY=3.12                         # major.minor only (no patch)
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

### A.2 On local machine — Transfer nexus code + wheel files to GPU server

```bash
# Transfer nexus_wheels folder and nexus code.
# Replace `user` with your GPU server login name. The remote `~/` expands to
# that user's home directory on the server — no need to hardcode `/home/<name>`.
scp -r nexus_wheels user@gpu-server:~/
scp -r nexus        user@gpu-server:~/
```

> If SSH uses a non-standard port, add the `-P port_number` option. Example: `scp -P 22222 -r nexus_wheels user@gpu-server:~/`

### A.3 On GPU server — Offline installation

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

## 🅱️ Method B — Docker image transfer *(fully isolated environment, when Docker is available)*

If Docker is installed on the GPU server, this method is the most reliable.

### B.1 On local machine — Write Dockerfile

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

### B.2 On local machine — Build and save image

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

### B.3 Transfer image to GPU server

```bash
scp nexus-env.tar.gz user@gpu-server:~/
```

### B.4 On GPU server — Load and run image

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

## 🧭 Method selection criteria

| Situation | Recommended Method |
|---|---|
| No Docker on GPU server, local machine is Linux | Method A (matching platform) |
| No Docker on GPU server, local machine is macOS/Windows | Method A (platform must be specified) |
| Docker is installed on GPU server | Method B |
| Entire team needs to share the same environment | Method B |

---

## 📌 Step C — Verify install on the GPU node

> **Purpose:** Confirm the offline install actually works before wiring cron sync. Run all commands while SSH-connected to the GPU server.

### C.1 Verify installation

```bash
cd ~/nexus
source ~/.nexus/activate.sh   # or: nexus-activate

python -c "import mlflow; print('mlflow:', mlflow.__version__)"
python -c "from nexus.logger import make_logger; print('logger OK')"
```

### C.2 Start local MLflow server (inside GPU server)

```bash
bash scheduled_sync/start_local_mlflow.sh
```

> This MLflow is only accessible from within the GPU server (`127.0.0.1:5100`). To access it from your local PC, use SSH tunneling:
>
> ```bash
> # On local PC terminal (tunnel GPU server MLflow to local)
> ssh -L 5100:127.0.0.1:5100 user@gpu-server
> # Then access http://localhost:5100 in the local browser
> ```

### C.3 Run smoke test

```bash
python tests/smoke_test.py
```

All items must show `[PASS]`, same as on the local PC.

### C.4 Integrated test with actual training code (optional)

If you have modified the trainer to use `make_logger`, run a short training session (e.g., 100 steps) to verify metrics are recorded. See [`11_LOGGER_SETUP.md`](11_LOGGER_SETUP.md) for the integration diff.

```python
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

After training starts, verify metrics accumulate in real time in the MLflow UI (`http://localhost:5100`) via SSH tunneling.

### ✅ Step C checklist

- [ ] `import mlflow` succeeds on the GPU server
- [ ] `bash scheduled_sync/start_local_mlflow.sh` runs without errors and the UI is reachable via SSH tunnel
- [ ] All `[PASS]` in `python tests/smoke_test.py`
- [ ] *(Optional)* Live metrics confirmed in the MLflow UI after a short training run

Once Step C passes, proceed to [`12_SCHEDULED_SYNC.md`](12_SCHEDULED_SYNC.md) to wire the cron sync to the central NEXUS server.

---

## 🛠️ Troubleshooting

### ⚠️ `externally-managed-environment` error during `pip install`

This error (PEP 668) is raised by distros that mark the system Python as externally managed — typically Ubuntu 23.04+ / Debian 12+ or any Python 3.11+ system install. Add the `--break-system-packages` flag when bootstrapping `virtualenv`:

```bash
pip install --no-index --find-links ~/nexus_wheels --break-system-packages virtualenv
```

After creating and activating a virtual environment (e.g. `python3 -m virtualenv venv`, or `python3.X -m virtualenv venv` to pin a specific version), the venv's internal pip is used and this error will not occur again.

### ⚠️ `ModuleNotFoundError: No module named 'pkg_resources'`

Starting from setuptools version 70+, `pkg_resources` has been removed from wheels. Switch to the last stable version that includes `pkg_resources` (69.5.1):

```bash
# On local machine: download pinned version (uses GPU_PY / GPU_PLATFORM from A.1)
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
# Packages with binary wheels (uses GPU_PY / GPU_PLATFORM from A.1)
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

### ⚠️ `tbparse` import error (`protobuf` version conflict)

A `protobuf` version conflict may occur between MLflow and TensorBoard:

```bash
pip install "protobuf>=3.20,<5.0"
```

### ⚠️ SSH connection keeps dropping (long-running sessions)

```bash
# Run in background with nohup + log file
nohup bash scheduled_sync/start_local_mlflow.sh > mlflow_local.log 2>&1 &
```

Or use `tmux` / `screen`:

```bash
tmux new -s nexus
bash scheduled_sync/start_local_mlflow.sh
# Ctrl+B, D to detach
```

### ⚠️ Local MLflow server won't start (`start_local_mlflow.sh`)

```bash
# Check existing processes
lsof -i :5100

# Kill all master + worker processes, then restart
lsof -ti :5100 | xargs kill
bash scheduled_sync/start_local_mlflow.sh
```

> **Why `kill $(cat .mlflow_local.pid)` doesn't work:** MLflow uses gunicorn internally to spawn multiple worker processes. The PID file only stores the master PID, so killing only the master leaves workers as orphan processes. Terminating by port kills both master and all workers at once.

### ⚠️ `MLflow server connection failed` in `smoke_test.py`

1. Verify MLflow server is running: `lsof -i :5100`
2. Verify correct URI is being used: `http://127.0.0.1:5100` (use IP directly instead of localhost)
3. Verify firewall is not blocking the port: `curl http://127.0.0.1:5100/health`

### ⚠️ Smoke test reports `[FAIL] DualLogger`

Almost always this means the local MLflow at `:5100` is not running, since dual mode exercises both TB and MLflow paths. Re-run `bash scheduled_sync/start_local_mlflow.sh`, confirm the UI is reachable in a browser, then re-run the smoke test.

---

## 🗺️ Next steps

After Step C passes:

1. **Wire cron sync to central server** → [`12_SCHEDULED_SYNC.md`](12_SCHEDULED_SYNC.md)
2. **Pipeline B alternative** (post-upload, no live sync) → [`13_POST_UPLOAD.md`](13_POST_UPLOAD.md)

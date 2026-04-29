# 🖥️ NEXUS MLflow Server Setup Guide

> **Audience:** For those setting up an Ubuntu Linux PC as an internal MLflow server for the first time
>
> **Environment:** Ubuntu 22.04 LTS / Internal company LAN / Connected to Blackwell GPU server via SCP

---

## 📋 Overview of Steps

```
Step 0  Verify install on local PC first  (recommended)
Step 1  Understand network topology
Step 2  Basic server PC setup
Step 3  Install MLflow and configure directories
Step 4  Functional test
Step 5  Firewall / port configuration
Step 6  Register systemd service
Step 7  Verify team member access
Step 8  Verify Blackwell connection
```

---

## 💻 Step 0 — Verify install on local PC first *(recommended)*

> **Why:** Confirm that NEXUS itself works on a familiar machine before touching the server. If the smoke test fails locally, it will fail on the server too — debug it once on your laptop.

### 0-1. Clone the repository

```bash
git clone https://github.com/jonghochoi/nexus.git
cd nexus
```

### 0-2. Verify Python version

```bash
python3 --version
```

Must be `Python 3.8` or higher. 3.10 or 3.11 is recommended.

### 0-3. Install environment

```bash
bash setup.sh
source ~/.nexus/activate.sh   # or: nexus-activate (if you ran setup.sh --alias)
```

The venv lives at `~/.nexus/venv` — **outside** the repo — so replacing or re-cloning nexus sources does not wipe it.

### 0-4. Verify installation

```bash
python -c "import mlflow; print('mlflow:', mlflow.__version__)"
python -c "import tbparse; print('tbparse OK')"
python -c "from nexus.logger import make_logger; print('logger OK')"
```

If all three lines output without errors, the package install is sound.

### 0-5. Start local MLflow + run smoke test

```bash
bash scheduled_sync/start_local_mlflow.sh   # boots :5100
python tests/smoke_test.py                  # all items must report [PASS]
```

The smoke test creates real runs under the `nexus_smoke_test` experiment on `http://localhost:5100`. Open the UI in a browser to confirm the runs are visible.

> 🛠️ **If smoke_test.py fails connecting to MLflow:** verify the URI is `http://127.0.0.1:5100` (not `localhost`), check `lsof -i :5100`, and confirm `curl http://127.0.0.1:5100/health` returns `OK`. Common operations and troubleshooting for `start_local_mlflow.sh` are also covered in [`21_AIRGAPPED_GPU_SERVER_SETUP.md` Troubleshooting](21_AIRGAPPED_GPU_SERVER_SETUP.md#-troubleshooting).

✅ **Step 0 done when:** `python tests/smoke_test.py` shows `All tests passed!` and `nexus_smoke_test` runs are visible in the local MLflow UI. Proceed to Step 1.

---

## 🔍 Step 1 — Understand Network Topology

Before the actual installation, confirm that both servers are on the same internal network.

### 1-1. Check the MLflow server (spare PC) IP address

**Run on the spare PC:**

```bash
ip addr show | grep "inet " | grep -v 127.0.0.1
```

**Expected output example:**

```
inet 192.168.1.42/24 brd 192.168.1.255 scope global ens3
```

> 💡 The `192.168.1.42` portion is this server's IP address. Share this with your team members.

---

### 1-2. Verify connectivity from Blackwell to the MLflow server

**Run on the Blackwell server:**

```bash
ping 192.168.1.42 -c 4   # Replace IP with the value found above
```

**Expected output (success):**

```
PING 192.168.1.42 (192.168.1.42) 56(84) bytes of data.
64 bytes from 192.168.1.42: icmp_seq=1 ttl=64 time=0.412 ms
64 bytes from 192.168.1.42: icmp_seq=2 ttl=64 time=0.388 ms
64 bytes from 192.168.1.42: icmp_seq=3 ttl=64 time=0.401 ms
64 bytes from 192.168.1.42: icmp_seq=4 ttl=64 time=0.395 ms

--- 192.168.1.42 ping statistics ---
4 packets transmitted, 4 received, 0% packet loss
```

> ✅ If you see `0% packet loss`, both servers are on the same network.
>
> ❌ If you see `Request timeout` or `100% packet loss`, contact your network administrator to check the VLAN settings for both servers.

---

## 🛠️ Step 2 — Basic Server PC Setup

**All subsequent commands are run on the spare PC (MLflow server).**

### 2-1. System update

```bash
sudo apt update && sudo apt upgrade -y
```

**Expected output:**

```
Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease
...
Calculating upgrade... Done
0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.
```

> ⏱️ If there are many updates, this may take several minutes.

---

### 2-2. Verify Python environment

```bash
python3 --version
pip3 --version
```

**Expected output:**

```
Python 3.10.12
pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)
```

> ⚠️ MLflow will not work with Python versions below 3.8. Ubuntu 22.04 comes with Python 3.10 by default, so this should not be an issue in most cases.

---

### 2-3. Install required packages

```bash
sudo apt install -y python3-pip python3-venv git curl net-tools
```

**Expected output:**

```
Reading package lists... Done
Building dependency tree... Done
The following NEW packages will be installed:
  python3-venv net-tools ...
Setting up python3-venv (3.10.6-1~22.04) ...
```

---

## 📦 Step 3 — Install MLflow and Configure Directories

### 3-1. Create working directory

```bash
sudo mkdir -p /opt/nexus-mlflow
sudo chown $USER:$USER /opt/nexus-mlflow
cd /opt/nexus-mlflow
```

**Verify:**

```bash
ls -la /opt/nexus-mlflow
# drwxr-xr-x 2 yourname yourname 4096 Apr 18 10:00 .
```

---

### 3-2. Create Python virtual environment and install MLflow

```bash
python3 -m venv venv
source venv/bin/activate
pip install mlflow==2.13.0
```

**Expected output (on successful installation):**

```
Collecting mlflow==2.13.0
  Downloading mlflow-2.13.0-py3-none-any.whl (24.5 MB)
...
Successfully installed mlflow-2.13.0 ...
```

**Verify installation:**

```bash
mlflow --version
```

**Expected output:**

```
mlflow, version 2.13.0
```

---

### 3-3. Create data storage directories

```bash
mkdir -p /opt/nexus-mlflow/mlruns       # Experiment metadata DB
mkdir -p /opt/nexus-mlflow/artifacts    # Checkpoint and config file storage
mkdir -p /opt/nexus-mlflow/sync_inbox   # Blackwell SCP receive directory
```

**Verify directory structure:**

```bash
tree /opt/nexus-mlflow
```

**Expected output:**

```
/opt/nexus-mlflow
├── artifacts/       ← Where best.pth, last.pth, etc. will be stored
├── mlruns/          ← MLflow experiment DB (metrics, params, tags)
├── sync_inbox/      ← Temporary receive folder for SCP transfers from Blackwell
└── venv/            ← Python virtual environment
```

> 💡 **Check disk space in advance:**
>
> ```bash
> df -h /opt/nexus-mlflow
> ```
>
> Since checkpoints accumulate in `artifacts/`, we recommend a drive with at least **100GB** of free space. If space is limited, you can change the path to an external HDD.

---

## ✅ Step 4 — Functional Test (Manual Run)

Before registering the service, run it manually first to verify it works correctly.

```bash
cd /opt/nexus-mlflow
source venv/bin/activate

mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri /opt/nexus-mlflow/mlruns \
    --artifacts-destination /opt/nexus-mlflow/artifacts \
    --serve-artifacts
```

> 🔑 **Why `--serve-artifacts` (proxied artifact storage)?**
>
> With `--default-artifact-root <local-path>` (the old style), MLflow **clients** (e.g. the GPU server running `mlflow.log_artifacts()` or `tb_to_mlflow.py --upload_artifacts`) try to write directly to that local filesystem path. Remote clients have no access to `/opt/nexus-mlflow/artifacts` on the server, so the upload fails with `Permission denied`.
>
> `--serve-artifacts` combined with `--artifacts-destination` makes the **server** accept artifact uploads over HTTP and persist them on its own disk. New runs get an `mlflow-artifacts:/` URI automatically; clients need no extra config.
>
> If you previously ran with `--default-artifact-root`, runs created before the switch keep their original on-disk `artifact_uri` — they continue to work via the server's filesystem access; only new runs use the proxied path.

**Expected output:**

```
[2025-04-18 10:15:32 +0900] [12345] [INFO] Starting gunicorn 21.2.0
[2025-04-18 10:15:32 +0900] [12345] [INFO] Listening at: http://0.0.0.0:5000
[2025-04-18 10:15:32 +0900] [12345] [INFO] Using worker: sync
[2025-04-18 10:15:32 +0900] [12346] [INFO] Booting worker with pid: 12346
```

> ✅ If you see `Listening at: http://0.0.0.0:5000`, it is working correctly.
>
> At this point, accessing `http://localhost:5000` in a browser **on the same PC** should display the MLflow UI.

Press `Ctrl + C` to stop when done.

---

## 🔥 Step 5 — Firewall / Port Configuration

### 5-1. Install and verify SSH server (sshd)

> ⚠️ **Freshly formatted PCs often do not have `openssh-server` installed.** If the SSH daemon is not running, Blackwell will encounter a `ssh: connect to host 192.168.1.42 port 22: Connection refused` error in Step 8. Verify that sshd is actually running before opening the firewall.

**1) Check SSH daemon status:**

```bash
sudo systemctl status ssh
```

**Expected output (when running normally):**

```
● ssh.service - OpenBSD Secure Shell server
     Loaded: loaded (/lib/systemd/system/ssh.service; enabled; ...)
     Active: active (running) since ...
```

- `Active: active (running)` → Proceed to Step 5-2.
- `Unit ssh.service could not be found` or `inactive (dead)` → Install/start using step 2) below.

**2) Install and start `openssh-server`:**

```bash
sudo apt update
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
```

**3) Verify port 22 is listening:**

```bash
sudo ss -tlnp | grep :22
```

**Expected output:**

```
LISTEN 0  128  0.0.0.0:22  0.0.0.0:*  users:(("sshd",...))
```

> ✅ If the above output appears, sshd is listening on port 22 correctly.

---

### 5-2. Check current firewall status

```bash
sudo ufw status
```

**Expected output (when inactive):**

```
Status: inactive
```

**Expected output (when active):**

```
Status: active

To                         Action      From
--                         ------      ----
22/tcp                     ALLOW       Anywhere
```

---

### 5-3. Open required ports

```bash
# SSH (required for Blackwell SCP transfer)
sudo ufw allow ssh

# MLflow UI and API (team member access + Blackwell integration)
sudo ufw allow 5000/tcp comment 'NEXUS MLflow Server'
```

---

### 5-4. Enable firewall

```bash
sudo ufw enable
```

**Expected output:**

```
Command may disrupt existing ssh connections. Proceed with operation (y|n)? y
Firewall is active and enabled on system startup
```

**Verify settings:**

```bash
sudo ufw status verbose
```

**Expected output:**

```
Status: active
Logging: on (low)
Default: deny (incoming), allow (outgoing), disabled (routed)
New profiles: skip

To                         Action      From
--                         ------      ----
22/tcp                     ALLOW IN    Anywhere
5000/tcp (NEXUS MLflow)    ALLOW IN    Anywhere
```

> ✅ If `5000/tcp` shows `ALLOW IN`, team members can connect.

---

## ⚙️ Step 6 — Register systemd Service (Auto-start)

Register the MLflow server to start automatically even after a PC reboot.

### 6-1. Check current username

```bash
echo $USER
# Example: jonghochoi
```

### 6-2. Create service file

```bash
sudo tee /etc/systemd/system/nexus-mlflow.service > /dev/null << EOF
[Unit]
Description=NEXUS MLflow Tracking Server
Documentation=https://github.com/jonghochoi/nexus
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/nexus-mlflow
Environment=PATH=/opt/nexus-mlflow/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/opt/nexus-mlflow/venv/bin/mlflow server \\
    --host 0.0.0.0 \\
    --port 5000 \\
    --backend-store-uri /opt/nexus-mlflow/mlruns \\
    --artifacts-destination /opt/nexus-mlflow/artifacts \\
    --serve-artifacts
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

**Verify creation:**

```bash
cat /etc/systemd/system/nexus-mlflow.service
```

Confirm that the `User=` line shows your account name correctly.

---

### 6-3. Register and start service

```bash
# Reload systemd (recognize new service file)
sudo systemctl daemon-reload

# Register for auto-start on boot
sudo systemctl enable nexus-mlflow

# Start immediately
sudo systemctl start nexus-mlflow
```

---

### 6-4. Verify service status

```bash
sudo systemctl status nexus-mlflow
```

**Expected output (when running normally):**

```
● nexus-mlflow.service - NEXUS MLflow Tracking Server
     Loaded: loaded (/etc/systemd/system/nexus-mlflow.service; enabled)
     Active: active (running) since Fri 2025-04-18 10:20:00 KST; 5s ago
   Main PID: 13579 (mlflow)
      Tasks: 5 (limit: 4096)
     Memory: 120.3M
        CPU: 1.241s
     CGroup: /system.slice/nexus-mlflow.service
             └─13579 /opt/nexus-mlflow/venv/bin/python mlflow server ...

Apr 18 10:20:00 nexus-server systemd[1]: Started NEXUS MLflow Tracking Server.
Apr 18 10:20:01 nexus-server mlflow[13579]: [INFO] Listening at: http://0.0.0.0:5000
```

> ✅ If `Active: active (running)` is shown, the service is running correctly.
>
> ❌ If `Active: failed` is shown, use the log command below to identify the cause.
>
> ```bash
> sudo journalctl -u nexus-mlflow -n 50 --no-pager
> ```

---

### 6-5. Useful service management commands

```bash
# Stop service
sudo systemctl stop nexus-mlflow

# Restart service (after configuration changes)
sudo systemctl restart nexus-mlflow

# View live logs
sudo journalctl -u nexus-mlflow -f

# View all logs for today
sudo journalctl -u nexus-mlflow --since today
```

#### Register the commands above as bash aliases

Run the following block once on the MLflow server. It appends a set of `mlflow-*` aliases to `~/.bashrc` so the commands above can be invoked with a short name.

```bash
cat >> ~/.bashrc <<'EOF'

# --- NEXUS MLflow service aliases ---
alias mlflow-status='sudo systemctl status nexus-mlflow'
alias mlflow-start='sudo systemctl start nexus-mlflow'
alias mlflow-stop='sudo systemctl stop nexus-mlflow'
alias mlflow-restart='sudo systemctl restart nexus-mlflow'
alias mlflow-logs='sudo journalctl -u nexus-mlflow -f'
alias mlflow-logs-today='sudo journalctl -u nexus-mlflow --since today'
# --- end NEXUS MLflow service aliases ---
EOF

# Apply to the current shell without reopening the terminal
source ~/.bashrc
```

**Alias quick reference:**

| Alias                | Equivalent command                            |
| -------------------- | --------------------------------------------- |
| `mlflow-status`      | `sudo systemctl status nexus-mlflow`          |
| `mlflow-start`       | `sudo systemctl start nexus-mlflow`           |
| `mlflow-stop`        | `sudo systemctl stop nexus-mlflow`            |
| `mlflow-restart`     | `sudo systemctl restart nexus-mlflow`         |
| `mlflow-logs`        | `sudo journalctl -u nexus-mlflow -f`          |
| `mlflow-logs-today`  | `sudo journalctl -u nexus-mlflow --since today` |

> 💡 Verify the aliases are registered with `alias | grep mlflow-`.
>
> 💡 To remove them later, open `~/.bashrc` and delete the block between the `NEXUS MLflow service aliases` markers, then run `source ~/.bashrc`.

---

## 👥 Step 7 — Verify Team Member Access

### 7-1. Confirm final access address

```bash
hostname -I | awk '{print $1}'
# Example: 192.168.1.42
```

Share the following address with your team members:

```
http://192.168.1.42:5000
```

### 7-2. Connection test from team member PC

When a team member opens the above URL in their browser and sees the MLflow UI shown below, the setup is successful.

```
┌─────────────────────────────────────────┐
│  MLflow                                 │
│  ┌─────────────────────────────────┐    │
│  │ Experiments                     │    │
│  │  · Default                      │    │
│  └─────────────────────────────────┘    │
│                                         │
│  No runs logged yet.                    │
└─────────────────────────────────────────┘
```

### 7-3. Troubleshooting checklist when unable to connect

```bash
# Run on the MLflow server PC

# 1. Verify the service is running
sudo systemctl status nexus-mlflow

# 2. Verify port 5000 is actually open
ss -tlnp | grep 5000
# Expected output: LISTEN 0 ... 0.0.0.0:5000

# 3. Verify the firewall is not blocking the port
sudo ufw status | grep 5000
# Expected output: 5000/tcp   ALLOW IN    Anywhere
```

---

## 🔗 Step 8 — Configure Blackwell → MLflow Server Connection

### 8-1. Generate SSH key on Blackwell

**Run on the Blackwell server:**

```bash
# Generate SSH key (skip if already exists)
ssh-keygen -t ed25519 -C "blackwell-to-nexus" -f ~/.ssh/nexus_key
```

**Expected output:**

```
Generating public/private ed25519 key pair.
Enter passphrase (empty for no passphrase):    ← Just press Enter (no passphrase)
Enter same passphrase again:                   ← Just press Enter
Your identification has been saved in /home/user/.ssh/nexus_key
Your public key has been saved in /home/user/.ssh/nexus_key.pub
The key fingerprint is:
SHA256:xxxxxxxxxxxxxxxxxxxx blackwell-to-nexus
```

> 💡 Leave the passphrase empty so that automated tools like cron can execute without password prompts.

---

### 8-2. Register public key on the MLflow server

**Run on the Blackwell server:**

```bash
ssh-copy-id -i ~/.ssh/nexus_key.pub USER@192.168.1.42
# Replace USER with the MLflow server account name, and IP with the actual value
```

**Expected output:**

```
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s)
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed
USER@192.168.1.42's password:    ← Enter the MLflow server account password (one time only)

Number of key(s) added: 1

Now try logging into the machine, with:   "ssh 'USER@192.168.1.42'"
and check to make sure that only the key(s) you wanted were added.
```

---

### 8-3. Verify key-based access (without password)

```bash
ssh -i ~/.ssh/nexus_key USER@192.168.1.42 "echo 'NEXUS connection successful'"
```

**Expected output:**

```
NEXUS connection successful
```

> ✅ If the above message appears without a password prompt, SCP automation is ready.

---

### 8-4. SCP file transfer test

```bash
# Transfer test file
echo "nexus test" > /tmp/nexus_test.txt
scp -i ~/.ssh/nexus_key /tmp/nexus_test.txt USER@192.168.1.42:/opt/nexus-mlflow/sync_inbox/

# Verify file arrived on MLflow server
ssh -i ~/.ssh/nexus_key USER@192.168.1.42 "cat /opt/nexus-mlflow/sync_inbox/nexus_test.txt"
```

**Expected output:**

```
nexus test
```

---

### 8-5. Full sync pipeline validation

Once the SSH key and SCP transfer are confirmed, the server setup is complete.

For the full sync pipeline setup (config file, pre-flight check, cron registration, multi-user pattern, verification checklist), follow **[`12_SCHEDULED_SYNC.md`](12_SCHEDULED_SYNC.md)**.

---

## 🗂️ Final Configuration Summary

After setup is complete, the structure will be as follows:

```
[Spare PC — NEXUS MLflow Server]
  IP:      192.168.1.42 (example)
  OS:      Ubuntu 22.04
  Service: nexus-mlflow (systemd, auto-starts after reboot)
  Port:    5000 (open to entire internal network)

  Storage locations:
  /opt/nexus-mlflow/
  ├── mlruns/          ← Experiment metadata (metrics, params, tags)
  ├── artifacts/       ← Checkpoints
  │   └── <run_id>/
  │       └── checkpoints/
  │           ├── best.pth
  │           └── last.pth
  └── sync_inbox/      ← Blackwell SCP receive folder

[Team member access]
  Browser → http://192.168.1.42:5000

[Blackwell → NEXUS connection]
  SSH key: ~/.ssh/nexus_key (automatic transfer without password)
  Method:  SCP → mlflow export/import (cron every 5 minutes)
```

---

## ⚠️ Common Issues

| Symptom | Cause | Solution |
|---|---|---|
| Cannot access browser | Firewall blocking | `sudo ufw allow 5000/tcp` |
| `port 22: Connection refused` | `openssh-server` not installed or sshd not running on MLflow server | See Step 5-1 → `sudo apt install -y openssh-server && sudo systemctl enable --now ssh` |
| Service fails to start | Path error | Check logs with `journalctl -u nexus-mlflow -n 30` |
| SCP keeps asking for password | Key not registered | Re-run `ssh-copy-id` |
| Disk full | Artifact accumulation | Check with `df -h` and delete old run artifacts |
| Server not starting after reboot | Not enabled | `sudo systemctl enable nexus-mlflow` |
| `tb_to_mlflow.py --upload_artifacts` → `PermissionError: /opt/nexus-mlflow/artifacts/...` | Server was started with `--default-artifact-root <local-path>`, which forces the remote client to write directly to that path. | Restart the server with `--artifacts-destination /opt/nexus-mlflow/artifacts --serve-artifacts` (see Step 4 / 6-2). Clients will then upload artifacts over HTTP through the server. |

# 🖥️ NEXUS MLflow 서버 구축 가이드

> **대상:** Ubuntu Linux PC를 사내 MLflow 서버로 설정하는 과정을 처음 해보는 분
>
> **환경:** Ubuntu 22.04 LTS / 사내 LAN 내부망 / Blackwell GPU 서버와 SCP 연결

---

## 📋 전체 단계 개요

```
Step 1  네트워크 구조 파악
Step 2  서버 PC 기본 세팅
Step 3  MLflow 설치 및 디렉토리 구성
Step 4  동작 테스트
Step 5  방화벽 / 포트 설정
Step 6  systemd 서비스 등록
Step 7  팀원 접속 확인
Step 8  Blackwell 연결 확인
```

---

## 🔍 Step 1 — 네트워크 구조 파악

본격적인 설치 전에 두 서버가 같은 사내망에 있는지 확인합니다.

### 1-1. MLflow 서버(남는 PC)의 IP 확인

**남는 PC에서 실행:**

```bash
ip addr show | grep "inet " | grep -v 127.0.0.1
```

**기대 출력 예시:**

```
inet 192.168.1.42/24 brd 192.168.1.255 scope global ens3
```

> 💡 `192.168.1.42` 부분이 이 서버의 IP입니다. 팀원들과 공유할 주소입니다.

---

### 1-2. Blackwell에서 MLflow 서버로 연결 확인

**Blackwell 서버에서 실행:**

```bash
ping 192.168.1.42 -c 4   # IP는 위에서 확인한 값으로 교체
```

**기대 출력 (성공 시):**

```
PING 192.168.1.42 (192.168.1.42) 56(84) bytes of data.
64 bytes from 192.168.1.42: icmp_seq=1 ttl=64 time=0.412 ms
64 bytes from 192.168.1.42: icmp_seq=2 ttl=64 time=0.388 ms
64 bytes from 192.168.1.42: icmp_seq=3 ttl=64 time=0.401 ms
64 bytes from 192.168.1.42: icmp_seq=4 ttl=64 time=0.395 ms

--- 192.168.1.42 ping statistics ---
4 packets transmitted, 4 received, 0% packet loss
```

> ✅ `0% packet loss` 가 나오면 두 서버가 같은 망에 있는 것입니다.
>
> ❌ `Request timeout` 또는 `100% packet loss` 가 나오면 네트워크 담당자에게 두 서버의 VLAN 설정을 확인 요청해야 합니다.

---

## 🛠️ Step 2 — 서버 PC 기본 세팅

**이후 모든 명령은 남는 PC(MLflow 서버)에서 실행합니다.**

### 2-1. 시스템 업데이트

```bash
sudo apt update && sudo apt upgrade -y
```

**기대 출력:**

```
Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease
...
Calculating upgrade... Done
0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.
```

> ⏱️ 업데이트 항목이 많으면 수 분이 걸릴 수 있습니다.

---

### 2-2. Python 환경 확인

```bash
python3 --version
pip3 --version
```

**기대 출력:**

```
Python 3.10.12
pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)
```

> ⚠️ Python 3.8 미만이면 MLflow가 동작하지 않습니다.
> Ubuntu 22.04는 기본으로 Python 3.10이 설치되어 있어서 대부분 문제없습니다.

---

### 2-3. 필수 패키지 설치

```bash
sudo apt install -y python3-pip python3-venv git curl net-tools
```

**기대 출력:**

```
Reading package lists... Done
Building dependency tree... Done
The following NEW packages will be installed:
  python3-venv net-tools ...
Setting up python3-venv (3.10.6-1~22.04) ...
```

---

## 📦 Step 3 — MLflow 설치 및 디렉토리 구성

### 3-1. 작업 디렉토리 생성

```bash
sudo mkdir -p /opt/nexus-mlflow
sudo chown $USER:$USER /opt/nexus-mlflow
cd /opt/nexus-mlflow
```

**확인:**

```bash
ls -la /opt/nexus-mlflow
# drwxr-xr-x 2 yourname yourname 4096 Apr 18 10:00 .
```

---

### 3-2. Python 가상환경 생성 및 MLflow 설치

```bash
python3 -m venv venv
source venv/bin/activate
pip install mlflow==2.13.0
```

**기대 출력 (설치 완료 시):**

```
Collecting mlflow==2.13.0
  Downloading mlflow-2.13.0-py3-none-any.whl (24.5 MB)
...
Successfully installed mlflow-2.13.0 ...
```

**설치 확인:**

```bash
mlflow --version
```

**기대 출력:**

```
mlflow, version 2.13.0
```

---

### 3-3. 데이터 저장 디렉토리 생성

```bash
mkdir -p /opt/nexus-mlflow/mlruns       # 실험 메타데이터 DB
mkdir -p /opt/nexus-mlflow/artifacts    # 체크포인트, 설정 파일 저장소
mkdir -p /opt/nexus-mlflow/sync_inbox   # Blackwell SCP 수신 디렉토리
```

**디렉토리 구조 확인:**

```bash
tree /opt/nexus-mlflow
```

**기대 출력:**

```
/opt/nexus-mlflow
├── artifacts/       ← best.pth, env_cfg.yaml 등이 저장될 곳
├── mlruns/          ← MLflow 실험 DB (metrics, params, tags)
├── sync_inbox/      ← Blackwell에서 SCP로 전송되는 임시 수신 폴더
└── venv/            ← Python 가상환경
```

> 💡 **디스크 용량 미리 확인:**
>
> ```bash
> df -h /opt/nexus-mlflow
> ```
>
> `artifacts/` 에 체크포인트가 쌓이므로, 최소 **100GB 이상** 여유가 있는 드라이브를 권장합니다. 여유가 부족하면 외장 HDD 경로로 변경하면 됩니다.

---

## ✅ Step 4 — 동작 테스트 (수동 실행)

서비스 등록 전에 먼저 수동으로 실행해서 정상 동작을 확인합니다.

```bash
cd /opt/nexus-mlflow
source venv/bin/activate

mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri /opt/nexus-mlflow/mlruns \
    --default-artifact-root /opt/nexus-mlflow/artifacts
```

**기대 출력:**

```
[2025-04-18 10:15:32 +0900] [12345] [INFO] Starting gunicorn 21.2.0
[2025-04-18 10:15:32 +0900] [12345] [INFO] Listening at: http://0.0.0.0:5000
[2025-04-18 10:15:32 +0900] [12345] [INFO] Using worker: sync
[2025-04-18 10:15:32 +0900] [12346] [INFO] Booting worker with pid: 12346
```

> ✅ `Listening at: http://0.0.0.0:5000` 가 보이면 성공입니다.
>
> 이 상태에서 **같은 PC의 브라우저**에서 `http://localhost:5000` 접속 시 MLflow UI가 보여야 합니다.

확인이 끝나면 `Ctrl + C` 로 종료합니다.

---

## 🔥 Step 5 — 방화벽 / 포트 설정

### 5-1. SSH 서버(sshd) 설치 및 실행 확인

> ⚠️ **새로 포맷한 PC에는 `openssh-server`가 설치되어 있지 않은 경우가 많습니다.**
> SSH 데몬이 떠 있지 않으면 Step 8에서 Blackwell이 접속을 시도할 때
> `ssh: connect to host 192.168.1.42 port 22: Connection refused` 에러가 발생합니다.
> 방화벽을 열기 전에 sshd가 실제로 동작 중인지 먼저 확인해야 합니다.

**1) SSH 데몬 상태 확인:**

```bash
sudo systemctl status ssh
```

**기대 출력 (정상 동작 시):**

```
● ssh.service - OpenBSD Secure Shell server
     Loaded: loaded (/lib/systemd/system/ssh.service; enabled; ...)
     Active: active (running) since ...
```

- `Active: active (running)` → 다음 5-2로 진행하면 됩니다.
- `Unit ssh.service could not be found` 또는 `inactive (dead)` → 아래 2)로 설치/기동.

**2) `openssh-server` 설치 및 기동:**

```bash
sudo apt update
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
```

**3) 22번 포트 LISTEN 확인:**

```bash
sudo ss -tlnp | grep :22
```

**기대 출력:**

```
LISTEN 0  128  0.0.0.0:22  0.0.0.0:*  users:(("sshd",...))
```

> ✅ 위 출력이 보이면 sshd가 정상적으로 22번 포트에서 대기 중입니다.

---

### 5-2. 현재 방화벽 상태 확인

```bash
sudo ufw status
```

**기대 출력 (비활성화 상태):**

```
Status: inactive
```

**기대 출력 (활성화 상태):**

```
Status: active

To                         Action      From
--                         ------      ----
22/tcp                     ALLOW       Anywhere
```

---

### 5-3. 필요한 포트 열기

```bash
# SSH (Blackwell SCP 전송에 필요)
sudo ufw allow ssh

# MLflow UI 및 API (팀원 접속 + Blackwell 연동)
sudo ufw allow 5000/tcp comment 'NEXUS MLflow Server'
```

---

### 5-4. 방화벽 활성화

```bash
sudo ufw enable
```

**기대 출력:**

```
Command may disrupt existing ssh connections. Proceed with operation (y|n)? y
Firewall is active and enabled on system startup
```

**설정 확인:**

```bash
sudo ufw status verbose
```

**기대 출력:**

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

> ✅ `5000/tcp` 가 `ALLOW IN` 상태이면 팀원들이 접속할 수 있습니다.

---

## ⚙️ Step 6 — systemd 서비스 등록 (자동 시작)

PC가 재부팅되어도 MLflow 서버가 자동으로 시작되도록 등록합니다.

### 6-1. 현재 사용자 이름 확인

```bash
echo $USER
# 예: jonghochoi
```

### 6-2. 서비스 파일 생성

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
    --default-artifact-root /opt/nexus-mlflow/artifacts
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

**생성 확인:**

```bash
cat /etc/systemd/system/nexus-mlflow.service
```

`User=` 라인에 본인 계정명이 올바르게 들어가 있는지 확인합니다.

---

### 6-3. 서비스 등록 및 시작

```bash
# systemd 재로드 (새 서비스 파일 인식)
sudo systemctl daemon-reload

# 부팅 시 자동 시작 등록
sudo systemctl enable nexus-mlflow

# 지금 바로 시작
sudo systemctl start nexus-mlflow
```

---

### 6-4. 서비스 상태 확인

```bash
sudo systemctl status nexus-mlflow
```

**기대 출력 (정상 동작 시):**

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

> ✅ `Active: active (running)` 이 보이면 서비스가 정상적으로 동작 중입니다.
>
> ❌ `Active: failed` 가 보이면 아래 로그 명령으로 원인을 확인합니다.
>
> ```bash
> sudo journalctl -u nexus-mlflow -n 50 --no-pager
> ```

---

### 6-5. 유용한 서비스 관리 명령어

```bash
# 서비스 중지
sudo systemctl stop nexus-mlflow

# 서비스 재시작 (설정 변경 후)
sudo systemctl restart nexus-mlflow

# 실시간 로그 보기
sudo journalctl -u nexus-mlflow -f

# 오늘 로그 전체 보기
sudo journalctl -u nexus-mlflow --since today
```

---

## 👥 Step 7 — 팀원 접속 확인

### 7-1. 접속 주소 최종 확인

```bash
hostname -I | awk '{print $1}'
# 예: 192.168.1.42
```

팀원들에게 아래 주소를 공유합니다:

```
http://192.168.1.42:5000
```

### 7-2. 팀원 PC에서 접속 테스트

팀원 PC 브라우저에서 위 주소로 접속했을 때 아래와 같은 MLflow UI가 보이면 성공입니다.

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

### 7-3. 접속이 안 될 때 체크리스트

```bash
# MLflow 서버 PC에서 실행

# 1. 서비스가 실행 중인지 확인
sudo systemctl status nexus-mlflow

# 2. 5000번 포트가 실제로 열려 있는지 확인
ss -tlnp | grep 5000
# 기대 출력: LISTEN 0 ... 0.0.0.0:5000

# 3. 방화벽이 포트를 막고 있지 않은지 확인
sudo ufw status | grep 5000
# 기대 출력: 5000/tcp   ALLOW IN    Anywhere
```

---

## 🔗 Step 8 — Blackwell → MLflow 서버 연결 설정

### 8-1. Blackwell에서 SSH 키 생성

**Blackwell 서버에서 실행:**

```bash
# SSH 키 생성 (이미 있으면 스킵)
ssh-keygen -t ed25519 -C "blackwell-to-nexus" -f ~/.ssh/nexus_key
```

**기대 출력:**

```
Generating public/private ed25519 key pair.
Enter passphrase (empty for no passphrase):    ← 그냥 Enter (비밀번호 없이)
Enter same passphrase again:                   ← 그냥 Enter
Your identification has been saved in /home/user/.ssh/nexus_key
Your public key has been saved in /home/user/.ssh/nexus_key.pub
The key fingerprint is:
SHA256:xxxxxxxxxxxxxxxxxxxx blackwell-to-nexus
```

> 💡 passphrase는 비워두어야 cron 등에서 비밀번호 입력 없이 자동 실행됩니다.

---

### 8-2. 공개 키를 MLflow 서버에 등록

**Blackwell 서버에서 실행:**

```bash
ssh-copy-id -i ~/.ssh/nexus_key.pub USER@192.168.1.42
# USER는 MLflow 서버의 계정명, IP는 실제 값으로 교체
```

**기대 출력:**

```
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s)
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed
USER@192.168.1.42's password:    ← MLflow 서버 계정 비밀번호 입력 (이번 1회만)

Number of key(s) added: 1

Now try logging into the machine, with:   "ssh 'USER@192.168.1.42'"
and check to make sure that only the key(s) you wanted were added.
```

---

### 8-3. 키 기반 접속 확인 (비밀번호 없이)

```bash
ssh -i ~/.ssh/nexus_key USER@192.168.1.42 "echo 'NEXUS 연결 성공'"
```

**기대 출력:**

```
NEXUS 연결 성공
```

> ✅ 비밀번호 입력 없이 위 메시지가 출력되면 SCP 자동화 준비가 완료된 것입니다.

---

### 8-4. SCP 파일 전송 테스트

```bash
# 테스트 파일 전송
echo "nexus test" > /tmp/nexus_test.txt
scp -i ~/.ssh/nexus_key /tmp/nexus_test.txt USER@192.168.1.42:/opt/nexus-mlflow/sync_inbox/

# MLflow 서버에서 파일 도착 확인
ssh -i ~/.ssh/nexus_key USER@192.168.1.42 "cat /opt/nexus-mlflow/sync_inbox/nexus_test.txt"
```

**기대 출력:**

```
nexus test
```

---

### 8-5. NEXUS 동기화 스크립트 최초 실행 테스트

```bash
# nexus 프로젝트 디렉토리에서 실행
cd /path/to/nexus

bash scheduled_sync/sync_mlflow_to_server.sh \
    --experiment       sharpa_hand_rl \
    --remote           USER@192.168.1.42:/opt/nexus-mlflow/sync_inbox \
    --remote_nexus_dir /opt/nexus \
    --local_uri        http://127.0.0.1:5100 \
    --remote_uri       http://127.0.0.1:5000 \
    --ssh_key          ~/.ssh/nexus_key
```

> 💡 `--remote_nexus_dir`은 NEXUS 서버에 nexus 저장소가 clone된 경로입니다.
> 스크립트가 SSH로 접속해 `${remote_nexus_dir}/scheduled_sync/import_delta.py`를 실행하므로 필수입니다.

**기대 출력:**

```
[2025-04-18 10:30:00] MLflow delta sync: sharpa_hand_rl
  [1/3] Exporting delta from local MLflow (http://127.0.0.1:5100)...
  [OK] Delta exported (12 KB)
  [2/3] Transferring delta to 192.168.1.42...
  [OK] Transfer complete
  [3/3] Importing delta on remote server...
  [OK] Import complete
  [DONE] Delta sync complete at 2025-04-18 10:30:05
```

> ℹ️ 최초 실행 시에는 전체 데이터가 한 번에 전송되고, 이후 실행부터는 증분(delta)만 전송됩니다.
> "새 데이터가 없으면" `[OK] No new data since last sync. Nothing to transfer.`로 즉시 종료됩니다 (정상).

---

## 🗂️ 최종 구성 요약

구축이 완료되면 아래 구조가 됩니다.

```
[남는 PC — NEXUS MLflow 서버]
  IP:      192.168.1.42 (예시)
  OS:      Ubuntu 22.04
  서비스:   nexus-mlflow (systemd, 재부팅 후 자동 시작)
  포트:     5000 (사내망 전체 개방)

  저장 위치:
  /opt/nexus-mlflow/
  ├── mlruns/          ← 실험 메타데이터 (metrics, params, tags)
  ├── artifacts/       ← 체크포인트, 설정 파일
  │   └── <run_id>/
  │       ├── configs/
  │       │   ├── env_cfg.yaml
  │       │   └── reward_fn.py
  │       └── checkpoints/
  │           ├── best.pth
  │           └── last.pth
  └── sync_inbox/      ← Blackwell SCP 수신 폴더

[팀원 접속]
  브라우저 → http://192.168.1.42:5000

[Blackwell → NEXUS 연결]
  SSH 키:  ~/.ssh/nexus_key (비밀번호 없이 자동 전송)
  방식:    SCP → mlflow export/import (5분마다 cron)
```

---

## ⚠️ 자주 발생하는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| 브라우저 접속 안 됨 | 방화벽 차단 | `sudo ufw allow 5000/tcp` |
| `port 22: Connection refused` | MLflow 서버에 `openssh-server` 미설치 또는 sshd 미실행 | Step 5-1 참고 → `sudo apt install -y openssh-server && sudo systemctl enable --now ssh` |
| 서비스 시작 실패 | 경로 오류 | `journalctl -u nexus-mlflow -n 30` 로 로그 확인 |
| SCP 비밀번호 계속 물어봄 | 키 등록 안 됨 | `ssh-copy-id` 재실행 |
| 디스크 꽉 참 | artifact 누적 | `df -h` 확인 후 오래된 run artifact 삭제 |
| 재부팅 후 서버 안 뜸 | enable 안 됨 | `sudo systemctl enable nexus-mlflow` |

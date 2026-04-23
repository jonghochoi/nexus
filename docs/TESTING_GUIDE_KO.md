# 🧪 NEXUS 단계별 검증 가이드

> 로컬 PC에서 먼저 동작을 확인하고, GPU 서버로 넘어가는 순서로 진행합니다.  
> 각 Phase를 완료할 때마다 체크리스트를 확인하세요.

---

## 📑 목차

- [💻 Phase 1-A — 로컬 PC 검증](#phase-1--로컬-pc-검증)
- [📤 Phase 1-B — 기존 TensorBoard 파일로 실제 업로드 테스트](#phase-1-b--기존-tensorboard-파일로-실제-업로드-테스트)
- [📦 Phase 2 — GPU 서버 의존성 설치 (인터넷 없는 환경)](#phase-2--gpu-서버-의존성-설치-인터넷-없는-환경)
- [🖥️ Phase 3 — GPU 서버 검증](#phase-3--gpu-서버-검증)
- [🔄 Phase 4 — 서버 간 Sync 검증](#phase-4--서버-간-sync-검증)
- [🛠️ 문제 해결 (Troubleshooting)](#문제-해결-troubleshooting)

---

## 💻 Phase 1-A — 로컬 PC 검증

> **목적:** GPU 서버에 올리기 전에 로컬에서 모든 기능이 정상 동작하는지 확인합니다.

### A-1. 코드 받기

```bash
git clone https://github.com/jonghochoi/nexus.git
cd nexus
```

### A-2. Python 버전 확인

```bash
python3 --version
```

`Python 3.8` 이상이어야 합니다. 3.10 또는 3.11을 권장합니다.

### A-3. 환경 설치

```bash
bash setup.sh
```

설치가 끝나면 가상환경을 활성화합니다:

```bash
source venv/bin/activate
```

> 터미널 프롬프트 앞에 `(venv)` 가 표시되면 활성화된 것입니다.

### A-4. 설치 확인

```bash
python -c "import mlflow; print('mlflow:', mlflow.__version__)"
python -c "import tbparse; print('tbparse OK')"
python -c "from logger import make_logger; print('logger OK')"
```

세 줄 모두 오류 없이 출력되면 설치 성공입니다.

### A-5. 로컬 MLflow 서버 실행

```bash
bash scheduled_sync/start_local_mlflow.sh
```

예상 출력:

```
[NXS] Starting local MLflow server on 127.0.0.1:5100...
[NXS] MLflow ready at http://127.0.0.1:5100
```

브라우저에서 `http://localhost:5100` 을 열어 MLflow UI가 보이는지 확인합니다.

> 서버가 이미 실행 중이라는 메시지가 나와도 괜찮습니다.

### A-6. 스모크 테스트 실행

스모크 테스트는 NEXUS의 핵심 기능(패키지 설치, MLflow 연결, 로깅, 검증)을 자동으로 확인합니다.

```bash
# nexus 디렉토리 최상위에서 실행
python tests/smoke_test.py
```

모든 항목이 `[PASS]` 로 표시되어야 합니다:

```
  [PASS]  패키지 임포트
  [PASS]  MLflow 서버 연결
  [PASS]  MLflowLogger 로깅
  [PASS]  make_logger 팩토리
  [PASS]  DualLogger (Dual)

  모든 테스트 통과! NEXUS가 정상적으로 동작합니다.
```

테스트가 끝나면 MLflow UI(`http://localhost:5100`)에서 `nexus_smoke_test` 실험 아래에 새로 생성된 run들이 보여야 합니다.

### ✅ Phase 1-A 체크리스트

- [ ] `bash setup.sh` 오류 없이 완료
- [ ] `source venv/bin/activate` 후 `(venv)` 프롬프트 확인
- [ ] 세 가지 import 명령어 모두 성공
- [ ] 브라우저에서 `http://localhost:5100` MLflow UI 접근 성공
- [ ] `python tests/smoke_test.py` 모두 `[PASS]`
- [ ] MLflow UI에서 `nexus_smoke_test` 실험 및 run 확인

---

## 📤 Phase 1-B — 기존 TensorBoard 파일로 실제 업로드 테스트

> **목적:** 이미 가지고 있는 tfevents 파일을 MLflow에 업로드하고, 데이터가 정확히 옮겨졌는지 검증합니다.  
> Phase 1의 체크리스트를 모두 통과한 뒤 진행하세요 (MLflow 서버가 실행 중이어야 합니다).

---

### B-1. tfevents 파일 위치 확인

먼저 업로드할 tfevents 파일이 어디 있는지 확인합니다.

```bash
# 디렉토리 안에 tfevents 파일이 있는지 확인
ls /path/to/your/logs/run_001/
```

예상 출력:
```
events.out.tfevents.1700000000.hostname.12345.0
```

> tfevents 파일은 보통 `events.out.tfevents.` 로 시작합니다.  
> 여러 개 있어도 괜찮습니다 — 스크립트가 폴더를 재귀적으로 탐색합니다.

---

### B-2. 드라이런(Dry Run) — 업로드 없이 파싱 결과만 미리 보기

실제 업로드 전에 `--dry_run` 옵션으로 어떤 메트릭이 몇 개나 파싱되는지 먼저 확인합니다.

```bash
cd nexus/
source venv/bin/activate

python post_upload/tb_to_mlflow.py \
    --tb_dir    /path/to/your/logs/run_001 \
    --dry_run
```

예상 출력 (메트릭 요약 테이블):

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

테이블이 보이면 파싱이 정상적으로 된 것입니다. 메트릭 이름과 데이터 수가 예상과 맞는지 확인하세요.

---

### B-3. 실제 업로드 실행

드라이런에서 내용을 확인했으면 실제 업로드를 진행합니다.

> **주의:** `tb_to_mlflow.py`의 기본 URI는 `:5000`이지만, 로컬 MLflow 서버는 `:5100`에서 실행됩니다.  
> `--tracking_uri`를 반드시 명시하세요.

```bash
python post_upload/tb_to_mlflow.py \
    --tb_dir       /path/to/your/logs/run_001 \
    --experiment   robot_hand_rl \
    --run_name     ppo_baseline_v1 \
    --tracking_uri http://127.0.0.1:5100 \
    --tags         researcher=kim seed=42 task=in_hand_reorientation
```

스크립트가 메트릭 요약 테이블을 보여주고 업로드 여부를 물어봅니다:

```
Upload the above data to MLflow? (y/n):
```

`y` 를 입력하면 업로드가 시작됩니다. 완료 후 다음과 같이 Run ID가 출력됩니다:

```
✓ Upload complete!
  Run ID      : a1b2c3d4e5f6...
  Data points : 2,100  (3 batches)
  UI URL      : http://127.0.0.1:5100
```

**Run ID를 복사해두세요** — 다음 단계 검증에서 필요합니다.

#### 사용 가능한 옵션 정리

| 옵션 | 설명 | 예시 |
|---|---|---|
| `--tb_dir` | tfevents 파일이 있는 폴더 경로 (필수) | `--tb_dir ./logs/run_001` |
| `--experiment` | MLflow 실험 이름 (기본값: `robot_hand_rl`) | `--experiment my_exp` |
| `--run_name` | MLflow run 이름 (기본값: 폴더명+타임스탬프) | `--run_name ppo_v1` |
| `--tracking_uri` | MLflow 서버 주소 (기본값: `:5000`) | `--tracking_uri http://127.0.0.1:5100` |
| `--tags` | 추가 태그들 (`key=value` 형식, 공백으로 구분) | `--tags researcher=kim seed=42` |
| `--dry_run` | 업로드 없이 파싱 결과만 출력 | `--dry_run` |
| `--upload_artifacts` | tfevents 파일 자체도 MLflow artifact로 첨부 | `--upload_artifacts` |

---

### B-4. 업로드 정확도 검증

업로드된 데이터가 원본 tfevents와 값이 일치하는지 자동으로 비교합니다.

```bash
python post_upload/verify_upload.py \
    --run_id       a1b2c3d4e5f6...   \
    --tb_dir       /path/to/your/logs/run_001 \
    --tracking_uri http://127.0.0.1:5100
```

세 가지 항목을 검사합니다:

| 검사 항목 | 의미 |
|---|---|
| Tag list fully matched | TB에 있던 모든 메트릭 이름이 MLflow에도 존재 |
| Data point counts matched | 각 메트릭의 데이터 포인트 수가 동일 |
| Values within tolerance | 모든 값이 오차 범위(기본 `1e-6`) 이내에서 일치 |

모두 통과하면:

```
✓ All checks passed! TB -> MLflow porting is accurate.
```

---

### B-5. MLflow UI에서 결과 확인

브라우저에서 `http://localhost:5100` 을 열어 업로드된 run을 확인합니다.

1. 왼쪽 사이드바에서 실험 이름(`robot_hand_rl`)을 클릭합니다.
2. run 목록에서 방금 업로드한 run(`ppo_baseline_v1`)을 클릭합니다.
3. **Metrics** 탭 → 메트릭 이름을 클릭하면 학습 곡선 그래프가 나옵니다.
4. **Parameters** 탭 → `--tags`로 지정한 태그들이 보여야 합니다.

여러 run을 선택해서 **Compare** 버튼을 누르면 곡선을 나란히 비교할 수 있습니다.

---

### ✅ Phase 1-B 체크리스트

- [ ] `ls` 로 tfevents 파일 존재 확인
- [ ] `--dry_run` 실행 후 메트릭 파싱 결과 확인
- [ ] 실제 업로드 실행 후 Run ID 기록
- [ ] `verify_upload.py` 세 항목 모두 `✓ PASS`
- [ ] MLflow UI에서 학습 곡선 그래프 확인

---

## 📦 Phase 2 — GPU 서버 의존성 설치 (인터넷 없는 환경)

> **문제:** GPU 서버는 인터넷이 없어서 `pip install`이 되지 않습니다.  
> **해결책:** 인터넷이 되는 머신에서 패키지 파일(.whl)을 미리 다운로드해서 SCP로 옮깁니다.

두 가지 방법 중 상황에 맞는 것을 선택하세요.

---

### 방법 A — pip wheel 오프라인 전송 *(Docker 불필요, 권장)*

이 방법은 Docker 없이도 사용할 수 있으며, Python 패키지만 전송하므로 용량이 작습니다.

#### A-1. 로컬 머신에서 — wheel 파일 다운로드

**중요:** 로컬 PC와 GPU 서버의 OS/Python 버전이 다를 수 있습니다.  
GPU 서버가 `Linux x86_64` + `Python 3.12` 이라면 아래처럼 플랫폼을 명시해야 합니다.

```bash
# nexus 폴더 안에서 실행
mkdir nexus_wheels

pip download \
    --platform manylinux2014_x86_64 \
    --python-version 3.12 \
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

> **플랫폼 값 확인 방법:**  
> GPU 서버에 SSH 접속 후 `python3 -c "import platform; print(platform.machine())"` 실행.  
> `x86_64` → `manylinux2014_x86_64` 사용.

**Python 버전 확인:**  
GPU 서버에서 `python3 --version` 실행 후 버전을 맞춰줍니다.  
예: `Python 3.12.x` → `--python-version 3.12`

#### A-2. 로컬 머신에서 — nexus 코드 + wheel 파일을 GPU 서버로 전송

```bash
# nexus_wheels 폴더와 nexus 코드 전송
scp -r nexus_wheels user@gpu-server:/home/user/
scp -r nexus        user@gpu-server:/home/user/
```

> SSH 포트가 다르다면 `-P 포트번호` 옵션을 추가하세요.  
> 예: `scp -P 22222 -r nexus_wheels user@gpu-server:/home/user/`

#### A-3. GPU 서버에서 — 오프라인 설치

GPU 서버에 SSH로 접속한 뒤:

```bash
cd /home/user/nexus

# 1단계: 시스템 pip으로 virtualenv 먼저 설치 (venv 모듈 대체)
pip install --no-index --find-links /home/user/nexus_wheels --break-system-packages virtualenv

# 2단계: virtualenv로 가상환경 생성
python3.12 -m virtualenv venv
source venv/bin/activate

# 3단계: setuptools 고정 버전 설치 (70+ 버전은 pkg_resources 미포함)
pip install --force-reinstall --no-index --find-links /home/user/nexus_wheels "setuptools==69.5.1"

# pip 자체 업그레이드 (오프라인 가능)
pip install --upgrade pip

# wheel 파일로 나머지 패키지 오프라인 설치
pip install \
    --no-index \
    --find-links /home/user/nexus_wheels \
    mlflow==2.13.0 \
    tbparse==0.0.8 \
    tensorboard==2.16.2 \
    tensorboardX \
    pandas \
    rich
```

> **왜 `virtualenv`인가:**  
> Ubuntu/Debian에서 `python3.12-venv`는 apt 패키지라 pip wheel로 전송할 수 없습니다.  
> `virtualenv`는 pip 패키지이므로 wheel로 오프라인 전송이 가능하며, 사용법은 venv와 동일합니다.

---

### 방법 B — Docker 이미지 전송 *(완전한 격리 환경, Docker가 있을 때)*

GPU 서버에 Docker가 설치되어 있다면 이 방법이 가장 안정적입니다.

#### B-1. 로컬 머신에서 — Dockerfile 작성

`nexus/` 폴더에 아래 `Dockerfile` 을 생성합니다:

```dockerfile
FROM python:3.10-slim

WORKDIR /nexus

# 패키지 설치 (빌드 시점에 인터넷 사용)
RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    tbparse==0.0.8 \
    tensorboard==2.16.2 \
    tensorboardX \
    pandas \
    rich

# nexus 코드 복사
COPY . /nexus/

CMD ["bash"]
```

#### B-2. 로컬 머신에서 — 이미지 빌드 및 저장

```bash
cd nexus/

# 이미지 빌드
docker build -t nexus-env:latest .

# 이미지를 파일로 저장 (압축)
docker save nexus-env:latest | gzip > nexus-env.tar.gz

# 파일 크기 확인
ls -lh nexus-env.tar.gz
```

> 이미지 크기는 보통 500MB~1GB 수준입니다.

#### B-3. GPU 서버로 이미지 전송

```bash
scp nexus-env.tar.gz user@gpu-server:/home/user/
```

#### B-4. GPU 서버에서 — 이미지 로드 및 실행

```bash
# 이미지 로드
docker load < /home/user/nexus-env.tar.gz

# 컨테이너 실행 (nexus 폴더 마운트)
docker run --rm -it \
    -v /home/user/nexus:/nexus \
    -p 5100:5100 \
    nexus-env:latest bash
```

컨테이너 안에서 이후 모든 명령을 실행합니다.

---

### 방법 선택 기준

| 상황 | 권장 방법 |
|---|---|
| Docker가 GPU 서버에 없고 로컬이 Linux | 방법 A (플랫폼 일치) |
| Docker가 GPU 서버에 없고 로컬이 macOS/Windows | 방법 A (플랫폼 명시 필수) |
| GPU 서버에 Docker가 설치되어 있음 | 방법 B |
| 팀 전체가 동일 환경을 공유해야 함 | 방법 B |

---

## 🖥️ Phase 3 — GPU 서버 검증

> GPU 서버에 SSH 접속한 상태에서 진행합니다.

### 3-1. 설치 확인

```bash
cd /home/user/nexus
source venv/bin/activate  # 방법 A의 경우

python -c "import mlflow; print('mlflow:', mlflow.__version__)"
python -c "from logger import make_logger; print('logger OK')"
```

### 3-2. 로컬 MLflow 서버 실행 (GPU 서버 내부)

GPU 서버에서 로컬 MLflow를 띄웁니다 (인터넷 불필요, loopback 전용):

```bash
bash scheduled_sync/start_local_mlflow.sh
```

> 이 MLflow는 GPU 서버 내부에서만 접근 가능합니다 (`127.0.0.1:5100`).  
> 외부에서 접근하려면 SSH 터널링을 사용합니다:
>
> ```bash
> # 로컬 PC 터미널에서 (GPU 서버 MLflow를 로컬로 터널링)
> ssh -L 5100:127.0.0.1:5100 user@gpu-server
> # 이후 로컬 브라우저에서 http://localhost:5100 접근 가능
> ```

### 3-3. 스모크 테스트 실행

```bash
python tests/smoke_test.py
```

로컬 PC와 동일하게 모두 `[PASS]` 가 나와야 합니다.

### 3-4. 실제 학습 코드 통합 테스트 (선택 사항)

PPO 코드를 수정했다면 짧은 학습 (예: 100 step)을 돌려서 실제 메트릭이 기록되는지 확인합니다.

```python
# 학습 코드 초기화 부분에 (docs/LOGGER_SETUP.md 참고)
from logger import make_logger
import os

self.writer = make_logger(
    mode="dual",
    log_dir=output_dir,
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

학습 시작 후 SSH 터널링으로 MLflow UI(`http://localhost:5100`)에서 메트릭이 실시간으로 쌓이는지 확인합니다.

### ✅ Phase 3 체크리스트

- [ ] GPU 서버에서 `import mlflow` 성공
- [ ] `bash scheduled_sync/start_local_mlflow.sh` 오류 없이 실행
- [ ] `python tests/smoke_test.py` 모두 `[PASS]`
- [ ] (선택) PPO 실행 후 MLflow UI에서 메트릭 확인

---

## 🔄 Phase 4 — 서버 간 Sync 검증

> GPU 서버의 실험 데이터를 중앙 MLflow 서버(NEXUS 서버)로 동기화하는 단계입니다.  
> 이 단계는 NEXUS 서버가 준비된 이후에 진행합니다.

### 4-1. Pipeline A — Delta Sync (MLflow incremental)

GPU 서버에서 아래 명령을 한 번 실행해서 동기화를 테스트합니다. 매 실행은 **증분(delta)** 방식으로, 이전 싱크 이후 새로 기록된 step만 전송합니다.

```bash
bash scheduled_sync/sync_mlflow_to_server.sh \
    --experiment       robot_hand_rl \
    --remote           user@nexus-server:/data/mlflow_delta_inbox \
    --remote_nexus_dir /opt/nexus
```

> 💡 `--remote_nexus_dir`는 NEXUS 서버에 설치된 nexus 경로입니다 (예: `/opt/nexus`). 서버에서 `import_delta.py`를 찾기 위해 필요합니다.

성공하면 NEXUS 서버의 MLflow UI에서 run이 나타납니다. 로컬 상태 파일(`/tmp/nexus_delta_{experiment}.json`)에 각 run·tag의 마지막 싱크 step이 기록됩니다.

**두 번째 실행 시 확인:** 새 메트릭이 없다면 `[OK] No new data since last sync.` 메시지와 함께 SCP가 생략됩니다.

**자동화 (cron 등록):**

```bash
crontab -e
# 아래 줄 추가 (5분마다 실행):
*/5 * * * * bash /home/user/nexus/scheduled_sync/sync_mlflow_to_server.sh \
    --experiment       robot_hand_rl \
    --remote           user@nexus-server:/data/mlflow_delta_inbox \
    --remote_nexus_dir /opt/nexus \
    >> /home/user/nexus_sync.log 2>&1
```

### 4-2. Pipeline B — 일회성 배치 업로드 (post_upload 검증)

기존 tfevents 파일이 있다면 학습 종료 후 일회성으로 업로드할 수 있습니다. 스케줄 싱크가 필요하다면 Pipeline A를 사용하세요.

```bash
# NEXUS 서버에서 직접 실행하거나, GPU 서버에서 --tracking_uri 지정
python post_upload/tb_to_mlflow.py \
    --tb_dir       /path/to/logs/run_001 \
    --experiment   robot_hand_rl \
    --run_name     ppo_baseline_v1 \
    --tracking_uri http://nexus-server:5000 \
    --tags         researcher=kim seed=42

# 업로드 후 검증
python post_upload/verify_upload.py \
    --run_id       <위에서 출력된 RUN_ID> \
    --tb_dir       /path/to/logs/run_001 \
    --tracking_uri http://nexus-server:5000
```

### ✅ Phase 4 체크리스트

- [ ] `sync_mlflow_to_server.sh` 수동 실행 성공
- [ ] NEXUS 서버 MLflow UI에서 synced run 확인
- [ ] cron 등록 후 자동 실행 확인 (`cat /home/user/nexus_sync.log`)
- [ ] (선택) `verify_upload.py` 검증 통과

---

## 🛠️ 문제 해결 (Troubleshooting)

### ⚠️ MLflow 서버가 시작되지 않을 때

```bash
# 기존 프로세스 확인
lsof -i :5100

# 마스터 + worker 전체 종료 후 재시작
lsof -ti :5100 | xargs kill
bash scheduled_sync/start_local_mlflow.sh
```

> **`kill $(cat .mlflow_local.pid)` 로 종료하면 안 되는 이유:**  
> MLflow는 내부적으로 gunicorn을 사용해 worker 프로세스를 여러 개 띄웁니다.  
> PID 파일에는 마스터 PID만 저장되므로, 마스터만 죽이면 worker들이 고아 프로세스로 남습니다.  
> 포트 기준으로 종료하면 마스터와 worker 전체를 한 번에 정리할 수 있습니다.

### ⚠️ `pip install` 시 `externally-managed-environment` 오류가 날 때

Python 3.12 + Ubuntu 계열에서 시스템 Python에 직접 pip 설치를 차단하는 오류입니다 (PEP 668).  
`virtualenv` 부트스트랩 설치 시 `--break-system-packages` 플래그를 추가하세요:

```bash
pip install --no-index --find-links /home/user/nexus_wheels --break-system-packages virtualenv
```

이후 `python3.12 -m virtualenv venv`로 가상환경을 생성하고 활성화하면,  
venv 내부 pip을 사용하므로 이 오류가 다시 발생하지 않습니다.

### ⚠️ `ModuleNotFoundError: No module named 'pkg_resources'` 가 날 때

setuptools 70+ 버전부터 `pkg_resources`가 wheel에서 제거되었습니다.  
`pkg_resources`가 포함된 마지막 안정 버전(69.5.1)으로 교체하세요:

```bash
# 로컬 머신: 고정 버전 다운로드
pip download --platform manylinux2014_x86_64 --python-version 3.12 \
    --only-binary=:all: -d ./nexus_wheels "setuptools==69.5.1"

scp nexus_wheels/setuptools-69.5.1*.whl user@gpu-server:/home/user/nexus_wheels/

# GPU 서버: 강제 재설치
pip install --force-reinstall --no-index --find-links /home/user/nexus_wheels "setuptools==69.5.1"

# 확인
python -c "import pkg_resources; print('OK')"
```

### ⚠️ `pip download --only-binary` 에서 일부 패키지가 실패할 때

일부 패키지는 binary wheel이 없어서 소스 빌드가 필요합니다.  
`--only-binary=:all:` 대신 개별 패키지를 분리하세요:

```bash
# binary wheel 있는 것
pip download --platform manylinux2014_x86_64 --python-version 3.12 \
    --only-binary=:all: -d ./nexus_wheels \
    mlflow==2.13.0 pandas rich virtualenv

# binary wheel 없는 것은 별도로
pip download -d ./nexus_wheels \
    tbparse==0.0.8 tensorboardX
```

GPU 서버 설치 시 `--no-build-isolation` 옵션이 필요할 수도 있습니다:

```bash
pip install --no-index --find-links ./nexus_wheels --no-build-isolation \
    tbparse==0.0.8
```

### ⚠️ SSH 연결이 자꾸 끊길 때 (장시간 실행)

```bash
# nohup + 로그 파일로 백그라운드 실행
nohup bash scheduled_sync/start_local_mlflow.sh > mlflow_local.log 2>&1 &
```

또는 `tmux`/`screen`을 사용하세요:

```bash
tmux new -s nexus
bash scheduled_sync/start_local_mlflow.sh
# Ctrl+B, D 로 detach
```

### ⚠️ `tbparse` 임포트 오류 (`protobuf` 버전 충돌)

MLflow와 TensorBoard 간 `protobuf` 버전 충돌이 생길 수 있습니다:

```bash
pip install "protobuf>=3.20,<5.0"
```

### ⚠️ smoke_test.py 에서 `MLflow 서버 연결 실패`

1. MLflow 서버가 실행 중인지 확인: `lsof -i :5100`
2. 올바른 URI 사용 중인지 확인: `http://127.0.0.1:5100` (localhost 대신 IP 직접 입력)
3. 방화벽으로 포트가 막혔는지 확인: `curl http://127.0.0.1:5100/health`

---

## ⚡ 빠른 참조 — 주요 명령어

```bash
# 환경 활성화
source venv/bin/activate

# 로컬 MLflow 시작
bash scheduled_sync/start_local_mlflow.sh

# 스모크 테스트
python tests/smoke_test.py

# 스모크 테스트 (다른 서버 URI)
python tests/smoke_test.py --tracking_uri http://nexus-server:5000

# MLflow 서버 종료 (마스터 + worker 전체)
lsof -ti :5100 | xargs kill

# GPU 서버 MLflow를 로컬에서 접근 (SSH 터널)
ssh -L 5100:127.0.0.1:5100 user@gpu-server
```

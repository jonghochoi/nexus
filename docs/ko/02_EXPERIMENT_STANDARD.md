# 🧭 팀 실험 관리 표준 (NEXUS Experiment Standard)

> **문서 목적:** NEXUS MLflow 서버에 쌓이는 실험 데이터를 팀 전체가 일관되게 관리하기 위한 표준입니다. 이 문서에 정의된 규칙은 모든 팀원이 준수해야 합니다.
>
> **최초 작성:** YYYY-MM-DD · **최종 수정:** YYYY-MM-DD · **관리자:** @jonghochoi

---

## 📌 한 줄 요약

> **MLflow = 숫자. Confluence = 판단. 이름 = 규칙. 실험 전 = 가설.**

---

## 목차

- [0. 도구 역할 분리 원칙](#0-도구-역할-분리-원칙)
- [1. Experiment 구조](#1-experiment-구조)
- [2. Run 이름 규칙](#2-run-이름-규칙)
- [3. Tags 규칙](#3-tags-규칙)
- [4. Run 구조 — 단독 vs Parent/Child](#4-run-구조-단독-vs-parent-child)
- [5. Artifact 관리](#5-artifact-관리)
- [6. 실험 생명주기](#6-실험-생명주기)
- [7. Failed Run 처리 규칙](#7-failed-run-처리-규칙)
- [8. Sim-to-Real 연결 규칙](#8-sim-to-real-연결-규칙)
- [9. MLflow UI 활용 가이드](#9-mlflow-ui-활용-가이드)
- [10. 절대 하지 말아야 할 것](#10-절대-하지-말아야-할-것)
- [11. 체크리스트](#11-체크리스트)
- [12. Confluence 실험 페이지 템플릿](#12-confluence-실험-페이지-템플릿)

---

## 🔀 0. 도구 역할 분리 원칙

> **MLflow와 Confluence의 역할을 절대로 섞지 않습니다.** MLflow에 해석을 쓰거나, Confluence에 수치를 나열하면 둘 다 쓸모없어집니다.

| | 📊 MLflow | 📝 Confluence |
|---|---|---|
| **질문** | *"무슨 숫자가 나왔는가?"* | *"왜 그 결과가 나왔고, 다음은 무엇인가?"* |
| **담는 것** | 숫자, 곡선, 체크포인트 | 판단, 결정, 가설 |
| **작성 주체** | 학습 코드 (자동) | 사람 (의도적으로) |
| **갱신 시점** | 매 epoch | 실험 그룹마다 |

### 0-1. MLflow에 기록하는 것

> ⚠️ **원칙:** 학습 중 생성된 숫자는 MLflow에만 기록합니다.

| 항목 | 예시 |
|---|---|
| 📈 **Metrics (시계열)** | `losses/actor_loss`, `performance/RLTrainFPS`, `info/kl` |
| ⚙️ **하이퍼파라미터** | `lr`, `gamma`, `e_clip`, reward weights |
| 🏷️ **재현성 Tags** | `experiment`, `researcher`, `task`, `hardware` |
| 💾 **Artifacts** | `best.pth`, `last.pth` |
| ℹ️ **Run 메타데이터** | 시작 시각, 소요 시간, 사용 GPU, status |
| 🔗 **Sim-to-Real 연결** | Real 평가 Run의 `sim_run_id` tag |

### 0-2. Confluence에 기록하는 것

> ⚠️ **원칙:** 사람이 직접 써야 하는 문장은 Confluence에만 기록합니다.

| 항목 | 예시 |
|---|---|
| 💡 **가설** | *"fingertip tactile reward가 slip 이벤트를 20% 이상 줄일 것이다"* |
| 🎯 **실험 의도** | 이 ablation을 설계한 이유, 어떤 질문에 답하는 실험인지 |
| 🔍 **해석** | *"v2가 stability를 개선했지만 고회전 구간에서 reward collapse 발생"* |
| 💥 **실패 분석** | 관찰된 edge case, 추정 원인 |
| ✅ **결정** | 팀이 다음에 무엇을 하기로 했는지, 그 이유 |
| 🔗 **MLflow 링크** | 관련 Run의 직접 URL (수치는 복사하지 않음) |

### 0-3. Confluence 페이지 구조

```
[Space] Robot Hand — Dexterous Manipulation Research
│
├── 📋 Project Overview
│   ├── Research goals and milestones
│   └── Infrastructure guide (MLflow 접근, 이름 규칙)
│
├── 🧪 Experiment Log                ← 실험 그룹마다 1페이지
│   ├── [2025-Q2] Reward Shaping Search
│   ├── [2025-Q2] Tactile Feedback Integration
│   └── [2025-Q3] Sim-to-Real Transfer
│
├── 🔍 Ablation Studies
│   ├── Contact Force Threshold Sweep
│   └── DOF Masking Strategy Comparison
│
├── 📌 Decision Log                  ← 팀 결정만, 원시 데이터 없음
│
└── 📚 Reference
    ├── Robot Hand kinematic constraints
    └── PhysX contact model behavior notes
```

### 0-4. Decision Log 형식

팀 결정은 Decision Log 페이지에 누적 기록합니다. 실험 페이지의 상세 내용을 읽지 않아도 팀 방향을 파악할 수 있게 합니다.

| 날짜 | 결정 내용 | 근거 | 출처 | 담당 |
|---|---|---|---|---|
| 2025-04-16 | tactile reward 채택 (v2, reweight) | slip 감소 확인 | [링크] | 팀 |
| 2025-04-10 | PPO → SAC 전환 보류 | SAC sample cost 과다 | [링크] | @name |

---

## 🗂️ 1. Experiment 구조

### 1-1. Experiment 계층 설계 원칙

Experiment는 **"함께 비교하고 싶은 Run들의 묶음"** 입니다. 서로 다른 Experiment에 있는 Run은 MLflow UI에서 나란히 비교하기 어렵습니다.

```
NEXUS MLflow
│
├── 📁 baseline_ppo           ← PPO 기준선 확립 실험
├── 📁 reward_shaping         ← 보상 함수 탐색 전반
├── 📁 ablation_contact       ← contact force 관련 ablation
├── 📁 ablation_tactile       ← tactile 관련 ablation
├── 📁 sim2real_transfer      ← sim-to-real 검증 실험
└── 📁 real_robot_eval        ← 실제 로봇 평가 전용
```

### 1-2. Experiment 목록

| Experiment 이름 | 용도 | 담당자 |
|---|---|---|
| `baseline_ppo` | PPO 기준 성능 확립. seed 다양성 실험 포함 | 전체 |
| `reward_shaping` | 보상 함수 구조/가중치 탐색 | 전체 |
| `ablation_contact` | contact force reward 관련 ablation | 전체 |
| `ablation_tactile` | tactile/fingertip 관련 ablation | 전체 |
| `sim2real_transfer` | Sim policy → Real 이전 검증 | 전체 |
| `real_robot_eval` | 실제 로봇 평가 결과 (Real 전용) | 전체 |

### 1-3. Experiment 신설 규칙

> ⚠️ Experiment는 함부로 만들지 않습니다. 많아질수록 관리가 어려워집니다.

- 기존 Experiment에 속하지 않는 완전히 새로운 연구 방향일 때만 신설
- 신설 전 **팀 채널에 공유 → 팀 합의 → 생성**
- Experiment 이름 규칙: `<목적>` 또는 `<목적>_<범위>` (소문자, 언더스코어)

---

## ✏️ 2. Run 이름 규칙

### 2-1. 네이밍 포맷

```
<연구자>_<방법론>_<핵심변수>_<버전>
```

| 요소 | 설명 | 예시 |
|---|---|---|
| `<연구자>` | 본인 이름 약칭 (팀 내 고유) | `kim`, `lee`, `park` |
| `<방법론>` | 알고리즘 또는 접근 방식 | `ppo`, `tactile`, `contact` |
| `<핵심변수>` | 이 실험에서 바꾼 것 | `contact0.3`, `lr1e4`, `seed42` |
| `<버전>` | 동일 설정 재실행 구분 | `v1`, `v2`, `v3` |

### 2-2. 네이밍 예시

```
✅ 올바른 예시
kim_ppo_seed42_v1
kim_ppo_seed123_v1
lee_tactile_fingertip_v2
lee_tactile_deform_v1
park_contact_weight0.5_v1
park_contact_ablation_v3

❌ 잘못된 예시
test_run             ← 무슨 실험인지 알 수 없음
ppo_1234             ← 누가 돌렸는지 알 수 없음
kim_experiment       ← 무엇을 바꿨는지 알 수 없음
final_real           ← 버전이 없음
```

### 2-3. 버전 관리 규칙

- 항상 `v1` 부터 시작
- 동일한 목적의 재실험은 버전 +1 (삭제하지 않고 누적)
- 설정이 크게 바뀌면 방법론 부분을 바꿔서 새 이름 사용

---

## 🏷️ 3. Tags 규칙

### 3-1. 필수 Tags

아래 Tags는 모든 Run에 반드시 포함되어야 합니다.
`experiment`는 `--experiment` 인수에서 자동으로 채워지며, 나머지 세 개는 직접 지정해야 합니다.

| Tag 키 | 설명 | 예시 값 |
|---|---|---|
| `experiment` | MLflow Experiment 이름 (자동 설정) | `robot_hand_rl` |
| `researcher` | 실험 수행자 | `kim` |
| `task` | 실험 태스크 이름 | `in_hand_reorientation` |
| `hardware` | 사용 하드웨어 | `robot_22dof` |

> ⚠️ 위 4개 Tag 중 하나라도 빠지면 재현 및 비교가 불가능합니다.

### 3-2. 선택 Tags *(필요 시 연구자가 추가)*

| Tag 키 | 사용 시점 | 예시 값 |
|---|---|---|
| `train` | 학습 방법론 명시 | `ppo`, `sac`, `td3` |

---

## 🧩 4. Run 구조 — 단독 vs Parent/Child

### 4-1. 단독 Run

대부분의 실험은 단독 Run으로 기록합니다.

```python
self.writer = make_logger(
    mode="dual",
    run_name="kim_ppo_contact0.3_v1",
    experiment_name="reward_shaping",
    tags={"experiment": "reward_shaping", "researcher": "kim", "task": "in_hand_reorientation", "hardware": "robot_22dof"},
    ...
)
```

### 4-2. Parent/Child Run *(Ablation/Sweep 전용)*

**같은 목적 아래 변수 하나만 바꿔가며 비교할 때** 사용합니다.
UI에서 parent를 펼치면 child들이 나란히 보여 비교가 편합니다.

```
✅ Parent/Child 사용 상황
- contact weight를 [0.1, 0.3, 0.5, 0.7] sweep
- seed를 [42, 123, 456] 반복
- lr을 [1e-3, 3e-4, 1e-4] sweep

❌ 단독 Run이 맞는 상황
- 서로 다른 연구자의 독립 실험
- 접근 방식 자체가 다른 실험
```

```python
# ablation 실행 예시
with mlflow.start_run(run_name="kim_contact_weight_sweep") as parent:
    mlflow.set_tags({"experiment": "ablation_contact", "researcher": "kim", "task": "in_hand_reorientation", "hardware": "robot_22dof"})

    for w in [0.1, 0.3, 0.5, 0.7]:
        with mlflow.start_run(run_name=f"kim_contact{w}_v1", nested=True):
            mlflow.log_param("reward.contact_weight", w)
            # ... 학습 ...
```

---

## 💾 5. Artifact 관리

### 5-1. 저장 대상 및 위치

| Artifact | MLflow 경로 | 저장 시점 | 보존 기간 |
|---|---|---|---|
| `best.pth` | `checkpoints/best.pth` | best reward 갱신 시 (덮어쓰기) | 영구 |
| `last.pth` | `checkpoints/last.pth` | `save_freq` 마다 (덮어쓰기) | 30일 |

### 5-2. 체크포인트 정책

```
best.pth  →  영구 보존. 삭제 금지.
last.pth  →  30일 경과 후 팀 합의 하에 삭제 가능.
중간 체크포인트 (ep_100_...) →  MLflow에 올리지 않음. 로컬에만 보관.
```

> 💡 `best.pth`와 `last.pth`는 갱신 시 덮어쓰기됩니다. 중간 체크포인트를 보존하고 싶으면 로컬 `nn_dir/` 디렉토리를 활용하세요.

### 5-3. 디스크 사용량 모니터링

```bash
# MLflow 서버에서 주기적으로 확인
du -sh /opt/nexus-mlflow/artifacts/
df -h /opt/nexus-mlflow/
```

artifacts가 **50GB를 초과하면 팀 회의**를 통해 오래된 run의 `last.pth` 정리를 검토합니다.

---

## 🚦 6. 실험 생명주기

실험 하나가 시작부터 끝날 때까지 따라야 하는 흐름입니다.

```
1️⃣  실험 전
    └─ Confluence에 실험 페이지 작성 (가설 + 목적 먼저)
    └─ Experiment 이름, Run 이름 결정
    └─ 필수 Tags 확인 (agent_cfg에 포함 여부)

2️⃣  학습 실행
    └─ make_logger() 호출 시 자동으로:
       params, tags 기록
    └─ 학습 중: status=running, metrics 실시간 기록
    └─ 5분마다 NEXUS 서버로 자동 동기화 (cron)

3️⃣  학습 완료
    └─ status=done (또는 failed) 자동 기록
    └─ best.pth, last.pth 업로드

4️⃣  실험 후
    └─ Confluence 실험 페이지 업데이트
       (결과 해석, 실패 원인, 다음 방향)
    └─ 팀 공유 (슬랙 등)
    └─ Decision Log 업데이트 (결정이 있는 경우)
```

---

## ❌ 7. Failed Run 처리 규칙

> **Failed Run은 삭제하지 않습니다.**

실패한 실험도 중요한 정보입니다. 같은 실수를 반복하지 않기 위해 반드시 남깁니다.

| 해야 할 것 | 하지 말아야 할 것 |
|---|---|
| `fail_reason` tag에 원인 기록 | MLflow에서 Run 삭제 |
| `status=failed` 확인 | 실패 원인 기록 없이 재실험 |
| Confluence 페이지 "실패 분석" 섹션 작성 | 실패 사실 공유하지 않기 |

```
fail_reason 작성 예시:
  "OOM: num_envs=8192에서 GPU 메모리 초과"
  "KL explosion: lr=1e-3이 너무 높음, 0.01 이후 발산"
  "reward collapse: contact_weight=1.0이 다른 항목 압도"
```

---

## 🔗 8. Sim-to-Real 연결 규칙

실제 로봇 평가 Run은 **반드시** 해당 Sim Run과 연결해야 합니다.

```python
# real_robot_eval Experiment에서
self.writer = make_logger(
    experiment_name="real_robot_eval",
    run_name="kim_real_20250418",
    tags={
        "experiment": "real_robot_eval",
        "researcher": "kim",
        "task": "in_hand_reorientation",
        "hardware": "robot_22dof",
        "sim_run_id": "abc123def456",   # ← Sim Run ID 필수
    },
    ...
)
```

> `sim_run_id` 없이 Real 평가 Run을 만들지 않습니다. Sim-to-Real 실패 시 원인 추적이 불가능해집니다.

> 💡 **Pipeline B 업로드 시**: tfevents 디렉토리 옆에 `run_meta.json` (`{"sim_run_id": "..."}`)을 떨어뜨려 두면 `upload_tb.py`가 자동으로 감지합니다. `--experiment real_robot_eval`로 업로드하면 `sim_run_id`가 필수 태그로 격상되어 누락 시 업로드가 차단됩니다. 상세: [`13_POST_UPLOAD.md`](../13_POST_UPLOAD.md) §5.

---

## 📊 9. MLflow UI 활용 가이드

### 자주 쓰는 필터

```
특정 연구자 실험 보기:
  tags.researcher = "kim"

특정 태스크 실험 보기:
  tags.task = "in_hand_reorientation"

특정 하드웨어 실험 보기:
  tags.hardware = "robot_22dof"

최근 1주일 실험:
  start_time >= "2025-04-11"
```

### Run 비교 방법

1. 비교할 Run들을 체크박스로 선택
2. 상단 **"Compare"** 버튼 클릭
3. Params diff / Metrics 곡선 겹쳐보기 가능

> ⚠️ 서로 다른 Experiment의 Run은 Compare가 되지 않습니다. 비교하려면 같은 Experiment 안에 있어야 합니다.

---

## 🛑 10. 절대 하지 말아야 할 것

| ❌ 금지 행위 | 이유 |
|---|---|
| MLflow에서 Run 삭제 | 팀 실험 이력 영구 소실 |
| Confluence에 metric 수치 직접 기재 | 수치가 변경되면 Confluence가 거짓말을 함 |
| MLflow Run description에 해석 작성 | 팀에게 보이지 않고 묻혀버림 |
| 이름 규칙 무시하고 Run 생성 | 나중에 누구 실험인지, 무엇을 한 건지 알 수 없음 |
| "빠른 실험"이라며 Confluence 페이지 생략 | 나중에 결과를 해석할 수 없게 됨 |
| 가설 없이 실험 시작 | 결과를 봐도 해석이 안 됨 |
| 실패한 실험 Confluence 페이지를 비워두기 | 팀이 같은 실수를 반복하게 됨 |
| `sim_run_id` 없이 Real 평가 Run 생성 | Sim-to-Real 추적 불가 |
| 필수 Tags 누락 | 재현 불가능한 실험이 됨 |

---

## 📋 11. 체크리스트

### 실험 시작 전

- [ ] Confluence 실험 페이지 작성 완료 (가설 포함)
- [ ] Experiment 이름 결정 (기존 목록에서 선택 또는 팀 합의)
- [ ] Run 이름 규칙 확인 (`<연구자>_<방법론>_<핵심변수>_<버전>`)
- [ ] `agent_cfg`에 필수 Tags 항목 포함 확인
  - [ ] `experiment`
  - [ ] `researcher`
  - [ ] `task`
  - [ ] `hardware`
- [ ] 로컬 MLflow 서버 실행 확인 (`bash start_local_mlflow.sh`)

### 실험 완료 후

- [ ] MLflow에서 `status=done` 확인
- [ ] `best.pth` artifact 업로드 확인
- [ ] Confluence 실험 페이지 업데이트 (결과 해석, 다음 방향)
- [ ] 팀 채널에 결과 공유 (MLflow 링크 포함)
- [ ] 결정 사항이 있으면 Decision Log 업데이트

### 실험 실패 시

- [ ] `fail_reason` tag 기록
- [ ] Confluence 실험 페이지 "실패 분석" 섹션 작성
- [ ] Run 삭제하지 않기 (**절대 삭제 금지**)

---

## 📝 12. Confluence 실험 페이지 템플릿

> 아래 양식을 복사해서 Confluence 새 페이지로 만드세요.

### 🔬 [실험명] — [한 줄 설명]

| Experiment | Run 이름 | 담당자 | 시작일 | 상태 |
|---|---|---|---|---|
| `experiment_이름` | `연구자_방법론_핵심변수_v1` | @이름 | YYYY-MM-DD | 🟡 계획 중 / 🔄 진행 중 / ✅ 완료 / ❌ 실패 / ⏸️ 보류 |

#### ✍️ [실험 전] 가설 & 설계

> 학습 시작 전에 작성합니다.

| 해결하려는 문제 | 가설 (측정 가능한 예측) | 성공 판단 기준 |
|---|---|---|
| | | `지표명 > 목표치 @ N steps` |

변경 사항 — 한 번에 하나만 바꿉니다.

| 항목 | Baseline | 이번 실험 | 변경 이유 |
|---|---|---|---|
| | | | |

#### 📊 [실험 후] 결과

| 항목 | 내용 |
|---|---|
| MLflow 링크 | [링크]() — 수치는 Confluence에 옮기지 않습니다 |
| 가설 검증 | ✅ 확인 / ❌ 기각 / ⚠️ 부분 확인 |
| 관찰된 패턴 | |
| 예상과 달랐던 점 | |
| 다음 실험 방향 | (구체적인 액션 아이템) |
| 팀 결정 | ✅ 채택 / ⚠️ 조건부 / ❌ 기각 / ⏸️ 보류 — [Decision Log](링크) |

실패 분석 (실패가 없어도 작성):

| 조건 | 관찰된 현상 | 추정 원인 |
|---|---|---|
| | | |

---

## 📚 관련 문서

| 문서 | 링크 | 설명 |
|---|---|---|
| NEXUS 서버 접속 주소 | `http://192.168.1.42:5000` | MLflow UI |
| 팀 합의 원칙 (캐노니컬) | [`00_PRINCIPLES.md`](../00_PRINCIPLES.md) | 영문 요약 — 모든 원칙의 단일 진실 소스 |
| 서버 구축 가이드 | [`20_MLFLOW_SERVER_SETUP.md`](../20_MLFLOW_SERVER_SETUP.md) | 서버 설치/운영 |
| 로거 설정 가이드 | [`11_LOGGER_SETUP.md`](../11_LOGGER_SETUP.md) | 학습 코드에 로거 붙이는 방법 |
| Pipeline B 업로드 가이드 | [`13_POST_UPLOAD.md`](../13_POST_UPLOAD.md) | 사후 업로드 CLI 심층 설명 |

---

## 🔄 문서 개정 이력

| 날짜 | 버전 | 변경 내용 | 작성자 |
|---|---|---|---|
| YYYY-MM-DD | v1.0 | 최초 작성 | @jonghochoi |

---

> 이 문서에 대한 수정 제안은 GitHub Issue 또는 팀 채널로 알려주세요.

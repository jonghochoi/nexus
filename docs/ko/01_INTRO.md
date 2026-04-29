# 🚀 NEXUS 프로젝트 소개

> **[NXS] NEXUS** — 팀 강화학습 실험의 로그를 한 곳에 모으고, 함께 분석하기 위한 허브입니다.
>
> *모든 실험 데이터가 NEXUS에서 수렴합니다.*

---

## 🤔 왜 이 프로젝트가 필요한가요?

지금 우리 팀의 실험 로그는 어디에 있나요?

아마도 각자의 로컬 PC, 또는 학습 서버의 개인 디렉토리에 흩어져 있을 겁니다. 누군가 *"저번에 돌렸던 실험이랑 비교해볼 수 있어요?"* 라고 물어보면, 로그 파일을 찾아서 복사하고, TensorBoard를 각자 열어서 화면을 캡처해서 공유하는 과정을 거쳐야 합니다.

이 프로젝트는 그 불편함을 없애기 위해 만들었습니다.

> 💡 **핵심 목표**
>
> 팀원 누구나, 언제든지, 모든 실험 결과를 **한 곳 (NEXUS)** 에서 비교할 수 있게 한다.

---

## 🔥 우리가 해결하려는 문제들

| # | 문제 | 현재 상황 |
|:---:|---|---|
| 1️⃣ | **실험 결과가 개인에게 묶여 있다** | 다른 사람 실험과 비교하려면 파일을 직접 받아야 함 |
| 2️⃣ | **"저번 실험 어떻게 했었더라?"를 반복한다** | 하이퍼파라미터, seed 등이 개인 노트에 흩어짐 |
| 3️⃣ | **장시간 학습 중 진행 상황을 공유하기 어렵다** | 학습이 끝나고 나서야 결과를 알게 됨 |
| 4️⃣ | **"왜 이 방향으로 결정했지?"를 나중에 알 수 없다** | 팀의 결론과 다음 방향이 어디에도 기록되지 않음 |

---

## 🛠️ NEXUS의 해결책

### 도구 역할 분리

```
  ●──●──●──●──●   (모든 실험 데이터)
        │
      NEXUS
     /     \
  📊        📝
MLflow      Confluence
"무슨 숫자?" "왜 그랬고, 다음은?"
```

| 도구 | 담당 | 질문 |
|:---:|---|---|
| 📊 **MLflow** | 실험 데이터 (숫자) | *"무슨 숫자가 나왔는가?"* |
| 📝 **Confluence** | 팀의 판단 (언어) | *"왜 그 결과가 나왔고, 다음에 무엇을 할 것인가?"* |

> ⚠️ **가장 중요한 원칙:** 두 도구의 역할을 절대로 섞지 않습니다.
> MLflow에 해석을 쓰거나, Confluence에 수치를 나열하면 둘 다 쓸모없어집니다.

### 인프라 제약 해결

학습 서버는 인터넷이 차단되어 있습니다. NEXUS는 이 제약을 **두 가지 파이프라인**으로 우회합니다.

---

## 🔀 두 가지 로깅 방식

### 🅰️ 방식 A: 듀얼 로거 *(TensorBoard + MLflow 동시 기록)*

학습 코드를 **딱 2곳**만 수정합니다. 기존 TensorBoard 로그는 그대로 유지되면서, MLflow에도 동시에 기록됩니다.

```python
# ❌ 변경 전
from tensorboardX import SummaryWriter
self.writer = SummaryWriter(log_dir=output_dir)

# ✅ 변경 후 — [NXS] 듀얼 로깅 활성화
from nexus.logger import make_logger
self.writer = make_logger(mode="dual", tb_dir=output_dir, run_name=..., ...)
```

> `write_stats()` 는 **단 한 줄도 바꾸지 않습니다.**

| 구분 | 내용 |
|---|---|
| ✅ 장점 | 기존 TensorBoard 워크플로우 완전 유지 + 5분 단위 실시간 모니터링 |
| ⚠️ 단점 | 초기 설정 (로컬 MLflow 서버 실행, cron 설정) 필요 |

---

### 🅱️ 방식 B: 사후 업로드 *(코드 변경 없음, 당장 시작 가능)*

학습 코드를 전혀 건드리지 않습니다.

```
기존 학습 코드 (변경 없음)
    ↓ tfevents 파일 생성
    ↓ SCP 전송
[NXS] NEXUS 서버로 업로드
    ↓
팀 전체가 MLflow UI에서 비교 🎉
```

| 구분 | 내용 |
|---|---|
| ✅ 장점 | 코드 수정 없이 바로 시작 가능 |
| ⚠️ 단점 | 학습 중 실시간 모니터링 불가 |

---

## 💬 실제로 어떻게 달라지나요?

### 😓 지금 (Before)

```
팀원 A: "저번에 contact force reward 넣었을 때 결과 어떻게 됐어요?"
팀원 B: "잠깐만요, 서버에서 로그 찾아볼게요..."  (5분 후)
         "이거인데, tensorboard로 열어봐야 해요"
팀원 A: "그냥 캡처해서 슬랙에 올려줄 수 있어요?"
```

### 😊 앞으로 (After NEXUS)

```
팀원 A: "저번에 contact force reward 넣었을 때 결과 어떻게 됐어요?"
팀원 B: "[NXS] NEXUS에서 experiment 필터하면 바로 나와요.
         ppo_tactile_v2 랑 baseline 같이 선택하면 곡선 겹쳐서 볼 수 있어요."
팀원 A: "오 바로 보이네요. Confluence에 분석도 달려있네요."
```

---

## 📝 Confluence는 어떻게 활용하나요?

MLflow가 숫자를 담당한다면, Confluence는 **팀의 판단**을 담당합니다.

| 시점 | 작성 내용 |
|---|---|
| 🕐 **실험 전** | 왜 이 실험을 하는지, 어떤 결과를 기대하는지 (가설) |
| 🕓 **실험 후** | 결과 해석, 실패 원인 분석, 다음 방향 |
| 🔗 **항상** | MLflow 링크 (수치는 직접 적지 않음) |
| ✅ **결론** | 이 실험을 바탕으로 무엇을 하기로 했는지 |

> ⚠️ **가장 중요한 규칙:** 실험을 시작하기 **전에** 가설을 먼저 씁니다.
> 결과를 보고 나서 목적을 쓰면 사후합리화가 됩니다.

Confluence 페이지 템플릿과 팀 실험 관리 전체 규칙은 [`02_EXPERIMENT_STANDARD.md`](02_EXPERIMENT_STANDARD.md) 문서(Section 12)를 참조하세요.

---

## 🗺️ 단계별 도입 계획

```
Phase 1 ──────────────────────────── 지금 당장 (코드 변경 없음)
  ✅ 완료된 실험부터 NEXUS에 업로드 (tb_to_mlflow.py)
  ✅ 팀원들이 MLflow UI 탐색
  ✅ Confluence 실험 페이지 템플릿 적용
  🎯 목표: [NXS] NEXUS가 어떻게 생겼는지 팀 전체가 파악

Phase 2 ───────────────── 듀얼 로거 도입 (코드 2줄 수정)
  ✅ make_logger(mode="dual") 로 교체
  ✅ 기존 TensorBoard는 그대로 유지
  ✅ 새 실험부터 NEXUS에 실시간 기록
  🎯 목표: 개인 TensorBoard 대신 NEXUS를 먼저 보는 습관

Phase 3 ──── MLflow 중심 전환 (선택, 팀 합의 후)
  ✅ mode="mlflow" 로 전환
  ✅ Confluence + NEXUS가 팀 실험 관리 표준
  🎯 목표: 팀 연구 경험의 완전한 중앙화
```

---

## ❓ 자주 묻는 질문

<details>
<summary><strong>Q. TensorBoard를 완전히 버려야 하나요?</strong></summary>

아닙니다. `mode="dual"` 을 사용하면 TensorBoard와 MLflow를 **동시에** 씁니다. 기존 방식이 완전히 보존됩니다. TensorBoard는 로컬 디버깅용으로 여전히 유용합니다.

</details>

<details>
<summary><strong>Q. 설정이 복잡하지 않나요?</strong></summary>

방식 B(사후 업로드)는 설정이 거의 없습니다. 학습 후 명령 하나로 업로드합니다. 방식 A는 초기 설정이 필요하지만, 한 번 해두면 이후에는 **자동으로** 동작합니다.

</details>

<details>
<summary><strong>Q. 학습 서버가 인터넷이 안 되는데 어떻게 MLflow를 쓰나요?</strong></summary>

학습 서버 내부에 로컬 MLflow 서버를 하나 띄웁니다. 학습 코드는 이 **로컬 서버에만** 씁니다 (인터넷 불필요). 이후 주기적으로 SCP를 통해 NEXUS 서버로 동기화합니다.

</details>

<details>
<summary><strong>Q. NEXUS(MLflow)를 보려면 어디에 접속하나요?</strong></summary>

팀 NEXUS 서버 주소는 별도로 공지합니다. 브라우저에서 접속하면 됩니다.

</details>

<details>
<summary><strong>Q. 실험마다 꼭 Confluence 페이지를 만들어야 하나요?</strong></summary>

최소한 MLflow 태그에 `researcher`, `task`, `seed` 는 반드시 입력해 주세요. Confluence 페이지는 팀 논의가 필요한 **실험 그룹 단위**로 작성하면 됩니다.

</details>

---

## 🚀 시작하기

### 1️⃣ 환경 설치

```bash
git clone https://github.com/jonghochoi/nexus.git
cd nexus
bash setup.sh --alias            # venv 는 ~/.nexus/venv 에 설치됩니다 (repo 외부)
source ~/.bashrc
nexus-activate                   # 어느 경로/터미널에서든 활성화 가능
# alias 없이 쓰려면: source ~/.nexus/activate.sh
```

### 2️⃣ 완료된 실험 업로드 *(방식 B, 코드 수정 없음)*

```bash
cd post_upload/
python tb_to_mlflow.py \
    --tb_dir    /path/to/your/logs/run_001 \
    --experiment robot_hand_rl \
    --run_name  ppo_baseline_v1 \
    --tags      researcher=yourname seed=42 task=in_hand_reorientation \
    --tracking_uri http://<nexus-server>:5000
# [NXS] Upload complete ✓
```

### 3️⃣ 새 실험에 듀얼 로거 적용 *(방식 A, 선택)*

[`11_LOGGER_SETUP.md`](../11_LOGGER_SETUP.md) 를 참고하세요. 기존 학습 코드를 최소한의 수정으로 NEXUS 로거에 연결할 수 있습니다.

---

## 🌟 NEXUS가 기대하는 것

단기적으로는 실험 결과 공유의 불편함이 줄어들 것입니다. 하지만 장기적으로 기대하는 것은 더 큽니다.

모든 실험이 NEXUS에 쌓이고, 모든 결정의 근거가 Confluence에 남으면 — **우리 팀의 연구 경험이 사라지지 않습니다.** 팀원이 바뀌어도, 6개월이 지나도, 우리가 어떤 시행착오를 거쳐 지금의 방향에 도달했는지 추적할 수 있습니다.

```
  ●──●──●──●──●   모든 실험이 수렴하는 곳
        │
      NEXUS
```

Dexterous Manipulation이라는 어려운 문제를 풀어가는 과정에서, 우리가 쌓은 지식과 경험이 NEXUS에 남아있기를 바랍니다.

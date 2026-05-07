# 🇰🇷 NEXUS 한글 트랙

> **새 팀원을 위한 한글 온보딩 + 운영 표준 가이드입니다.**
>
> 영문 기술 문서는 [`docs/`](../) 상위 디렉토리에 있습니다. 이 트랙은 한국어 사용 팀원이 동기·도구 역할·실험 관리 규칙을 빠르게 익히기 위한 자기완결적 경로입니다.

---

## 읽는 순서

| 순서 | 문서 | 무엇을 다루나요 |
|:---:|---|---|
| 0️⃣ | [`../README.md`](../../README.md) | 프로젝트 개요 (3분, 영문) |
| 0️⃣ | [`../00_PRINCIPLES.md`](../00_PRINCIPLES.md) | 팀 합의 + 엔지니어링 원칙 (5분, 영문) |
| 1️⃣ | [`01_INTRO.md`](01_INTRO.md) | NEXUS 동기, 두 파이프라인 개념, FAQ |
| 2️⃣ | [`02_EXPERIMENT_STANDARD.md`](02_EXPERIMENT_STANDARD.md) | 실험 명명·태그·라이프사이클·Confluence 템플릿 |

> 💡 영문이 부담스럽더라도 `00_PRINCIPLES.md`는 꼭 읽어주세요. 팀이 합의한 모든 규칙을 한 페이지로 모은 짧은 문서입니다.

---

## 다음 단계 *(역할별)*

실험을 직접 진행하거나 도구를 셋업할 때 참고할 영문 가이드입니다.

| 역할 | 다음 문서 |
|---|---|
| 학습 코드를 NEXUS 로거에 연결하려는 사람 | [`../10_ARCHITECTURE.md`](../10_ARCHITECTURE.md) → [`../11_LOGGER_SETUP.md`](../11_LOGGER_SETUP.md) → [`../12_SCHEDULED_SYNC.md`](../12_SCHEDULED_SYNC.md) |
| 완료된 tfevents를 사후 업로드하려는 사람 | [`../13_POST_UPLOAD.md`](../13_POST_UPLOAD.md) |
| 중앙 MLflow 서버를 구축하려는 운영자 | [`../20_MLFLOW_SERVER_SETUP.md`](../20_MLFLOW_SERVER_SETUP.md) *(Step 0에 로컬 PC 검증 포함)* |
| 인터넷 차단된 GPU 노드를 셋업하려는 운영자 | [`../21_AIRGAPPED_GPU_SERVER_SETUP.md`](../21_AIRGAPPED_GPU_SERVER_SETUP.md) *(Step 1에서 Method A/B로 오프라인 설치, Step 2에서 smoke test 검증)* |
| Pipeline A cron sync을 운영하려는 운영자 | [`../12_SCHEDULED_SYNC.md`](../12_SCHEDULED_SYNC.md) *(Verification checklist 포함)* |
| 옵션 기능을 살펴보려는 사람 | [`../30_ADVANCED_FEATURES.md`](../30_ADVANCED_FEATURES.md), [`../31_CHART_SETTINGS_GUIDE.md`](../31_CHART_SETTINGS_GUIDE.md) |


# 🧪 [실험 페이지 템플릿]

> **사용법:** 이 파일을 복사해서 Confluence에 새 페이지로 만드세요.
> `[ ]` 표시된 항목은 실험 전에 반드시 채워야 합니다.
> 실험 전 작성 항목과 실험 후 작성 항목이 구분되어 있습니다.

---

# 🔬 [실험명] — [한 줄 설명]

**Experiment:** `reward_shaping` *(해당 MLflow Experiment 이름)*
**Run 이름:** `kim_contact_weight_v1` *(MLflow Run 이름 규칙 준수)*
**담당자:** @이름
**시작일:** YYYY-MM-DD
**상태:** 🟡 계획 중 / 🔄 진행 중 / ✅ 완료 / ❌ 실패 / ⏸️ 보류

---

## ✍️ [실험 전 작성] 목적 & 가설

> ⚠️ **이 섹션은 학습을 시작하기 전에 작성해야 합니다.**
> 결과를 보고 나서 작성하면 사후합리화가 됩니다.

### 해결하려는 문제

```
현재 어떤 문제가 있는지, 어떤 현상을 관찰했는지 구체적으로 서술합니다.

예시:
  현재 baseline_ppo 실험에서 grasp 성공률이 40% 수준에서 정체됨.
  MLflow 로그를 보면 episode 후반부에 contact force가 급격히 증가하다가
  물체를 튕겨내는 패턴이 반복됨.
  → contact force를 reward에 포함해서 이 패턴을 줄일 수 있을지 확인.
```

### 가설

> 구체적이고 **측정 가능한** 예측을 씁니다. "좋아질 것 같다"는 가설이 아닙니다.

```
예시:
  contact force reward (weight=0.3)를 추가하면,
  10M step 기준 grasp_success_rate가 baseline(40%)보다 15%p 이상 향상될 것이다.
```

**가설이 맞다고 판단하는 기준:**

```
예시:
  grasp_success_rate > 55% at 10M steps
  contact_force_mean < 5.0 N (현재 평균 12.0 N)
```

### 이 실험이 중요한 이유

```
예시:
  Sim-to-Real에서 가장 큰 실패 원인이 excessive contact force임.
  이를 reward로 제어하면 Real 이전 성공률이 높아질 것으로 기대.
```

---

## ⚙️ [실험 전 작성] 실험 설계

### 변경 사항 (Baseline 대비)

| 항목 | Baseline 값 | 이번 실험 값 | 변경 이유 |
|---|---|---|---|
| `reward.contact_weight` | 0.0 (없음) | 0.3 | 가설 검증 |
| 기타 | 동일 | 동일 | — |

> 💡 한 번에 하나만 바꿉니다. 여러 개를 동시에 바꾸면 원인을 알 수 없습니다.

### 비교 기준 (Baseline)

| 항목 | 내용 |
|---|---|
| Baseline Run | [kim_ppo_seed42_v1](http://192.168.1.42:5000/...) |
| Baseline Experiment | `baseline_ppo` |
| 비교 지표 | `episode_rewards/step`, `eval/grasp_success_rate` |

### MLflow 설정

```python
# make_logger() 호출 시 사용한 설정
experiment_name = "reward_shaping"
run_name        = "kim_contact_weight_v1"
tags = {
    "method":          "ppo",
    "component":       "reward",
    "ablation_target": "contact_weight",
    "baseline_run_id": "abc123def456",
}
```

### 학습 환경

| 항목 | 값 |
|---|---|
| Seed | 42 |
| Isaac Lab 버전 | 1.2.0 |
| PhysX Solver | TGS |
| Num Envs | 4096 |
| Max Steps | 50M |
| 사용 GPU | Blackwell GPU 0 |

---

## 📊 [실험 후 작성] MLflow 링크

> ⚠️ 수치를 여기에 직접 적지 않습니다. MLflow 링크를 통해 확인합니다.

| Run 이름 | MLflow 링크 | 상태 | 비고 |
|---|---|---|---|
| `kim_contact_weight_v1` | [링크](http://192.168.1.42:5000/...) | ✅ 완료 | 메인 실험 |
| `kim_contact_weight_v2` | [링크](http://192.168.1.42:5000/...) | ✅ 완료 | 재실험 (seed 변경) |

**→ 수치 비교는 위 MLflow 링크에서 "Compare" 기능으로 확인하세요.**

---

## 🔍 [실험 후 작성] 결과 해석

> 수치가 아닌 **의미와 패턴** 을 씁니다.
> "success_rate가 73.2%였다"가 아니라 "baseline 대비 유의미하게 향상되었다"처럼.

### 가설 검증 결과

```
✅ 가설 확인됨 / ❌ 가설 기각됨 / ⚠️ 부분적으로 확인됨

예시:
  ⚠️ 부분적으로 확인됨.
  success_rate는 baseline 대비 향상되었으나 목표치(55%)에는 미치지 못함.
  contact_force_mean은 기대대로 감소했으나, 특정 구간에서 reward collapse 관찰.
```

### 관찰된 주요 패턴

```
예시:
  - 학습 초반(0~5M step)에는 contact reward가 오히려 exploration을 방해함
  - 5M step 이후부터 contact_force_mean이 안정적으로 감소
  - episode 후반부의 "튕겨내기" 패턴은 70% 감소
  - 단, 물체 rotation 30° 이상 구간에서 여전히 불안정
```

### 예상과 달랐던 점

```
예시:
  - contact_weight=0.3이 예상보다 강하게 작용해 초반 학습 속도가 느려짐
  - success_rate 향상폭이 예상(15%p)보다 작았음 (실제: 8%p)
  - rotation 구간 불안정성은 contact reward와 무관한 별도 원인으로 보임
```

---

## 💥 [실험 후 작성] 실패 분석 / Edge Cases

> 실패가 없어도 이 섹션을 작성합니다. "문제없음"도 기록입니다.

### 관찰된 실패 패턴

| 조건 | 관찰된 현상 | 추정 원인 |
|---|---|---|
| 물체 rotation > 30° | grasp slip 반복 | contact reward가 angular slip을 포착 못함 |
| contact_weight > 0.5 | reward collapse (3M step 이후) | 단일 reward 항목이 전체를 압도 |
| 학습 초반 1M step | exploration 저하 | contact 패널티가 너무 강하게 작용 |

### Sim artifact 의심 사항

```
예시:
  PhysX contact normal 방향이 실제 물체 표면과 불일치하는 것으로 의심.
  Isaac Lab 1.2.0에서 concave mesh의 contact point 계산 오류 가능성.
  → 다음 실험 전 PhysX contact visualization으로 확인 필요.
```

---

## 🔜 [실험 후 작성] 다음 실험 방향

> 구체적인 액션 아이템으로 작성합니다. "더 연구가 필요하다"는 안 됩니다.

- [ ] contact_weight를 [0.1, 0.2, 0.3] sweep → Ablation Study로 진행 | 담당: @kim
- [ ] rotation 구간 불안정성 원인 파악 → PhysX contact visualization 확인 | 담당: @lee
- [ ] 학습 초반 contact_weight를 낮게 시작해서 curriculum으로 높이는 방식 시도 | 담당: @kim

---

## ✅ [실험 후 작성] 팀 결정

> 이 실험을 바탕으로 팀이 내린 결정을 기록합니다.

**결정 내용:**

```
예시:
  contact force reward 구조 자체는 유효함을 확인.
  단, weight=0.3은 너무 강하며 0.1~0.2 범위에서 재탐색 필요.
  rotation 구간 불안정성은 별도 이슈로 분리해서 추적.
```

**채택 여부:** ✅ 채택 / ⚠️ 조건부 채택 / ❌ 기각 / ⏸️ 보류

**결정일:** YYYY-MM-DD
**결정 참여자:** @kim, @lee, @park

> 이 결정은 [Decision Log](링크) 에도 기록되었습니다.

---

## 📎 첨부 자료

| 자료 | 링크/위치 |
|---|---|
| MLflow Run (메인) | [kim_contact_weight_v1](http://192.168.1.42:5000/...) |
| env_cfg.yaml | MLflow artifact > configs/ |
| reward_fn.py | MLflow artifact > configs/ |
| best.pth | MLflow artifact > checkpoints/ |

---

*이 페이지 템플릿에 대한 개선 제안은 @jonghochoi에게 알려주세요.*

# `upload_eval.py` 수동 테스트 가이드

> ⚠️ **이 문서는 임시 파일입니다.** `claude/mlflow-artifact-upload-3EDwm` 브랜치 머지 전에 통과 확인 후 삭제(또는 해당 commit revert) 해 주세요. 정식 사용자 문서는 [`docs/13_POST_UPLOAD.md`](docs/13_POST_UPLOAD.md) 입니다.

자동화로 검증한 항목(아티팩트 트리, HTTP `Content-Type`, URL 인코딩 등)은 e2e 스크립트로 이미 통과 확인됨. 이 가이드는 **브라우저로만 확인 가능한 부분** + 운영자 환경에서 한 번 직접 돌려보면 좋은 시나리오를 모았습니다.

---

## 0. 사전 준비

```bash
# 1) 브랜치 체크아웃
git fetch origin
git checkout claude/mlflow-artifact-upload-3EDwm

# 2) venv (없으면)
bash setup.sh
source ~/.nexus/activate.sh        # 또는 ~/.nexus/venv/bin/activate

# 3) 로컬 MLflow 띄우기 (기본 포트 5100)
bash scheduled_sync/start_local_mlflow.sh

# 4) 브라우저로 접속
#    http://127.0.0.1:5100
```

> 다른 포트를 쓰고 있다면 아래 명령들의 `--tracking_uri http://127.0.0.1:5100` 값을 환경에 맞게 바꿔주세요.

---

## 1. 부모 run 시드 (먼저 한 번)

`upload_eval.py` 는 **이미 존재하는** run 에 붙는 스크립트라 먼저 베이스 run 이 필요합니다. 가짜 tfevents 로 짧게 하나 만듭니다.

```bash
# tfevents 만들기 (Python 스니펫)
mkdir -p /tmp/nexus_manual/tb/baseline_v1
python -c "
from tensorboardX import SummaryWriter
w = SummaryWriter('/tmp/nexus_manual/tb/baseline_v1')
for s in range(50):
    w.add_scalar('train/episode_reward', 10 + s*0.5, s)
    w.add_scalar('train/loss', 1.0/(s+1), s)
w.close()
"

# upload_tb.py 로 run 생성 (이름: baseline_v1)
echo "y" | python post_upload/upload_tb.py \
    --tb_dir /tmp/nexus_manual/tb/baseline_v1 \
    --tracking_uri http://127.0.0.1:5100 \
    --experiment manual_test \
    --run_name baseline_v1 \
    --tags researcher=me seed=1 task=manual_check \
    --no_verify
```

**브라우저 확인**: `http://127.0.0.1:5100` → `manual_test` 실험 → `baseline_v1` run 이 나타나면 OK.

---

## 2. 평가 산출물 디렉토리 만들기

진짜 mp4 가 있으면 베스트지만, 없어도 됩니다. 짧은 mp4 가 있으면 폴더에 넣어 두세요.

```bash
mkdir -p /tmp/nexus_manual/eval/baseline_v1/plots

# 진짜 mp4 가 있다면 cp 로 옮기고, 없으면 ffmpeg 으로 생성
ffmpeg -y -f lavfi -i testsrc=duration=5:size=320x240:rate=15 \
    /tmp/nexus_manual/eval/baseline_v1/rollout.mp4 2>/dev/null

# 옵션: GIF 미리보기
ffmpeg -y -i /tmp/nexus_manual/eval/baseline_v1/rollout.mp4 \
    -vf "scale=160:-1,fps=8" \
    /tmp/nexus_manual/eval/baseline_v1/rollout_preview.gif 2>/dev/null

# 리포트와 스코어
echo "# Eval report — baseline_v1" > /tmp/nexus_manual/eval/baseline_v1/report.md
echo '* success_rate: 0.87' >> /tmp/nexus_manual/eval/baseline_v1/report.md
echo '{"success_rate": 0.87, "mean_return": 132.4}' \
    > /tmp/nexus_manual/eval/baseline_v1/metrics.json

ls -la /tmp/nexus_manual/eval/baseline_v1/
```

---

## 3. 핵심 시나리오 — mp4 인라인 재생 ⭐

이 단계가 **이번 PR 의 메인 가치**입니다. 자동화로는 HTTP `Content-Type: video/mp4` 까지만 확인했으니, 실제 재생 동작은 여기서 손으로 봐야 합니다.

```bash
# dry-run 으로 미리보기
python post_upload/upload_eval.py \
    --tracking_uri http://127.0.0.1:5100 \
    --experiment manual_test --run_name baseline_v1 \
    --eval_dir /tmp/nexus_manual/eval/baseline_v1 \
    --eval_id manual_001 \
    --metrics success_rate=0.87 mean_return=132.4 \
    --tags observer_commit=abc1234 \
    --dry_run

# 진짜 업로드
echo "y" | python post_upload/upload_eval.py \
    --tracking_uri http://127.0.0.1:5100 \
    --experiment manual_test --run_name baseline_v1 \
    --eval_dir /tmp/nexus_manual/eval/baseline_v1 \
    --eval_id manual_001 \
    --metrics success_rate=0.87 mean_return=132.4 \
    --tags observer_commit=abc1234
```

### 브라우저 체크리스트

`baseline_v1` run 페이지를 열고 좌측 **Artifacts** 패널에서:

- [ ] `eval/manual_001/` 디렉토리가 보인다
- [ ] 그 아래 `index.html`, `rollout.mp4`, `rollout_preview.gif`, `report.md`, `metrics.json` 이 모두 있다
- [ ] **`index.html` 클릭 → 우측 미리보기에 페이지가 렌더링된다** (다운로드 안 됨)
- [ ] 페이지 안 `<video>` 플레이어에서 **재생 버튼이 보이고 mp4 가 재생된다**
- [ ] 재생 컨트롤로 **시킹(드래그) 이 동작한다** ← Range request 지원 여부 확인
- [ ] `rollout_preview.gif` 도 같은 페이지에 인라인으로 보인다
- [ ] `report.md`, `metrics.json` 링크 클릭 시 같은 Artifact 패널에서 열린다

추가:

- [ ] 좌측 **Metrics** 패널에 `eval/success_rate`, `eval/mean_return` 이 차트로 보인다
- [ ] **Tags** 에 `eval.last_id=manual_001`, `eval.observer_commit=abc1234` 이 박혀 있다

---

## 4. 두 번째 평가 번들 — 같은 run 에 누적

같은 mp4 폴더로 다른 `eval_id` 만 바꿔서 한 번 더:

```bash
echo "y" | python post_upload/upload_eval.py \
    --tracking_uri http://127.0.0.1:5100 \
    --experiment manual_test --run_name baseline_v1 \
    --eval_dir /tmp/nexus_manual/eval/baseline_v1 \
    --eval_id manual_002 \
    --metrics success_rate=0.91
```

체크:

- [ ] Artifacts 트리에 `eval/manual_001/` 과 `eval/manual_002/` 둘 다 있다
- [ ] `eval.last_id` 태그가 `manual_002` 로 갱신됐다
- [ ] `eval/success_rate` 차트에 점이 두 개(또는 그 이상) 찍혀 있다
- [ ] 두 번째 `index.html` 도 정상 재생된다

---

## 5. `--no-index` — 자동 HTML 안 만들기

observer 가 자기 페이지를 동봉하거나, 아예 mp4 만 던지고 싶을 때:

```bash
echo "y" | python post_upload/upload_eval.py \
    --tracking_uri http://127.0.0.1:5100 \
    --experiment manual_test --run_name baseline_v1 \
    --eval_dir /tmp/nexus_manual/eval/baseline_v1 \
    --eval_id manual_no_index \
    --no-index
```

체크:

- [ ] `eval/manual_no_index/` 안에 `index.html` 이 **없다**
- [ ] mp4 / gif / md / json 만 그대로 올라가 있다

---

## 6. 사용자가 자기 `index.html` 동봉

```bash
mkdir -p /tmp/nexus_manual/eval_user_idx
cp /tmp/nexus_manual/eval/baseline_v1/rollout.mp4 /tmp/nexus_manual/eval_user_idx/
cat > /tmp/nexus_manual/eval_user_idx/index.html <<'HTML'
<!doctype html><html><body>
<h1>Observer's own report page</h1>
<p>This text proves the auto-generator did not overwrite us.</p>
<video controls src="rollout.mp4" width=400></video>
</body></html>
HTML

echo "y" | python post_upload/upload_eval.py \
    --tracking_uri http://127.0.0.1:5100 \
    --experiment manual_test --run_name baseline_v1 \
    --eval_dir /tmp/nexus_manual/eval_user_idx \
    --eval_id manual_user_idx
```

스크립트 출력에 다음 줄이 보여야 합니다:

```
eval_dir already contains index.html — leaving it alone.
```

체크:

- [ ] Artifacts 의 `eval/manual_user_idx/index.html` 을 클릭하면 **"Observer's own report page"** 텍스트가 보인다
- [ ] 자동 생성 배너(`Generated by upload_eval.py on ...`) 가 **나타나지 않는다**

---

## 7. 잘못된 run_name 은 dry-run 에서도 차단되어야 한다

```bash
python post_upload/upload_eval.py \
    --tracking_uri http://127.0.0.1:5100 \
    --experiment manual_test --run_name oops_typo \
    --eval_dir /tmp/nexus_manual/eval/baseline_v1 \
    --dry_run
echo "exit code: $?"
```

체크:

- [ ] `[ERROR] No run found in experiment 'manual_test' with run_name='oops_typo'.` 출력
- [ ] exit code = `1`

---

## 8. 파일명에 공백 / em-dash

```bash
mkdir -p "/tmp/nexus_manual/eval_spaces"
cp /tmp/nexus_manual/eval/baseline_v1/rollout.mp4 \
   "/tmp/nexus_manual/eval_spaces/run 5 — best.mp4"

echo "y" | python post_upload/upload_eval.py \
    --tracking_uri http://127.0.0.1:5100 \
    --experiment manual_test --run_name baseline_v1 \
    --eval_dir /tmp/nexus_manual/eval_spaces \
    --eval_id manual_spaces
```

체크 (Artifacts 의 `eval/manual_spaces/index.html` 을 열고):

- [ ] `<strong>run 5 — best.mp4</strong>` 처럼 **사람이 읽을 수 있는 파일명** 이 보인다
- [ ] 그 아래 `<video>` 플레이어에서 **mp4 가 재생된다** (URL-encoding 으로 정상 fetch 됨을 확인)

---

## 9. `--history` 가 두 종류를 잘 분리하는지

```bash
# eval 만
python post_upload/upload_eval.py --history

# tb 만
python post_upload/upload_tb.py --history
```

체크:

- [ ] `upload_eval.py --history` 출력에 위에서 만든 eval 번들들이 있고, `Kind` 컬럼이 `eval` 이다
- [ ] `upload_tb.py --history` 출력에는 §1 의 `baseline_v1` (`Kind=tb`) 만 보이고 eval 항목은 안 섞인다
- [ ] `python post_upload/verify_tb.py --from-last` 가 §1 의 tb run 으로 잘 잡힌다 (eval 레코드를 잘못 끌고 오지 않음)

---

## 10. 정리

```bash
# MLflow 서버 종료
lsof -ti :5100 | xargs -r kill

# 픽스처 삭제
rm -rf /tmp/nexus_manual ./mlruns_training ./mlflow_training.log .mlflow_local.pid

# 테스트로 쌓인 history 도 지우고 싶다면:
#   rm ~/.nexus/history.json
```

---

## 11. 머지 전 마무리

이 가이드 통과 후:

```bash
# 이 파일과 해당 commit 제거
git rm MANUAL_TEST_upload_eval.md
git commit -m "chore: drop manual test guide"
# 또는 가이드 추가 commit 자체를 revert/drop
```

PR 본문에 **§3, §4, §5, §6, §8 의 브라우저 체크리스트 모두 통과** 라고 한 줄 적어 두면 리뷰어가 같은 환경에서 재현할 필요가 줄어듭니다.

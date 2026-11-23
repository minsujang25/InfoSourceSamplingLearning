# Gate 0E — 로컬 Mac 실행 안내 (2026-11-23)

**현재 구현:** `paper-b-gate0e-peer-channel-implementation` 브랜치
**완료:** 30개 회귀 테스트 및 Gate 0E-0 48회 N24 T6 smoke test GitHub Actions PASS
**미실행:** Gate 0E-1 480회 N100 T200 및 그 이후 장기 실험
**보존:** Paper B v5.2, IIIa 생산 결과 및 IIIb 사전 포함규칙

[GitHub Actions smoke 실행](https://github.com/minsujang25/InfoSourceSamplingLearning/actions/runs/45273017872)

## 1. 이미 로컬에 preanalysis 브랜치가 있는 경우

아래 명령은 **Git 저장소 최상위 디렉터리**에서 실행한다.

```bash
git status --short
git fetch origin refs/heads/paper-b-gate0e-peer-channel-implementation
git switch -c paper-b-gate0e-peer-channel-implementation FETCH_HEAD
```

`git status --short`가 수정 파일을 보인다면 먼저 별도 커밋,
stash 또는 백업으로 보존한다. 같은 이름의 로컬 브랜치가
이미 존재하면 `git switch <branch>` 후
`git merge --ff-only FETCH_HEAD`를 사용한다.

`origin/...` 이름을 이용한 `git switch --track`은 로컬
`remote.origin.fetch` 설정에 따라 참조가 없을 수 있으므로
위처럼 **명시적 브랜치 fetch + FETCH_HEAD**를 사용한다.

## 2. 설치 상태 및 설계 확인

이미 사용 중인 `paper-b` Conda 환경에서 실행한다.

```bash
conda activate paper-b
python -m paper_b.experiments.check_gate0e_design
python -m unittest paper_b.test_gate0e_peer_channels -v
python -m paper_b.experiments.run_gate0e_local \
  --phase mechanism --workers 4 --dry-run \
  --output-dir "$HOME/Research/paperb_gate0e_results"
```

설계 명세는 전체 14,928회의 *계획된* 실행을 포함하지만,
dry-run은 아무 시뮬레이션도 시작하지 않는다.

## 3. 본 실험: Gate 0E-1 (480 runs, T=200)

아래를 별도 로컬 디스크 경로에서 실행한다. **iCloud Drive
안의 저장소에서 실행하더라도 결과 위치는 iCloud 밖으로 지정.**

```bash
mkdir -p "$HOME/Research/paperb_gate0e_results"

python -m paper_b.experiments.run_gate0e_local \
  --phase mechanism \
  --workers 4 \
  --output-dir "$HOME/Research/paperb_gate0e_results"
```

사전설계에 따라:
- matched seeds: **6921–6940**, 총 20개 (재사용하지 않는다).
- `flat` prior, N=100, T=200, K=1, epsilon=.05, credit=20.
- 4개 채널 E/M/S/L × 3 sender (null, adaptive,
  credibility_aware) × 2 receiver (adaptive, frozen).
- `source_posterior`, `tau_social=1`, t0 centroid audit,
  first_audit frozen ranking, lambda=1.
- 시작 26기간에 대한 시민별 메시지 요청/수신/audit 로그 저장.
- worker 수는 Mac CPU 및 메모리에 맞춰 2 또는 4 등으로
  조정 가능하며 과도한 병렬 실행을 피한다. Worker 수는
  scientific design ID와 별도로 실행 자원만 제어한다.

### 중단되었을 때

**동일한 phase와 결과 디렉터리**를 그대로 사용한다.

```bash
python -m paper_b.experiments.run_gate0e_local \
  --phase mechanism --workers 4 --resume \
  --output-dir "$HOME/Research/paperb_gate0e_results"
```

`--resume`은 동일한 design hash의 유효한 완료 블록만 건너뛴다.
실험 중 코드를 수정하면 design ID가 달라질 수 있으므로,
한번 시작한 실험은 그 코드를 동결한 상태로 완료한다.

## 4. 완료 후 확인

`run_gate0e_local`은 마지막에 matched pilot validator를 실행한다.
반드시 `Pilot gate: PASS`, `observed runs: 480/480`을 확인한다.

결과 경로 구조:

```text
~/Research/paperb_gate0e_results/
  gate0e_mechanism/
    pilot_<DESIGN_ID>/
      pilot_manifest.json
      pilot_gate.json
      pilot_summary.json
      runs.csv
      gate0b_sender_damage.csv
      gate0e_channel_contrasts.csv
      gate0e_peer_event_period_summary.csv
      gate0d_sender_events.csv.gz
      shards/*.json.gz
      ...
    paper_b_pilot_<DESIGN_ID>_shareable.zip
```

중요한 것:
- `pilot_gate.json`의 `pass:true`.
- `pilot_summary.json`의 `matched_t0_audits_equal:true`,
  `extended_edge_maps_equal:true`, `n_gate0e_channel_contrasts:80`.
- `gate0e_channel_contrasts.csv`에는 seed×receiver×sender
  (20×2×2=80)의 E/M/S/L 차이 분해가 있다.
- `gate0e_peer_event_period_summary.csv`에는 매 기간 channel별
  실제 요청/전달 수량이 있다.
- `gate0d_sender_events.csv.gz`는 개별 시민 기반 원자료로
  큰 용량이므로 필요할 때만 공유하고, 우선 shareable ZIP을
  검토용으로 업로드한다.

결과의 효과 크기와 신뢰구간은 seed-block 단위로 계산한다.
2·4개 seed 진단 값과 20개 seed 분석을 단순 합쳐 추론하지 않는다.
결과 확인 후에도 원래의 IIIa·IIIb와 v5.2 내용은 자동 수정하지 않는다.

## 5. 추가 실험

`--phase smoke`는 CI에서 이미 통과한 48회 미니 진단을
Mac에서도 다시 돌리는 선택지다.

`--phase stability`는 100개 새 seed로 2,400회
N100 T400을 실행한다. Gate 0E-1 검토 전에는 시작하지 말 것.

원래 12,000회 Gate 1 production은 이 wrapper에 포함되지 않는다.
기존 고정된 highS/lowS 실험의 초기 조건 구현과
추가 검증이 완료돼야만 별도 실행할 수 있다.

# v6 로컬 실행 가이드 — 첫 technical pilot

이번 단계는 **기술 검증**이다. 기존 논문, production, Gate 0E-2를 덮어쓰지 않는다. 모델 코드는 변경하지 않았다. 새 실행기는 기존 run_condition과 structural_designs를 직접 호출해 centroid / first_audit를 명시한다. 기존 run_canonical_production 명령은 사용하지 않는다.

## 1. 저장소 및 환경

로컬 저장소 root에서 먼저 상태 확인. iCloud/OneDrive의 파일 읽기 오류가 있으면 cloud 동기화 밖의 별도 clone을 권장한다. 로컬 변경을 무시하거나 reset하지 말 것.

```bash
git status --short
git fetch origin refs/heads/paper-b-v6-theory-reconstruction
git switch -c paper-b-v6-theory-reconstruction FETCH_HEAD
git rev-parse HEAD
conda activate paper-b
```

이미 로컬 브랜치가 있으면 마지막 switch 대신 다음을 사용한다. 작업 트리가 clean일 때만 진행한다.

```bash
git switch paper-b-v6-theory-reconstruction
git merge --ff-only FETCH_HEAD
```

## 2. 기존 회귀 검증과 신규 실행기 문법 검사

```bash
python -m py_compile paper_b/experiments/run_v6_pilot.py
python -m unittest paper_b.test_gate0a_initial_audit paper_b.test_gate0b_canonical_pilot paper_b.test_gate0c_credibility_aware paper_b.test_gate0d_mechanism paper_b.test_gate0e_peer_channels -v
python -m paper_b.experiments.check_gate0e_design
```

실패하면 다음 단계 중지. Mesa 3.5.1, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2, networkx 3.3은 이전 Gate 0E2 환경. Dry-run에서 실제 설치 버전과 upstream fingerprint를 확인한다. Upstream fingerprint는 이전 754d7c373ff4b9f6eb24e1b0044917d559f69397531a053d5ab64d314cd67774와 동일할 것으로 예상되나 실제 확인 필요. v6 fingerprint와 design ID는 신규 값이다.

## 3. Dry-run

```bash
python -m paper_b.experiments.run_v6_pilot \
  --phase technical --workers 4 \
  --output-dir "$HOME/Research/paperb_v6_results" \
  --dry-run
```

기대: seeds 8001,8002; N100 T26; H=16, G=8, C=32 조건/seed; 112 grid + 6 parity 비교. No simulations in dry-run.

## 4. Technical pilot

```bash
python -m paper_b.experiments.run_v6_pilot \
  --phase technical --workers 4 \
  --output-dir "$HOME/Research/paperb_v6_results"
```

중단 후 재개는 코드, 환경, 경로, phase를 그대로 두고 위 명령에 --resume을 추가한다. 완료된 seed/suite shard를 검증 후 재사용한다. 손상된 shard가 발견되면 실패 내용을 보존하고 원인을 확인한다. 임의로 파일을 삭제하거나 다른 fingerprint 결과를 합치지 않는다.

## 5. 결과 전달

v6_technical_<design_id>/paper_b_v6_technical_<design_id>_shareable.zip을 전달한다. v6_gate.json이 pass=true, status=technical_only, grid_runs=112, parity_runs=6, blocks=6이면 기술 검증 통과. gate가 pass=false 또는 incomplete이거나 traceback이 나오면 출력과 오류를 공유한다. PASS는 과학적 결론이나 본 생산 승인과 동일하지 않다.

scientific phase는 계획만 미리 확인 가능:
```bash
python -m paper_b.experiments.run_v6_pilot --phase scientific \
  --output-dir "$HOME/Research/paperb_v6_results" --dry-run
```
실제 scientific 실행은 코드에서 잠금. Technical 결과 검토 및 전체 subgroup temporal/request 측정 사양 동결 후 활성화할 것. 500-seed production 또는 robustness는 아직 실행하지 않는다.

## 검증 상태
GitHub 원본 코드 및 새 파일을 정적 대조했다. ChatGPT 작업 환경은 Mesa 미설치 및 GitHub DNS 차단으로 로컬 checkout/실행 테스트를 수행할 수 없었다. 위 regression 및 technical pilot이 실제 검증이다. 오류 시 기존 과학 코드 수정 없이 새 runner부터 점검한다.

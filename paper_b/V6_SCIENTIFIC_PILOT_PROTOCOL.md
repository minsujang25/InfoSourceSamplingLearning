# V6 structural scientific pilot — frozen 2026-11-24

Technical prerequisite: uploaded paper_b_v6_technical_f2f3c133bd4c_shareable.zip has gate pass=true, 112 grid runs, 6 representative parity runs, 6 blocks and T26. Its upstream fingerprint is 754d7c373ff4b9f6eb24e1b0044917d559f69397531a053d5ab64d314cd67774. Retain that batch unchanged. This protocol activates the previously declared H/G scientific pilot only; it does not authorize final production, robustness variants or C bridge.

## Design
- Preflight: seed 8001, H16/G8, T26, 24 grid + 2 representative parity runs. Required for the new measurement implementation. Gate tied to exact runner fingerprint and software versions.
- Scientific: seeds 6001–6020 inclusive, H16/G8, T400, 480 grid + 40 representative parity runs. 320 H + 160 G. Reuse legacy seed numbers intentionally for version comparison, not independent confirmation. No changes to the scientific model, upstream runner, old technical runner, or old design manifests.
- Common specification per V6_MODEL_SPECIFICATION.md: N100, epsilon .05, C20, K1, interval5, source_posterior, tau1, centroid t0 audit and first_audit frozen ranks, null/fixed_biased. H: 2x2x2x2; G: 2x2x2. This is a structural pilot, not a strategic-sender experiment; keep prior Gate0E2 results separate.
- New standalone run_v6_scientific module preserves run_v6_pilot and its old scientific lock; do not use the old command for T400.

## Passive measurement
Capture the instance returned by the upstream constructor during run_condition, only to read its completed histories. No model or RNG writes, no altered update order. Citizen posterior arrays include initial state at step0 and post-period t at step t+1. Export group/role mean belief, squared bias, population variance, MSE and mean posterior SD at every existing belief checkpoint plus initial state and terminal. Full-horizon request counts are reconstructed from realized_reliance * C (integer allocation by construction), with integer/budget checks, cumulative at checkpoints; split credibility audit versus substantive theta. Requests are not delivered messages, unique source observations or evidence precision. Gateway peer count overlaps same/other peer categories. Export explicit zero counts and undefined shares when denominator zero. Existing Q/W/I, ranks, acquisition and evidence network checkpoints retained.

## Analysis freeze
Terminal paired high-minus-low contrasts in each prior/sender/receiver/group, HxS difference-in-differences and damage high-minus-low (fixed-minus-null within structure, then high-minus-low). Primary puzzle: H, high-S, group +1, fixed-biased, adaptive; report all groups and conditions. W and Q separately, MSE and bias/variance, joint seed sign frequencies. No inference about causal mediation or source provenance. Monte Carlo MCSE and 10,000 whole-seed paired bootstrap draws, RNG 20261124; n=1 preflight no intervals. Scientific output permits analysis of group trajectory and cumulative actual requests at T200/T400. Compare legacy only with matched historical raw data and verified version manifests, not pooled outcomes. All directions retained; no sign-based expansion. Existing uniform gateway benchmark must use consistent peer-conditional denominator.

## Local execution
From clean branch checkout, conda activate paper-b:
```bash
python -m py_compile paper_b/experiments/run_v6_scientific.py paper_b/experiments/v6_scientific_measurement.py paper_b/experiments/v6_scientific_analysis.py
python -m unittest paper_b.test_v6_scientific_measurement paper_b.test_v6_pilot_parity -v
python -m paper_b.experiments.run_v6_scientific --phase preflight --workers 4 --output-dir "$HOME/Research/paperb_v6_results"
python -m paper_b.experiments.run_v6_scientific --phase scientific --workers 4 --output-dir "$HOME/Research/paperb_v6_results" --dry-run
python -m paper_b.experiments.run_v6_scientific --phase scientific --workers 4 --output-dir "$HOME/Research/paperb_v6_results"
```
The last command is allowed only after matching preflight PASS. Worker count may change (does not alter design ID); use 4 initially, increase to 8 only with adequate RAM. Each seed/suite shard is written atomically after complete validation and representative logger parity. Resume identical phase/design with --resume; corrupt shards fail closed. Consolidation is two-pass, one shard at a time. Shareable ZIP contains consolidated CSV.gz and JSON, not duplicate raw shard edge records; retain shards locally for further audit. If upload size is excessive, share manifest, gate, seed_contrasts, contrast_summary, group_terminal, group_belief_checkpoints and group_request_checkpoints first. Final gate pass means technical integrity of the scientific sample, not scientific confirmation or production authorization.

## Verification and scope
Actual local compile, unit tests, preflight and long-horizon run required. No claim that code was executed remotely. Remaining scientific issues: precision cap, alternative acquisition and audit schedules, C fixed-biased bridge, strategic surveillance/segmentation continuity, eventual final seed size; all require separate frozen designs. No model change or new full production in this step.

# Gate 0E-2 stability analysis and manuscript triage — 2026-11-23

**Status:** planning/readiness memo; **NO Gate 0E-2 simulations run** and **NO manuscript or prior production rewritten**.

**Base:** `paper-b-gate0e-peer-channel-implementation` @ `e98cb992cc75772e8fd8453cd079efbd393045cf`. Preserve v5.2 (2026-10-29), Experiment IIIa production, Experiment IIIb's frozen inclusion rule, and the original 12,000-run Gate 1 proposal. This document is a downstream review of the frozen Gate 0E preanalysis, not a retrospective amendment to it.

## 1. Source and Gate 0E-1 audit

Verified against the uploaded transition package of 2026-11-23 (not new simulations):
- `pilot_gate.json`: PASS, errors=0, matched blocks 80/80, runs 480/480, fixed horizon, finite outcomes, matched fingerprints and initial t=0 peer audits.
- `pilot_manifest.json`: `design_id=971c9ebb6f00`; scientific code fingerprint `754d7c373ff4b9f6eb24e1b0044917d559f69397531a053d5ab64d314cd67774`; 20 matched seeds 6921–6940; N=100, T=200; flat priors; E/M/S/L × null/myopic-adaptive/credibility-aware × adaptive/frozen; source_posterior, tau_social=1, first_audit freeze, centroid t=0, lambda=1, K=1, epsilon=.05, credit=20.
- The original 1,248,000 citizen-level event rows are NOT included in the small transition ZIP; the package has the summary with 12,480 seed-period channel observations covering periods 0–25.
- Recomputed from `gate0e_channel_contrasts.csv`, at T=200, L−E mean damage for myopic/adaptive +13.676, myopic/frozen +10.319, aware/adaptive +0.855, aware/frozen −1.128. L−S: respectively +6.338, +11.397, +0.544, +0.086. These are 20-seed exploratory Monte Carlo contrasts, **not** a confirmatory claim.
- Null MSE differs across channels (e.g. adaptive null: E 0.00027, M 0.01425, S 0.11812, L 0.00122). Show sender MSE, null MSE, and D together.
- From belief checkpoints, myopic/adaptive L−E mean D is +54.091 at the end of 26 executed periods, +23.752 after 100 periods, +13.676 after 200 periods: a strong reason to assess T=400 before claiming persistence.
- E/M/S/L algebra is telescoping accounting, not independent causal mediation or original provenance reconstruction. Muted involves deliberately wasted attention.
- **CSV import gotcha:** pandas normally converts the literal string `null` into missing data. Use `pd.read_csv(..., keep_default_na=False)` or an explicit NA policy when pivoting by `jammer_regime`.

## 2. Gate 0E-2: frozen scientific design

- **100 NEW matched seeds 7501–7600**, N=100, T=400, flat initial regime.
- 4 environments E/M/S/L × 3 sender regimes null / adaptive myopic / credibility_aware × 2 receiver modes adaptive/frozen; **2,400 individual runs = 400 blocks × 6 runs**.
- Keep initial centroid audit at t=0, first strategic refresh t=1 then 1,6,11,..., K=1, surveillance_interval=5, credit=20, epsilon=.05, source_posterior, tau_social=1, frozen_ranking_mode=first_audit, lambda=1.
- Do not change the model objective, peer-channel code, seed set, prior regime, estimator, or case inclusion after inspecting 0E-1. Validate the actual local scientific code fingerprint and package versions against the 0E-1 manifest before running.
- A new **design_id** is expected because seeds/T and any event logging option differ. That is not permission to change the scientific code fingerprint unnoticed.

### Preflight issue: no event trace by default

The checked-in `run_gate0e_local.py` defaults to `--sender-event-period-limit 0` for stability. In `consolidate_gate0e.validate_channel_blocks`, event-based t=0 audit equality is then unavailable (`matched_t0_audits_equal=None`), although run-level finite/fingerprint and block gates still operate. Prefer the existing **non-invasive** `--sender-event-period-limit 2` option for Gate 0E-2, recording only periods 0 and 1. Expected 2400 × 100 × 2 = **480,000 citizen event rows**—less than the Gate 0E-1 1,248,000 event rows—and enabling checks of matched t=0 audits, unchanged M/S/L edge maps, and zero peer deliveries in M beginning at t=1. This is logging only; it does change the run's design ID. Trace on/off parity has earlier tests but should remain a preflight check.

Before producing data, run on the user's Mac in the existing `paper-b` Conda environment and a local NON-iCloud result directory:

```bash
git status --short
git fetch origin refs/heads/paper-b-gate0e-peer-channel-implementation
# Verify or cleanly checkout EXACT scientific base commit before proceeding.
conda activate paper-b
python -m paper_b.experiments.check_gate0e_design
python -m unittest paper_b.test_gate0e_peer_channels -v
python -m paper_b.experiments.run_gate0e_local \
  --phase stability --workers 4 \
  --sender-event-period-limit 2 --dry-run \
  --output-dir "$HOME/Research/paperb_gate0e_results"
```

After verifying local commit, environment, resources, dry-run matrix, output path, and any code-fingerprint mismatch, use the same command WITHOUT `--dry-run`. Resume only with exactly the same phase/arguments/code/output and `--resume`. Never mix shards from different design IDs. The connected GitHub tool cannot execute the user's Mac job; the 2,400 runs are still pending.

## 3. Stability analysis: acceptance and reporting

1. **Integrity gate (hard):** 400 unique complete blocks, 2400 unique complete runs, N=100 and T=400 in every cell, 100 seed blocks, 24 cells per seed; all finite; correct `null` slot; code/config/version manifest, identical structural peer edge maps across M/S/L, identical matched initial beliefs, t=0 audit equivalence, M delivers 0 peer messages at t>=1. Preserve `pilot_gate.json`, `pilot_manifest.json`, SHA/hash, logs and run-level outputs. No scientific interpretation on failed gate.
2. **Outcome accounting (always, not sign-selected):** For each sender × receiver × channel show mean total sender MSE, null MSE and D, SD across seeds, plus seed-paired contrasts `M−E`, `S−M`, `L−S`, `L−E`, `Ω` (aware−myopic total peer), `Γ` (adaptive−frozen total peer).
3. **Uncertainty:** paired seed-difference SD, MCSE = SD/sqrt(100), 10,000 whole-seed bootstrap resamples retaining all cells, percentile 95% Monte Carlo intervals, and fraction of seed differences above/below zero. Report all contrasts regardless of sign. Seed, not citizen or message, is the replication unit.
4. **Time path:** read checkpoint `horizon_step` rather than conflating zero-indexed `period` with T; the current runner supplies end-of-step 1,2,3,6,11,26,51,100,101,151,200,300,400 checkpoints at T=400. If exact T=50 is not available, do NOT present period 50 (= step 51) as T=50. Show supported checkpoints and the change from the T=200 exploratory profile without pooling 20 and 100 seeds as if one preplanned design.
5. **Mechanism:** inspect early actual peer request/delivery and Jammer request/delivery, audit shares and mean peer precision. With 2-period event capture, these are only early mechanism checks; later channel histories require separately justified passive aggregate checkpoints or an additional diagnostic run. `W_peer` is proximate precision, not origin attribution.
6. **Decision rule (non-directional):** A stable and interpretable full pattern can justify one concise secondary main-text comparison with full channel matrix and methods in the Supplement. Instability, denominator/null artifacts, or heterogeneous/sign-changing results argue for Supplement or reviewer-response reserve. Do not promote only positive contrasts. The Gate 0E-2 result alone does not retrospectively prove the broader Gate 1 highS/lowS hypothesis or replace IIIa/IIIb.
7. **Sensitivity budget:** no automatic expansion to new objective tuning, network designs, or complete highS/lowS cross before assessing T=400; retain initial-audit and event-source assumptions and disclose rank-cliff limitations.

## 4. Manuscript revision triage

**P0 now (editorial and provenance; no new result claims):**
- Create an explicit old-production vs Gate 0A–0E specification crosswalk: t=0 audit/first strategic t=1, `truth_clone` versus structural-slot `null`, `legacy_batch` vs `source_posterior`, tau_social and first_audit versus pre_disruption. Do not silently relabel archived v5.2 results.
- Verify v5.2 §5.3's total MSE versus excess-damage difference: previously flagged −3.304 (total) versus −3.297 (null-adjusted) at p.22. Confirm against original source before a surgical correction.
- Precisely define source-level repeated peer precision, external-source independence, `A`, `Λ_t`, `W_T`, and scope of adversary optimization. Preserve double-blind separation from title page.

**P1 after validated Gate 0E-2:**
- Choose the main contribution around network-conditioned damage, peer competition and corrective opportunities rather than universal network protection.
- If warranted, report a concise E vs L comparison (absolute losses and D) and a bounded M/S/L check, explicitly labelled as exploratory diagnostic in SM. Do not write `L−S` as peer retransmission mediation.
- Revise Abstract, Introduction, §§3–5, Discussion **only after** the new scientific conditions and production figures are reconciled; do not mix old and new estimands in a figure/caption.

**P2 separate but high priority for broader manuscript claims:**
- Reconcile original Gate 1 `highS/lowS` prior/structural designs with `structural_designs.py` on a separate branch. Retain `fixed_biased` in the predeclared 12,000-run 500-seed T=400 design; a validated smaller highS/lowS T400 calibration must precede it.
- IIIb shared-bottleneck versus independent corrective paths remains a separate frozen diagnostic; its prior inclusion thresholds govern main/SM/reserve decisions.

**P3 submission QA:** anonymized manuscript + separate title page, clear definitions of T and period, matched comparison footnotes, figures/SM links, bibliography, transparent replication and AI/tool declarations, and verified reproducibility.

Deadline: **2026-12-15** for invited full manuscript submission, not acceptance. Minimize scope creep and protect the historical production archive.

## 5. Proposed order

**A.** Freeze review of Gate 0E-1 and baseline/code crosswalk → **B.** stability preflight and 2,400-run Gate 0E-2 → **C.** whole-seed results and time-path triage → **D.** targeted manuscript revision and SM → **E.** evaluate whether any distinct Gate 1 calibration is feasible before submission. Never merge Gate 0E diagnostics into historical IIIa/IIIb result tables without explicit design labels.

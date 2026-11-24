# V6 targeted robustness — frozen 2026-11-24

Purpose: assess whether the structural displacement–accuracy and gateway composition findings depend on independent external draws being treated as separate evidence, or the very steep attention allocation. Not a search for favorable results. Existing scientific model, historical runners and archive stay unchanged. Source baseline is the exact original ZIP 85faac503d4a; require SHA256, manifest, scientific and upstream fingerprints, identical software, and matched initial-state/structural fingerprints. See V6_SCIENTIFIC_PILOT_FINDINGS_2026-11-24.md.

## Frozen design
H high-S only, low/high H, and G low/high gateway, each null/fixed_biased x adaptive/frozen. 8 H + 8 G cells per seed per variant. Seeds 6001–6020, N100 T400 C20 K1 interval5, centroid/first_audit, source_posterior tau1. Two one-factor variants; no combined treatment.
- precision_cap5: epsilon .05; substantive external source variance = sigma_source^2/min(n_delivered,5); sample mean still uses ALL draws. Apply symmetrically to Expert and biased sender. Peer variance unchanged. Audits, credibility updating rule, message generation, requests and initial audit unchanged. This is a within-period receiver-side discount, not simulated correlated messages or a lifetime precision cap. Adaptive responses can change downstream via belief feedback. No strategic sender optimization examined here.
- epsilon20: epsilon .20, baseline unmodified evidence update; same recursive rank rule, 80% top choice, less steep lower-rank accessibility. Initial equal audit requests remain unchanged. Not an alternative functional-form test.

320 grid runs per variant = 640 additional runs; 80 representative uninstrumented parity runs = 720 actual runs. Existing baseline reused, no full re-run. Scope deliberately excludes low-S H; cannot assess full HxS interaction robustness in this batch. Both receivers retained to test behavioral dependence. These are paired sensitivity diagnostics, not independent confirmatory seeds.

## Implementation and validation
New standalone run_v6_robustness imports prior scientific runner and applies a process-local scoped replacement of config epsilon or Citizen substantive Gaussian update method. No permanent edits to model, legacy runner, or scientific fingerprint. Precision cap retains original arithmetic except external effective n. Logger records the actual capped added precision. Validate actual rank probability spectrum from edges, cap bound Q <= 5 * state-learning periods for unit-SD external sources, prior scientific validation and one representative parity comparison per seed/suite/variant. Reuse same random seeds/initial graph but histories may diverge after treatment.
Preflight: one seed 6001, T26, 16 cells each baseline/cap5/epsilon20 = 48 grid +6 parity. Exact t0 audit event signatures across all three variants (exclude pre-audit substantive probability, which epsilon changes). Replay two preselected original T400 representative H/G runs against archive terminal citizen beliefs, group Q/W/MSE and integer theta requests, tolerance 1e-12 for floats. Total preflight 56 runs. Robustness execution requires matching preflight PASS with same fingerprint/environment/archive.
No sign-based changes, no silent failed-cell omission. All runs retained. If technical checks fail, diagnose before interpreting or proceeding.

## Analysis
Within each seed/suite/prior/group/receiver: high-low MSE, direct X theta request counts/shares, Q/W/I, bias/variance; joint divergence sign indicators. Both sender regimes; fixed-minus-null damage at each structural level and high-low damage difference. Variant contrast minus matched baseline contrast; use 10000 paired whole-seed bootstrap draws RNG 20261124 and MCSE as in scientific analysis. Report all groups/receivers and directions. High-S aligned +1 adaptive is primary H; population D is primary G. No ratio of means as paired estimand. Inspect relative and absolute dependence separately. Shared output contains consolidated variant data and baseline contrasts, not all baseline raw records; preserve original ZIP and local shards.

## Local execution
From clean branch and paper-b environment:
```bash
python -m py_compile paper_b/experiments/run_v6_robustness.py paper_b/experiments/v6_precision_cap.py paper_b/experiments/v6_robustness_analysis.py
python -m unittest paper_b.test_v6_precision_cap paper_b.test_v6_scientific_measurement paper_b.test_v6_pilot_parity -v
python -m paper_b.experiments.run_v6_robustness --phase preflight --workers 4 --output-dir "$HOME/Research/paperb_v6_results"
python -m paper_b.experiments.run_v6_robustness --phase robustness --workers 8 --output-dir "$HOME/Research/paperb_v6_results"
```
Default archive location is output-dir/v6_scientific_85faac503d4a/paper_b_v6_scientific_85faac503d4a_shareable.zip; if moved, use --baseline-zip with exact original path in both commands. Optional --dry-run verifies archive and prints design without simulations. Resume identical phase with --resume; new code or software changes design ID. Use fewer workers if RAM pressure; workers not part of design. Share paper_b_v6_robustness_<id>_robustness_shareable.zip; retain raw local shards. Preflight gate must report 48 grid, 6 parity, archive_replay_pass=true, matched_t0_audit=true. Full gate 640 grid, 80 parity. Gate only validates implementation, not direction of results or production approval.

## Limits and next decision
Unit tests and actual preflight require local execution; no ABM execution claimed here. Cap is an explicitly selected single moderate setting, not evidence of empirical calibration. Cross-period correlation, alternative acquisition functional form, audit timing, off-center truth, strategic-sender variants and topology remain untested. Do not generalize this robustness to them. After inspecting this fixed batch, proceed to fixed-biased C bridge and decide final production scope; no automatic expansion.

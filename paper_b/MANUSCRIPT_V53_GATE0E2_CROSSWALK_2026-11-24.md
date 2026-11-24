# Paper B v5.3 working editorial crosswalk (2026-11-24)

**Status:** WORKING REVISION, not a frozen/submitted manuscript. Full LaTeX working source and an anonymous compiled PDF were produced from the user's exact `v5.2_AnonymousSource.zip` in the 2026-11-23 transition package; this document tracks the changes in GitHub. The original v5.2 and archived IIIa/IIIb are NOT altered.

## Scientific compatibility firewall

| Dimension | Archived canonical v5.2 Experiment 1–2 | Gate 0E-2 (new separate diagnostic) |
| --- | --- | --- |
| Primary design | Homophily × segregation; ordinary vs Expert-linked gateway opportunities | Elite-only E, muted-peer M, stale-peer S, updated-live-peer L |
| Seeds / cells | 500 seeds 6001–6500; 12,000 completed production runs | 100 new seeds 7501–7600; 2,400 completed runs |
| Initial belief design | Low/high segregation under the canonical structural designs | Flat priors only |
| Horizon | T=400 | T=400 |
| Frozen ranking | pre_disruption, before t0 credibility audit | first_audit, after common non-disruptive t0 audit |
| Jammer/second-source t0 | Archived legacy initialization and myopic t0 strategy | Common, non-disruptive centroid credibility audit for all sender regimes, first strategic refresh t1 |
| Attacker conditions | Null, fixed biased, bounded myopic adaptive; adaptive tested in selected cells | Explicit null, myopic adaptive, one-step credibility-aware with lambda=1 |
| Baseline caveat | Archived manuscript calls the second slot inert null; generic earlier runner defaults could call historical J0 truth_clone | Explicit null preserves source slot, but sends no substantive messages after t0; do not equate with truth_clone |
| Peer evidence | Source-posterior, tau_social=1 in the archived manuscript | source_posterior, tau_social=1, plus delivered-SD snapshot for stale |
| Key outcome | MSE total, matched null-relative excess damage D, A / Lambda / W diagnostics | Sender MSE, null MSE and D; M−E, S−M, L−S, L−E exact contrast accounting |
| Interpretation | Network-conditioned total structural replacements | Separate total peer-channel intervention effects; NOT independent mediation, and NOT upstream provenance |
| Code audit | Canonical production archive retained | design b7a571b4818a, scientific code fingerprint 754d7c373ff4b9f6eb24e1b0044917d559f69397531a053d5ab64d314cd67774 |

**No cross-design pooled MSE or causal comparison** is licensed by this crosswalk. Gate 1's prespecified 12,000-run highS/lowS design remains unexecuted and is not replaced by Gate 0E-2.

## Rechecked Gate 0E-2 results

Source: the actual user-uploaded `paper_b_pilot_b7a571b4818a_shareable.zip`, especially `pilot_gate.json`, `runs.csv`, `gate0e_channel_contrasts.csv`, and `belief_checkpoints.csv`. The uploaded package's validation PASS is 2,400/2,400 runs, 400/400 complete matched blocks, 0 reported errors. Actual event log limit was zero, so the new batch does **not** independently test t0 audit identity at individual-event level (`matched_t0_audits_equal=null`). Gate 0E-1 completed that diagnostic with the **same scientific fingerprint**.

At T400, mean L−E null-relative damage by sender/reliance:
- Myopic/adaptive +7.0697 (bootstrap percentile interval [5.753, 8.586]); L−S +4.6868.
- Myopic/frozen +5.7168 ([5.001, 6.461]); L−S +6.6718.
- Credibility-aware/adaptive +0.5953 ([0.399, 0.796]); L−S +0.4238.
- Credibility-aware/frozen −0.9565 ([−1.173, −0.740]); L−S +0.0727, interval [−0.045, 0.190].

These intervals are **seed-block Monte Carlo uncertainty**, not statistical inference about real-world populations. All four E/M/S/L contrasts are reported in the working Supplement. Myopic/adaptive M−E is +101.570, because M intentionally wastes peer request budget: it is NOT a natural disconnected-network counterfactual. Stale null MSE is nontrivial; interpretation MUST include total sender MSE and null MSE, not D alone.

In the same 100 seed trajectories, myopic L−E declines from 12.748 to 7.070 (adaptive) and 10.978 to 5.717 (frozen) between T200 and T400. Do not attribute the difference between 20-seed Gate 0E-1 and 100-seed Gate 0E-2 entirely to horizon; the within-batch trajectory is more informative.

## Edits in v5.3 working anonymous source

1. **Abstract & Introduction:** add a narrowly framed Gate 0E-2 result and distinguish it from the archived gateway/homophily production; do not claim definitive peer retransmission effects.
2. **Model §3:** explicitly distinguish pre_disruption versus first_audit frozen ranks; archived myopic t0 optimization versus corrected centroid audit with strategic refresh t1; state aware sender's joint information-set/objective changes.
3. **Design §4:** new 4-arm channel experiment subsection, exact telescoping definitions, all 24 conditions per matched seed; warn against mediation/provenance readings.
4. **Results §5:** one 8-row E/L main table with actual sender MSE, matched null MSE, D and peer-damage contrast; briefly report L−S and within-seed T200→T400 stability.
5. **Results §5.3 arithmetic:** total-MSE difference is 2.442−5.746=−3.304, *not* −3.297; the latter is null-adjusted excess damage (historically reported MCSE 0.0877).
6. **Discussion:** replace statements that an elite-only/peers diagnostic has *never* been run; retain the remaining unverified path-origin and forward-looking attacker limitations.
7. **Supplement:** 16-cell E/M/S/L table, four-arm contrast table, bootstrap uncertainty, T200→T400 matched trajectory and explicit event-log limitation.
8. **Unchanged:** Theory substantive conjectures, archived IIIa results/figures and IIIb inclusion rule, anonymous/main manuscript separation; author title page remains separate.

## Editorial decision

For now retain Gate 0E-2 as a **secondary, independently labeled mechanism diagnostic**: concise E/L results in the main text, 4-arm details in the Supplement. Do NOT relabel the canonical 12k figures as products of this experiment. New empirically grounded scientific hypotheses are not claimed retrospectively confirmatory. Prioritize a qualitative review of the different source-initialization regimes and the manuscript's breadth before declaring v5.3 submission-ready.

Review checkpoints: verify the original archived §5.3 MCSE from per-seed original data; verify no use of truth_clone for the stated canonical null baseline; decide whether the optional short t0 event audit is proportionate to the deadline; check anonymous submission package and references. The full working LaTeX source is in the current conversation's generated artifact, not yet copied into GitHub as a complete manuscript tree.

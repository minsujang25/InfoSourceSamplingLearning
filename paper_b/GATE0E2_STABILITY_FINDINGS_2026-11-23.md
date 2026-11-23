# Gate 0E-2 stability findings — 100 seeds, T=400

Date: 2026-11-23  
Status: **results audited from user-supplied shareable ZIP; mechanism diagnostic, NOT original Gate 1 confirmatory design**  
Source ZIP: `paper_b_pilot_b7a571b4818a_shareable.zip` (keep the original archive separately; not committed).  
Design ID: `b7a571b4818a`. Scientific code fingerprint: `754d7c373ff4b9f6eb24e1b0044917d559f69397531a053d5ab64d314cd67774`, **identical to Gate 0E-1**.  
This report does **not** update the frozen v5.2 manuscript, change historical IIIa/IIIb results, or amend the prospective Gate 0E design.

## 1. Execution and integrity

The uploaded `pilot_gate.json` reports `pass=true`, 0 errors, 400/400 matched channel blocks and 2,400/2,400 runs; all runs finite, exact T=400, frozen-ranking invariant and Jammer refresh/hold checks pass. Initial beliefs and M/S/L edge maps agree according to `pilot_summary.json`. The delivered package contains 2,400 `runs.csv` rows, 1,600 null-relative sender-damage rows and 400 E/M/S/L contrast rows (100 seeds × 2 receivers × 2 non-null senders). Seed range 7501–7600; flat priors; N=100, K=1, epsilon=.05, credit=20, initial centroid audit, first_audit receiver freeze, source_posterior, tau_social=1, lambda=1.

**Limitation:** `sender_event_period_limit=0`; no citizen event records, so `matched_t0_audits_equal=null` (not false). Matching of period-0 individual audit output and post-t0 muted message delivery cannot be rechecked from this shareable package alone. Gate 0E-1 had detailed events and passed these checks with the same scientific code fingerprint. For any additional audit, use a *separate* short event-logging validation, not a rerun or overwrite of the 2,400-run batch.

## 2. Primary T=400 results

`D=MSE_sender-MSE_null`, within same seed, network and receiver. E = elite only, M = extended muted, S = extended stale, L = extended live. Numbers are seed means, rounded to 3 decimals.

| Sender / receiver | D_E | D_M | D_S | D_L | M-E (slots) | S-M (static signal) | L-S (updating) | L-E (total peer) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Myopic / adaptive | 5.575 | 107.145 | 7.958 | 12.645 | +101.570 | -99.187 | +4.687 | +7.070 |
| Myopic / frozen | 9.111 | 10.117 | 8.156 | 14.828 | +1.006 | -1.961 | +6.672 | +5.717 |
| Aware / adaptive | 6.189 | 6.882 | 6.360 | 6.784 | +0.694 | -0.522 | +0.424 | +0.595 |
| Aware / frozen | 7.394 | 7.560 | 6.364 | 6.437 | +0.166 | -1.195 | +0.073 | -0.956 |

All contrasts telescope to L-E up to floating point precision; this is **accounting**, NOT independent causal mediation and NOT evidence of the originating route of an adversarial rumor.

10,000 whole-seed bootstrap draws (fixed local analysis RNG, seed 20261123) give:

| Sender / receiver | L-E [bootstrap 95%] | L-S [bootstrap 95%] |
|---|---|---|
| Myopic / adaptive | +7.070 [5.753, 8.586] | +4.687 [3.157, 6.246] |
| Myopic / frozen | +5.717 [5.001, 6.461] | +6.672 [6.015, 7.361] |
| Aware / adaptive | +0.595 [0.399, 0.796] | +0.424 [0.324, 0.522] |
| Aware / frozen | -0.956 [-1.173, -0.740] | +0.073 [-0.045, 0.190] |

Sign fractions: L-E positive 93%, 99%, 69%, 17% (same row order); L-S positive 83%, 100%, 84%, 57%. These describe Monte Carlo variation across model seed blocks, not external-world confidence intervals.

Cross-sender `Ω= (L-E)_aware-(L-E)_myopic`: adaptive receiver -6.474 [bootstrap -7.917, -5.201], frozen -6.673 [-7.285, -6.104].
Receiver interaction `Γ=(L-E)_adaptive-(L-E)_frozen`: myopic +1.353 [0.204, 2.639], aware +1.552 [1.324, 1.771]. The myopic Γ has marked seed heterogeneity (51% of seeds positive; median +0.101), so avoid describing it as universal.

## 3. Horizon stability — same 100 seeds, distinct from comparing two batches

The earlier 20-seed Gate 0E-1 reported T=200 L-E as +13.676, +10.319, +0.855 and -1.128, and L-S as +6.338, +11.397, +0.544 and +0.086 (same condition order). Signs of these *mean* contrasts persist at Gate 0E-2 T=400. The better time comparison uses *the same 100 Gate 0E-2 seeds* at available checkpoints:

| Sender / receiver | L-E at T=26 | T=100 | T=200 | T=400 | L-S at T=200 | T=400 |
|---|---:|---:|---:|---:|---:|---:|
| Myopic / adaptive | +49.562 | +21.952 | +12.748 | +7.070 | +7.422 | +4.687 |
| Myopic / frozen | +72.883 | +21.002 | +10.978 | +5.717 | +11.842 | +6.672 |
| Aware / adaptive | +0.075 | +0.534 | +0.584 | +0.595 | +0.465 | +0.424 |
| Aware / frozen | -1.418 | -1.002 | -0.964 | -0.956 | +0.101 | +0.073 |

Within these same 100 seed blocks, T400−T200 L-E changes: myopic/adaptive -5.678 [bootstrap -6.539, -4.867]; myopic/frozen -5.261 [-5.713, -4.817]. Both changes are negative in *all 100 seeds*. Aware/adaptive +0.011; aware/frozen +0.008, broadly stable. Early dynamics at steps 2/3 can have opposite signs or very large transient values; do not claim uniformly positive peer effects at all periods.

**Interpretation:** myopic sender damage is strongly transient and declines with continued learning, whereas credibility-aware sender contrasts are small but much more persistent. Persistent positive L-S for the myopic conditions is bounded by the specified model and does NOT itself identify disinformation passed along via peers.

## 4. Absolute-error and evidence warnings

Always show *total sender MSE* and *null MSE* alongside D. Stale null MSE is non-negligible at T400:
- Adaptive receivers: E .0001, M .0033, S .1253, L .0008.
- Frozen receivers: E .0001, M .0107, S .3208, L .0006.

Aware/frozen L-S in incremental D is +0.073, but **L-S in total sender MSE is -0.248**: sign differs because S's null MSE is much worse. Hence 'more resilient' must never be inferred from D alone. For myopic/adaptive M has D of 107.145: this is an extreme artificial *wasted attention* counterfactual, not what naturally follows from just removing peer ties.

`peer_reliance` and `peer_realized` are not confirmed actual post-t0 deliveries under M: these fields measure reliance/request allocations and show nonzero quantities under muted. `W_peer_precision_share` is 0 under muted, consistent with no delivered peer evidence in the precision update, but not an event-level delivery log. `W_peer_precision_share` is immediate incoming-source precision, not reconstructed upstream provenance or causal influence. With the sender regime's endogeneity, M/S/L need not have the same realized allocation paths despite identical source slots.

## 5. Recommended manuscript policy

**Do not reopen v5.2 or IIIa/IIIb automatically.** The stable E/M/S/L differences are a credible prospective **secondary mechanism diagnostic** after validating common-mode code, but they are not a clean peer-retransmission effect and are not a substitute for the separately frozen 12,000-run highS/lowS Gate 1 experiment.

Immediate manuscript QA: current v5.2's Abstract and Discussion explicitly say the elite-only versus peer-connected addition test has not been done. If Gate 0E-2 is promoted, update these statements and the applicable experiment-design sections *together*, while distinguishing archived models/parameters (and fixed biased production) from exploratory Gate 0E. Main-text possible: one bounded table (E versus L, total sender MSE and D) plus short note that M/S/L separates delivery and updating interventions. Full 4-arm matrices, bootstrap CIs, and the transient trajectory belong in the Supplement unless their role becomes central and consistent with IIIa.

Before revising substantive claims: verify the old-vs-new model specification crosswalk, clarify `null` versus `truth_clone` and `first_audit` versus `pre_disruption`, audit source-level within-period precision, and consider whether full Gate 1 is computationally feasible by the 2026-12-15 special issue deadline. Do not substitute Gate 0E-2's flat priors for highS/lowS. Do not optimize post hoc on positive peer contrasts.

Data parsing caveat: pandas defaults treat the literal `null` sender regime as missing; use `pd.read_csv(..., keep_default_na=False)` in reproducibility scripts.

## 6. Immediate next steps

1. Archive original result ZIP and freeze design ID, code fingerprint and full validation output.
2. Independently reproduce paired contrast tables and full-seed bootstrap from stored `gate0e_channel_contrasts.csv`; analyze checkpoint trajectories using `horizon_step`.
3. Optional short matched event-logging check solely for omitted t0 audit/delivery invariants; do not rerun 2400 merely to change tracing.
4. Map v5.2 §4/§5/§6 language to actual historical versus new run settings; decide if any 0E-2 result merits main text, supplementary, or reviewer-reserve placement.

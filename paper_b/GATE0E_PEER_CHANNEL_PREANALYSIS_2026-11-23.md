# Gate 0E — Peer-channel mechanism identification: prospective design freeze

Date: 2026-11-23
Parent: `paper-b-gate0d-mechanism-identification`
Status: **DESIGN PREPARED — NO NEW MECHANISM RESULTS OBSERVED OR ACCEPTED**
Paper B v5.2, Experiment IIIa production, and the separate IIIb diagnostic remain frozen.

## 1. Why this experiment is necessary

In the canonical `extended` network, citizens have two elite sources
(Expert and Jammer) **plus two randomly chosen citizen peers**.
In `elite_only`, they have only Expert and Jammer.
Both source choice and credibility auditing depend on the source set, so
`extended - elite_only` changes the number of alternatives, the allocation
of the fixed request budget, the audit sample composition, the ranking
process, and the informational content. It is not a pure peer-transmission
effect. In the existing `source_posterior` rule, repeated peer messages
within a period are counted as one source-level observation, so evidence
precision is already source specific. W_jammer and W_peer are *proximate
incoming precision*, not reconstructed origin-of-information paths.

Gate 0E introduces **source-preserving peer-message interventions** to
identify informative counterfactuals. These are interventions on the
information channel, not a path-specific causal mediation estimator.

## 2. Network treatments (same citizen population)

| Code | Structural peers? | Messages after t=0? | Interpretation |
|---|---|---|---|
| E = `elite_only` | No | No peers exist | Historical no-peer anchor |
| M = `extended_muted` | Same two peers as L/S | Peer requests occur but messages are discarded after generation | Peer *slots and attention dilution*, with no substantive peer deliveries after t=0 |
| S = `extended_stale` | Same two peers as L/M | Peer messages draw from each sender's fixed **pre-audit** initial belief mean and SD for all t>=1 | Static peer-content availability without feedback from peers' subsequent learning |
| L = `extended_live` | Same two peers as L/M/S | Current t-period sender belief mean and SD | Endogenous updated peer messages, original extended mode |

**Crucial start-up constraint:** all three extended arms (M,S,L) use
exactly the *same genuine initial peer messages* at t=0. They are changed
**only from t=1 onward**. Otherwise the initial credibility calibration
would be an unmeasured additional treatment. E does not have peer slots
and therefore cannot have the same t=0 ranking; E is an external anchor,
not a within-extended direct channel contrast.

- Jammer initial centroid audit remains fixed for every sender regime.
- All citizens keep the exact same structural source sets, edge directions,
  source count, credit=20, epsilon=.05, audit probability 1/(t+1),
  theta/delta updating, and sender access within M/S/L.
- For M, call the underlying peer sampling operation (to consume the same
  initial number of random draws as a delivered peer source) but discard
  its message payload for the recipient. The recipient still allocates
  requests to that source, and source-specific posterior/audit updates see
  no peer messages. This is **wasted attention**, not passive removal of
  the peer from the source set. Future ranks may adapt endogenously.
- For S, store the initial (pre-t=0) peer mean and SD. Its outgoing
  messages from t>=1 use those fixed parameters. Critically, the
  `source_posterior` likelihood must use this *delivered channel's
  SD*, not the evolving peer's internal posterior SD; otherwise
  inference is internally inconsistent. Implement via an explicit
  per-source outgoing observation-SD snapshot rather than changing
  sender's internal state.
- L remains bitwise equivalent to the parent extended implementation
  when event tracing is disabled. No peer strategy intervention
  in the baseline model.
- The source identities and degree for M/S/L must be compared
  by exact, per-ego edge-set fingerprints **within each seed**.

## 3. Scientific estimands — pair at the complete matched-seed block

Define MSE_T(s,n,r,a) for seed s, network treatment n in {E,M,S,L},
receiver reliance r in {adaptive,frozen}, and sender
a in {null,myopic_adaptive,credibility_aware}.

For each network/receiver/sender, define null-relative damage

    D_s(n,r,a) = MSE_T(s,n,r,a) - MSE_T(s,n,r,null).

Report MSE for the actual sender **and** the null condition separately.

Within each (s,r,a):

    C_slots  = D_s(M,r,a) - D_s(E,r,a)
    C_signal = D_s(S,r,a) - D_s(M,r,a)
    C_update = D_s(L,r,a) - D_s(S,r,a)
    C_total  = D_s(L,r,a) - D_s(E,r,a)
             = C_slots + C_signal + C_update

This is an **exact contrast-accounting identity**. It is **not**
a causal mediation/decomposition into mutually independent physical
pathways: each information intervention can change future credibility
rankings, request allocations, peer beliefs, and even future adaptive
attacker messages. C_slots includes wasted attention and other
consequences of adding silent peers. C_update is the effect of allowing
endogenous peer beliefs to determine outgoing message content, not
a clean measurement of a uniquely originating deceptive rumor.

The cross-sender difference in total peer effect is

    Ω_r = C_total(r,credibility_aware)
        - C_total(r,myopic_adaptive).

The adaptive-versus-frozen receiver interaction under each sender is

    Γ_a = [D_s(L,adaptive,a) - D_s(E,adaptive,a)]
        - [D_s(L,frozen,a) - D_s(E,frozen,a)].

Retain the original Gate 1 pre-frozen orientation when comparing these
with the earlier proposed B and Γ contrasts. Positive C_total means
peer addition raises **incremental** attack damage, not necessarily
total MSE or compromised accuracy.

## 4. Information-set and temporal safeguards

- Freeze the sender specification (including the myopic adaptive
  baseline and lambda=1 credibility-aware option) before analysis.
  The original fixed-biased sender remains a separately reportable
  comparator but does not need to be present in all new channel
  intervention cells.
- The existing `credibility_aware` strategy observes current
  audience credibility and ranking; this is a joint information-set
  and objective change vs the historical myopic sender.
- `source_posterior` and tau_social=1 remain fixed. Retaining
  the same expectation formulas across regimes matters more than
  selecting a favorable posterior interpretation after results.
- Intervention starts at t=1, AFTER the corrected t=0 centroid audit
  with common initial sender/peer messages in M/S/L.
- The stale peer signals are independent draw realizations from the
  sender's initial signal parameters, not replayed past messages.
  They are *not* assumed truthful or representative of the state.
- Frozen receiver's primary rule: `first_audit`. Historical
  `pre_disruption` only as a separate sensitivity with its own
  design ID. Never pool these as one ranking design.
- M and S can change the actual allocation and source credibility
  over time: *do not claim* all three extended treatments have
  identical realized request shares after t=1. Their **choice sets
  and budgets** stay matched, not their endogenous histories.

## 5. Required passive, event-level diagnostics

At each checkpoint t in {0,1,2,5,10,25,50,100,200,399} (if in horizon),
log per-period AND/or per-citizen:

- learning target (audit vs theta), source count, initial and current
  Jammer/peer ranks, ranked substantive request probabilities;
- requested credits for Expert/Jammer/peers separately and actual
  *delivered* messages for each channel (zero after t=0 in M);
- number of citizens who requested each channel, and who received
  >=1 message from it;
- pre/post mu_theta, sd_theta, mu_delta[Jammer], sd_delta[Jammer],
  and assigned evidence precision by *immediate* source class;
- sender first-attack message mean, refresh times and K=1 cohort
  centroid, and finite-state counts;
- within-extended edge-map, initial belief, and t=0 audit-output
  fingerprints, to confirm identical starting conditions.

A passive recording switch must have exact **on/off parity** in MSE,
terminal beliefs, message means, and reliance when using the same
seed/configuration. The event collector must read incoming messages
before `commit_pending_update()` clears the buffers, as demonstrated
in Gate 0D. No post-hoc reconstruction of discarded messages.

## 6. Staged implementation and promotion criteria

Gate 0E-0 (unit and smoke):
  - N=24, T=6, 2 matched seeds (6901–6902); all four channel modes,
    three senders (null,myopic,aware), and two receiver modes:
    2×4×3×2 = **48 small model runs**.
  - Verify absolute t=0 equivalence in M/S/L, period-1 onset,
    exact source maps, no t>=1 peer deliveries under M,
    exactly fixed initial outgoing peer mean/SD in S,
    unchanged live results vs parent, no missing cells,
    deterministic read-only instrumentation.
  - Run lambda=1 only; no outcome-based parameter tuning.

Gate 0E-1 (exploratory mechanism):
  - N=100, T=200; 20 NEW matched seeds 6921–6940,
    flat prior; 4×3×2×20 = **480 runs**.
  - Interpret as diagnostic, not confirmatory. Publish/show all
    contrasts regardless of sign or magnitude.

Gate 0E-2 (stability pilot; subject to computational feasibility):
  - N=100, T=400; same 4 network treatments and 3 senders,
    2 receiver regimes and 100 predeclared seeds 7501–7600:
    **2,400 runs**.
  - This is distinct from historical IIIa/IIIb and the original
    Gate 1 confirmatory experiment. Promotion is based on validation
    success and stability of error estimates, NOT choosing the most
    positive peer-network contrast. The full-channel diagnostic remains
    a supplement candidate unless it materially clarifies the main
    theoretical claim without conflicting with frozen IIIa/IIIb.

## 7. Gate 1 original confirmatory design — separate and preserved

Original preanalysis ambition (before Gate 0E channel interventions):

  500 matched seeds 7001–7500
  × 2 networks (elite_only, extended_live)
  × 3 senders (null, fixed_biased, myopic_adaptive)
  × 2 receivers (adaptive, frozen)
  × 2 initial segregation settings (highS, lowS)
  = **12,000 T=400 production runs**, N=100.

Do not silently substitute the exploratory Gate 0D `flat`,
`consensus`, or `polarized` categories for highS/lowS. The
current `run_matched_pilot.py` understands those three local
regimes but does **not** directly implement the original highS/lowS
initial-belief design from `structural_designs.py`; implementing
this crosswalk is a distinct acceptance gate.

For the original confirmatory design, primary outcomes are T=400
MSE, null-relative damage D, and two paired contrasts: peer
addition in adaptive myopic D, and the peer×receiver interaction.
The optional credibility-aware sender is a **prospective secondary
extension**, not a silent replacement for the originally specified
myopic treatment. Keep the myopic arm even if its initial attack
is extreme. Distinguish `source_posterior` tau=1,
`centroid` initial audit and first_audit freezing from archived
legacy production settings.

The 12,000-run Gate 1 stage should start only after a smaller
matched T=400 calibration sample passes all graph, event, finite
MSE, time horizon, initial-ranking, and consistency checks.

## 8. Statistical and manuscript reporting

- Seed is the unit of randomization/inference; report paired
  seed differences, average contrast, sd(diff)/sqrt(n), and
  bootstrap percentile 95% CIs resampling **whole matched seed
  blocks**, retaining all treatment cells per seed.
- Preserve the two original primary contrast definitions and
  report both without selecting the one with a desirable sign.
  Declare all other network-channel and sender-mechanism contrasts
  exploratory and present their complete matrix.
- Plot total sender MSE, null MSE, D, and t=1/T=200/T=400 trajectories.
  A lower D with a larger null loss is not automatically greater
  resilience.
- Distinguish incoming-source precision from upstream
  disinformation provenance. Do not infer precise rumor routes
  from W_peer or peer inclusion alone.
- If channels do not produce distinct findings or mechanisms remain
  inseparable, document the null and retain Gate 0E in Supplement
  or as reviewer-response material, without reopening the v5.2
  main Theory/Abstract.
- Experiment IIIb's independent-versus-shared corrective-path
  intervention and frozen inclusion rule remain **separate** from
  Gate 0E; neither supersedes the other.

## 9. Implementation order

1. Add `peer_signal_mode` only on a new child branch, default
   `live`, and make its effect conditional on t>=1 and on
   citizen-to-citizen (not Expert/Jammer) messages.
2. Add explicit initial outgoing mean/SD snapshot for `stale`
   and a corresponding delivered-channel SD in
   `bayesian_update_theta_from_sources`; preserve live path
   byte-for-byte when the option is absent.
3. Implement M drop-after-draw without rerouting or reallocating
   credits automatically. Do not change the audit sampling rule.
4. Add `network_channel_mode` to a dedicated matched runner
   or validated wrapper, rather than mutating historical
   canonical production.
5. Add per-source request and delivery event checks, including
   graph/t=0 parity and optional-trace invariance.
6. Freeze the machine-readable matrix and code SHA before
   T=200/T=400 batches; run smoke before expansion.
7. Keep separate output folders and deterministic design hashes
   for all channel, receiver, prior and frozen-rank variants.

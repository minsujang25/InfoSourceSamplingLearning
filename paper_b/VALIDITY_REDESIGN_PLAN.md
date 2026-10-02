# Paper B Validity Redesign — Frozen Pre-Run Plan

## Status

This branch is intentionally separate from `paper-b-theory-reconstruction`.
All previously reported Experiment III/IIIb/IV outputs remain archived results
from the pre-validity-revision specification. They are not overwritten.

The purpose of this branch is to resolve four implementation questions before
any new 500-seed production run:

1. repeated peer statements and posterior precision;
2. the meaning of the J=0 counterfactual;
3. the timing of the frozen-reliance counterfactual;
4. sender-side boundary conditions.

No manuscript claim is conditioned on the direction of the redesigned results.

## V2 receiver-side specification

### 1. Source-level peer evidence

Legacy state updating flattens all messages in a period and estimates the
variance of the pooled sample mean. Under concentrated acquisition this can make
repeated statements from one peer behave like many independent observations of
truth.

The redesign adds an explicit `peer_evidence_mode`.

- `legacy_batch`: exact backwards-compatible implementation.
- `source_posterior`: source-level likelihood aggregation.

Under `source_posterior`, source (s)'s period-t signal is summarized by its
sample mean. For a citizen peer, the likelihood variance is the sender's
pre-update posterior variance and is **not divided by the number of repeated
draws from that peer**. Repetition reveals the peer's current stated location;
it does not create independent epistemic evidence beyond the peer's posterior.

For an elite source whose messages are generated as conditionally independent
unit-variance signals, repeated draws retain variance (1/n_s).

The receiver combines the prior and all source-level likelihoods by Gaussian
precision weighting.

This is the primary validity redesign. Process noise or forgetting is not
introduced unless this source-level correction still produces pathological
early certainty.

### 2. Jammer regimes

The old boolean `jammer_active` remains available for exact legacy
reproduction, but redesigned experiments use an explicit `jammer_regime`:

- `adaptive`: current posterior-uncertainty, myopic audience-adaptive Jammer;
- `fixed_biased`: fixed source centered on the Jammer's underlying position;
- `truth_clone`: legacy J=0, a second truthful unit-variance source;
- `null`: structural slot and RNG consumption retained, but its messages do
  not enter belief or credibility updating and it is behaviorally ranked last.

Primary redesigned adversarial perturbation contrasts use `adaptive - null`.
`truth_clone` is retained as a distinct channel-compromise counterfactual,
not called "no adversary."

The fixed-biased sender is a sender-side boundary condition, not a replacement
for the adaptive Jammer and not a target-tuned rescue specification.

### 3. Frozen reliance

The legacy `first_audit` frozen rule remains reproducible.

The redesigned primary frozen rule is `pre_disruption`: before period 0, all
citizens receive the same deterministic source-ranking initialization based on
the baseline source states:

- Expert: truth-centered location;
- citizen peer: peer's initial posterior mean;
- Jammer: underlying sender position, before strategic optimization.

No substantive belief is updated and the main simulation RNG stream is not
consumed by this initialization. Adaptive and frozen conditions therefore begin
from the same pre-disruption behavioral ranking. Frozen keeps that ranking;
adaptive may subsequently revise it at scheduled audits.

### 4. Audit schedule

The current audit schedule (p_t=1/(t+1)) remains unchanged in the first
redesign diagnostic so that the effect of the three changes above can be
isolated. It must be reported explicitly in the manuscript.

The validation output records the realized number and timing of audits per
citizen. A different audit schedule is considered only if the redesigned model
still exhibits a validity problem.

## Primary diagnostic, not production

Run 24 matched seeds before any 500-seed rerun.

### Experiment IV validity block

Factors:

- H in {low, high};
- S in {low, high};
- reliance in {adaptive, frozen};
- sender in {null, adaptive, fixed_biased}.

The truth-clone condition is retained in a smaller compatibility block rather
than crossed into the full primary diagnostic.

Primary questions:

1. Does posterior certainty still collapse before the substantive dynamics?
2. Does adaptive reliance still reduce dominant-path Expert reachability and
   increase citizen-only closed reliance?
3. Does the H x S contrast differ materially between adaptive and frozen
   reliance?
4. Does sender regime change the sign or magnitude of the H x S effect?
5. Are baseline performance and adversarial displacement distinct?

### Experiment III validity block

Factors:

- redundancy in {low, high};
- reliance in {adaptive, frozen};
- sender in {null, adaptive, fixed_biased}.

Primary questions:

1. Does nominal structural redundancy increase dominant-path Expert
   reachability?
2. Does adaptive reliance convert that structural capacity into distributed
   effective dependence or closed citizen reliance?
3. Is any apparent adversarial benefit separable from baseline learning?
4. Is effective reliance more concentrated on gateways in the nominally
   high-redundancy structure?

## Required diagnostics

### Belief/precision dynamics

At fixed checkpoints including 0, 1, 2, 5, 10, 25, 50, 100, 150, 199:

- mean and median posterior SD;
- SD-floor share;
- MSE, RMSE, MAE;
- squared population displacement;
- belief variance.

### Matched movement

For matched sender conditions at the citizen level:

- signed movement: (mu_i^{sender}-mu_i^{null});
- absolute movement;
- change in absolute truth error;
- change in squared truth error.

### Effective-network diagnostics

From the top-ranked/dominant-reliance graph:

- Expert-reaching share;
- Jammer-reaching share;
- citizen-only-cycle share;
- cycle-size distribution;
- same-group closed-cycle share;
- checkpoint persistence of terminal cycles.

From weighted Lambda:

- source incoming reliance;
- effective-reliance HHI/concentration;
- gateway incoming-reliance share;
- structural vs effective homophily;
- peer reliance.

Top-1 graphs are called **dominant-reliance skeletons**, not causal influence
graphs.

### Sender diagnostics

- Jammer message trajectory;
- posterior response gain kappa;
- number/timing of credibility audits;
- direct Jammer reliance/message counts by group.

## Decision gates

A redesigned 500-seed production run is permitted only if:

1. all numerical/correctness gates pass;
2. source-level peer evidence prevents repeated peer statements from creating
   artificial near-zero observation variance;
3. the scientific meaning of null, truth-clone, fixed-biased, and adaptive
   sender regimes is verified by unit/smoke tests;
4. adaptive and frozen conditions start from identical pre-disruption rankings
   under the redesigned frozen rule;
5. posterior-certainty trajectories and MSE dynamics are inspected before
   choosing the final production specification;
6. no result-dependent parameter tuning is introduced.

## Manuscript status during redesign

The current LaTeX v2 is a **pre-validity-revision manuscript**. It is not
updated numerically until this diagnostic closes.

The intended theoretical spine remains:

[
A \rightarrow R_t \rightarrow \Pi_t \rightarrow \Lambda_t
\rightarrow X_t \rightarrow \text{beliefs}.
]

Final endpoints are separated into:

- baseline epistemic performance;
- adversarial perturbation / matched belief displacement.

The central network claim to test is not "redundancy is good" or "homophily is
bad." It is:

> Structural opportunities do not mechanically determine effective
> informational dependence.

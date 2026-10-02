# Paper B Canonical Production Plan — Frozen Specification

## Status

The validity-redesign and final-calibration stages are closed. This document
freezes the canonical production specification for the Social Networks Paper B.

No parameter tuning, model redesign, horizon search, or sender-objective
modification is permitted after this point without creating a separately named
post-production robustness analysis.

## Frozen model specification

```text
N = 100
T = 400
epsilon = .05
credit = 20
K = 1
surveillance interval = 5
MIN_SD = 1e-8

peer_evidence_mode = source_posterior
tau_social = 1
frozen_ranking_mode = pre_disruption
```

The final-calibration decision fixed T=400 because the precommitted T=200
stability gate failed in selected primary cells while the corrected model
remained numerically stable through T=400.

## Production seeds

Use exactly 500 new matched seeds:

```text
6001-6500
```

The same seed set is used in Experiments III and IV.

These seeds are distinct from the validity and calibration seed sets.

## Sender hierarchy

The paper distinguishes three sender environments.

### 1. Null sender

Primary baseline. The Jammer slot remains structurally available and consumes
the matched RNG stream, but contributes no substantive evidence and is
behaviorally ranked last.

### 2. Fixed biased sender

Primary persistent-information stress condition. The source remains centered on
its underlying biased position. This condition identifies resilience to a
credible/persistent biased source without conflating the result with a dynamic
sender objective.

### 3. Myopic adaptive Jammer

Secondary sender robustness/application. The current posterior-uncertainty
Jammer remains unchanged and is evaluated only under adaptive reliance in
selected cells. It is not crossed with frozen reliance in production because
the frozen x adaptive-Jammer combination represents an immutable-trust
bait-and-switch stress case rather than a clean adaptation counterfactual.

The fixed-biased sender is therefore the main adversarial/persistent-source
stress environment. The adaptive Jammer is retained as a secondary sender-side
boundary condition.

## Experiment IV — homophily x prior segregation

### IV-A. Primary baseline/network-mechanism block

Cross:

- H in {low, high}
- S in {low, high}
- reliance in {adaptive, frozen}
- sender = null

Per seed: 8 runs.

Primary outcomes:

- MSE, RMSE, MAE;
- squared population displacement;
- belief variance;
- effective homophily;
- peer reliance;
- dominant Expert reachability;
- citizen-only cycle share;
- same-group closed-cycle share;
- incoming-reliance HHI/top-five share.

Primary estimands:

[
I^A_{HS},qquad I^F_{HS},qquad I^A_{HS}-I^F_{HS}.
]

The same high-S high-H minus low-H contrast is estimated for effective
homophily and same-group closed dependence.

### IV-B. Persistent biased-source stress block

Cross:

- H in {low, high}
- S = high
- reliance in {adaptive, frozen}
- sender = fixed_biased

Per seed: 4 runs.

The corresponding high-S null cells from IV-A provide the paired baseline.

Primary stress estimands:

- sender damage within each H x reliance cell;
- high-H minus low-H difference in sender damage;
- adaptive-minus-frozen difference in the homophily effect on sender damage.

### IV-C. Secondary myopic adaptive-Jammer block

Cross:

- H in {low, high}
- S = high
- reliance = adaptive
- sender = adaptive

Per seed: 2 runs.

The corresponding adaptive high-S null cells provide the paired baseline.

This block is secondary and does not determine the central network theory.

### Experiment IV total

[
8+4+2=14
]

runs per seed, or

[
14	imes500=7000
]

runs.

## Experiment III — corrective redundancy

### III-A. Primary baseline and persistent-stress block

Cross:

- redundancy in {low, high}
- reliance in {adaptive, frozen}
- sender in {null, fixed_biased}

Per seed: 8 runs.

Primary estimands:

1. high-minus-low redundancy contrast in null-sender MSE;
2. high-minus-low redundancy contrast in gateway incoming reliance;
3. high-minus-low redundancy contrast in incoming-reliance concentration;
4. fixed-biased sender damage within each redundancy x reliance cell;
5. high-minus-low difference in fixed-biased sender damage;
6. adaptive-minus-frozen difference in each redundancy contrast.

### III-B. Secondary myopic adaptive-Jammer block

Cross:

- redundancy in {low, high}
- reliance = adaptive
- sender = adaptive

Per seed: 2 runs.

The corresponding adaptive null cells provide the paired baseline.

### Experiment III total

[
8+2=10
]

runs per seed, or

[
10	imes500=5000
]

runs.

## Total production workload

[
(14+10)	imes500
=
12000
]

T=400 simulations.

The workload should be executed with resumable per-seed/per-experiment shards.

## Canonical checkpoints

Store at minimum:

```text
0, 1, 2, 5, 10, 25, 50, 100, 150, 199, 299, 399
```

Period 399 is the canonical terminal checkpoint. Period 199 is retained only
for horizon-sensitivity reporting and continuity with the calibration stage.

## Main inferential unit

The matched simulation seed is the inferential unit.

For every primary contrast report:

- matched-seed mean;
- Monte Carlo standard error;
- approximate 95% Monte Carlo interval;
- median;
- 10% trimmed mean;
- sign share;
- leave-one-out mean range.

No iid run-level standard error is used for factorial contrasts.

## Outcome hierarchy

### Primary loss outcome

[
MSE_T=rac1Nsum_i(mu_{i,T}-	heta^*)^2.
]

### Supporting loss outcomes

- RMSE;
- MAE.

### MSE decomposition

[
MSE=(armu-	heta^*)^2+operatorname{Var}(mu).
]

Report squared displacement and belief variance separately.

### Effective-network outcomes

Treat these as descriptive network configurations, not causal mediation
coefficients:

- effective homophily;
- peer reliance;
- dominant Expert/Jammer reachability;
- citizen-only cycles;
- same-group closed cycles;
- incoming-reliance HHI;
- top-five incoming-reliance share;
- gateway incoming-reliance share in Experiment III.

The dominant-reliance skeleton is not called a causal influence graph.

## Canonical interpretation hierarchy

### Experiment IV

The primary question is baseline collective learning under the interaction of
structural homophily, segregated priors, and reliance flexibility.

The calibrated expectation is that frozen reliance can produce more rigid
same-group closure and substantially larger H x S learning failure than
adaptive reliance. Production results may differ; no sign is required.

### Experiment III

The primary question is whether structural corrective redundancy changes both
baseline learning and resilience to a persistent biased source, and how that
change is expressed through effective reliance on corrective gateways.

The redundancy effect is not assumed to be adaptive-specific.

### Sender-side interpretation

The fixed-biased source is the main persistent-stress environment.

The myopic adaptive Jammer is a secondary boundary condition/application.
Production does not reinterpret it as a forward-looking strategic agent.

## Production safety gates

The production aggregation step must fail if any of the following occur:

1. fewer or more than 500 unique seeds are observed;
2. the observed seed set differs from 6001-6500;
3. any experiment/seed block is incomplete;
4. the expected run count differs from 12,000;
5. any run fails to reach T=400;
6. any terminal MSE or terminal belief is non-finite;
7. terminal MSE >= 1,000,000;
8. absolute terminal belief >= 10,000;
9. a frozen x adaptive-Jammer cell appears;
10. tau_social differs from 1;
11. peer_evidence_mode differs from source_posterior;
12. frozen_ranking_mode differs from pre_disruption.

Posterior floor incidence is reported but is not expected to determine the
production model after the completed validity stage unless a new numerical
failure appears.

## Output architecture

Keep raw resumable shards locally.

Create a compact review bundle containing:

- production manifest;
- production gate;
- run-level terminal metrics;
- cell summaries;
- matched seed-level contrasts;
- contrast summaries;
- adaptive-minus-frozen contrast table;
- sender-vs-null damage table;
- group-level sender-damage summaries for Experiment IV;
- period-199/399 horizon summaries;
- effective-network checkpoint summaries;
- provenance/design document.

The review bundle should be a single ZIP suitable for upload back into ChatGPT.

A larger full-output bundle may remain local and is not required for review.

## Production freeze rule

Once this plan is committed, production code may be corrected only for
implementation bugs that prevent the frozen design from being executed as
written. Any substantive specification change requires a new documented
robustness analysis and does not overwrite the canonical production run.

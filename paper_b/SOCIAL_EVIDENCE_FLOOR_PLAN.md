# Paper B Social-Evidence Floor Diagnostic — Frozen Pre-Run Plan

## Purpose

The first validity-redesign diagnostic showed that replacing pooled peer-message
variance with source-level peer posterior variance removes within-period
pseudo-replication but does not prevent rapid recursive social overconfidence.
Posterior SD still approached the numerical floor by approximately periods
25--50 in many redesigned conditions.

This diagnostic changes exactly one receiver-side assumption:

[
V_{j	o i,t}=sigma_{j,t}^2+	au_{social}^2.
]

The additive term represents communication/social-evidence uncertainty: even a
highly confident peer's stated opinion is not treated as a perfectly precise,
independent observation of the state.

No sender objective, audit schedule, network topology, epsilon, credit budget,
or initial-belief design is changed in this round.

## Backward compatibility

The model default is `tau_social=0`, which reproduces the current
`source_posterior` likelihood:

[
V_{j	o i,t}=sigma_{j,t}^2.
]

The diagnostic compares `tau_social in {0,1}`. Elite-source likelihoods are
unchanged and continue to use independent unit-variance message logic.

## Frozen diagnostic sample

Use exactly eight matched seeds:

```text
5101-5108
```

with:

```text
N = 100
T = 200
epsilon = .05
credit = 20
K = 1
surveillance interval = 5
MIN_SD = 1e-8
peer_evidence_mode = source_posterior
frozen_ranking_mode = pre_disruption
reliance_mode = adaptive
```

Frozen reliance is not crossed in this round. The purpose is to isolate the
social-evidence floor while avoiding the special immutable-trust/adaptive-
Jammer bait-and-switch counterfactual found in the previous diagnostic.

## Selected Experiment IV cells

### Baseline-network block

Cross:

- H in {low, high}
- S in {low, high}
- sender = null
- tau_social in {0,1}

This block asks whether the corrected social-evidence model still produces:

- different same-group closed dependence under homophily + segregation;
- stable baseline epistemic performance;
- delayed/non-pathological posterior certainty.

### Sender-stress block

Cross:

- H in {low, high}
- S = high
- sender in {adaptive, fixed_biased}
- tau_social in {0,1}

This block checks whether sender credibility boundary conditions alter the
network pattern once recursive overconfidence is controlled.

## Selected Experiment III cells

Cross:

- redundancy in {low, high}
- sender in {null, adaptive, fixed_biased}
- tau_social in {0,1}

This block asks whether nominal structural redundancy still changes gateway
concentration and dominant Expert reachability when peer evidence has a
positive social uncertainty floor.

## Run count

Per seed:

```text
Exp IV baseline-network: 2 H x 2 S x 1 sender x 2 tau = 8
Exp IV sender-stress:    2 H x 1 S x 2 sender x 2 tau = 8
Exp III:                 2 R x 3 sender x 2 tau       = 12
----------------------------------------------------------
Total per seed                                      = 28
```

Eight seeds therefore produce:

[
8	imes28=224
]

runs.

## Required checkpoints

At minimum:

```text
0, 1, 2, 5, 10, 25, 50, 100, 150, 199
```

Record:

- posterior SD mean and median;
- numerical-floor share;
- MSE, RMSE, MAE;
- squared population displacement;
- belief variance;
- dominant Expert reachability;
- dominant Jammer reachability;
- citizen-only cycle share;
- same-group cycle share;
- cycle-size diagnostics;
- effective incoming-reliance HHI;
- top-five incoming-reliance share;
- gateway incoming-reliance share in Experiment III;
- effective homophily and peer-reliance mass.

## Matched tau contrasts

For every otherwise identical cell compute:

[
Y(	au=1)-Y(	au=0)
]

for posterior, loss, and effective-network metrics.

Sender-vs-null contrasts are also retained separately.

## Pre-run decision rule

`tau_social=1` is eligible for consideration as the final receiver-side
specification only if all of the following hold.

### A. Early numerical-floor collapse is materially reduced

Across the tau=1 **null-sender baseline-network cells**, evaluated across the
eight matched seeds:

- median posterior-SD floor share at t=25 must be < .05;
- median posterior-SD floor share at t=50 must be < .25.

Failure of either threshold means tau=1 is not sufficient to solve recursive
social overconfidence.

### B. Numerical behavior remains stable

- every run is finite;
- fixed horizon T=200 is reached;
- no run has terminal MSE >= 1,000,000;
- no run has absolute terminal belief >= 10,000.

These are numerical safety gates, not substantive success criteria.

### C. Network claims are re-evaluated rather than preserved by assumption

Two previously observed directions are treated as diagnostic targets:

1. in Exp IV high-S/null, high H should be compared with low H on same-group
   closed dependence;
2. in Exp III/null, high redundancy should be compared with low redundancy on
   gateway incoming-reliance concentration.

If either direction disappears or reverses under tau=1, the corresponding
network claim is not carried into production. This does **not** invalidate the
receiver specification by itself; it means the earlier network pattern was
sensitive to recursive certainty.

### D. No result-dependent tuning

No value other than tau=0 and tau=1 is run before this diagnostic is classified.
A different tau value requires a new documented diagnostic rather than
post-result search.

## Production gate

No 500-seed production run and no manuscript numerical update occurs until this
224-run diagnostic is inspected and classified.

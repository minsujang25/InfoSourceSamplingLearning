# Paper B Measurement Audit Plan

## Status and scope

This audit occurs **after** the frozen canonical production
`c9d09daad143`. It does not redesign the behavioral model.

The purpose is to verify that the manuscript's network measurements support the
claimed distinction between structural opportunity and effective informational
dependence.

The audit has two phases:

1. **Phase A — no-rerun structural benchmarking and outcome re-expression**
2. **Phase B — passive precision-network logging on an exact canonical rerun**

No manuscript figures are rebuilt until both phases are classified.

---

# Phase A — Structural baselines and canonical-result re-expression

Phase A uses the existing canonical production outputs. No simulations are run.

## A1. Uniform-use structural baseline

For every seed and structural condition, construct the network obtained if each
citizen used every structurally available source with equal weight:

[
U^A_{ij}=
rac{A_{ij}}{sum_k A_{ik}}.
]

This is a measurement benchmark, not a behavioral counterfactual.

For every weight-based Lambda statistic with a unique uniform-use analogue,
report:

[
Delta^{Lambda-A}=M(Lambda)-M(U^A).
]

Required structural benchmarks:

- structural peer homophily (H^A);
- uniform-use total incoming HHI;
- uniform-use top-five incoming share;
- Experiment III uniform-use gateway incoming share using **all source mass**;
- Experiment III peer-conditional gateway share using peer edges only;
- effective-minus-structural homophily:
  [
  H^Lambda-H^A.
  ]

The denominator must be explicit for every gateway statistic.

### Dominant-skeleton exception

A top-1 dominant-reliance skeleton does not have a unique deterministic
uniform-use counterpart because all available ties are tied under (U^A).
Therefore top-1 cycle/reachability statistics are **not** assigned an arbitrary
structural-excess score.

Instead, they are reported as rank-induced configurations and interpreted
alongside structural opportunity measures. Any randomized-tie or Markov
baseline, if later desired, must be labeled as a separate robustness analysis.

## A2. Null-slot ranking audit

The code invariant is:

[
	exttt{null source is always moved to the final behavioral rank}.
]

The audit must verify at runtime that the null Jammer slot is last-ranked for
every citizen/checkpoint in null-sender conditions.

The null slot remains structurally present. On state-learning rounds it can
receive the epsilon-tail request probability and yield no evidence, so the null
estimand is an **inert available slot with possible attention wastage**, not a
removed source with reallocated attention.

## A3. Total-loss 2 x 2 presentation

For Experiments III and IV, report total loss rather than only incremental
sender damage.

Required cells:

[
{	ext{null},	ext{fixed biased}}
	imes
{	ext{adaptive},	ext{frozen}}.
]

Report at least:

- MSE;
- RMSE;
- MAE;
- fixed-biased minus null damage;
- adaptive minus frozen total-loss difference within each information
  environment.

This determines whether reconfigurability is beneficial when the dominant
threat is endogenous closure versus persistent external bias.

## A4. Effect-size scaling

Report:

- terminal cell-level MSE and RMSE;
- expected initial MSE benchmark;
- terminal MSE as a fraction of initial MSE;
- Experiment IV group-mean separation where available;
- group-specific fixed-biased damage.

The purpose is to distinguish Monte Carlo precision from substantive magnitude.

## A5. Horizon trajectories

Using existing checkpoint outputs, report:

[
Tin{100,200,300,400}
]

for key null conditions, especially Experiment IV high-H/high-S adaptive and
frozen cells.

This audit distinguishes:

- persistent failure;
- slow corrective learning;
- transient top-1 closure.

No new horizon is searched.

---

# Phase B — Precision-contribution network W

## B1. Motivation

Under `source_posterior`, acquisition probability and Bayesian precision
contribution are not the same object.

For a sampled peer (j):

[
q_{ij,t}
=
rac{1}{sigma_{j,t}^2+	au_{social}^2},
]

regardless of how many times the same peer is sampled in that state-learning
period.

For an elite or biased information source (s) with (n_{is,t}) independent
messages:

[
q_{is,t}
=
rac{n_{is,t}}{sigma_s^2}.
]

If a source is not sampled, or if the period is a credibility-audit period,
its state-learning precision contribution is zero.

This asymmetry is intentional:

- elite/source messages are modeled as fresh independent observations;
- repeated peer statements are modeled as repeated expressions of one current
  posterior opinion.

The manuscript must state this explicitly.

## B2. Cumulative evidence-precision network

For each citizen-source pair define cumulative added evidence precision:

[
Q_{ij,T}
=
sum_{t=0}^{T-1} q_{ij,t}.
]

Normalize within ego:

[
W_{ij,T}
=
rac{Q_{ij,T}}
{sum_k Q_{ik,T}}.
]

(W_T) is called the **evidence-precision network**.

It measures each source's share of the source precision actually added to the
citizen's Gaussian state updates over the observed horizon.

It is **not** called a causal influence network.

## B3. Required W metrics

At minimum calculate at checkpoints:

[
Tin{25,50,100,200,300,400}
]

and terminal (T=400):

- Expert precision share;
- Jammer/fixed-source precision share;
- peer precision share;
- evidence-network peer homophily (H^W);
- peer precision mass;
- incoming evidence-precision HHI;
- top-five incoming precision share;
- Experiment III gateway incoming precision share;
- dominant evidence-source Expert/Jammer/citizen-cycle reachability;
- dominant same-group evidence-cycle share.

For the central manuscript claims compare:

[
A,qquad Lambda,qquad W.
]

## B4. Passive-logging requirement

The precision logger must:

- make no RNG calls;
- make no changes to sampling, ranking, message generation, posterior
  arithmetic, or sender optimization;
- only record quantities already computed by the frozen canonical update.

The rerun uses exactly:

[
6001,ldots,6500
]

and the identical 12,000-condition canonical grid.

## B5. Behavioral identity gate

The passive-logging rerun is valid only if it reproduces the frozen canonical
production behavior.

Compare against local reference production `c9d09daad143`.

Required seed-by-condition identity checks:

- terminal MSE/RMSE/MAE;
- terminal citizen posterior means;
- terminal citizen posterior SDs;
- belief checkpoint MSEs;
- Lambda checkpoint metrics;
- condition labels and structural fingerprints.

Maximum absolute numerical difference must be:

[
le 10^{-12}
]

for floating outputs included in the gate.

Any failure blocks interpretation of (W) until the logging implementation is
corrected.

## B6. Claim-classification rule

No claim is required to survive in advance.

### Experiment IV

Reassess separately:

1. structural homophily (H^A);
2. acquisition homophily (H^Lambda);
3. evidence homophily (H^W);
4. dominant closure under Lambda;
5. dominant closure under W;
6. adaptive versus frozen learning trajectories.

If high homophily in Lambda does not imply high homophily or closure in W, the
manuscript must distinguish acquisition from evidence weighting explicitly.

### Experiment III

Separate:

1. structural corrective opportunity;
2. uniform-use structural gateway concentration;
3. behavioral acquisition concentration beyond A;
4. evidence-precision concentration in W;
5. baseline learning;
6. resilience to fixed-biased information.

If the gateway-concentration result is mostly structural, the main claim is
written as a structural redundancy result rather than an endogenous-Lambda
result.

---

# Deferred supplementary analyses

The following are **not** part of the measurement-audit gate.

## Audit-frequency comparative statics

Potential later supplement:

- vary source-reassessment frequency;
- test whether endogenous-closure costs and persistent-bias exposure costs move
  in opposite directions.

This would operationalize reconfigurability as a comparative-static dimension
rather than only adaptive/frozen.

## Tau sign robustness

Potential later supplement:

[
	au_{social}in{0.5,1,2}.
]

The purpose would be sign/qualitative robustness only, not tuning.

The substantive rationale for (	au=1) is that social communication carries
one unit of residual uncertainty on the same scale as a single unit-variance
elite signal, so a peer's expressed posterior cannot become arbitrarily more
precise as communicated evidence merely because the peer is personally
confident.

Neither deferred analysis is run until the A/Lambda/W measurement audit is
classified.

---

# Measurement-audit decision sequence

1. Run Phase A on the existing canonical production.
2. Inspect A versus Lambda benchmarks and total-loss/horizon diagnostics.
3. Run the passive W logger on a small smoke sample.
4. Verify exact behavioral identity with the corresponding canonical runs.
5. Run the full 500-seed passive measurement rerun.
6. Verify the full behavioral identity gate.
7. Compare A, Lambda, and W.
8. Freeze the final claim hierarchy.
9. Only then rebuild manuscript figures/tables.
10. Decide whether the optional audit-frequency and tau robustness supplements
    add enough value to run.

The canonical behavioral evidence remains `c9d09daad143`; the measurement
rerun does not replace it unless a behavioral-identity failure reveals an
implementation problem.

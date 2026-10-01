# Experiment IIIb Diagnostic Plan — Path Independence vs Shared Bottlenecks

## Status

Experiment IIIa is frozen and remains the canonical completed production
experiment. Experiment IIIb is an optional mechanism diagnostic motivated by
the identification ambiguity revealed by IIIa.

The diagnostic is not pre-committed for manuscript inclusion. Its purpose is to
determine whether the paper gains enough network-mechanism value to justify
adding it to the main text or Supplementary Materials, or whether it is better
retained as a reviewer-response reserve.

The design and inclusion logic in this document are frozen **before** inspecting
Experiment IIIb results.

## Motivation

Experiment IIIa manipulated nominal corrective-route multiplicity by changing a
non-gateway citizen from

```text
one gateway peer + one non-gateway peer
```

to

```text
two gateway peers.
```

The production run showed that this reduced incremental Jammer damage on
average, but it also raised no-Jammer loss. The manipulation therefore changed
both route multiplicity and the concentration/composition of immediate peer
opportunities.

Experiment IIIb asks a narrower network question:

> Holding immediate focal-neighborhood composition and global gateway load
> fixed, does independence of corrective paths matter?

This is a path-organization diagnostic, not a rescue experiment for IIIa.

## Structural design

The default N=100 citizen population is partitioned once per matched seed into:

```text
10 Expert gateways
40 relay citizens
50 focal citizens
```

Role assignment is randomized by seed and then fixed across all counterfactual
conditions.

### Elite opportunities

All citizens have structural Jammer access.

Each gateway has:

```text
Expert + Jammer
```

Each relay has:

```text
one gateway + Jammer
```

Each focal citizen has:

```text
two fixed relays + Jammer
```

The same nodes occupy the same roles in both path structures.

### Shared-bottleneck condition

For a focal citizen with fixed relay sources r1 and r2:

```text
focal -> r1 -> g1 -> Expert
      -> r2 -> g1 -> Expert
```

The two nominal corrective paths share the same gateway bottleneck.

The maximum number of internally vertex-disjoint designed corrective paths is:

```text
1
```

### Independent-path condition

The same focal citizen keeps the same two immediate relay sources:

```text
focal -> r1 -> g1 -> Expert
      -> r2 -> g2 -> Expert
```

with g1 != g2.

The maximum number of internally vertex-disjoint designed corrective paths is:

```text
2
```

### Balanced rewiring

Relay pairs are grouped into four-relay blocks.

For gateways g1 and g2:

```text
SHARED
    r1,r2 -> g1
    r3,r4 -> g2

INDEPENDENT
    r1,r3 -> g1
    r2,r4 -> g2
```

This preserves the gateway relay-indegree vector exactly.

Therefore the counterfactual holds fixed:

- focal immediate source sets;
- node roles;
- per-node source degree;
- direct elite access;
- total number of relay-to-gateway edges;
- gateway relay-indegree;
- initial beliefs;
- Jammer opportunity;
- reliance mode and J=1/J=0 pairing.

The manipulated quantity is local overlap of the two designed corrective paths.

## Diagnostic grid

Primary diagnostic:

```text
50 matched seeds
x 2 path structures
x 2 reliance modes
x 2 Jammer states
= 400 runs
```

Frozen parameters:

```text
seeds        = 4001-4050
N citizens   = 100
T            = 200
K            = 1
epsilon      = 0.05
credit       = 20
surveillance = 5
MIN_SD       = 1e-8
prior        = flat
Jammer        = posterior-uncertainty formulation
```

Run:

```bash
PAPER_B_WORKERS=12 bash scripts/run_paper_b_exp3b_diagnostic.sh
```

## Outcomes

The paper-level outcome remains population MSE:

```text
D_p^a = MSE(J=1,p,a) - MSE(J=0,p,a)
```

for path structure p in {shared, independent} and reliance mode a.

The primary IIIb path contrast is:

```text
P^a = D_independent^a - D_shared^a
```

Negative values mean that independent corrective paths reduce incremental
Jammer damage relative to a shared bottleneck.

Because IIIb directly manipulates paths for 50 focal citizens, the same
J=1/J=0 contrast is also computed for the focal subgroup. Focal outcomes are a
mechanism diagnostic, not a replacement for the population-level estimand.

The output separately reports:

```text
MSE_independent,J=1 - MSE_shared,J=1
MSE_independent,J=0 - MSE_shared,J=0
```

for both the full population and focal citizens.

This decomposition prevents a smaller D from being interpreted as greater
resilience when it is generated primarily by a worse no-Jammer baseline.

The adaptive-minus-frozen path interaction is retained as secondary:

```text
P^A - P^F.
```

Experiment IIIb does not require this interaction to be nonzero in order for
path independence to be theoretically informative.

## Pre-analysis decision rule

The diagnostic result will be classified only after the 50-seed bundle is
complete.

### Main-text candidate

IIIb becomes a candidate for brief main-text treatment only when all of the
following descriptive conditions hold:

1. the adaptive population path contrast is negative;
2. its median and trimmed distribution are directionally consistent rather
   than being generated by a small number of seeds;
3. the focal-citizen path contrast points in the same direction;
4. independent paths reduce absolute J=1 MSE;
5. a worse J=0 baseline does not account for most of the reduction in D;
6. the result is stable at the fixed late horizon and free of numerical
   pathology.

A main-text decision would still be made in light of manuscript length and the
clarity of the larger Results sequence.

### Supplementary-material candidate

IIIb is a supplementary mechanism check when the independent-path contrast is
directionally consistent but small, Monte Carlo-noisy, or not sufficiently
important to justify another main-text experiment.

### Reviewer-response reserve

IIIb remains outside the submitted manuscript when the result is approximately
null, unstable, difficult to interpret, or primarily generated by a no-Jammer
baseline difference.

The result is retained so that a reviewer concern about gateway concentration,
path overlap, or bottleneck identification can be answered with an already-run
matched counterfactual.

### Theory-reopening case

The theoretical framing is reopened only if IIIb produces a strong and
systematic result that materially changes the interpretation of IIIa—for
example, if path independence rather than nominal route count clearly emerges
as the dominant structural mechanism, or if independent paths systematically
increase vulnerability.

## Relationship to IIIa and IV

IIIa remains a substantive result about nominal route multiplicity under a
realistic concentration trade-off.

IIIb asks whether two nominally available routes differ when their overlap is
isolated.

Experiment IV remains the clean test that structural homophily becomes
consequential when paired with segregated priors.

Together, these experiments can support the broader Paper B proposition:

```text
structural network statistics do not determine resilience by themselves;
the organization and effective use of available informational pathways matter.
```

Whether IIIb appears in the submitted manuscript is explicitly left open until
the diagnostic result is inspected.

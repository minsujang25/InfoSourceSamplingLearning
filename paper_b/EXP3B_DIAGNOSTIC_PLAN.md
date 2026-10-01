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

This rule is frozen before inspecting any Experiment IIIb outcome.

The primary classification quantity is the adaptive population contrast:

```text
P^A = D_independent^A - D_shared^A
```

For the 50-seed diagnostic, "clearly lower D" is operationalized as a negative
matched-seed mean whose approximate 95% Monte Carlo precision interval
(mean +/- 1.96 x MCSE) also lies below zero. This interval describes simulation
precision for the matched-seed mean; it is not a model-based sampling interval.

The J=0 baseline check is:

```text
B^A = MSE_independent,J=0^A - MSE_shared,J=0^A
```

so (B^A<0) means that independent paths have a smaller no-Jammer baseline
cost. The J=1 absolute-loss contrast and the focal-citizen contrast are reported
as supporting diagnostics, but they do not silently replace the frozen
population-level rule.

### Case A — main-text candidate

If independent paths show **clearly lower** (D) than the shared-bottleneck
condition and the J=0 baseline cost is also smaller:

> **Main-text candidate**

The detailed topology and audit remain in Supplementary Materials. The 50-seed
result does not itself become the final manuscript estimate; this case warrants
considering a 500-seed confirmation before inclusion.

### Case B — directional but small / noisy

If (P^A<0) but the Monte Carlo precision interval overlaps zero, or the
pattern is otherwise directionally sensible but too small/noisy to support a
main-text mechanism claim:

> **Supplementary mechanism check**

The current IIIa interpretation remains unchanged. A 500-seed escalation is
not automatic.

### Case C — null, baseline-driven, or difficult to interpret

If the contrast is approximately null or positive, if a worse J=0 baseline is
doing the substantive work, or if the decomposition is otherwise difficult to
interpret:

> **Do not include in the submitted manuscript**

Retain the diagnostic as reviewer-response reserve.

### Case D — strong result that materially changes IIIa

If IIIb produces a strong, systematic result that materially changes the
interpretation of IIIa—for example, if path independence clearly dominates
nominal route count as the relevant structural mechanism, or if independent
paths systematically increase vulnerability:

> **Reopen the theoretical Results structure**

Only this case warrants reopening the paper's theoretical architecture rather
than treating IIIb as an optional diagnostic.

The 50-seed runner is the only currently authorized IIIb execution. Whether to
run 500 matched seeds is decided only after the 50-seed bundle is classified
under Cases A-D.

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

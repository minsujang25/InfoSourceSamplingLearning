# Rank-Competition Unification Plan
## Frozen before execution — October 3, 2026

This stage follows the completed Experiment-1 mechanism and scope-condition
analyses.

Its purpose is to test whether three results can be organized by one
rank-competition framework:

1. Experiment 1: corrective Expert crowding-out;
2. Experiment 2: structural protection of corrective gateways;
3. fixed-biased stress: biased-source crowding-in for aligned citizens.

No behavioral rule is changed.

---

# 1. Deterministic reconstruction

Before any new learning runs, reconstruct period-0 rankings for canonical seeds
6001--6500.

## Experiment 2

For non-gateway citizens under the null sender:

- low corrective-route multiplicity:
  one gateway peer + one ordinary peer;
- high corrective-route multiplicity:
  two gateway peers.

Record:

- share whose top-ranked peer is a gateway;
- best gateway rank;
- total gateway acquisition probability;
- probability of at least one gateway request within the 20-request budget.

## Fixed-biased Experiment 1 stress

For high prior segregation and low/high homophily, record Jammer rank by citizen
group:

- group -1;
- group +1.

The fixed-biased Jammer is centered at 4 and is not forced last.

---

# 2. Targeted passive reruns

## A. Experiment 2 null

Cross:

\[
\text{multiplicity}\in\{\text{low},\text{high}\}
\times
\text{reliance}\in\{\text{adaptive},\text{frozen}\}
\]

for 500 seeds.

Total:

\[
2\times2\times500=2{,}000
\]

runs.

Record source-rank diagnostics over time for non-gateway citizens:

- top-ranked peer is gateway share;
- mean best-gateway rank;
- gateway acquisition-probability mass;
- implied gateway inclusion probability.

Also retain W gateway metrics already defined in the measurement architecture.

## B. Experiment 1 fixed-biased stress

Cross:

\[
H\in\{\text{low},\text{high}\}
\times
\text{reliance}\in\{\text{adaptive},\text{frozen}\}
\]

under high prior segregation for 500 seeds.

Total:

\[
2\times2\times500=2{,}000
\]

runs.

Record by citizen group:

- Jammer rank distribution;
- mean Jammer acquisition probability;
- Jammer inclusion-period share;
- Jammer W precision share;
- terminal group-specific MSE.

Total new runs:

\[
\boxed{4{,}000}.
\]

---

# 3. Reference identity

All 4,000 conditions already exist in passive measurement rerun
\`087afed31ccf\`.

The new targeted rerun must match the corresponding reference runs with:

\[
\max |\Delta|\le 10^{-12}
\]

for substantive run outputs and exact structural/initial fingerprints.

The logger makes no RNG calls.

---

# 4. Checkpoints

Rank diagnostics are recorded before acquisition at:

\[
t\in\{0,1,2,5,10,25,50,100,150,199,299,399\}.
\]

Evidence-channel group summaries are recorded after:

\[
T\in\{25,50,100,200,400\}.
\]

---

# 5. Interpretation tests

## Experiment 2: rank protection

The proposed mechanism is supported if high corrective-route multiplicity
raises the probability that corrective gateway information occupies a
high-accessibility position even before credibility adaptation.

The strongest structural prediction is:

> when both local peer alternatives are gateways, local peer ranking cannot
> displace corrective access toward an ordinary peer.

Adaptive re-ranking may modify the magnitude but is not required for the
structural protection result.

## Fixed-biased stress: crowding-in

The mirror mechanism is supported if the aligned positive group initially gives
the biased source substantially more favorable rank/access than the negative
group.

A stronger claim that high homophily increases biased-source crowding-in is
made only if high H itself improves the Jammer's rank/access. If high H instead
reduces direct Jammer rank while damage remains high, the stress result must be
interpreted as a combination of biased-source exposure and reduced corrective
access, not simple crowding-in.

---

# 6. Unification decision

After the audit, use one of two framings.

### Unified rank-competition framing

If both tests support it:

\[
\text{opportunity structure}
\rightarrow
\text{relative source rank}
\rightarrow
\text{accessibility}
\rightarrow
\text{evidence allocation}
\rightarrow
\text{learning}.
\]

Experiment 1 creates corrective crowding-out, Experiment 2 structurally
protects corrective rank, and aligned biased sources can crowd into favorable
positions.

### Partial unification

If fixed-biased damage is not explained by Jammer crowding-in, retain the common
rank framework for Experiments 1--2 but treat persistent biased-source damage as
a separate evidence-semantics stress result.

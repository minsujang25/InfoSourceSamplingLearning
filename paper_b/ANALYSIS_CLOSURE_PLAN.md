# Paper B final analysis-closure plan
## Frozen before execution — October 3, 2026

This is the last planned analytical closure stage before the v5 manuscript rewrite.
It is deliberately passive and narrow. No behavioral rule, simulation parameter,
or substantive mechanism is changed, and no new model runs are required.

## 1. Terminal group-gap diagnostic

Use the existing terminal citizen beliefs from passive measurement rerun
`087afed31ccf`. Restrict the primary diagnostic to Experiment 1 / historical IV,
null sender, full `H x S x adaptive/frozen` factorial, seeds 6001--6500.

For each seed and cell define

\[
G_T = |\bar\mu_{+,T}-\bar\mu_{-,T}|.
\]

Also reconstruct the matched initial gap `G_0` from the frozen initial-belief
generator and report `G_T/G_0` when `G_0>0`.

Pre-specified contrasts:

1. `H x S` interaction in `G_T`;
2. high-segregation high-minus-low homophily contrast in `G_T`.

Interpretation rule: use this only as a descriptive check on the established
claim that the high-H/high-S learning failure is dominated by residual
dispersion/subgroup separation. It does not replace terminal MSE and does not
create a new polarization claim. If the gap pattern is weak or inconsistent,
leave it in the supplement or omit it from the manuscript.

## 2. Structural opportunity-network descriptives

Reconstruct the canonical Experiment-1 citizen-to-citizen peer opportunity
network `A_peer` directly from `exp4_homophily_source_maps` for seeds 6001--6500.
External Expert/null/Jammer source slots are excluded from these topology
summaries so that universal or inert elite slots do not mechanically distort
peer-network statistics.

Report by low/high structural homophily:

- peer outdegree (sanity check; fixed at `d=2`);
- peer indegree mean and SD;
- directed reciprocity;
- transitivity of the undirected projection;
- same-group peer-edge share;
- E-I index `(external - internal)/(external + internal)`.

Interpretation rule: these are reporting-completeness/SNA descriptives, not new
mechanisms. They belong in the Supplement unless they reveal an unexpected
structural artifact.

## 3. Gate

The closure stage passes only if:

- all 4,000 Experiment-1 null run-level group-gap summaries are recovered;
- all 1,000 structural seed x homophily network summaries are reconstructed;
- canonical design parameters match;
- peer outdegree is exactly two for every citizen/network;
- all reported closure metrics are finite.

Passing this gate closes the analysis program. The next step is figure
architecture and the v5 Theory/Results rewrite, not additional robustness
searching.

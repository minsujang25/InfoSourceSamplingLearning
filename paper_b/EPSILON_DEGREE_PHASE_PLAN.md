# Epsilon-by-Degree Scope-Condition Plan
## Frozen before execution — October 2, 2026

This stage follows:

- canonical production \`c9d09daad143\`;
- mechanism robustness \`9b5c35506618\`;
- exact W-channel decomposition \`56b730d60198\`.

The W-channel decomposition supports retaining Expert-specific congenial
crowding-out as the primary mechanism. The next task is to map the range of
epsilon and peer degree over which this mechanism produces a substantively
meaningful learning penalty.

---

# 1. Grid

Use:

\[
\epsilon\in\{.02,.05,.10,.20,.30\}
\]

and:

\[
d\in\{2,3,4\}.
\]

For each \((\epsilon,d)\) specification, run the full Experiment-1 null
factorial:

\[
H\in\{\text{low},\text{high}\}
\times
S\in\{\text{low},\text{high}\}
\times
R\in\{\text{adaptive},\text{frozen}\}.
\]

Use 500 matched seeds 6001--6500.

The full surface therefore contains:

\[
15\times8\times500=60{,}000
\]

runs.

Four specifications already exist and must be reused rather than re-simulated:

- d=2, epsilon=.05 from canonical production;
- d=2, epsilon=.10 from mechanism robustness;
- d=2, epsilon=.20 from mechanism robustness;
- d=4, epsilon=.05 from mechanism robustness.

The new production burden is therefore:

\[
11\times8\times500=\boxed{44{,}000}
\]

new runs.

---

# 2. Primary outcome surface

The main phase diagram reports the homophily-by-segregation MSE interaction,
separately for adaptive and frozen reliance:

\[
\Delta_{H\times S}MSE
=
(MSE_{HH}-MSE_{LH})
-
(MSE_{HL}-MSE_{LL}).
\]

This directly retains the manuscript's claim that homophilous opportunity is
most consequential when prior beliefs are segregated.

A second, more mechanism-specific surface reports the high-segregation
homophily penalty:

\[
\Delta_H^{S=high}MSE
=
MSE_{highH,highS}
-
MSE_{lowH,highS}.
\]

Both should be retained.

---

# 3. Mechanism surfaces

For each specification report:

1. initial/period-0 Expert rank distribution;
2. mean Expert acquisition probability;
3. Expert inclusion probability implied by rank and 20 requests;
4. terminal Expert evidence share in W;
5. terminal W dominant Expert reach;
6. terminal same-group W enclosure as a downstream diagnostic.

The primary mechanism link is:

\[
(\epsilon,d)
\rightarrow
\text{Expert rank depth/accessibility}
\rightarrow
\text{Expert evidence contribution}
\rightarrow
\text{learning penalty}.
\]

---

# 4. Degree nesting

For each seed and homophily treatment:

- the d=3 peer set must contain both d=2 peers;
- the d=4 peer set must contain both d=2 peers.

This preserves the interpretation of degree as adding alternatives rather than
replacing the original opportunity set.

---

# 5. Scope-condition interpretation

The phase diagram is intended to identify where crowding-out is substantively
important.

Expected comparative statics:

\[
\epsilon\uparrow
\Rightarrow
\text{lower-rank access increases}
\Rightarrow
\text{crowding penalty decreases},
\]

and:

\[
d\uparrow
\Rightarrow
\text{more congenial competitors}
\Rightarrow
\text{Expert rank depth increases}
\Rightarrow
\text{crowding penalty increases}.
\]

No threshold separating "important" from "unimportant" is selected before
observing the surface. Report the continuous surface and describe regions
rather than dichotomizing post hoc.

---

# 6. Existing-reference reuse

The combined surface must mark whether each cell comes from:

- canonical production;
- prior mechanism robustness;
- new phase-diagram production.

Existing reference outputs are copied into the combined analysis only after
checking:

- design labels;
- seed range;
- peer degree;
- epsilon;
- null sender;
- T=400;
- source-posterior peer evidence;
- tau_social=1;
- pre-disruption frozen initialization.

---

# 7. Main tables/figures

## Figure A — Scope-condition heatmaps

Two heatmaps:

- frozen H x S MSE interaction;
- adaptive H x S MSE interaction.

Axes:

- x: epsilon;
- y: peer degree.

## Figure B — Mechanism heatmaps

At minimum:

- Expert inclusion/accessibility;
- W Expert evidence share.

## Figure C — Mechanism/outcome collapse

Across all 15 specifications, plot:

- x: Expert accessibility or W Expert;
- y: H x S learning penalty or high-S homophily penalty.

This is descriptive and not a causal mediation estimate.

---

# 8. Empirical interpretation

The manuscript should interpret epsilon and degree as scope parameters.

- Small epsilon represents strongly concentrated attention on the top-ranked
  source.
- Larger epsilon represents flatter attention across ranked alternatives.
- d represents the number of peer alternatives competing within the local
  information opportunity set.

After the surface is known, connect the empirically plausible range of these
parameters to social-signature/core-discussion-network evidence without
claiming direct calibration unless an external dataset is actually used.

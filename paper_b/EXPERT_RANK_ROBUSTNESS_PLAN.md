# Expert-Rank Mechanism Audit and Targeted Robustness Plan
## Frozen before execution — October 2, 2026

This stage follows canonical behavioral production \`c9d09daad143\` and the
passive A--Lambda--W measurement audit \`087afed31ccf\`.

It does not reopen the receiver specification.

The purpose is to test the proposed mechanism behind Experiment IV:

\[
\text{homophilous opportunity}
\rightarrow
\text{congenial peer crowding}
\rightarrow
\text{Expert rank demotion}
\rightarrow
\text{lower corrective acquisition}
\rightarrow
\text{closed dependence}
\rightarrow
\text{slower corrective learning}.
\]

"Congenial crowding-out" is a working mechanism label. It refers to rank
competition under a limited acquisition budget, not to Granovetterian tie
strength.

---

# Phase 1 — Existing-output Expert mechanism audit

No learning simulation is required for the frozen-rank reconstruction.

For canonical seeds 6001--6500 and Experiment IV high segregation:

1. reconstruct the exact low-/high-homophily opportunity networks;
2. reconstruct the exact paired initial beliefs;
3. apply the canonical pre-disruption distance ranking;
4. keep the null sender last, exactly as in canonical production;
5. record every citizen's Expert rank, same-group peer count, and the number of
   peers that outrank the Expert;
6. map Expert rank to the canonical epsilon-greedy acquisition probability;
7. compare reconstructed mean Expert acquisition to the period-0 canonical
   Lambda checkpoint.

The expected rank-based acquisition probabilities for four sources at
epsilon=.05 are:

\[
p_1=.95,\qquad
p_2=.0475,\qquad
p_3=.002375,\qquad
p_4=.000125.
\]

With credit 20, the corresponding expected request counts are 19, .95, .0475,
and .0025.

The audit must report:

- Expert rank 1/2/3 shares under low and high H;
- mean Expert acquisition probability;
- mean same-group peer count;
- mean count of peers that outrank Expert;
- rank distribution conditional on 0/1/2 same-group peers;
- rank distribution by citizen group;
- canonical adaptive/frozen Expert acquisition paths from the passive
  measurement output, when supplied.

The audit is descriptive/mechanistic. It is not a causal mediation analysis.

---

# Phase 2 — epsilon robustness

Run only the Experiment IV null block.

Freeze:

- N=100;
- T=400;
- source_posterior peer evidence;
- tau_social=1;
- pre_disruption frozen initialization;
- credit=20;
- K=1;
- high-group shift=3;
- residual SD=1;
- canonical seeds 6001--6500;
- peer degree d=2.

New epsilon values:

\[
\epsilon\in\{.10,.20\}.
\]

Canonical epsilon=.05 is taken from the frozen production rather than
re-simulated.

For each epsilon, cross:

- low/high structural homophily;
- low/high prior segregation;
- adaptive/frozen reliance.

Thus Phase 2 adds:

\[
2 \times 2 \times 2 \times 2 \times 500 = 8{,}000
\]

new null-sender simulations.

Pre-specified comparative-static prediction:

\[
\epsilon\uparrow
\Rightarrow
\text{rank penalty weakens}
\Rightarrow
\text{Expert demotion becomes less consequential}
\Rightarrow
\text{frozen }H\times S\text{ learning penalty shrinks}.
\]

No sign or magnitude threshold is required in advance.

Primary outputs:

- terminal MSE and MSE decomposition;
- Expert acquisition path;
- Expert rank shares at selected checkpoints;
- A/Lambda/W homophily;
- Lambda and W same-group closed-dependence diagnostics;
- W dominant Expert reach.

---

# Phase 3 — peer-degree robustness

Run only the Experiment IV null block at:

\[
d=4,\qquad \epsilon=.05.
\]

Use the same canonical seeds and the full:

- low/high H;
- low/high S;
- adaptive/frozen

factorial.

This adds:

\[
2\times2\times2\times500=4{,}000
\]

new simulations.

The d=4 structural generator must preserve the first two peer selections from
the d=2 design for each matched seed/ego, then add two further peers from the
same common-random-number stream.

Working analytical motivation:

If each peer independently outranks the Expert with probability q, the Expert's
rank is approximately

\[
R_E=1+\operatorname{Binomial}(d,q).
\]

Under recursive rank acquisition, increasing d can therefore push a corrective
source geometrically deeper into the attention hierarchy even though the
source remains structurally available.

The simulation is used to test this comparative static under the actual
correlated network and prior construction, not to assume the binomial
approximation is exact.

Primary outputs match Phase 2.

---

# Interpretation gate

After Phases 1--3:

- If Expert demotion explains the frozen high-H/high-S acquisition pattern and
  the epsilon/degree comparative statics follow the predicted direction,
  elevate congenial crowding-out to the primary Experiment-1 mechanism.
- If the comparative statics are weak or inconsistent, retain Expert rank as a
  descriptive mechanism diagnostic and do not build the theory around it.

Only after this gate should the Social Networks theory section be rewritten
around the mechanism.

---

# Later stages

After the mechanism gate:

1. rewrite the theory in Social Networks language;
2. make the model specification self-contained;
3. rename experiments and potentially ambiguous network terms;
4. optionally run elite-evidence-semantics robustness;
5. optionally add empirical calibration or clustered topology.

The optional analyses do not block the mechanism gate.

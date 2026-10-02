# W-Channel Decomposition Plan
## Frozen before execution — October 2, 2026

This stage follows canonical production \`c9d09daad143\`, the A--Lambda--W
measurement audit, and mechanism robustness design \`9b5c35506618\`.

The purpose is to distinguish two channels through which increasing epsilon may
reduce the Experiment-1 high-homophily/high-segregation learning penalty:

1. increased access to the corrective Expert;
2. increased access to other-group peers.

No receiver, ranking, acquisition, or sender rule is changed.

---

# 1. Targeted design

Run only Experiment 1 (historical Experiment IV), high prior segregation,
frozen reliance, null sender.

Cross:

\[
H\in\{\text{low},\text{high}\},
\qquad
\epsilon\in\{.05,.10,.20\},
\]

with canonical peer degree \(d=2\) and seeds 6001--6500.

Total:

\[
2\times3\times500=\boxed{3{,}000}
\]

passive-logging reruns.

All other parameters remain canonical:

- \(N=100\);
- \(T=400\);
- credit \(=20\);
- \(K=1\);
- \(\tau_{\text{social}}=1\);
- source-posterior peer evidence;
- pre-disruption frozen ranking;
- high-group shift \(=3\);
- residual initial-belief SD \(=1\).

---

# 2. Exact evidence channels

For citizen \(i\), cumulative source precision is decomposed into:

\[
Q^E_i
\]

for the Expert,

\[
Q^S_i
\]

for same-group peers,

\[
Q^X_i
\]

for other-group peers, and

\[
Q^B_i
\]

for the disruptive-source slot.

Under the null sender, \(Q^B_i=0\).

Normalize within citizen:

\[
W^c_i=\frac{Q^c_i}
{Q^E_i+Q^S_i+Q^X_i+Q^B_i}.
\]

The principal channels are:

- \(W^E\): Expert evidence share;
- \(W^S\): same-group-peer evidence share;
- \(W^X\): other-group-peer evidence share;
- \(W^{E+X}=W^E+W^X\): corrective/cross-cutting evidence share.

The decomposition uses the exact source-specific precision already added to the
Gaussian state update.

---

# 3. Accessibility measures

For each citizen and source class, record the fraction of state-learning periods
in which at least one positive-precision observation from that class enters the
update:

- \(I^E\): Expert inclusion-period share;
- \(I^S\): same-group-peer inclusion-period share;
- \(I^X\): other-group-peer inclusion-period share.

These measures distinguish:

\[
\epsilon
\rightarrow
\text{source accessibility}
\rightarrow
\text{cumulative evidence allocation}.
\]

They are passive measurement counters and make no RNG calls.

---

# 4. Exact checkpoints

Record channel summaries after exactly:

\[
T\in\{25,50,100,200,400\}
\]

completed periods.

This avoids the historical period-versus-horizon indexing ambiguity.

---

# 5. Behavioral-identity gate

The logging rerun is valid only if substantive outputs reproduce the frozen
reference runs.

## epsilon=.05

Compare seed-by-condition against canonical production
\`c9d09daad143\`.

## epsilon=.10 and .20

Compare seed-by-condition against mechanism robustness production
\`9b5c35506618\`.

Required tolerance for floating behavioral outputs:

\[
\max |\Delta| \le 10^{-12}.
\]

At minimum compare:

- terminal MSE/RMSE/MAE;
- mean belief;
- belief variance;
- squared displacement;
- structural and initial-state fingerprints.

Any identity failure blocks channel interpretation.

---

# 6. Primary channel test

Within the high-H frozen cell, compare epsilon changes in:

\[
W^E,\quad W^X,\quad W^S,
\]

and

\[
I^E,\quad I^X,\quad I^S.
\]

Interpretation:

### Expert-dominant channel

If \(W^E\) and \(I^E\) rise strongly while \(W^X\) changes little, retain
Expert-specific congenial crowding-out.

### Cross-group-dominant channel

If \(W^X\) and \(I^X\) account for most of the change, broaden the mechanism to
cross-cutting exposure crowding-out.

### Joint channel

If both Expert and other-group evidence increase materially, define the
mechanism more generally as crowding-out of non-congenial corrective exposure.

No threshold is tuned after observing results.

---

# 7. Mechanism-collapse diagnostic

For each seed-condition run retain:

- terminal MSE;
- \(W^E\);
- \(W^X\);
- \(W^{E+X}\);
- \(I^E\);
- \(I^X\).

The next diagnostic will compare terminal MSE against:

1. \(W^E\);
2. \(W^X\);
3. \(W^{E+X}\).

The objective is descriptive: assess whether experimental cells collapse onto
a common nonlinear evidence-accessibility/error relationship.

This is not treated as causal mediation.

---

# 8. Decision sequence

1. run deterministic/passive-identity smoke checks;
2. run all 3,000 targeted decompositions;
3. require the full reference-identity gate to pass;
4. classify the Expert versus cross-group channel;
5. run the mechanism-collapse diagnostic;
6. freeze the mechanism wording;
7. only then design the \((\epsilon,d)\) phase diagram.

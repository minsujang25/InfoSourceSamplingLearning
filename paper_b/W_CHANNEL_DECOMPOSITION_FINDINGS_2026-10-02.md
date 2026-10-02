# W-Channel Decomposition Findings
## Design 56b730d60198 — October 2, 2026

This note records the exact W-channel decomposition for Experiment 1
(historical Experiment IV), high prior segregation, frozen reliance, null
sender, d=2, epsilon in {.05,.10,.20}, and seeds 6001--6500.

Reference designs:

- canonical production: \`c9d09daad143\`;
- mechanism robustness: \`9b5c35506618\`.

---

# 1. Production and identity gate

The targeted passive-logging design passed all gates:

- 1,500 / 1,500 shards;
- 3,000 / 3,000 runs;
- canonical base match TRUE;
- all terminal MSE finite;
- all runs at T=400;
- null-last share minimum = 1.0;
- max citizen W-share sum deviation = 2.22e-16;
- max null precision = 0;
- structural and initial-state fingerprints matched reference runs;
- max behavioral difference relative to reference outputs = 0.0 across
  21,000 numerical comparisons.

The decomposition therefore adds measurement only; it does not alter behavior.

---

# 2. High-H frozen channel decomposition

Terminal cell means:

| epsilon | MSE | W Expert | W same-group peer | W other-group peer | I Expert | I other-group peer |
|---:|---:|---:|---:|---:|---:|---:|
| .05 | .176056 | .197195 | .798519 | .004286 | .244729 | .015470 |
| .10 | .026339 | .268405 | .721766 | .009829 | .380046 | .039299 |
| .20 | .001640 | .394138 | .587990 | .017873 | .627261 | .096115 |

Increasing epsilon therefore opens both the Expert and other-group channels,
but the Expert channel is much larger in both level and change.

From epsilon=.05 to .20:

- Expert W share rises by +.19694;
- other-group-peer W share rises by +.01359;
- combined corrective/cross-cutting W share rises by +.21053.

Thus approximately 93.5% of the increase in normalized corrective/cross-cutting
evidence share comes from the Expert.

Using absolute cumulative precision instead of normalized W:

- Expert precision rises from 616.56 to 901.65 (+285.09);
- other-group-peer precision rises from 6.11 to 39.26 (+33.15).

Approximately 89.6% of the increase in total Expert-plus-other-group precision
therefore comes from the Expert.

The same pattern holds from .05 to .10 and from .10 to .20.

---

# 3. Accessibility channel

The inclusion-period measure gives the same interpretation.

Under high H:

\[
I^E:
.245 \rightarrow .380 \rightarrow .627,
\]

whereas

\[
I^X:
.015 \rightarrow .039 \rightarrow .096.
\]

The other-group channel becomes more accessible as epsilon increases, but it
remains much less frequently accessed than the Expert.

The evidence-share trajectories are already separated by T=25 and remain
stable thereafter. The mechanism is therefore not generated only at the end of
the simulation.

---

# 4. Low-H comparison

Under low H, the Expert is much more accessible at every epsilon:

| epsilon | MSE | W Expert | W other-group peer | I Expert | I other-group peer |
|---:|---:|---:|---:|---:|---:|
| .05 | .001183 | .531056 | .019794 | .613629 | .185495 |
| .10 | .000547 | .600203 | .039428 | .746701 | .303778 |
| .20 | .000346 | .661502 | .068423 | .872058 | .486771 |

Other-group exposure is much larger in low-H networks than in high-H networks,
but terminal error is already very small. This reinforces the interpretation
that the principal high-H mechanism is loss of corrective Expert access rather
than cross-group exposure alone.

---

# 5. Mechanism-collapse diagnostic

The mechanism-collapse diagnostic is descriptive rather than causal.

At the seed-run level, across all 3,000 runs:

- Spearman(MSE, W Expert) = -.939;
- Spearman(MSE, W other-group peer) = -.927;
- Spearman(MSE, W Expert + W other-group peer) = -.941.

Because all three channels co-move with epsilon and homophily, these raw
correlations alone do not identify the dominant channel.

A more informative comparison uses a common quadratic relationship with
log10(MSE), evaluated with grouped cross-validation by seed.

Across all 3,000 runs:

- W Expert: CV R2 ≈ .898;
- W other-group peer: CV R2 ≈ .793;
- W Expert + W other-group peer: CV R2 ≈ .898;
- Expert inclusion I Expert: CV R2 ≈ .918;
- other-group inclusion I X: CV R2 ≈ .781.

Within the high-H runs only (n=1,500):

- W Expert: CV R2 ≈ .840;
- W other-group peer: CV R2 ≈ .753;
- W Expert + W other-group peer: CV R2 ≈ .841;
- Expert inclusion: CV R2 ≈ .848;
- other-group inclusion: CV R2 ≈ .776.

Adding the other-group W channel to W Expert improves high-H descriptive fit
only trivially. The inclusion result is even clearer.

These fits are not causal mediation estimates; they are used only to determine
whether the experimental cells follow a common accessibility/error pattern.

---

# 6. Mechanism wording decision

The decomposition supports retaining an Expert-specific mechanism as the
primary interpretation.

Recommended wording:

> Homophilous opportunity becomes consequential when congenial peers crowd a
> corrective Expert downward in a steep attention hierarchy. The resulting
> learning cost depends on whether lower-ranked corrective information remains
> behaviorally accessible. Increasing epsilon also opens cross-group peer
> exposure, but the dominant evidence-channel change in the present design is
> restored Expert access.

Cross-group exposure should therefore be treated as a secondary spillover
rather than replacing the Expert-specific mechanism with a broader
cross-cutting-exposure mechanism.

A slightly more general theoretical wording can still acknowledge that the
same ranking mechanism may crowd out other non-congenial sources in other
settings.

---

# 7. Implication for the next phase diagram

The W-channel decomposition resolves the channel-identification question.

The next analysis should map the scope condition in the epsilon-by-degree
plane, with the Expert-access mechanism retained as the principal mechanism.

The most informative primary outcome is now:

\[
\Delta_H MSE(\epsilon,d)
=
MSE_{highH,highS}-MSE_{lowH,highS},
\]

reported separately for adaptive and frozen reliance.

The corresponding mechanism panels should report:

- Expert rank depth;
- Expert inclusion probability/share;
- W Expert evidence share.

A full H x S interaction can remain a validation outcome, but the phase diagram
need not spend simulation capacity on every low-S cell if sparse low-S
sentinels confirm that the degree/epsilon surface is specific to prior
segregation.

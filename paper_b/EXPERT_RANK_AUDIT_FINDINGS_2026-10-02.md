# Expert-Rank Mechanism Audit — Findings
## October 2, 2026

This note records the Experiment-IV Expert-rank audit using canonical seeds
6001--6500 and the passive measurement output \`087afed31ccf\`.

The audit does not add a new learning simulation. Frozen pre-disruption ranks
are reconstructed exactly from the canonical structural maps and paired initial
beliefs. Adaptive Expert-acquisition trajectories are read from the canonical
Lambda checkpoints.

---

## 1. Rank-based acquisition scale

With four sources and epsilon=.05:

\[
(p_1,p_2,p_3,p_4)
=
(.95,.0475,.002375,.000125).
\]

With credit 20, expected requests by rank are:

\[
(19,\ .95,\ .0475,\ .0025).
\]

The probability of receiving at least one request in a state-learning period
is approximately:

\[
(1,\ .6222,\ .0464,\ .0025).
\]

Thus moving the Expert from rank 2 to rank 3 has a large behavioral
consequence even though the Expert remains structurally available.

---

## 2. Exact frozen Expert-rank reconstruction under high prior segregation

### Low structural homophily

- mean same-group peer count: 1.003 of 2;
- mean peers outranking Expert: 0.909;
- Expert rank 1: 30.87%;
- Expert rank 2: 47.38%;
- Expert rank 3: 21.75%;
- mean Expert acquisition probability: .316287.

### High structural homophily

- mean same-group peer count: 1.801 of 2;
- mean peers outranking Expert: 1.632;
- Expert rank 1: 6.924%;
- Expert rank 2: 22.968%;
- Expert rank 3: 70.108%;
- mean Expert acquisition probability: .078353.

The reconstructed acquisition probabilities match the canonical period-0
Lambda checkpoints.

---

## 3. Same-group peers almost directly determine Expert demotion

Across the canonical high-segregation construction, a same-group peer lies
closer to the citizen's prior than the Expert about 90.55% of the time.
An other-group peer does so only about 0.17% of the time.

Conditional on same-group peer count, the Expert-rank distribution is nearly
the same in the low- and high-homophily treatments.

### Zero same-group peers

Expert is rank 1 about 99.4--99.7% of the time.

### One same-group peer

Expert is rank 2 about 90.3--90.5% of the time.

### Two same-group peers

Expert is rank 3 about 86.3--86.4% of the time.

The homophily treatment therefore changes the prevalence of rank-demoting local
configurations rather than changing how a given local configuration is ranked.

Same-group peer-count shares are approximately:

| Homophily | 0 same | 1 same | 2 same |
|---|---:|---:|---:|
| Low H | .248 | .501 | .251 |
| High H | .010 | .179 | .811 |

This provides the direct mechanism behind the frozen Expert-acquisition
difference.

---

## 4. Adaptive Expert acquisition is early-loaded but not a one-audit artifact

Under high segregation, frozen Expert acquisition remains fixed by design.

| Period | Low-H frozen | High-H frozen |
|---:|---:|---:|
| 0 | .3163 | .0784 |
| 10 | .3163 | .0784 |
| 100 | .3163 | .0784 |
| 399 | .3163 | .0784 |

Adaptive reliance begins from exactly the same pre-disruption ranking.

| Period | Low-H adaptive | High-H adaptive |
|---:|---:|---:|
| 0 | .3163 | .0784 |
| 1 | .3284 | .1404 |
| 5 | .3414 | .1909 |
| 10 | .2988 | .2215 |
| 25 | .2233 | .1993 |
| 100 | .1377 | .1430 |
| 199 | .1073 | .1203 |
| 399 | .0853 | .0997 |

In the high-H condition, Expert acquisition almost triples from .078 at period
0 to a local maximum around .221 at period 10. It later declines as peer
credibility evolves, but remains above the frozen level at the terminal
horizon.

The mechanism is therefore best described as **early credibility reassessment
with continuing but declining opportunities for re-ranking**, not as either
continuous high-frequency adaptation or a one-time first-audit effect.

---

## 5. Mechanism interpretation

The audit supports the following sequence:

\[
\text{homophilous opportunity}
\rightarrow
\text{more congenial peers}
\rightarrow
\text{Expert rank demotion}
\rightarrow
\text{lower corrective acquisition}.
\]

This resolves the apparent puzzle that frozen low-H networks can have high
acquisition/evidence homophily and still learn well. With one congenial peer,
the Expert is usually rank 2 and remains behaviorally reachable. With two
congenial peers, the Expert is usually rank 3, where recursive rank acquisition
reduces its expected use sharply.

Closed dependence and reduced Expert reach should therefore be interpreted as
downstream network configurations associated with this rank-crowding process,
not as the primitive behavioral mechanism.

The remaining robustness question is whether the substantive learning penalty
changes in the predicted direction when:

1. epsilon increases and the adjacent-rank penalty weakens;
2. peer degree increases and more congenial alternatives can outrank the
   Expert.

Those tests are precommitted in \`EXPERT_RANK_ROBUSTNESS_PLAN.md\`.

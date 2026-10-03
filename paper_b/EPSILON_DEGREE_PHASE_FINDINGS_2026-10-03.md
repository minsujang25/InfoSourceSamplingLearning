# Epsilon-by-Degree Phase Diagram Findings
## Design 3f9c496208d5 — October 3, 2026

This note records the full Experiment-1 null phase diagram over:

\[
\epsilon\in\{.02,.05,.10,.20,.30\},
\qquad
d\in\{2,3,4\},
\]

with 500 matched seeds, low/high homophily, low/high prior segregation, and
adaptive/frozen reliance.

Existing canonical/mechanism specifications were reused and 44,000 new runs
were added, yielding a complete 60,000-run surface.

---

# 1. Production gate

The phase-diagram gate passed:

- expected combined rows: 60,000;
- observed combined rows: 60,000;
- duplicate rows: 0;
- missing rows: 0;
- extra rows: 0;
- expected new runs: 44,000;
- observed new runs: 44,000;
- expected new shards: 5,500;
- observed new shards: 5,500;
- all terminal MSE finite;
- canonical base match TRUE;
- degree nesting PASS.

The d=3 and d=4 opportunity sets preserve the canonical d=2 peers and add
additional alternatives.

---

# 2. Main outcome surface

The homophily-by-segregation terminal-MSE interaction is:

## Adaptive reliance

| d \\ epsilon | .02 | .05 | .10 | .20 | .30 |
|---:|---:|---:|---:|---:|---:|
| 2 | .16134 | .01418 | .00148 | .00012 | .00002 |
| 3 | 1.39997 | .34977 | .06334 | .00395 | .00070 |
| 4 | 3.31553 | 1.52897 | .56303 | .05901 | .00818 |

## Frozen reliance

| d \\ epsilon | .02 | .05 | .10 | .20 | .30 |
|---:|---:|---:|---:|---:|---:|
| 2 | .80328 | .17484 | .02578 | .00129 | .00026 |
| 3 | 4.22366 | 2.36403 | .91824 | .09025 | .00947 |
| 4 | 6.60516 | 5.51422 | 3.69324 | 1.07175 | .19247 |

The high-segregation homophily penalty is nearly identical to the H x S
interaction throughout the surface. The largest discrepancy is below .002.
Thus the surface is driven almost entirely by the high-segregation condition,
while low-segregation homophily effects remain negligible.

---

# 3. Both pre-specified comparative statics hold monotonically

For every peer degree and both reliance modes:

\[
\epsilon\uparrow
\Rightarrow
\Delta_{H\times S}MSE\downarrow.
\]

For every epsilon and both reliance modes:

\[
d\uparrow
\Rightarrow
\Delta_{H\times S}MSE\uparrow.
\]

There are no reversals in either comparative static.

This strongly supports the proposed scope mechanism:

- flatter rank-based attention preserves lower-ranked corrective access;
- more peer alternatives deepen corrective-source crowding.

---

# 4. The d=3 surface resolves the empirical-scope concern

Degree 3 lies smoothly between d=2 and d=4.

Examples under frozen reliance:

- epsilon=.05:
  - d=2: .175
  - d=3: 2.364
  - d=4: 5.514
- epsilon=.10:
  - d=2: .026
  - d=3: .918
  - d=4: 3.693
- epsilon=.20:
  - d=2: .001
  - d=3: .090
  - d=4: 1.072

Thus the mechanism is not restricted to the d=4 stress case. Increasing the
number of peer competitors shifts the epsilon range over which crowding-out is
consequential.

---

# 5. Scope condition rather than universal failure

The results define a continuous scope condition rather than a single robust
effect.

At d=2, increasing epsilon to .20--.30 nearly eliminates the learning penalty
under both adaptive and frozen reliance.

At d=3, substantial frozen penalties persist through epsilon=.10, but become
small by epsilon=.20 and very small by .30.

At d=4, crowding remains consequential over a much wider exploration range:

- frozen penalty = 3.69 at epsilon=.10;
- frozen penalty = 1.07 at epsilon=.20;
- frozen penalty = .192 even at epsilon=.30.

Adaptive re-ranking shifts the surface downward but does not remove the same
scope condition. At d=4, adaptive penalties remain .563 at epsilon=.10 and
.059 at .20.

The manuscript should therefore state explicitly that crowding-out is most
important in environments combining:

1. segregated priors;
2. multiple congenial alternatives;
3. strongly concentrated rank-based attention.

---

# 6. Initial Expert rank depth

Under high homophily and high segregation, mean Expert rank depends on degree
but not epsilon:

- d=2: 2.632;
- d=3: 3.447;
- d=4: 4.263.

At epsilon=.05, the rank distributions are:

## d=2

- rank 1: 6.9%;
- rank 2: 23.0%;
- rank 3: 70.1%.

## d=3

- rank 1: 4.2%;
- rank 2: 7.8%;
- rank 3: 27.0%;
- rank 4: 61.0%.

## d=4

- rank 1: 3.3%;
- rank 2: 3.9%;
- rank 3: 9.5%;
- rank 4: 29.8%;
- rank 5: 53.5%.

Peer multiplicity therefore deepens corrective-source displacement
monotonically.

---

# 7. Expert accessibility surface

The initial probability that the Expert is sampled at least once in a
20-request state-learning period under high H/high S is:

| d \\ epsilon | .02 | .05 | .10 | .20 | .30 |
|---:|---:|---:|---:|---:|---:|
| 2 | .150 | .245 | .380 | .627 | .807 |
| 3 | .070 | .105 | .164 | .321 | .510 |
| 4 | .046 | .062 | .088 | .166 | .293 |

This surface is the near mirror image of the MSE surface.

Across the 15 high-H/high-S specifications:

- under frozen reliance, Spearman correlation between Expert inclusion and
  terminal MSE is approximately -.982;
- under adaptive reliance, the corresponding Spearman correlation is -1.000.

Using log10 terminal MSE:

- frozen Pearson correlation with initial Expert inclusion is approximately
  -.989;
- adaptive Pearson correlation is approximately -.920.

The association is descriptive rather than a mediation estimate, but it
strongly supports lower-rank accessibility as the relevant scope parameter.

---

# 8. Evidence-network surface

The evidence network follows the same broad pattern.

In high-H/high-S frozen conditions, terminal Expert W share increases with
epsilon and decreases with peer degree.

Examples:

- d=2: .137 -> .197 -> .268 -> .394 -> .498 as epsilon rises .02 -> .30;
- d=3: .066 -> .087 -> .117 -> .185 -> .260;
- d=4: .044 -> .054 -> .066 -> .098 -> .142.

Dominant Expert reach also rises with epsilon and falls with degree.

Closed same-group evidence configurations move in the opposite direction.

These are downstream network manifestations of the accessibility mechanism.

One qualification is important under adaptive reliance: terminal Expert W
share is not perfectly monotonic at high degree. For d=4 it peaks around
epsilon=.20 and is slightly lower at .30, even while MSE continues to improve.
Thus terminal Expert share is not itself a welfare criterion. Early corrective
access and the subsequent improvement of peer information can reduce the need
for continued direct Expert dependence.

---

# 9. Frozen versus adaptive reliance

Adaptive re-ranking lowers the crowding penalty throughout the surface, but
does not change the direction of the epsilon or degree comparative statics.

The frozen-minus-adaptive gap is especially large in the intermediate region
where lower-ranked sources are neither completely inaccessible nor almost
universally sampled.

This supports treating reconfigurability as a mitigating mechanism rather than
a substitute for lower-rank accessibility.

---

# 10. Manuscript interpretation

The phase diagram resolves the main robustness concern raised by the strong
epsilon=.05 effect.

The result should not be stated as:

> homophily generically produces slow learning.

It should be stated as:

> Homophilous opportunity can produce severe corrective-source crowding when
> prior beliefs are segregated, local peer competition is sufficiently dense,
> and attention is strongly concentrated toward top-ranked sources. Flatter
> attention and lower peer multiplicity preserve access to corrective
> information and sharply reduce the penalty.

The canonical epsilon=.05, d=2 specification is one point on this surface, not
a universal benchmark.

The phase diagram should become a main-text scope-condition figure rather than
being relegated to a robustness appendix.

---

# 11. Next analytical step

With the Experiment-1 mechanism and scope conditions now mapped, the next
analysis should test whether the same rank-competition logic unifies:

1. Experiment 2 corrective-route multiplicity through gateway-rank protection;
2. fixed-biased stress through biased-source crowding-in.

The objective is to determine whether the two experiments and the stress
condition can be presented as three manifestations of a common structural
rank-competition framework.

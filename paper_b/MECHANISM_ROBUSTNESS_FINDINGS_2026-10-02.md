# Mechanism Robustness Findings
## Design 9b5c35506618 — October 2, 2026

This note records the precommitted Experiment-IV mechanism robustness results.

Reference canonical design: \`c9d09daad143\`.

Robustness design:

- epsilon robustness: d=2, epsilon in {.10,.20};
- degree robustness: d=4, epsilon=.05;
- null sender only;
- full low/high H x low/high S x adaptive/frozen factorial;
- 500 matched seeds 6001--6500;
- T=400;
- source_posterior peer evidence;
- tau_social=1;
- pre_disruption frozen initialization.

The production gate passed:

- 1,500 / 1,500 blocks;
- 12,000 / 12,000 runs;
- canonical base match TRUE;
- d=4 nested extension TRUE;
- fixed horizon TRUE;
- all terminal MSE finite;
- null source last-ranked in every run.

---

# 1. Epsilon comparative statics strongly support rank-based crowding-out

Canonical d=2, epsilon=.05 HxS MSE interactions:

- adaptive: +.014181;
- frozen: +.174843.

At epsilon=.10:

- adaptive: +.001482;
- frozen: +.025783.

At epsilon=.20:

- adaptive: +.000124;
- frozen: +.001293.

Relative to the canonical epsilon=.05 result, the HxS penalty falls by:

- adaptive: 89.5% at epsilon=.10 and 99.1% at epsilon=.20;
- frozen: 85.3% at epsilon=.10 and 99.3% at epsilon=.20.

The directional prediction is therefore strongly supported:

\[
\epsilon\uparrow
\Rightarrow
\text{weaker rank penalty}
\Rightarrow
\text{weaker homophily-by-segregation learning penalty}.
\]

All 500 frozen-seed HxS contrasts remain positive at epsilon=.10. At
epsilon=.20 the frozen contrast remains positive in 99.6% of seeds, but its
magnitude is nearly eliminated.

---

# 2. Why epsilon matters: lower-rank accessibility, not rank ordering

Frozen Expert ranks do not change with epsilon because frozen source ordering
is fixed. What changes is the acquisition probability attached to each rank.

For d=2, four sources are available. Expert rank distributions in the high-H,
high-S frozen condition are approximately:

- rank 1: 6.9%;
- rank 2: 23.0%;
- rank 3: 70.1%.

The probability that the Expert is sampled at least once within a 20-request
state-learning period changes sharply:

- epsilon=.05:
  - rank 2: .622;
  - rank 3: .046;
- epsilon=.10:
  - rank 2: .848;
  - rank 3: .165;
- epsilon=.20:
  - rank 2: .969;
  - rank 3: .478.

Using the fixed high-H rank distribution, the approximate probability that a
citizen samples the Expert at least once rises from about .245 at epsilon=.05
to .380 at epsilon=.10 and .627 at epsilon=.20.

This explains why terminal mean Expert acquisition probability changes only
moderately while learning improves dramatically. The important mechanism is
whether lower-ranked corrective information remains behaviorally reachable at
all, not only its average acquisition share.

---

# 3. Degree robustness strongly supports congenial crowding-out

At d=4 and epsilon=.05, the HxS MSE interaction becomes:

- adaptive: +1.528973;
- frozen: +5.514216.

Relative to canonical d=2, this is approximately:

- 108 times larger under adaptive reliance;
- 32 times larger under frozen reliance.

High-H/high-S terminal MSE reaches:

- adaptive: 1.541;
- frozen: 5.593.

Low segregation remains near zero-error in the same d=4 topology, so the
degree result is not a generic loss of learning from adding peers. The severe
failure appears specifically when high peer multiplicity combines with
segregated priors and homophilous opportunity.

---

# 4. Degree deepens Expert rank demotion

Under d=4, epsilon=.05, high-H/high-S frozen Expert ranks at period 0 are:

- rank 1: 3.28%;
- rank 2: 3.93%;
- rank 3: 9.47%;
- rank 4: 29.83%;
- rank 5: 53.49%.

Thus about 83.3% of citizens place the Expert at rank 4 or 5.

With six available sources and epsilon=.05, the probability of at least one
Expert request in a 20-request period is approximately:

- rank 1: ~1;
- rank 2: .622;
- rank 3: .046;
- rank 4: .00237;
- rank 5: .000119.

The implied mean probability of sampling the Expert at least once is only about
.062 in the high-H d=4 frozen condition, compared with about .301 under low H.

The d=4 result therefore follows the pre-specified analytical prediction:
additional congenial alternatives push the corrective source geometrically
deeper into the recursive attention hierarchy.

---

# 5. Adaptive re-ranking helps but does not fully overcome d=4 crowding

In high-H/high-S d=4:

- adaptive Expert acquisition begins at .033 at period 0;
- rises to about .098 by period 10;
- remains about .074 at period 399.

Frozen Expert acquisition stays at .033.

Adaptive re-ranking therefore still partially restores corrective access, but
the rank disadvantage created by four homophilous peers is sufficiently deep
that large residual error remains even under adaptive reliance.

This sharpens the interpretation of reconfigurability: it can mitigate rank
displacement, but its effectiveness is constrained by the depth of the
opportunity hierarchy.

---

# 6. Dominant-skeleton closure is downstream, not the primitive mechanism

The epsilon robustness is especially informative about the interpretation of
network closure.

The frozen Lambda top-1 same-group cycle share under high H remains about .641
at both epsilon=.10 and epsilon=.20 because the rank ordering does not change.
Yet the frozen HxS MSE interaction falls from .0258 to .00129 between these two
epsilon values.

Similarly, the W dominant same-group-cycle share in high-H/high-S frozen runs
is approximately .130 at both epsilon=.10 and epsilon=.20, while terminal MSE
falls from .0263 to .00164.

Therefore a dominant-skeleton closure measure cannot be the primitive
explanation of the learning effect.

The more defensible causal ordering is:

\[
\text{opportunity composition}
\rightarrow
\text{source rank depth}
\rightarrow
\text{lower-rank accessibility}
\rightarrow
\text{cumulative evidence allocation}
\rightarrow
\text{learning speed}.
\]

Closed dependence remains an informative network signature of severe
crowding, especially at epsilon=.05 and d=4, but it should not be presented as
the sole mechanism.

---

# 7. Evidence-network consequences remain substantively meaningful

At d=4, high H versus low H strongly changes the evidence network.

High-S adaptive:

- W same-group dominant-cycle share: .085 -> .577;
- W dominant Expert reach: .832 -> .420.

High-S frozen:

- W same-group dominant-cycle share: .247 -> .886;
- W dominant Expert reach: .753 -> .114.

Thus severe rank crowding is translated into an evidence network with very low
corrective reach and strong same-group enclosure. These measures are best
interpreted as downstream network configurations generated by the acquisition
mechanism.

---

# 8. Interpretation gate

The precommitted mechanism gate PASSES.

The results support elevating congenial crowding-out to the primary
Experiment-IV mechanism, with one refinement:

> Homophilous opportunity is consequential because congenial peers can push a
> corrective source deeper into a rank-based attention hierarchy. The effect
> depends on the accessibility of lower-ranked sources, which is controlled by
> epsilon and amplified by peer multiplicity. Adaptive credibility
> reassessment can reverse part of this displacement, while closed acquisition
> and evidence configurations are downstream network manifestations of the
> resulting accessibility pattern.

The next manuscript stage should therefore rewrite the theory around:

1. structural opportunity versus consequential dependence;
2. opportunity-constrained source ranking;
3. congenial crowding-out under limited attention;
4. lower-rank accessibility as a key behavioral parameter;
5. adaptive re-ranking as partial restoration of corrective access;
6. evidence enclosure as a network consequence rather than the primitive
   behavioral cause.

No further receiver redesign is warranted by this robustness exercise.

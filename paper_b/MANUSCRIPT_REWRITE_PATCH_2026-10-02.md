# Social Networks Paper B — Post-Measurement Manuscript Rewrite Patch
## October 2, 2026

This file supersedes the October 1 manuscript-revision patch for substantive
interpretation. The October 1 patch should be retained as an audit-history
artifact because it documents the pre-validity-revision interpretation.

The behavioral evidence base remains canonical production design
`c9d09daad143`. The passive measurement rerun `087afed31ccf` reproduces
that behavior exactly and adds the evidence-precision network (W).

The revised paper should now be organized around:

[
A ightarrow R_t ightarrow Pi_t ightarrow Lambda_t ightarrow W_t
ightarrow 	ext{belief dynamics}.
]

The central question is:

> How do fixed relational opportunities become acquisition patterns and
> evidence-weighting relations, and when do those transformations support
> collective learning or persistent error?

Strategic disinformation remains an application and stress environment, not the
paper's primary theoretical object.

---

# 1. Recommended title direction

The old title over-centers the adaptive Jammer:

> Strategic Disinformation, Adaptive Reliance, and Network Resilience:
> An Agent-Based Model of Collective Belief Formation

The revised manuscript should foreground network dependence.

## Preferred working title

> **From Network Opportunity to Informational Dependence: Adaptive Reliance,
> Evidence Weighting, and Collective Learning**

## Alternative with stress-environment continuity

> **From Network Opportunity to Informational Dependence: Collective Learning
> under Persistent Biased Information**

## Alternative retaining resilience language

> **Network Opportunity, Effective Dependence, and Collective Resilience:
> An Agent-Based Model of Social Learning**

The phrase "strategic disinformation" can remain in the abstract, experiment
description, and secondary sender analysis without defining the title.

---

# 2. Replacement abstract

Social networks define whom citizens can consult, but structural ties do not
determine whose information ultimately shapes belief updating. This paper
develops an agent-based model that separates three layers of informational
dependence: structural opportunity, behavioral acquisition, and the precision
that sampled sources contribute to Bayesian state updating. Citizens repeatedly
evaluate connected information sources and allocate future attention according
to learned credibility. The model therefore transforms a fixed opportunity
network (A) into a behavioral acquisition network (Lambda_t) and an
evidence-precision network (W_t).

Matched simulations identify two distinct network mechanisms. First, when
initial beliefs are segregated, high structural homophily produces much slower
corrective learning under fixed than adaptive reliance. Homophilous acquisition
alone does not explain the difference: low-homophily frozen networks can also
become strongly homophilous while learning accurately. The sharper distinction
appears in the evidence network, where high homophily combined with fixed
reliance generates substantially greater same-group closure and lower dominant
reachability of corrective expert information. Second, increasing corrective
redundancy improves learning and resilience to persistent biased information,
but the associated concentration on corrective gateways is primarily a
structural property of the opportunity network rather than an endogenous
behavioral concentration effect. Acquisition and evidence weighting largely
preserve that structural redirection.

The results show that neither homophily nor concentration has a fixed epistemic
meaning. Their consequences depend on where network structure directs
dependence and on whether reliance can be reconfigured. Reconfigurability is
also conditional: it reduces endogenous closure but can increase exposure to a
persistent biased source. The paper therefore shifts attention from network
connectivity alone to the process by which available ties become consequential
informational dependence.

---

# 3. Introduction — replacement contribution and findings architecture

The manuscript should no longer open the findings section with a Jammer-damage
comparison. The introduction should first establish the measurement problem.

## 3.1 Problem statement

A social network specifies opportunities for contact, not the informational
weight those contacts ultimately receive. A citizen may remain connected to the
same set of sources while changing which sources are sampled, and sampled
sources need not contribute equally to belief updating. This creates a
three-layer distinction:

[
A
ightarrow
Lambda_t
ightarrow
W_t.
]

Here (A) is the structural opportunity network, (Lambda_t) records
behavioral acquisition probabilities across available sources, and (W_t)
records the cumulative share of source precision actually added to citizens'
state updates.

The distinction matters for social-learning models because common network
statistics can change meaning across these layers. A highly homophilous
opportunity network need not generate a closed evidence network. Conversely, a
network can become behaviorally concentrated while improving learning if
dependence is directed toward corrective nodes.

## 3.2 Two network questions

The paper asks two questions.

First, when do homophilous opportunities become a closed informational system?
Experiment IV combines structural homophily with weak or strong prior
segregation and compares adaptive with frozen reliance.

Second, what does corrective redundancy do when accurate information reaches
most citizens through gateway nodes? Experiment III changes local access to
corrective gateways and tests whether the resulting structure improves
learning and resilience.

These experiments separate two conceptually different processes:

- **reconfiguration of dependence within a fixed opportunity structure**;
- **restructuring of the opportunity network itself**.

## 3.3 Revised findings paragraph

Three results organize the paper.

First, structural homophily by itself does not identify informational closure.
Under high prior segregation, adaptive and frozen citizens can both exhibit
strong same-group acquisition, yet frozen reliance produces far greater
residual error. The difference is sharper in the evidence-precision network:
high homophily under frozen reliance generates substantially more same-group
dominant closure and weaker dominant reachability of corrective expert
information. The same closed configuration remains associated with much slower
corrective convergence through (T=400).

Second, corrective redundancy improves baseline learning and reduces damage
from a persistent biased source under both adaptive and frozen reliance. The
mechanism is primarily structural. High redundancy directs opportunity toward
corrective gateways, and both the acquisition and evidence networks largely
preserve that structural redirection. Concentration on gateways is therefore
not an emergent pathology; it is part of the beneficial architecture being
manipulated.

Third, reconfigurability has a threat-dependent value. Adaptive reliance
substantially limits endogenous closure in the homophily experiment, but the
same openness can increase total error when a persistent biased source remains
available and repeatedly supplies fresh evidence. Reconfigurability is thus not
uniformly beneficial or harmful.

## 3.4 Revised contribution paragraph

The paper makes three contributions.

**First, it distinguishes structural opportunity, acquisition, and evidence
weighting.** The same network can look different depending on whether edges
represent who can be consulted, who is behaviorally sampled, or whose evidence
actually contributes posterior precision.

**Second, it identifies closure and reconfigurability as distinct from
homophily.** Same-group acquisition can be high without severe learning
failure. The more consequential configuration is persistent closure in the
evidence network, especially when segregated priors and fixed reliance align.

**Third, it shows that network concentration has no invariant epistemic sign.**
Concentration can sustain within-group closure, but concentration on corrective
gateways can improve both learning and resilience. The effect of concentration
depends on where structural opportunities direct dependence.

---

# 4. Theory section — revised backbone

## 4.1 Structural opportunity

Let (A_{ij}in{0,1}) indicate whether source (j) is available to citizen
(i). (A) constrains possible information acquisition but does not imply
equal or persistent use of those opportunities.

A uniform-use structural benchmark is:

[
U^A_{ij}
=
rac{A_{ij}}{sum_k A_{ik}}.
]

All structural-to-effective comparisons should use a clearly matched
denominator.

## 4.2 Credibility ranking and acquisition

Citizens maintain credibility beliefs that induce a source ranking (R_t).
An epsilon-greedy acquisition policy (Pi_t) translates that ranking into
probabilities of sampling structurally available sources.

The behavioral acquisition network is:

[
Lambda_t=AodotPi_t.
]

(Lambda_t) describes expected acquisition, not causal influence and not the
precision entering Bayesian updates.

## 4.3 Evidence-precision network

Under the final receiver specification, repeated messages from one peer during
a period are treated as repeated expressions of one posterior opinion rather
than independent observations of truth.

For a sampled peer (j),

[
q_{ij,t}
=
rac{1}{sigma_{j,t}^2+	au_{	ext{social}}^2}.
]

For an elite or biased source that supplies (n_{is,t}) independent messages,

[
q_{is,t}
=
rac{n_{is,t}}{sigma_s^2}.
]

Define cumulative evidence precision:

[
Q_{ij,T}
=
sum_{t=0}^{T-1}q_{ij,t},
]

and normalized evidence dependence:

[
W_{ij,T}
=
rac{Q_{ij,T}}{sum_k Q_{ik,T}}.
]

(W) measures each source's share of precision actually added to the citizen's
Gaussian state updates. It is not a causal influence network because it does
not encode message direction or counterfactual belief change.

The paper should state explicitly that elite/source messages and peer
statements use different evidence semantics. Elite messages are modeled as
fresh independent observations; repeated peer statements are source-level
expressions of one current posterior belief.

## 4.4 Four network properties

The revised theory should separate four properties:

1. **structural composition** — who is available;
2. **acquisition composition** — who is sampled;
3. **evidence closure/concentration** — whose precision enters updating;
4. **reconfigurability** — whether the identity of consequential sources can
   change over time.

Homophily is one dimension of composition. It is not synonymous with closure.

## 4.5 Core propositions/conjectures

### C1. Opportunity and informational dependence are distinct

Network statistics measured on (A), (Lambda), and (W) need not coincide.
Claims about informational dependence should therefore be evaluated at the
network layer corresponding to the proposed mechanism.

### C2. Homophily becomes costly when it supports persistent closed dependence

High structural homophily should be most consequential when prior beliefs are
segregated and reliance is difficult to reconfigure. Under that combination,
same-group opportunities can be translated into persistent closed evidence
chains that slow corrective learning.

This conjecture concerns **closure**, not homophily alone.

### C3. Corrective redundancy works primarily through structural redirection

When accurate information is concentrated among corrective gateways, adding
routes to those gateways can improve learning by changing where structural
opportunity points. Acquisition and evidence weighting may reinforce that
architecture, but the gateway-concentration contrast need not be generated
endogenously by adaptive behavior.

### C4. Reconfigurability has threat-dependent value

Reconfigurability should reduce error when the main threat is endogenous closed
dependence. The same reconfigurability can raise exposure when an external
biased source persistently remains available and supplies repeated apparently
fresh evidence.

This is a comparative statement about information environments, not a
normative ranking of adaptive and frozen behavior.

---

# 5. Methods — final receiver and control specification

## 5.1 Peer evidence semantics

Replace any pooled-message variance formulation in the manuscript with the
final source-level update.

For peer (j), all messages from that peer in a state-learning period enter as
one source-level observation with:

[
mu^{obs}_{j,t}
=
overline{m}_{j,t},
]

[
V^{obs}_{j,t}
=
sigma_{j,t}^2+	au_{	ext{social}}^2.
]

The canonical value is:

[
	au_{	ext{social}}=1.
]

This prevents social certainty from recursively becoming arbitrarily precise
while preserving the informational content of peer beliefs.

## 5.2 Elite/source evidence

Elite and biased-source messages retain the independent-message interpretation.
If source (s) is sampled (n) times in the period and has unit message
variance, its observation variance is (1/n).

This difference is deliberate and should be explained as a distinction between
fresh source signals and repeated expressions of a peer's current opinion.

## 5.3 Sender regimes

Use the following terminology.

- **null:** inert available source slot; always behaviorally ranked last;
  epsilon-tail acquisition can be wasted but no state evidence is supplied.
- **fixed biased:** persistent source centered at the biased source position.
- **myopic adaptive sender:** current audience-responsive Jammer; secondary
  boundary condition.
- **truth clone:** historical compatibility condition only; not the primary
  control.

The manuscript should not call the adaptive sender forward-looking or
dynamically strategic.

## 5.4 Frozen reliance

Frozen reliance is initialized from the same pre-disruption source assessment
used by adaptive citizens and then kept fixed.

This replaces the old first-audit frozen condition that was contaminated by the
period-0 adaptive-sender shock.

## 5.5 Horizon and simulation sample

Canonical production uses:

[
N=100,qquad T=400,
]

with 500 matched seeds 6001--6500.

The longer horizon was chosen through a precommitted (T=200) versus (T=400)
calibration on the same stochastic trajectories. (T=200) failed the
precommitted stability gate in selected primary cells.

## 5.6 Measurement architecture

Report network measures in three layers.

### Structural benchmark (A)

Use equal allocation across structurally available sources.

### Acquisition network (Lambda)

Report expected acquisition probabilities and clearly label them behavioral
reliance/acquisition rather than influence weights.

### Evidence network (W)

Report cumulative source shares of precision added to state updates.

For gateway concentration and HHI, use matched denominators when comparing
(A), (Lambda), and (W). In Experiment III, report both:

- all-source gateway share;
- peer-conditioned gateway share.

Top-1 skeleton measures do not receive an arbitrary structural uniform-use
baseline because equal structural weights create ties.

---

# 6. Results architecture

The main Results should be reordered. Experiment IV should come before
Experiment III because it provides the cleaner transformation/reconfigurability
result.

---

## 6.1 Measurement check: A, Lambda, and W are different objects

Open Results with a short measurement result, not a substantive experiment.

The passive measurement rerun reproduces the canonical behavioral simulations
exactly: all run-level, terminal-belief, belief-checkpoint, and
Lambda-checkpoint differences are zero within numerical precision. The null
slot is last-ranked for all citizens in all null conditions.

The comparison confirms that acquisition shares and evidence shares can differ
substantially. For example, in Experiment IV high-segregation null conditions,
direct Expert acquisition is approximately .085--.100 under adaptive reliance,
whereas the Expert accounts for approximately .35 of cumulative evidence
precision. In Experiment III, direct Expert acquisition is similarly smaller
than the Expert's cumulative evidence share.

This establishes why Lambda should not be interpreted as the belief-update
influence network.

---

## 6.2 Experiment IV: rigid evidence closure slows corrective learning

### Baseline learning result

Under null information and low prior segregation, homophily has little
substantive effect.

Under high prior segregation, terminal MSE is:

| Reliance | Low H | High H |
|---|---:|---:|
| Adaptive | .00090 | .01511 |
| Frozen | .00118 | .17606 |

The canonical (H	imes S) MSE interaction is:

[
0.01418
]

under adaptive reliance and

[
0.17484
]

under frozen reliance.

The adaptive-minus-frozen interaction is approximately:

[
-0.16066.
]

The main interpretation is not non-learning. High-H/high-S frozen MSE declines
from approximately .273 around (T=100) to .176 by (T=400). Fixed reliance
therefore produces **substantially slower corrective learning over the observed
horizon**.

### Structural, acquisition, and evidence homophily

Under high segregation:

| H | Reliance | (H^A) | (H^Lambda) | (H^W) |
|---|---|---:|---:|---:|
| Low | Adaptive | .501 | .517 | approximately .52 |
| High | Adaptive | .900 | .916 | approximately .92 |
| Low | Frozen | .501 | .744 | .724 |
| High | Frozen | .900 | .988 | .981 |

The adaptive high-minus-low homophily contrast is therefore almost entirely
structural. Frozen reliance also increases same-group dependence strongly even
in low-H networks.

This result rules out a simple account in which the level of homophily itself
explains the high-H/high-S learning failure.

### Evidence closure

The sharper difference appears in the dominant evidence-precision skeleton.

In high-segregation conditions, the high-minus-low same-group closure contrast
is approximately:

[
+0.115
]

for adaptive reliance and

[
+0.388
]

for frozen reliance.

Dominant Expert reachability in the evidence network is approximately:

| Reliance | Low H | High H |
|---|---:|---:|
| Adaptive | .966 | .869 |
| Frozen | .965 | .577 |

The high-H/high-S frozen evidence network also becomes progressively more
closed over time: same-group evidence-cycle share rises from roughly .18 at
(T=25) to roughly .42 by (T=400), while dominant Expert reach declines from
about .82 to .58.

These configurations accompany the slower convergence under fixed reliance.
They are descriptive network mechanisms, not causal mediation estimates.

### Main Exp IV conclusion

> Homophily is not sufficient for learning failure. Severe residual error
> emerges when segregated priors, high structural homophily, and inflexible
> reliance jointly produce a persistent closed evidence configuration.

---

## 6.3 Experiment III: corrective redundancy is primarily a structural network result

Experiment III restricts direct Expert access to a fixed gateway subset and
manipulates the number of local corrective routes.

### Baseline learning

The high-minus-low redundancy contrast in null MSE is approximately:

[
-0.00677
]

under adaptive reliance and

[
-0.00960
]

under frozen reliance.

Thus redundancy improves baseline learning under both reliance modes.

### Structural gateway redirection

Using the peer-conditioned denominator, the high-minus-low gateway-share
contrast is:

[
+0.450
]

in the structural uniform-use benchmark.

The corresponding contrasts are approximately:

[
+0.479
]

in adaptive (Lambda),

[
+0.471
]

in frozen (Lambda),

[
+0.470
]

in adaptive (W), and

[
+0.478
]

in frozen (W).

The behavioral and evidence layers therefore add little to the structural
contrast. The main mechanism is that the high-redundancy topology itself
redirects opportunity toward corrective gateways, and subsequent acquisition
and evidence weighting largely preserve that redirection.

Peer-only incoming HHI leads to the same conclusion: the increase in
concentration is largely present in (A) before behavioral selection.

### Resilience to persistent biased information

Fixed-biased incremental MSE damage is approximately:

Adaptive:
[
11.12ightarrow9.35
]

from low to high redundancy.

Frozen:
[
9.36ightarrow7.66.
]

The high-minus-low damage contrasts are approximately:

[
-1.765
]

adaptive and

[
-1.695
]

frozen.

Corrective redundancy therefore improves both baseline learning and resilience,
but not because adaptive reliance uniquely creates gateway concentration.

### Main Exp III conclusion

> Corrective redundancy beneficially restructures access toward corrective
> gateways. Acquisition and evidence dependence largely realize a structural
> advantage already encoded in the opportunity network.

---

## 6.4 Reconfigurability depends on the information environment

This subsection compares total error rather than only incremental sender
damage.

### Experiment IV high-H/high-S

Null:
[
MSE_A=.0151,qquad MSE_F=.1761.
]

Fixed biased:
[
MSE_A=6.649,qquad MSE_F=6.502.
]

### Experiment III low redundancy

Null:
[
MSE_A=.0161,qquad MSE_F=.0178.
]

Fixed biased:
[
MSE_A=11.136,qquad MSE_F=9.376.
]

### Experiment III high redundancy

Null:
[
MSE_A=.00937,qquad MSE_F=.00817.
]

Fixed biased:
[
MSE_A=9.364,qquad MSE_F=7.672.
]

The correct conclusion is not that adaptive reliance is universally resilient.

> Reconfigurability is valuable when the dominant threat is endogenous closed
> dependence, but the same openness can increase exposure to persistent biased
> information.

The evidence-network interpretation reinforces this result because persistent
elite-like biased messages can contribute substantial cumulative precision even
when their acquisition share is smaller.

---

## 6.5 Persistent biased-source damage is subgroup-specific

In high-segregation Experiment IV, fixed-biased damage is concentrated almost
entirely in the initially aligned positive group.

Under adaptive reliance:

Low H:
[
D_{-}approx .232,qquad D_{+}approx12.615.
]

High H:
[
D_{-}approx .058,qquad D_{+}approx13.209.
]

The main text should include one small panel or concise paragraph making this
heterogeneity visible. Population-average damage around 6--7 should not be
interpreted as moderate uniform movement of the entire population.

---

## 6.6 Secondary boundary condition: myopic adaptive sender

The adaptive sender should be moved out of the main theoretical sequence.

Its early message remains extremely large under high initial uncertainty,
which rapidly damages credibility. The appropriate interpretation is:

> A myopic sender objective that ignores future credibility can
> self-undermine under high initial audience uncertainty.

Do not generalize this result into a broad claim that persistent bias is more
important than strategy in general.

Detailed sender trajectories belong in the Supplement.

---

# 7. Discussion — replacement architecture

## 7.1 Opportunity is not dependence

The first discussion point should return directly to the network contribution.

A structural graph answers who may be consulted. It does not determine who is
sampled, and sampling does not uniquely determine the evidence weight entering
belief updates.

The paper therefore distinguishes:

[
A,qquadLambda,qquad W.
]

This is not terminological refinement. Experiment III shows a case in which a
major network contrast is mostly structural, while Experiment IV shows a case
in which the consequential distinction emerges in the evidence configuration
generated within the same class of opportunities.

## 7.2 Homophily is not closure

The second point is that high same-group dependence is not sufficient to imply
poor learning.

Frozen low-H citizens develop substantial same-group acquisition and evidence
homophily but converge accurately. High-H/high-S frozen citizens instead
combine near-complete same-group evidence dependence with persistent
same-group closure and weakened corrective reachability.

The theoretically relevant distinction is therefore between **composition**
and **closure**.

## 7.3 Concentration has no invariant epistemic sign

Experiment III shows that a network can become structurally more concentrated
on a corrective gateway class and learn better.

This does not contradict theories in which excessive social influence can
reduce wisdom under equal-quality signals. The present model changes the
informational role of the nodes toward which opportunity is concentrated.

The general implication is:

> concentration should be evaluated jointly with the informational position
> of the nodes receiving dependence.

## 7.4 Reconfigurability is conditional

The third discussion point concerns flexibility.

Adaptive reliance sharply limits endogenous closure in Experiment IV. Yet fixed
biased information produces lower total MSE under frozen than adaptive reliance
in several cells.

The value of reconfigurability therefore depends on the threat environment:

[
	ext{closure risk}
quad	ext{versus}quad
	ext{persistent-source exposure}.
]

This tradeoff is a more defensible interpretation than calling adaptation
uniformly resilient.

## 7.5 Strategic disinformation as application

Strategic or biased senders remain useful because they stress-test the network
process, but they should not organize the manuscript.

The paper's network contribution survives even in null environments:

- Exp IV identifies differential convergence and evidence closure;
- Exp III identifies a beneficial structural redundancy effect.

Sender analyses then show how these network configurations matter under
persistent biased information.

---

# 8. Conclusion — replacement text

Structural ties do not by themselves determine informational dependence. In
the model, a fixed opportunity network is translated first into behavioral
acquisition and then into the precision that sampled sources contribute to
belief updating. These transformations can preserve structural patterns,
amplify them, or create qualitatively different dependence configurations.

The two experiments illustrate different sides of this process. In the
homophily experiment, severe residual error is not explained by same-group
acquisition alone. It appears when segregated priors and high structural
homophily are combined with inflexible reliance, producing a progressively
closed evidence network and slower corrective learning. In the redundancy
experiment, improved learning is primarily structural: additional corrective
routes redirect opportunity toward expert-connected gateways, and both
acquisition and evidence weighting largely preserve that advantage.

These results also qualify simple interpretations of flexibility and
concentration. Reliance reconfigurability limits endogenous closure but can
increase exposure to persistent biased information. Concentration can be
harmful when it traps dependence within closed groups, but beneficial when
network structure directs dependence toward corrective nodes.

The broader implication is that network resilience cannot be read from topology
alone or from behavioral sampling alone. What matters is the sequence through
which available relationships become consequential evidence. A network's
epistemic function depends on where opportunity points, how citizens allocate
attention, how sampled information enters belief updating, and whether those
patterns of dependence can be reconfigured over time.

---

# 9. Main-text figure and table architecture

Do not rebuild final figures until the manuscript uses this architecture.

## Figure 1 — Conceptual measurement architecture

Panels:

1. structural opportunity (A);
2. acquisition network (Lambda);
3. evidence-precision network (W).

Use a small stylized ego/network example showing that the same opportunity
edges can receive different acquisition and evidence weights.

## Figure 2 — Experiment IV: learning and evidence closure

Recommended panels:

A. terminal MSE for low/high H x low/high S, adaptive/frozen;
B. high-S (H^A), (H^Lambda), (H^W);
C. high-S same-group closure in Lambda and W;
D. high-H/high-S MSE and W-Expert reach trajectory through T=400.

This is the main mechanism figure.

## Figure 3 — Experiment III: structural corrective redundancy

Recommended panels:

A. structural peer-conditioned gateway share (A) versus Lambda versus W;
B. null terminal MSE low/high redundancy;
C. fixed-biased total MSE or sender damage low/high redundancy;
D. optional peer-only incoming HHI showing that the concentration contrast is
   largely structural.

## Figure 4 — Information-environment tradeoff

A compact total-loss figure:

[
{	ext{null},	ext{fixed biased}}
	imes
{	ext{adaptive},	ext{frozen}}.
]

Include Exp IV high-H/high-S and Exp III low/high redundancy.

A small final panel may show Exp IV group-specific fixed-biased damage.

## Table 1 — Model and final validation specification

Include:

- N=100;
- T=400;
- epsilon=.05;
- credit=20;
- K=1;
- tau_social=1;
- source_posterior;
- pre_disruption frozen;
- null/fixed/adaptive sender semantics;
- 500 canonical matched seeds.

## Table 2 — Main matched-seed results

Include:

- Exp IV HxS MSE interaction adaptive/frozen;
- adaptive-minus-frozen interaction;
- W same-group closure contrast;
- W Expert-reach contrast;
- Exp III high-minus-low null MSE;
- structural, Lambda, and W gateway contrasts;
- fixed-biased damage contrast.

Report mean, MCSE, Monte Carlo interval, median, and sign share where useful.

---

# 10. Supplement architecture

Move the following out of the main narrative:

1. validity-revision history;
2. tau=0 versus tau=1 social-evidence-floor diagnostic;
3. T=200 versus T=400 calibration;
4. truth-clone compatibility;
5. full posterior-SD trajectories;
6. Lambda versus W checkpoint tables;
7. adaptive-sender trajectory and period-0 self-undermining;
8. RMSE/MAE alternatives;
9. full MSE displacement/variance decomposition;
10. full group-specific damage tables;
11. Experiment IIIb historical bottleneck/path-independence diagnostic, clearly
    labeled as belonging to the pre-validity specification if retained for
    audit history rather than substantive evidence.

Do not mix pre-validity production numbers into the canonical result tables.

---

# 11. Claims to delete from the October 1 patch

The following October 1 claims are superseded and should not survive the next
draft.

- "effective influence" when the quantity is Lambda acquisition.
- "adaptive reliance turns structural redundancy into resilience."
- "gateway concentration is generated by adaptive reliance."
- "homophily sharply increases strategic-Jammer vulnerability" as the central
  Experiment IV result.
- "same-group reliance exceeds structure" as the main evidence for the Exp IV
  mechanism without distinguishing A, Lambda, and W.
- "cycles cause failure."
- "fixed-biased sources are more harmful because persistence matters more than
  strategy" as a general sender conclusion.
- any use of T=200 as the final production horizon.
- any use of truth-clone as the primary no-adversary baseline.

---

# 12. Immediate editing order

When applying this patch to the LaTeX manuscript, revise in the following
order:

1. title and abstract;
2. theory Sections 2.2--2.4;
3. Methods receiver update and measurement definitions;
4. simulation design and sender/control definitions;
5. Results order: measurement check -> Exp IV -> Exp III -> tradeoff ->
   subgroup capture -> adaptive-sender boundary;
6. Discussion;
7. conclusion;
8. figures/tables;
9. Supplement;
10. final consistency audit for old terminology and superseded numbers.

The manuscript should not be line-edited for prose before this structural
rewrite is complete.

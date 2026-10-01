# Social Networks Paper B — Manuscript Revision Patch
## Results-aligned revision after Experiments III, IIIb, and IV
### October 1, 2026

This file provides replacement manuscript text for the current 31-page draft
"Strategic Disinformation, Adaptive Reliance, and Network Resilience: An
Agent-Based Model of Collective Belief Formation."

The revision preserves the paper's core identity:

> structural network -> adaptive reliance -> effective influence -> collective resilience

The strategic Jammer remains a stress test for that receiver-side network
process rather than the manuscript's primary theoretical object.

Experiment IIIa is retained as a substantive but qualified redundancy result.
Experiment IIIb is treated as a Supplementary Materials diagnostic. Experiment
IV becomes the cleanest main-text mechanism result.

---

# Replacement Abstract

How can a fixed social network generate different levels of resilience to
strategic disinformation? This paper develops an agent-based model in which
citizens repeatedly update beliefs, learn which connected sources appear
credible, and adapt future reliance across those sources. Structural ties
therefore define opportunities for influence, while realized influence depends
on how citizens reweight those opportunities over time. The model embeds these
adaptive citizens in a networked information environment with an unbiased
Expert, peer communication, and a strategic disinformation source that adjusts
messages to audience beliefs. Matched counterfactual simulations isolate two
network mechanisms. First, adding a second local corrective route reduces
incremental disruption on average, but the benefit is not uniquely activated by
adaptive reliance and the high-redundancy structure also carries a higher
no-disruption baseline cost. A cleaner shared-bottleneck versus
vertex-disjoint-path diagnostic produces only a small and heterogeneous average
advantage for independent paths. Second, structural homophily has essentially
no effect on incremental disruption when initial beliefs are not segregated,
but it sharply increases vulnerability when homophily coincides with segregated
priors. In that condition, same-group effective reliance rises above the
structural level even though direct reliance on the disinformation source
remains small. The results show that structural connectivity alone does not
determine resilience. What matters is how available pathways are organized and
how adaptive reliance turns them into an effective network of informational
dependence.

---

# Introduction — replace the current findings/contributions block beginning
# with "The simulation experiments compare..." through the contributions
# paragraph

The analysis proceeds in two stages. The first uses the paper's original
communication environments as benchmark regimes: isolated elite exposure,
sparse peer communication, homophilous peer communication, and an extended
network with both peer ties and direct elite access. These benchmarks show how
complete information environments differ, but they bundle several structural
features at once. The second stage therefore introduces matched
counterfactual experiments that manipulate narrower network properties while
holding opportunity structures and initial conditions fixed across paired
worlds.

The paper focuses on two network questions. The first concerns corrective
redundancy. If accurate information can reach a citizen through more than one
local route, does this reduce the additional damage created by a strategic
disinformation source? The second concerns homophily. Does clustering among
similar citizens increase vulnerability by itself, or does it become
consequential only when social groups are also separated in their prior
beliefs? In both cases, the relevant object is not the structural graph alone
but the effective pattern of dependence generated as citizens repeatedly
reallocate reliance across available sources.

Three findings organize the revised analysis. First, greater local corrective
route multiplicity lowers incremental disruption on average in the matched
redundancy experiment. The result is narrower than a general claim that "more
redundancy improves resilience." The same structural change raises loss when
the Jammer is neutralized, and the redundancy effect is not meaningfully larger
under adaptive than frozen reliance. Second, an additional matched diagnostic
that holds focal neighborhoods and gateway load fixed finds only a modest and
heterogeneous average advantage for internally vertex-disjoint corrective
paths over routes sharing a bottleneck. This diagnostic reduces concern that
the main redundancy result is purely a no-Jammer baseline artifact, but it does
not identify path independence as a robust sufficient mechanism. Third,
homophily has almost no effect on incremental disruption when priors are not
segregated, yet substantially increases vulnerability when homophilous
structure coincides with separated initial beliefs. In that high-homophily,
high-segregation condition, effective same-group reliance becomes more
concentrated than the structural network itself.

The paper makes three contributions. First, it distinguishes relational
opportunity from realized influence. A structural tie determines who can be
sampled; adaptive reliance determines who actually enters belief formation
with consequential weight. Second, it treats network resilience as an
emergent property of a receiver-side learning process rather than as a fixed
attribute of topology. Structural redundancy, bottlenecks, and homophily
matter through the opportunities they create for adaptive dependence, not as
self-executing mechanisms. Third, the paper identifies a sharp interaction
between social structure and prior segregation: homophily is largely
inconsequential for incremental disruption when beliefs are not segregated,
but becomes strongly consequential when social and belief boundaries align.
Together, these results shift the focus from whether a manipulative source
remains structurally connected to whether the network continues to sustain
effective access to corrective information.

---

# Section 2.3 — replace the final two paragraphs of
# "Structural and Effective Influence Networks"

Peer communication can create resilience by increasing the number and
organization of routes through which corrective information remains reachable.
Route multiplicity alone need not be sufficient. Two nominally distinct paths
may depend on the same intermediate gateway and fail together, while two
structurally separated paths can preserve access when reliance becomes
concentrated on one part of the network. The network problem is therefore not
only how many paths exist, but whether those paths provide genuinely distinct
opportunities for effective informational dependence.

The same distinction applies to homophily. A homophilous graph restricts the
set of cross-group opportunities, but structural mixing need not translate
directly into realized influence. When priors are only weakly separated,
within-group communication may still carry information that has already moved
toward the truth. When prior beliefs are segregated, credibility learning can
instead amplify same-group dependence, making the effective network more
homophilous than the underlying opportunity structure. This yields the paper's
central theoretical sequence: structural ties constrain possible information
paths; adaptive reliance selects among those paths; the resulting effective
influence network determines whether corrective information remains capable of
offsetting strategic disruption.

---

# Section 2.4 — revised network conjectures

## C1. Strategic disruption as a stress test

A strategic disinformation source should create greater collective loss than
the otherwise identical no-disruption counterfactual when it can remain
credible enough to enter citizens' information sets. The size of that
incremental loss depends on the receiver-side network through which
alternative information is available.

## C2. Conditional value of corrective-route multiplicity

Additional local routes to corrective information can reduce incremental
disruption when they expand the set of usable alternatives to a strategic
source. The effect need not be monotonic or costless. Additional routes may
share bottlenecks, concentrate opportunity on the same gateways, or alter
learning even when no strategic disruption is present. Corrective redundancy
should therefore be evaluated relative to a matched no-disruption baseline
rather than interpreted as an unconditional improvement in collective
learning.

## C3. Adaptive activation of structural opportunity

If resilience arises specifically because citizens learn to reallocate
reliance across available corrective pathways, a structural redundancy effect
should be larger under adaptive than frozen reliance. A small
adaptive-minus-frozen contrast would instead imply that structural opportunity
itself, rather than adaptive activation of that opportunity, accounts for much
of the observed difference.

## C4. Homophily conditional on prior segregation

Structural homophily should have limited effect on incremental disruption when
social groups are not meaningfully separated in their initial beliefs. Its
effect should increase when homophilous ties coincide with segregated priors,
because credibility learning can reinforce same-group dependence and restrict
the effective reach of corrective information. Under this mechanism,
effective same-group reliance may exceed structural homophily even without
dominant direct reliance on the strategic source.

---

# Section 4 — insert after the existing benchmark simulation design

## 4.X Matched mechanism experiments

The benchmark communication environments compare complete information regimes
and therefore bundle differences in access, redundancy, and clustering. I add
two matched mechanism experiments to isolate narrower relational claims. Both
use $N=100$, a fixed evaluation horizon of $T=200$, and matched random
seeds. For every matched seed, the paired conditions share initial citizen
states and all structural features not assigned to the treatment. The Jammer
remains structurally present in both $J=1$ and $J=0$ conditions; the
$J=0$ counterfactual neutralizes adversarial content rather than deleting the
node. The primary outcome is terminal population mean squared error (MSE)
relative to the true state. Monte Carlo standard errors summarize simulation
precision across matched seeds.

### Experiment III: corrective-route multiplicity

Experiment III uses 500 matched seeds and crosses two local redundancy
conditions, adaptive versus frozen reliance, and active versus neutralized
Jammer states, yielding 4,000 runs. Peer degree is fixed at two. A fixed 10
percent gateway subset has direct Expert access in every cell. For
non-gateway citizens, the low-redundancy condition provides one gateway peer
and one non-gateway peer, yielding one short corrective route. The
high-redundancy condition provides two distinct gateway peers, yielding two
short corrective routes. Gateway citizens themselves retain the same peer
structure across conditions.

For redundancy level $r$ and reliance mode $a$, define incremental
disruption as

$
D_r^a=MSE_T(J=1,r,a)-MSE_T(J=0,r,a).
$

The structural redundancy contrast is
$D_{high}^a-D_{low}^a$. The adaptive-activation contrast is

$
(D_{high}^A-D_{low}^A)
-
(D_{high}^F-D_{low}^F).
$

The first quantity asks whether the high-redundancy structure changes
incremental Jammer damage. The second asks whether any such change depends on
adaptive reallocation of reliance.

### Experiment IV: structural homophily by prior segregation

Experiment IV uses 500 matched seeds in a $2\times2\times2$ design:
low versus high structural homophily, low versus high prior segregation, and
active versus neutralized Jammer states. Reliance is adaptive in every cell,
for a total of 4,000 runs. Each citizen has direct structural access to both
elite sources and exactly two citizen peers in every condition. Fixed group
labels are assigned once per seed and reused across all cells.

Low and high homophily correspond to same-group peer probabilities of .50 and
.90. Prior segregation reuses the same individual residual draw across
conditions. Under low segregation, initial belief means are centered on the
common residual. Under high segregation, the same residual is shifted by
$3G_i$, producing an expected group-mean separation of six units.

Within each homophily-segregation cell define

$
D_{H,S}=MSE_T(J=1,H,S)-MSE_T(J=0,H,S).
$

The primary estimand is the difference in the homophily effect across prior
segregation:

$
[D_{highH,highS}-D_{lowH,highS}]
-
[D_{highH,lowS}-D_{lowH,lowS}].
$

This contrast isolates whether structural homophily becomes more consequential
when it is aligned with segregated priors.

---

# Section 5 — revised Results architecture

## 5.1 Benchmark communication environments

Retain the existing benchmark section in shortened form. Its purpose in the
revised manuscript is descriptive and motivational: isolated exposure produces
the greatest disruption, while peer-enabled environments provide alternative
routes for corrective information. These bundled comparisons motivate the
matched mechanism experiments below but do not identify the independent effect
of redundancy or homophily.

## 5.2 Corrective-route multiplicity changes incremental disruption

Experiment III shows that the high-redundancy structure experiences less
incremental Jammer damage on average. Under adaptive reliance,

$
D_{low}^A=1.399,\qquad D_{high}^A=1.105,
$

so that

$
D_{high}^A-D_{low}^A=-0.294.
$

The Monte Carlo standard error is .093 and the approximate 95 percent Monte
Carlo interval is $[-0.476,-0.112]$. The median contrast is $-0.130$, the
5 percent trimmed mean is $-0.297$, and 55.8 percent of matched seeds produce
a negative contrast. Leave-one-out means remain negative throughout. The
average result is therefore not generated by a single extreme realization.

The effect is smaller and less precise when reliance is frozen. The
high-minus-low contrast is $-0.207$ with MCSE .106 and an approximate 95
percent Monte Carlo interval of $[-0.415,0.002]$. The resulting
adaptive-minus-frozen interaction is

$
-0.088
$

with MCSE .114 and an approximate interval of $[-0.311,0.136]$. The
simulation therefore does not support a strong claim that adaptive reliance is
necessary for, or substantially amplifies, the redundancy effect.

The absolute-loss decomposition further narrows the interpretation. Under
adaptive reliance, high redundancy lowers active-Jammer MSE by only

$
MSE_{high,J=1}-MSE_{low,J=1}=-0.076,
$

while increasing the no-Jammer baseline by

$
MSE_{high,J=0}-MSE_{low,J=0}=+0.218.
$

The $-0.294$ difference in incremental disruption should therefore not be
read as a (-0.294) unconditional improvement in collective learning. About
86 percent of the incremental-disruption contrast is associated with reduced
cross-sectional belief variance rather than a common shift in the population
mean.

This qualification reflects the structural treatment itself. Non-gateway
citizens move from one gateway peer plus one non-gateway peer to two gateway
peers. Route multiplicity, immediate peer composition, and concentration on
the fixed gateway set therefore change together. Experiment III establishes
that this higher-redundancy opportunity structure changes vulnerability; it
does not isolate path multiplicity as a sufficient causal mechanism.

### Supplementary diagnostic: shared bottleneck versus independent paths

A separate matched diagnostic in the Supplement holds each focal citizen's
immediate relay neighborhood and the global gateway-load vector fixed while
rewiring whether two nominal corrective routes share a gateway bottleneck or
reach distinct gateways. The diagnostic was first run on 50 matched seeds and
then extended once, using a disjoint additional set of 50 seeds, to assess
Monte Carlo stability. The cumulative $N=100$ result remains directionally
negative but heterogeneous.

Under adaptive reliance, the cumulative independent-minus-shared contrast is

$
P^A=-0.0335
$

with MCSE .0310 and an approximate 95 percent Monte Carlo interval of
$[-0.094,0.027]$. The median is $-0.0007$, the 5 percent trimmed mean is
$-0.0337$, and 54 percent of seed-level contrasts are negative. The
additional 50-seed cohort is weaker than the original cohort: its mean is
$-0.0137$, median is approximately zero, and 44 percent of seed-level
contrasts are negative.

The no-Jammer difference is essentially zero
($-0.00003$), whereas the active-Jammer difference is $-0.0335$. The
diagnostic therefore reduces concern that the average direction is produced by
a baseline penalty like the one present in Experiment III. It does not,
however, identify path independence as a robust sufficient mechanism. The
focal-citizen contrast is also negative on average
($-0.0607$, MCSE .0510) but remains Monte Carlo-imprecise. With frozen
reliance, the population path contrast is effectively zero. I therefore treat
this exercise as a mechanism check rather than as an additional main-text
result.

## 5.3 Homophily becomes consequential when priors are segregated

Experiment IV produces a sharper conditional network result. When prior
segregation is low, structural homophily has essentially no effect on
incremental disruption:

$
D_{highH,lowS}-D_{lowH,lowS}=-0.0002,
$

with MCSE .0025 and an approximate 95 percent Monte Carlo interval of
$[-0.0051,0.0047]$.

When prior segregation is high, the homophily effect increases to

$
D_{highH,highS}-D_{lowH,highS}=0.5444,
$

with MCSE .0609 and an approximate interval of $[0.425,0.664]$. The primary
homophily-by-segregation interaction is

$
0.5446
$

with MCSE .0611 and the same approximate interval after rounding. The median
interaction is .322, 68.4 percent of matched seeds are positive, the 5 percent
trimmed mean is .484, and leave-one-out means remain between .528 and .556.
The result is therefore not driven by a small number of extreme seeds.

The interaction is concentrated in population dispersion rather than a common
shift. Its MSE decomposition attributes approximately .519 of the .545 mean
interaction to increased cross-sectional belief variance and only .026 to
squared population displacement. The high-homophily, high-segregation cell
also has a much larger no-Jammer baseline MSE than the other cells. The primary
estimand remains incremental Jammer damage, so this baseline learning problem
and the additional vulnerability created by the Jammer should be kept
analytically distinct.

The reliance measures show how the same structural graph can become a more
segregated effective influence network. Mean structural peer homophily is about
.499 in the low-homophily cells and .899 in the high-homophily cells. Under
high prior segregation, effective same-group reliance rises above those
structural values, reaching approximately .587 in the low-H/high-S cell and
.957 in the high-H/high-S cell. By contrast, direct terminal reliance on the
Jammer remains small, approximately .009 and .015 in those two high-segregation
cells. The pattern is therefore more consistent with persistence through a
peer-mediated belief system than with continued dominant direct dependence on
the Jammer. These quantities describe a mechanism correspondence; they are not
a path-specific mediation estimate.

Taken together, Experiment IV supports a conditional claim: structural
homophily by itself does not increase incremental vulnerability in this model,
but it does so strongly when homophilous structure aligns with segregated
priors. Prior segregation changes not only where citizens begin but how
structural opportunities are converted into effective same-group dependence.

---

# Section 6 — replacement Discussion and Conclusion

The revised experiments shift the paper's central claim away from the idea that
particular structural features are intrinsically resilient or fragile.
Structural ties create opportunities for information flow, but those
opportunities acquire influence only through repeated source use. Network
resilience is therefore a property of the joint system formed by relational
architecture and adaptive reliance.

The first contribution is the distinction between structural opportunity and
effective informational dependence. A citizen may remain connected to the same
sources throughout a simulation while changing which of those sources enters
belief formation most often. The same structural graph can therefore support
different realized influence patterns. This distinction matters for strategic
disinformation because nominal reach is not equivalent to sustained influence:
a disruptive source can remain structurally present while corrective peer and
Expert pathways alter the balance of information that citizens actually use.

The second contribution is a more qualified account of redundancy. Experiment
III shows that a structure with two short corrective routes experiences less
incremental Jammer damage on average than a structure with one. That difference
does not establish that adaptive reliance uniquely activates redundancy, and
it should not be interpreted as an unconditional learning benefit. The
high-redundancy condition also has a higher no-Jammer loss, and the treatment
changes immediate peer composition as well as route count. The Supplementary
shared-bottleneck diagnostic sharpens this interpretation. Once immediate
focal neighborhoods and gateway load are held fixed, internally independent
paths retain a small negative average disruption contrast, but the effect is
heterogeneous and Monte Carlo-imprecise across 100 matched seeds. The combined
evidence suggests that corrective-route organization can matter without
supporting a simple rule that more or more-independent paths mechanically
produce resilience.

The third and strongest network-mechanism result concerns the alignment of
social and belief structure. Homophily alone has virtually no effect on
incremental Jammer damage when initial beliefs are not segregated. The same
structural homophily becomes strongly consequential when group identities are
paired with separated prior beliefs. In that condition, effective same-group
reliance exceeds structural homophily, while direct reliance on the Jammer
remains a small share of terminal dependence. The result is consistent with a
peer-mediated persistence mechanism: segregated priors change how citizens
evaluate and reuse information inside homophilous neighborhoods, turning a
structural tendency toward within-group contact into a more concentrated
effective influence network.

This finding also clarifies why homophily should not be treated as a sufficient
explanation for fragmentation. A high proportion of same-group ties can coexist
with low incremental vulnerability when beliefs are not already separated.
What changes under segregation is the interaction between structural
opportunity and receiver-side evaluation. Once within-group messages are more
likely to remain compatible with local beliefs, credibility learning and
repeated source use can reinforce the structural partition.

Several limitations remain. The network structures are stylized and fixed
within runs; the model does not allow citizens to create or sever ties.
Experiment III manipulates a deliberately local gateway structure and therefore
does not identify every dimension of network redundancy. The IIIb diagnostic
shows that removing its main gateway-composition confound does not yield a
large, stable path-independence effect. Experiment IV identifies a clean
homophily-by-prior-segregation interaction, but its effective-reliance measures
are descriptive mechanism evidence rather than causal mediation estimates.
The model also represents political belief as a scalar state and uses a
stylized strategic sender. These simplifications are intended to isolate the
receiver-side network process rather than reproduce a particular media
platform.

The broader implication is that resilience cannot be read directly from a
structural graph. Degree, redundancy, and homophily describe opportunities,
not realized influence. Whether those opportunities protect or expose a
population depends on how citizens allocate reliance among available sources
and on the belief environment in which that allocation occurs. In the model,
the strongest vulnerability appears when structural segregation and prior
segregation reinforce one another and are then reproduced through effective
same-group dependence.

Strategic disinformation is therefore best understood here as a stress test of
an adaptive social-learning system. The core network process is receiver-side:
citizens learn whom to trust, those choices reweight the structural network,
and the resulting effective influence network determines whether corrective
information continues to matter. This perspective shifts the question from
whether a manipulative source can remain connected to when a network preserves
effective pathways for collective correction.

---

# Supplementary Materials — new subsection

## S.X Experiment IIIb: shared bottlenecks and path independence

Experiment III changes both corrective-route multiplicity and the composition
of immediate peer opportunities. To assess whether the main result could be
attributed primarily to gateway concentration, I conducted a separate matched
diagnostic that isolates local path overlap.

Each matched seed assigns 100 citizens to 10 Expert gateways, 40 relay
citizens, and 50 focal citizens. Every focal citizen has the same two immediate
relay sources in both conditions. The shared-bottleneck condition routes those
two relays through the same gateway, whereas the independent-path condition
routes them through distinct gateways. The rewiring preserves each focal
citizen's immediate source set, per-node source degree, elite opportunity, and
the full gateway relay-indegree vector. Each focal therefore has two nominal
corrective routes in both conditions, but one versus two internally
vertex-disjoint routes.

The diagnostic crosses the two path structures with adaptive versus frozen
reliance and active versus neutralized Jammer states. The primary diagnostic
was run on 50 matched seeds (4001-4050). After inspecting that result, I ran one
pre-documented precision extension on a disjoint additional set of 50 seeds
(4051-4100). The original 50-seed classification is retained separately; the
cumulative 100-seed analysis is treated as a post-diagnostic precision check.

For adaptive reliance, the cumulative population contrast is

$
[D_{independent}-D_{shared}]=-0.0335
$

with MCSE .0310 and an approximate 95 percent Monte Carlo interval of
$[-0.094,0.027]$. The median is approximately zero
($-0.0007$), the 5 percent trimmed mean is $-0.0337$, and 54 percent of
seed-level contrasts are negative. The additional 50-seed cohort is weaker
than the original cohort: its mean is $-0.0137$, median is approximately
zero, and 44 percent of seed-level contrasts are negative.

The absolute-loss decomposition is informative. The independent-minus-shared
difference under $J=0$ is approximately $-0.00003$, effectively zero,
whereas the corresponding $J=1$ difference is $-0.0335$. The cumulative
focal-citizen path effect is $-0.0607$ with MCSE .0510 and an approximate
interval of $[-0.161,0.039]$. Under frozen reliance, the population path
contrast is effectively zero. The adaptive-minus-frozen contrast is
$-0.0335$ with MCSE .0310.

The diagnostic therefore serves two purposes. First, it shows that the
directional average observed in the main redundancy experiment is not generated
by a comparable no-Jammer baseline penalty once gateway load and immediate
focal neighborhoods are held fixed. Second, it shows that path independence by
itself is not a robust sufficient mechanism: the average effect is small,
heterogeneous across matched seeds, and Monte Carlo-imprecise. I therefore use
Experiment IIIb as a mechanism check rather than as a separate main-text
claim.

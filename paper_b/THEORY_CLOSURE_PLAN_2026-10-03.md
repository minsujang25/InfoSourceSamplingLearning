# Paper B theory-closure plan
## Frozen before execution — October 3, 2026

This is the final closure package before the v5 Theory rewrite. It contains
three pre-specified tasks. No new behavioral treatment, parameter condition, or
mechanism search is introduced.

## Task 1 — Early evidence-channel decomposition

### Purpose

Test whether the strong terminal relationship between corrective Expert access
and learning is already visible early enough to support a timing-based
interpretation rather than a terminal-dependence interpretation.

### Design

Reuse the already completed W-channel design exactly:

- Experiment 1 / high prior segregation;
- frozen ranking;
- null sender;
- structural homophily low/high;
- peer degree d=2;
- epsilon in {.05, .10, .20};
- C=20;
- N=100, T=400;
- seeds 6001--6500.

The only change is passive measurement: cumulative channel metrics are also
recorded after 10 completed periods. Existing checkpoints at 25, 50, 100, 200,
and 400 remain unchanged.

The rerun must pass the existing behavioral-identity gate against the frozen
canonical/mechanism results at tolerance 1e-12.

Primary diagnostic cells are high-H/high-S frozen/null at T=10 and T=25.

Report:

- absolute cumulative precision Q by Expert, same-group peer, and other-group
  peer channels;
- normalized cumulative precision W by the same channels;
- inclusion frequency I, defined as the fraction of substantive state-learning
  periods in which the channel contributed positive precision.

Terminal T=400 values are retained only as a benchmark.

### Interpretation rule

Use the early-channel result in the main mechanism discussion only if the
epsilon-related increase in corrective access/evidence is already clearly
visible by T=10 or T=25 and points in the same direction as terminal learning.
Do not claim formal temporal mediation. If early patterns are weak or unstable,
retain the diagnostic in the Supplement and keep the existing accessibility
argument.

## Task 2 — Adaptive attenuation surface

For every cell of the frozen/adaptive epsilon-by-degree phase surface, define

    Attenuation(epsilon,d)
      = 1 - I_adaptive(epsilon,d) / I_frozen(epsilon,d),

where I is the H-by-S terminal-MSE interaction.

Also report the absolute reduction

    I_frozen - I_adaptive.

The attenuation ratio is descriptive. It should not replace the raw interaction
surface because ratios become unstable when the denominator is close to zero.

Primary interpretation question:
Does adaptive re-ranking remove a smaller fraction of the frozen penalty in the
deep-crowding / low-accessibility corner of the surface?

This diagnostic belongs in the Supplement unless it materially sharpens the
Theory discussion of reconfigurability.

## Task 3 — Analytical q(m) curve

Let group-center separation from truth be m, with citizen and same-group peer
residuals independently distributed N(0,1):

    mu_i = m + e_i,
    mu_j = m + e_j.

With the Expert at truth (0), define

    q(m) = Pr(|mu_i - mu_j| < |mu_i|),

the probability that a same-group peer is initially closer to the citizen than
the truthful Expert.

Conditioning on e_i gives

    q(m)
      = E_e [
          Phi(e + |m+e|)
          - Phi(e - |m+e|)
        ].

Evaluate q(m) deterministically by numerical quadrature on m in [0,4]. The
implementation must reproduce the anchor values approximately:

- q(0) = .3524
- q(1) = .5485
- q(2) = .7924
- q(3) = .9088

This is an analytical scope bridge, not a new simulation result. It can support
the Theory claim that prior separation continuously changes the probability of
corrective-source rank displacement.

## Final gate

The theory-closure package passes only if:

1. the passive W-channel rerun passes behavioral identity at 1e-12;
2. T=10 and T=25 full Q/W/I rows exist for all epsilon x homophily cells;
3. all 15 epsilon-by-degree attenuation cells are recovered;
4. all frozen H-by-S interactions used as denominators are positive;
5. the q(m) anchor values match the stated numerical targets within 5e-4.

After this package passes, the analysis program is frozen again and the next
task is the Theory rewrite, followed by Introduction and Discussion.

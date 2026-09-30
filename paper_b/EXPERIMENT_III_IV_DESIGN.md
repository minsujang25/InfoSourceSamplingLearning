# Experiments III and IV — matched mechanism designs

This document freezes the computational design for the two mechanism experiments
that follow the four-environment baseline and adaptive-vs-frozen comparison.

The primary evaluation horizon is fixed at **T=200**. The targeted T=400
sensitivity run showed that late-horizon movement is rare and concentrated in a
small number of sparse-network trajectories, so production effort is allocated
to Monte Carlo replication rather than a longer primary horizon.

The primary production seed count is fixed at **500 matched seeds per
experiment**.

## Shared design principles

Both experiments use explicit structural source maps rather than relying on
separate RNG realizations to happen to match.

The model now supports:

- `structural_source_map`: an explicit citizen-to-source opportunity map;
- `fixed_group_ids`: group labels supplied independently of initial beliefs.

These overrides are used only for the matched mechanism experiments. The four
baseline environments retain their existing constructors.

Across J=1 and J=0, the Jammer node always remains structurally present. J=0
neutralizes adversarial content rather than removing the node.

For Experiments III and IV, the Jammer structural slot is universal. This is an
intentional isolation device: adversarial opportunity is held constant while
the experiment manipulates corrective-route redundancy or peer homophily.

Direct Expert access is localized to an exact seed-specific gateway subset.
The default share is 0.10, matching the model's existing
`elite_access_probability` default. This is a design parameter, not an
empirical estimate.

Peer degree is fixed at two.

## Experiment III — corrective-pathway redundancy

### Question

Does structural multiplicity of corrective routes create resilience capacity,
and does adaptive reliance activate that capacity?

### Matched block

```text
(seed, initial-belief regime)
```

The primary production run uses the flat prior regime only.

Within every block:

- the initial belief vector is identical across all conditions;
- the direct Expert gateway set is identical across all conditions;
- Jammer access is universal and identical across all conditions;
- peer degree is exactly two;
- J=1/J=0 share the same structure;
- adaptive/frozen share the same structure.

### Structural manipulation

Let G_E be the fixed set of citizens with direct Expert access.

For citizens not in G_E:

```text
LOW redundancy:
    one peer in G_E
    one peer outside G_E
    -> exactly one distinct length-2 Expert route

HIGH redundancy:
    two distinct peers in G_E
    -> exactly two distinct length-2 Expert routes
```

Citizens already in G_E receive the same two non-gateway peers in both
conditions. Their local structure therefore does not change.

The manipulation is deliberately local: it identifies the effect of adding a
second distinct short corrective route while holding direct elite access and
peer degree fixed. Longer paths may still exist endogenously.

### Factorial structure

```text
2 redundancy levels
x 2 reliance modes (adaptive, frozen)
x 2 Jammer states
= 8 runs per matched seed
```

With 500 seeds:

```text
500 x 8 = 4,000 production runs
```

### Primary estimands

For redundancy level r and reliance mode a:

```text
D_r^a = MSE_T(J=1, r, a) - MSE_T(J=0, r, a)
```

Structural redundancy effect:

```text
D_high^a - D_low^a
```

Activation interaction:

```text
(D_high^A - D_low^A)
-
(D_high^F - D_low^F)
```

The first term asks whether extra structural routes change disruption under
adaptive reliance. The interaction asks whether the redundancy effect depends
on adaptive behavioral activation.

### Production command

```bash
PAPER_B_WORKERS=12 bash scripts/run_paper_b_exp3_production.sh
```

Default seed range:

```text
2001-2500
```

The runner is resumable and writes one atomic shard per matched seed.

## Experiment IV — structural homophily x prior segregation

### Question

Does structural homophily become especially consequential when fixed social
groups are also separated in their initial beliefs?

### Fixed group labels

Citizen group labels are generated once per matched seed, balanced between -1
and +1, and then held fixed across all four H x S cells.

The labels do not change when prior segregation changes.

### Fixed elite access

The same exact Expert gateway subset is reused in every H x S cell.

Jammer access is universal in every cell.

Peer degree remains exactly two.

Therefore the homophily manipulation does not change direct elite opportunity
or degree.

### Homophily manipulation

Low structural homophily:

```text
P(peer drawn from same fixed group) = 0.50
```

High structural homophily:

```text
P(peer drawn from same fixed group) = 0.90
```

Low and high homophily use common random draws and common candidate rankings.
Changing the threshold therefore creates a paired structural counterfactual
rather than two unrelated network realizations.

The runner records realized structural peer homophily for every seed and
requires a clear empirical separation between the two conditions.

### Prior-segregation manipulation

The same individual residual draw is reused across low and high segregation.

For citizen i with fixed group G_i in {-1,+1}:

```text
LOW:
    mu_i,0 = e_i

HIGH:
    mu_i,0 = e_i + 3 G_i

e_i ~ N(0,1)
```

Thus the expected group-mean gap is:

```text
LOW:  0
HIGH: 6
```

while individual stochastic residuals are held constant across the
counterfactual.

### Factorial structure

The frozen primary Experiment IV specification is:

```text
2 homophily levels
x 2 prior-segregation levels
x 2 Jammer states
x adaptive reliance
= 8 runs per matched seed
```

Frozen reliance is not crossed into the primary Experiment IV design because
Experiment II already isolates adaptive-vs-frozen reliance and Experiment III
uses that contrast to identify route activation. Experiment IV focuses on the
conditional structural claim in C4.

With 500 seeds:

```text
500 x 8 = 4,000 production runs
```

### Primary estimand

Within each H x S cell:

```text
D_H,S = MSE_T(J=1,H,S) - MSE_T(J=0,H,S)
```

Homophily effect at segregation S:

```text
D_highH,S - D_lowH,S
```

Primary interaction:

```text
[D_highH,highS - D_lowH,highS]
-
[D_highH,lowS - D_lowH,lowS]
```

This is the clean test of the claim that structural homophily becomes more
consequential when it is paired with segregated priors.

### Production command

```bash
PAPER_B_WORKERS=12 bash scripts/run_paper_b_exp4_production.sh
```

Default seed range:

```text
3001-3500
```

## Why 500 matched seeds

The 20-seed calibration pilot showed that sparse-network disruption is strongly
right-skewed and that several cells have seed-level standard deviations near
3. With only 20 seeds, Monte Carlo standard errors in the noisiest cells were
roughly 0.6-0.7.

Holding that variance scale fixed, increasing from 20 to 500 seeds reduces
Monte Carlo standard error by:

```text
sqrt(20/500) = 0.20
```

so a 0.7 pilot MCSE corresponds to approximately 0.14 at 500 seeds.

The purpose of the 500-seed freeze is not to guarantee a particular inferential
threshold. It is to make Monte Carlo error small relative to the heavy
seed-to-seed variation observed in the sparse structures while keeping the
production grid tractable.

## Automatic gates

Experiment III must satisfy:

- all expected matched blocks present;
- all eight conditions per block;
- finite citizen states;
- exact T=200;
- identical initial states across the block;
- identical elite access across low/high redundancy;
- identical peer degree;
- exactly one vs two length-2 Expert routes for non-gateways;
- frozen Lambda dynamics equal zero.

Experiment IV must satisfy:

- all expected matched blocks present;
- all eight H x S x J conditions per block;
- finite citizen states;
- exact T=200;
- fixed group labels across cells;
- identical Expert access across cells;
- identical structure across low/high segregation at a given H;
- identical initial state across low/high homophily at a given S;
- clear realized low/high structural homophily separation;
- clear realized low/high prior-segregation separation.

Both runners create a compact shareable ZIP after the gate passes.

## CI status

GitHub CI runs deterministic structural checks plus one-seed end-to-end smoke
runs for both experiments. The smoke tests verify the complete pipeline without
running the 500-seed production grids.

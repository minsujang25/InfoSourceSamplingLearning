# Experiments III and IV — Final 500-Seed Production Audit

## Status

The canonical post-revision 500-seed production grids have completed under the
posterior-uncertainty Jammer and `MIN_SD=1e-8`.

Production IDs:

```text
Experiment III: 3e97dd95138b
Experiment IV:  8c6fca9d5694
```

Both grids use N=100 and T=200.

This audit separates computational validity from substantive interpretation.
Experiment IV cleanly supports the conditional homophily result. Experiment III
shows a stable redundancy contrast but does **not** support a strong claim that
adaptive reliance uniquely activates redundancy; moreover, much of the primary
Delta-MSE contrast reflects a change in the J=0 baseline.

## Computational gates

### Experiment III

```text
500 matched seeds
x 2 redundancy
x 2 reliance
x 2 Jammer
= 4,000 runs
```

Gate:

- all 4,000 runs finite;
- all 4,000 runs reach T=200;
- frozen Lambda invariant;
- exact mean local two-step Expert routes: low=1, high=2;
- max terminal MSE = 26.276;
- max |terminal citizen belief| = 6.379;
- max |Jammer message mean| = 53.462;
- max individual response gain = 25/26 = 0.961538;
- min Jammer objective denominator = 51/676 = 0.0754438;
- terminal SD-floor share = 0.5428.

The largest-MSE trajectories plateau before T=200. For example, seed 2065,
low-redundancy/frozen/J=1 has MSE 25.918 at T=50, 26.276 at T=100, and
26.276 at T=200. No pre-revision Jammer runaway reappears.

### Experiment IV

```text
500 matched seeds
x 2 homophily
x 2 prior segregation
x 2 Jammer
= 4,000 runs
```

Gate:

- all 4,000 runs finite;
- all 4,000 runs reach T=200;
- homophily manipulation gate passes;
- segregation manipulation gate passes;
- mean realized structural homophily = 0.4986 / 0.8986;
- mean realized prior segregation = 0.1599 / 6.0068;
- max terminal MSE = 15.255;
- max |Jammer message mean| = 53.133;
- max individual response gain = 25/26 = 0.961538;
- min Jammer objective denominator = 51/676 = 0.0754438;
- terminal SD-floor share = 0.7884.

The largest-MSE trajectory is stable rather than explosive. Seed 3293,
high-H/high-S/J=1 has MSE 15.319 at T=50, 15.289 at T=100, and 15.255 at
T=200.

## Experiment III — redundancy

Define

```text
D_r^a = MSE(J=1,r,a) - MSE(J=0,r,a).
```

### Cell disruption

```text
                         low redundancy     high redundancy
adaptive D                   1.399              1.105
frozen D                     1.090              0.883
```

### Redundancy effect

```text
Adaptive: D_high - D_low = -0.294
MCSE                       =  0.093
approx. 95% MC interval    = [-0.476, -0.112]
median                     = -0.130
5% trimmed mean            = -0.297
negative in                = 55.8% of seeds
leave-one-out mean range   = [-0.309, -0.258]

Frozen: D_high - D_low     = -0.207
MCSE                       =  0.106
approx. 95% MC interval    = [-0.415, +0.002]
median                     ~= 0
negative in                = 52.0% of seeds
```

The adaptive contrast is robust to tail trimming and no single seed controls
the mean.

### Absolute-loss decomposition

For adaptive reliance:

```text
high - low MSE under J=1 = -0.076
MCSE                      =  0.106

high - low MSE under J=0 = +0.218
MCSE                      =  0.036
```

Thus the -0.294 Delta-MSE redundancy effect is not equivalent to a -0.294
improvement in active-Jammer performance. The high-redundancy structure
slightly lowers mean J=1 MSE but also materially raises the no-Jammer
baseline.

The same distinction is visible on transformed loss:

```text
Adaptive high-low:
                    J=0       J=1       Delta(J1-J0)
MAE                +0.067     -0.039       -0.106
RMSE               +0.127     -0.080       -0.207
```

The active-Jammer RMSE improvement is modest, while the no-Jammer penalty is
clear.

### MSE decomposition

Adaptive disruption:

```text
low redundancy:
    Delta squared displacement = 0.419
    Delta belief variance      = 0.980
    Delta MSE                  = 1.399

high redundancy:
    Delta squared displacement = 0.378
    Delta belief variance      = 0.727
    Delta MSE                  = 1.105
```

The adaptive high-minus-low Delta-MSE effect decomposes to:

```text
squared displacement = -0.041
belief variance      = -0.253
total                = -0.294
```

About 86% of the mean contrast is therefore associated with the belief-variance
component.

### Adaptive activation interaction

```text
[(D_high-D_low)_adaptive - (D_high-D_low)_frozen]

mean                       = -0.088
MCSE                       =  0.114
approx. 95% MC interval    = [-0.311, +0.136]
median                     = -0.075
negative in                = 55.0% of seeds
5% trimmed mean            = -0.134
leave-one-out mean range   = [-0.127, -0.063]
```

On transformed losses:

```text
RMSE activation interaction = -0.093
MAE activation interaction  = -0.027
```

Neither provides evidence for a large adaptation-specific amplification of the
redundancy effect.

The interaction is also stable over the late fixed horizon:

```text
T=50   -0.056
T=100  -0.079
T=150  -0.085
T=200  -0.088
```

### Interpretation of Experiment III

The production run supports a narrow statement:

> In this structural manipulation, adding a second local corrective route
> reduces incremental Jammer damage on average under adaptive reliance.

It does **not** support the stronger statement:

> Adaptive reliance is necessary for, or strongly amplifies, the resilience
> benefit of redundancy.

The manipulation also changes the composition of immediate peer opportunities:
non-gateway citizens move from one gateway peer plus one non-gateway peer to two
gateway peers. This concentrates peer opportunities on the fixed 10% gateway
set. Because high redundancy raises J=0 loss, path multiplicity should not be
equated with an unconditional improvement in collective learning.

This result is consistent with the theory's existing caveat that multiplicity
alone can be insufficient when corrective paths are correlated or bottlenecked,
but a stronger causal claim about independent route redundancy would require a
design that holds gateway concentration/composition more tightly fixed.

## Experiment IV — homophily x prior segregation

Define

```text
D_H,S = MSE(J=1,H,S) - MSE(J=0,H,S).
```

### Cell disruption

```text
                         low S           high S
low H                    0.0107           0.3295
high H                   0.0105           0.8739
```

Homophily effect at low segregation:

```text
D_highH,lowS - D_lowH,lowS = -0.0002
MCSE                        =  0.0025
approx. 95% MC interval     = [-0.0051, +0.0047]
```

Homophily effect at high segregation:

```text
D_highH,highS - D_lowH,highS = +0.5444
MCSE                          =  0.0609
approx. 95% MC interval       = [+0.425, +0.664]
```

Primary H x S interaction:

```text
mean                       = +0.5446
MCSE                       =  0.0611
approx. 95% MC interval    = [+0.425, +0.664]
median                     = +0.3221
positive in                = 68.4% of seeds
5% trimmed mean            = +0.4837
10% trimmed mean           = +0.4463
leave-one-out mean range   = [+0.528, +0.556]
```

The interaction is therefore not an outlier-driven mean.

### Absolute baseline and active-Jammer loss

Mean MSE:

```text
                         J=0             J=1
low H, low S            0.0081          0.0188
high H, low S           0.0097          0.0202
low H, high S           0.4633          0.7928
high H, high S          3.1990          4.0729
```

High homophily plus segregated priors therefore generates both a large baseline
learning problem and an additional increase in vulnerability to the Jammer.
These are analytically distinct quantities and should be reported separately.

### MSE decomposition

The H x S interaction decomposes to:

```text
squared population displacement = +0.0255
belief variance                  = +0.5191
total MSE                        = +0.5446
```

Approximately 95% of the mean interaction is associated with increased
cross-sectional belief variance rather than a common population shift.

### Effective-reliance mechanism

Mean terminal J=1 quantities:

```text
                         H_A      H_Lambda   peer rel.  Expert rel. Jammer rel.
low H, low S            .499       .497       .948       .052       .0003
low H, high S           .499       .587       .928       .063       .0085
high H, low S           .899       .899       .948       .052       .0004
high H, high S          .899       .957       .908       .077       .0152
```

Prior segregation therefore pushes effective same-group reliance above the
structural benchmark, especially in the high-H/high-S cell. The Jammer's direct
terminal reliance remains a minority share; the disruption pattern is
consistent with persistence through the peer-mediated belief system rather than
continued dominant direct Jammer dependence.

This is a descriptive mechanism correspondence, not a path-specific mediation
estimate.

### Outcome-scale robustness

The primary estimand remains MSE.

The H x S interaction on MAE is also positive:

```text
mean = +0.152
MCSE =  0.016
median = +0.092
positive in 70.4% of seeds
```

The direct interaction on RMSE is much smaller:

```text
mean = +0.016
MCSE =  0.022
```

This compression is expected because RMSE is a nonlinear square-root
transformation and the high-H/high-S cell already has a much larger J=0
baseline. RMSE is therefore retained as a presentation metric rather than used
to redefine the factorial estimand.

### Fixed-horizon stability

The mean MSE H x S interaction is:

```text
T=50   0.528
T=100  0.539
T=150  0.541
T=200  0.545
```

The aggregate comparative static is stable at the fixed T=200 evaluation
horizon.

## Production-level theoretical assessment

### C4

Experiment IV provides clear support for the conditional claim:

> Structural homophily by itself has essentially no effect on incremental
> disruption when priors are not segregated, but it substantially increases
> vulnerability when structural homophily coincides with segregated prior
> beliefs.

The effective-reliance results further support the paper's distinction between
structural opportunity and effective informational dependence.

### C3

Experiment III requires narrower wording.

The simulation shows that extra local corrective-route multiplicity can reduce
incremental adversarial damage. It does not show that adaptive reliance is
necessary to activate that effect, and the high-redundancy structure carries a
no-Jammer learning cost.

Accordingly, the manuscript should not present Experiment III as a clean
demonstration that "more redundancy improves resilience through adaptation."
The defensible production result is that structural route multiplicity changes
vulnerability, while its value depends on how those routes are organized and
used. A cleaner independence/bottleneck manipulation would be needed for a
stronger route-redundancy causal claim.

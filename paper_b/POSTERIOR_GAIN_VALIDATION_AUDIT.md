# Posterior-Uncertainty Jammer Validation Audit

## Status

The first validation bundle under the posterior-uncertainty Jammer formulation
completed successfully.

Design:

```text
Targeted Exp III regression:
    seed 2173
    flat prior
    2 redundancy x 2 reliance x 2 Jammer
    = 8 runs

Small Exp III matched pilot:
    seeds 2001-2020
    flat prior
    2 redundancy x 2 reliance x 2 Jammer
    = 160 runs

Small Exp IV matched pilot:
    seeds 3001-3020
    2 homophily x 2 prior segregation x 2 Jammer
    = 160 runs
```

All runs use N=100 and T=200.

This is a validation gate, not a final manuscript result set.

## Gate 1: former runaway seed 2173 — PASS

The pre-revision Exp III run at seed 2173 reached terminal MSE 197.621 in the
high-redundancy, adaptive, active-Jammer condition and continued accelerating
through T=200.

Under the posterior-uncertainty Jammer, the same matched block completed with:

```text
max terminal MSE              = 3.2610
max |terminal citizen belief| = 4.4143
max |Jammer message mean|     = 53.0881
max individual response gain  = 25/26 = 0.961538
min objective denominator     = 51/676 = 0.0754438
```

The formerly unstable high-redundancy/adaptive/J=1 MSE trajectory is now:

```text
T=51   3.3946
T=101  3.3185
T=151  3.2800
T=200  3.2610
```

The matched low-redundancy/adaptive/J=1 trajectory is likewise stable:

```text
T=51   3.0843
T=101  2.9678
T=151  2.9202
T=200  2.9006
```

The former late-period message escalation is gone. At period 0 the high initial
posterior SD of five implies the common model-generated message mean 53.09.
After learning reduces posterior uncertainty, the optimized message rapidly
returns close to the Jammer's underlying position:

```text
period 0     m = 53.088
period 5     m ~= 4.051
period 10    m ~= 4.003
period 195   m ~= 4.000
```

The initial large message is finite and follows from the unchanged quadratic
objective under high initial posterior uncertainty; it is not clipping. Because
the initial posterior state and Jammer opportunity are matched across the
structural treatment cells, it is a common adversarial shock in the mechanism
comparisons.

## Gate 2: 20-seed Exp III matched pilot — PASS for stability; mechanism remains provisional

The Exp III computational gate passes:

```text
160/160 runs finite
160/160 runs at T=200
low/high local two-step Expert routes = 1/2
frozen Lambda invariant = true
max terminal MSE = 6.995
max |Jammer message mean| = 53.199
max individual response gain = 0.961538
min objective denominator = 0.0754438
```

For the primary MSE disruption contrast
`D = MSE(J=1)-MSE(J=0)`, the 20-seed means are:

```text
Adaptive:
    low redundancy D  = 1.186
    high redundancy D = 0.765
    high-low effect   = -0.422
    MCSE              = 0.339
    median            = -0.106

Frozen:
    low redundancy D  = 0.622
    high redundancy D = 0.476
    high-low effect   = -0.145
    MCSE              = 0.286
    median            = -0.002

Activation interaction:
    (high-low)_adaptive - (high-low)_frozen
                       = -0.276
    MCSE               = 0.337
    median             = -0.005
```

The adaptive high-minus-low absolute MSE decomposition is:

```text
J=1: -0.415
J=0: +0.0067
```

Thus, in this small validation sample, the negative adaptive disruption
contrast is generated primarily by lower active-Jammer MSE rather than by
deterioration of the J=0 baseline.

The direction of the adaptive redundancy effect is also negative on transformed
losses:

```text
high-low Delta RMSE = -0.254
high-low Delta MAE  = -0.165
```

The activation interaction is much less stable:

```text
MSE  = -0.276
RMSE = -0.178
MAE  = -0.043
```

With only 20 seeds, the activation result is too noisy for inference. Its MSE
median is near zero and the sign is split evenly across seeds. The 500-seed
production run remains necessary to determine whether adaptation materially
changes the redundancy effect.

The aggregate fixed-horizon effect is stable late in the run. Mean adaptive
high-minus-low Delta MSE is -0.425 at T=101, -0.423 at T=151, and -0.422 at
T=200.

## Gate 3: 20-seed Exp IV matched pilot — PASS for stability; C4 remains directional

The Exp IV computational and manipulation gates pass:

```text
160/160 runs finite
160/160 runs at T=200
mean structural H: 0.492 / 0.897
mean prior S:      0.154 / 5.943
max terminal MSE = 13.423
max |Jammer message mean| = 53.128
max individual response gain = 0.961538
min objective denominator = 0.0754438
```

Mean MSE disruption by cell is:

```text
low H,  low S   D = 0.0088
high H, low S   D = 0.0024
low H,  high S  D = 0.3468
high H, high S  D = 0.6109
```

The matched homophily effects are:

```text
low segregation:  high H - low H = -0.0064
high segregation: high H - low H = +0.2640
```

so the MSE H x S interaction is:

```text
+0.2705
MCSE   = 0.1986
median = +0.2763
positive in 60% of seeds
```

This preserves the directional C4 pattern in the small validation sample:
homophily has essentially no disruption effect under low prior segregation and
a positive effect under high prior segregation. Twenty seeds are not sufficient
to treat the interaction magnitude as a final estimate.

The terminal effective-reliance pattern is also directionally consistent with
effective informational isolation. Under J=1:

```text
low H, low S:   H_Lambda = 0.483
low H, high S:  H_Lambda = 0.563
high H, low S:  H_Lambda = 0.894
high H, high S: H_Lambda = 0.948
```

The MSE interaction is stable over the late fixed horizon:

```text
T=101  0.267
T=151  0.264
T=200  0.270
```

RMSE should not be used to define the H x S interaction. Because RMSE is a
nonlinear square-root transformation and baseline MSE is already much larger in
the high-H/high-S cell, an interaction computed directly on RMSE can change
sign even when the primary MSE interaction is positive. RMSE remains a
presentation metric; MSE remains the theoretical estimand.

## Gate 4: posterior-SD floor — REVALIDATION REQUIRED

The posterior-SD numerical floor remains active:

```text
Exp III terminal floor share = 0.544
Exp IV terminal floor share  = 0.776
```

Exp III floor incidence is concentrated especially in adaptive conditions
(roughly 0.84-0.86 in this pilot). Exp IV floor incidence ranges from roughly
0.69 to 0.87 across cells.

Floor-bound citizens are not exclusively truth-converged. In this validation
bundle, floor-bound terminal citizens have larger mean absolute and squared
errors than non-floor citizens.

The old `MIN_SD` sensitivity was run before the Jammer response rule began
using posterior uncertainty directly. It therefore cannot clear the current
model. The floor sensitivity must be rerun under the posterior-gain Jammer
before full production.

## Decision

The posterior-uncertainty Jammer revision clears the instability gate:

- the seed-2173 runaway channel is removed;
- the model-implied response-gain and curvature bounds hold;
- Exp III and Exp IV are finite and late-horizon stable in the validation grids;
- the intended Exp III and Exp IV comparative patterns remain non-degenerate.

The next gate is the prepared post-revision numerical-floor sensitivity:

```bash
PAPER_B_WORKERS=12 bash scripts/run_paper_b_posterior_gain_floor_sensitivity.sh
```

Only after that sensitivity is inspected should the 500-seed Exp III and Exp IV
production grids be rerun.

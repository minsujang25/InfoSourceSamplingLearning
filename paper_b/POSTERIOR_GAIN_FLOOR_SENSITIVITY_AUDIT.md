# Posterior-Gain Numerical-Floor Sensitivity Audit

## Status

The post-revision numerical-floor sensitivity clears the final pre-production
gate for Experiments III and IV.

The sensitivity was run under the canonical posterior-uncertainty Jammer, not
the superseded cross-sectional-dispersion response rule.

Design:

```text
MIN_SD in {1e-6, 1e-8, 1e-10}

Experiment III:
    10 matched flat-prior seeds
    x 2 redundancy
    x 2 reliance
    x 2 Jammer
    = 80 runs per floor

Experiment IV:
    10 matched seeds
    x 2 homophily
    x 2 prior segregation
    x 2 Jammer
    = 80 runs per floor
```

All runs use N=100 and T=200.

This is a numerical sensitivity audit. Ten seeds are not used for manuscript
inference.

## Computational gates

Every floor value passed all Experiment III and IV gates.

Across all three floors:

```text
max individual response gain  = 0.961538 = 25/26
min objective denominator     = 0.0754438 = 51/676
```

The maximum Jammer message mean was also invariant to the floor:

```text
Exp III: 53.1630
Exp IV:  53.1276
```

No numerical runaway was observed.

## Floor incidence changes substantially

The sensitivity is informative because changing MIN_SD materially changes how
many citizens finish exactly at the numerical floor.

```text
MIN_SD        Exp III floor share     Exp IV floor share
1e-6                 0.6214                  0.8131
1e-8                 0.5529                  0.7503
1e-10                0.5058                  0.6956
```

Thus the floor is active and the experiment does not merely compare three
effectively identical implementations.

## Experiment III: redundancy contrast is stable

Adaptive high-minus-low disruption on the primary MSE scale:

```text
MIN_SD        mean          median        MCSE
1e-6         -0.6500       -0.6097       0.5177
1e-8         -0.6219       -0.6098       0.5087
1e-10        -0.6127       -0.6098       0.5038
```

The corresponding transformed-loss contrasts are also stable:

```text
MIN_SD        Delta RMSE     Delta MAE
1e-6           -0.1405        -0.1796
1e-8           -0.1396        -0.1791
1e-10          -0.1395        -0.1801
```

The adaptive absolute MSE high-minus-low decomposition is:

```text
MIN_SD        J=1            J=0
1e-6         -0.8049        -0.1549
1e-8         -0.7553        -0.1333
1e-10        -0.7459        -0.1332
```

Thus the negative disruption contrast is not generated solely by deterioration
of the J=0 baseline.

At the seed level, the adaptive MSE redundancy contrast is extremely stable:

```text
corr(1e-6, 1e-8)   = 0.9994
corr(1e-8, 1e-10)  = 0.9999
```

The mean absolute seed-level difference between 1e-6 and 1e-10 is about 0.040,
small relative to the approximately 0.50 Monte Carlo SE at ten seeds.

## Experiment III: activation interaction remains a production question

The adaptive-minus-frozen redundancy interaction is:

```text
MIN_SD        mean          median        MCSE
1e-6         -0.0249        +0.2822       0.3698
1e-8         -0.0531        +0.2016       0.3594
1e-10        +0.0616        +0.1397       0.3955
```

The values fluctuate around zero and are much smaller than their ten-seed Monte
Carlo uncertainty. This does not indicate floor sensitivity; it indicates that
the adaptation-specific interaction is weak/noisy in the small calibration
sample.

The 500-seed production run is therefore required to determine whether adaptive
reliance materially changes the redundancy effect.

## Experiment IV: H x S interaction is stable

The primary MSE homophily x prior-segregation interaction is:

```text
MIN_SD        mean          median        MCSE       positive share
1e-6          0.4769         0.3810        0.2965          0.80
1e-8          0.4792         0.3810        0.2942          0.80
1e-10         0.4620         0.3810        0.2987          0.80
```

Seed-level stability is very high:

```text
corr(1e-6, 1e-8)   = 0.9998
corr(1e-8, 1e-10)  = 0.9986
```

The mean absolute seed-level interaction difference between 1e-6 and 1e-10 is
about 0.019, far below the approximately 0.30 ten-seed Monte Carlo SE.

Cell-level mean disruption is also nearly invariant. At MIN_SD=1e-8:

```text
low H,  low S   D = 0.0155
high H, low S   D = 0.0057
low H,  high S  D = 0.2910
high H, high S  D = 0.7603
```

The qualitative C4 pattern therefore survives the posterior-floor sensitivity.

## Run-level robustness

Across the 80 Experiment III runs per floor, terminal MSE values remain highly
correlated across numerical floors. The 1e-6 versus 1e-10 mean absolute
run-level MSE difference is about 0.035.

Across the 80 Experiment IV runs per floor, the corresponding mean absolute
difference is about 0.006.

The numerical floor can change individual micro-trajectories, especially in
frozen Experiment III realizations, but it does not materially determine the
aggregate mechanism contrasts targeted by the paper.

## Decision

The default

```text
MIN_SD = 1e-8
```

is retained as the canonical numerical guardrail.

It is not interpreted as a calibrated behavioral parameter.

The post-revision validation sequence is now complete:

1. posterior-uncertainty Jammer derivation — cleared;
2. former runaway seed 2173 regression — cleared;
3. 20-seed Exp III matched validation — cleared;
4. 20-seed Exp IV matched validation — cleared;
5. post-revision MIN_SD sensitivity — cleared.

Experiments III and IV are ready for the frozen 500-seed production runs:

```bash
PAPER_B_WORKERS=12 bash scripts/run_paper_b_exp3_production.sh
PAPER_B_WORKERS=12 bash scripts/run_paper_b_exp4_production.sh
```

The full production results, rather than these ten-seed sensitivity estimates,
determine the manuscript conclusions.

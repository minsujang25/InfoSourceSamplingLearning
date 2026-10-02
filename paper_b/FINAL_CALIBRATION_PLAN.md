# Paper B Final Calibration Plan — Horizon and Adaptive-Reliance Identification

## Status

This is the final calibration round before any new 500-seed production run.

The receiver-side specification is **not redesigned further** in this round.
The following are frozen:

[
	au_{social}=1,qquad
	exttt{peer\_evidence\_mode=source\_posterior},qquad
	exttt{frozen\_ranking\_mode=pre\_disruption}.
]

Other frozen settings:

```text
N = 100
epsilon = .05
credit = 20
K = 1
surveillance interval = 5
MIN_SD = 1e-8
seeds = 5101-5108
```

The purpose is limited to two questions:

1. Is the existing terminal horizon (T=200) adequate under the slower,
   validity-corrected dynamics relative to (T=400)?
2. Do the central network contrasts depend materially on adaptive versus frozen
   reliance under scientifically interpretable sender conditions?

No parameter search is permitted in this round.

## Why one T=400 run is sufficient

Every calibration simulation runs to (T=400). The state after 200 executed
periods is read from checkpoint period 199 and the state after 400 executed
periods from checkpoint period 399.

Thus (T=200) and (T=400) are compared on the **same stochastic trajectory**.
No separate T=200 simulation is run, avoiding unnecessary Monte Carlo noise.

## Sender restrictions

The adaptive Jammer is not crossed with frozen reliance in this calibration.
The previous validity diagnostic showed that immutable pre-disruption trust
combined with the adaptive Jammer's extreme initial message produces a special
bait-and-switch stress case rather than a clean adaptation counterfactual.

The final calibration therefore uses:

- `null`: baseline/no-substantive-Jammer slot;
- `fixed_biased`: a persistent biased source.

The fixed-biased sender is a robustness/stress condition. The null sender is
the primary mechanism-identification environment.

## Experiment IV calibration block

### Primary null block

Cross:

- structural homophily (Hin{low,high});
- prior segregation (Sin{low,high});
- reliance mode (in{adaptive,frozen});
- sender = null.

Per seed:

[
2H	imes2S	imes2R = 8
]

runs.

Primary quantities:

- MSE/RMSE/MAE;
- squared population displacement;
- belief variance;
- effective homophily;
- peer reliance;
- dominant Expert reachability;
- citizen-only cycle share;
- same-group closed-cycle share;
- incoming-reliance concentration.

Mechanism estimands:

[
I^{A}_{HS},qquad I^{F}_{HS},qquad I^{A}_{HS}-I^{F}_{HS}.
]

The same contrasts are computed for same-group closed dependence and effective
homophily. No direction is required in advance.

### Fixed-biased high-segregation stress block

Cross:

- (Hin{low,high});
- (S=high);
- reliance mode (in{adaptive,frozen});
- sender = fixed_biased.

Per seed: 4 runs.

These runs are paired with the corresponding high-S null runs to measure sender
damage under adaptive and frozen reliance.

## Experiment III calibration block

Cross:

- redundancy (in{low,high});
- reliance mode (in{adaptive,frozen});
- sender (in{null,fixed\_biased}).

Per seed:

[
2R	imes2A/F	imes2J = 8
]

runs.

Primary quantities:

- baseline MSE;
- fixed-biased sender damage relative to null;
- dominant Expert reachability;
- citizen-cycle share;
- gateway incoming-reliance share;
- effective incoming HHI;
- top-five incoming-reliance share.

Mechanism estimands include the redundancy contrast under adaptive and frozen
reliance and their difference.

## Total run count

Per seed:

```text
Exp IV null block                 8
Exp IV fixed-biased high-S block  4
Exp III block                     8
-----------------------------------
Total                            20
```

Eight matched seeds therefore produce:

[
8	imes20 = 160
]

T=400 simulations.

## Required checkpoints

At minimum:

```text
0, 1, 2, 5, 10, 25, 50, 100, 150, 199, 299, 399
```

The same trajectory supplies both the (T=200) and (T=400) comparison.

## Precommitted horizon-adequacy rule

For every run define:

[
r_i
=
rac{|MSE_{399}-MSE_{199}|}
{max(MSE_{199},10^{-6})}
]

and

[
a_i=|MSE_{399}-MSE_{199}|.
]

A run-level trajectory is classified as practically stable if either:

[
r_i<0.10
]

or

[
a_i<0.005.
]

The absolute-change clause prevents trivial near-zero MSE values from failing
only because a percentage denominator is tiny.

### Primary horizon gate

The existing (T=200) horizon is eligible for retention only if all of the
following hold in the **null-sender primary cells**:

1. at least 75% of the 8 matched seeds are practically stable in each primary
   cell;
2. the median posterior-SD floor share remains zero at both periods 199 and 399;
3. the key network contrast does not reverse sign between periods 199 and 399:
   - Experiment IV high-S high-H minus low-H same-group closed dependence;
   - Experiment III high-R minus low-R gateway incoming-reliance share.

If condition 1 fails but the key substantive contrasts are stable, the default
decision is to use (T=400) in production rather than modify the model.

No intermediate horizon is searched in this calibration.

## Precommitted contrast-stability rule

For each key matched contrast (C), compare (C_{199}) and (C_{399}).

A contrast is classified as stable if:

1. the signs agree, treating any \(|C|<0.005\) as effectively zero, and
2. either
   [
   rac{|C_{399}-C_{199}|}{max(|C_{199}|,10^{-6})}<0.25
   ]
   or
   [
   |C_{399}-C_{199}|<0.005.
   ]

Key contrasts:

### Experiment IV

- null-sender (H	imes S) interaction in MSE, separately for adaptive and
  frozen reliance;
- high-S high-H minus low-H same-group closed dependence, separately for
  adaptive and frozen reliance.

### Experiment III

- high-minus-low redundancy contrast in null-sender MSE, separately for
  adaptive and frozen reliance;
- high-minus-low redundancy contrast in gateway incoming-reliance share,
  separately for adaptive and frozen reliance.

The adaptive-minus-frozen difference is reported but is **not** required to
have a particular sign.

## Mechanism-identification rule

Adaptive versus frozen reliance is descriptive/mechanistic, not a success gate.

The calibration will classify each core network result into one of three cases:

1. **adaptive-specific/amplified** — materially larger under adaptive reliance;
2. **shared across reliance modes** — similar under adaptive and frozen;
3. **adaptive-attenuated/reversed** — smaller or opposite under adaptive.

No case is preferred ex ante. Manuscript theory will follow the observed case.

## Numerical safety gates

All runs must:

- reach exactly T=400;
- have finite terminal beliefs and losses;
- have terminal MSE < 1,000,000;
- have absolute terminal belief < 10,000.

Failure of a numerical safety gate blocks production.

## Production decision

After this 160-run calibration:

- if T=200 passes the precommitted horizon gate, production may retain T=200;
- otherwise production uses T=400;
- the receiver model remains tau_social=1 unless a numerical failure appears;
- adaptive/frozen interpretation is frozen from this calibration;
- no further model redesign or parameter tuning is performed before the
  500-seed production grid.

Manuscript numerical results are not updated until the production grid is
complete.

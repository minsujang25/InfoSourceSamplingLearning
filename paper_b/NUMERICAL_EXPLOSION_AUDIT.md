# Numerical Explosion Audit: polarized K=4 / peer-cap=5

## Status

Root cause identified and reproduced.

The failing diagnostic condition is:

- scenario: `group_polarized_k4_n5_delta`
- seed: `101`
- initial-belief regime: polarized
- jammer surveillance/segmentation: K = 4
- maximum citizen neighbors: 5
- credibility comparison: delta comparison

The same explosion occurs under Mesa 2.4 and Mesa 3.5, so it is not a migration error.

## Root cause

`Citizen.bayesian_update_sd_theta()` computes the Gaussian posterior **variance**

```python
posterior_sd_theta = (
    prior_sd_theta**2 * std_msgs**2
    / (prior_sd_theta**2 + std_msgs**2)
)
```

but stores that quantity as if it were a **standard deviation**.

Given prior standard deviation (s) and message standard deviation (	au), the
implemented recurrence is

[
s' = rac{s^2 	au^2}{s^2 + 	au^2},
]

whereas the corresponding posterior standard deviation is

[
s' = sqrt{rac{s^2 	au^2}{s^2 + 	au^2}}.
]

When message dispersion is large relative to the citizen's current uncertainty,
the legacy expression approaches

[
s' approx s^2.
]

For any (s>1), repeated theta-updating can therefore create a squaring
recurrence rather than Bayesian uncertainty contraction.

## Observed failing trajectory

The diagnostic workflow traced the maximum citizen theta standard deviation:

| period | max theta SD | max absolute theta mean |
|---:|---:|---:|
| 0 | 5 | 5.21 |
| 2 | 21.24 | 5.28 |
| 3 | 106.36 | 6.26 |
| 4 | 10,114 | 6.25 |
| 5 | 1.02e8 | 24.38 |
| 6 | 1.04e16 | 1.26e4 |
| 7 | 1.08e32 | 9.75e10 |
| 8 | 1.18e64 | 3.59e22 |
| 9 | 1.38e128 | 1.64e45 |
| 10 | non-finite | non-finite |

The uncertainty explosion clearly precedes the belief-mean explosion.

The mechanism is:

1. heterogeneous source messages produce a large `std_msgs`;
2. the variance is stored as an SD, increasing `sd_theta`;
3. citizen `mu_out()` uses that inflated `sd_theta` as the Normal message
   scale;
4. peer messages then become extremely dispersed;
5. subsequent theta updates push stored uncertainty toward (s^2) again;
6. extreme peer draws propagate through the group-matched network;
7. theta means eventually become non-finite.

The jammer message means remain moderate (roughly 4--6 in this diagnostic), so
the disruptive-jammer message optimization is **not** the direct source of the
numerical explosion.

## Shadow correction

A diagnostic-only shadow run replaced the legacy update with

```python
posterior_var = (
    prior_sd**2 * std_msgs**2
    / (prior_sd**2 + std_msgs**2)
)
posterior_sd = sqrt(posterior_var)
```

No other model logic was changed.

Under the same scenario and seed:

- maximum theta SD never exceeded its initial value of 5;
- it declined below 1 by period 8;
- theta means remained approximately within [-5, 5];
- no NaN or infinite states appeared through the diagnostic horizon.

A dimensional sanity check gives:

- prior SD = 5
- message SD = 100
- legacy stored value = 24.94
- correct posterior SD = 4.99

Thus the legacy formula turns a highly noisy observation into a roughly
five-fold *increase* in uncertainty, while the standard-deviation-consistent
formula produces the expected slight contraction.

## Classification

This is a **model implementation bug / variance-vs-standard-deviation mismatch**,
not merely floating-point overflow.

The floating-point failure is a downstream symptom of the incorrect recurrence.

## Recommended correction

The smallest theoretically defensible correction is to change
`bayesian_update_sd_theta()` so it returns the square root of the posterior
variance.

Do not clip beliefs or standard deviations to arbitrary bounds.

Because this is a substantive correctness fix rather than a Mesa API migration,
it should be committed in the Paper B branch (or a dedicated correctness-fix
branch) and followed by:

1. regression tests for posterior-SD contraction;
2. rerunning the diagnostic scenario;
3. rerunning the original parameter grid / archived-result audit to quantify
   how much historical results depended on the bug;
4. only then launching new Paper B simulations.

## Additional issue to audit later

`learn_delta()` also contains quantities named `sd_delta` whose update
formula is not obviously dimensionally a standard-deviation update. It did not
drive the observed theta explosion in this case, but it should be audited
separately before the full Paper B production grid.

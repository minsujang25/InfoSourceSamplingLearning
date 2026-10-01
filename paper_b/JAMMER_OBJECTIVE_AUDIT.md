# Jammer Objective Audit — Posterior-Uncertainty Revision

## Status

This document records the canonical Paper B Jammer formulation after the
500-seed Experiment III production audit.

The earlier reconstruction correctly removed the undocumented legacy
within-window recurrence and restored the Jammer's quadratic deviation cost.
It nevertheless used **cross-sectional segment dispersion** as if it were the
Bayesian uncertainty governing an individual citizen's response to a message.

That approximation is rejected here.

The current formulation keeps the same strategic objective, message variance,
surveillance timing, and segment-specific message rule, but derives response
gains from citizens' **posterior uncertainty about the state**.

No message clipping, exogenous message bound, or new strategic-cost parameter is
introduced.

## Why the previous response-gain approximation was rejected

The previous reconstruction used

```text
kappa_g = s_g^2 / (s_g^2 + 1)
```

where `s_g` was the cross-sectional standard deviation of citizen belief means
inside segment `g`.

That quantity measures audience disagreement. It is not the uncertainty in a
citizen's Gaussian posterior that determines how strongly a new unit-variance
observation updates that citizen.

The distinction became consequential in the 500-seed Experiment III run. One
high-redundancy, adaptive, active-Jammer trajectory at seed 2173 developed
rapidly increasing cross-sectional dispersion. Under the old approximation,

```text
s_g large
    -> kappa_g close to 1
    -> 1 - kappa_g^2 close to 0
    -> very large optimal message
    -> still larger cross-sectional dispersion.
```

The trajectory remained finite at T=200 but was dynamically unstable and
dominated the mean MSE contrast. This was not a floating-point bug and was not
caused by the posterior-SD numerical floor. It exposed a mismatch between the
quantity called "dispersion" and the quantity required by the Bayesian response
calculation.

## Citizen-level Gaussian response gain

Citizen `i` enters a surveillance refresh with current pre-update posterior

```text
theta_i ~ Normal(mu_i, sigma_i^2).
```

A Jammer message is drawn from a unit-variance signal centered on the chosen
segment message mean `m`.

The one-message Gaussian response gain is therefore

```text
kappa_i = sigma_i^2 / (sigma_i^2 + 1).
```

For the one-step strategic approximation, citizen `i`'s post-message mean is

```text
mu_i'(m) = (1-kappa_i) mu_i + kappa_i m.
```

The key point is that `sigma_i`, not cross-sectional disagreement among
citizens, governs responsiveness.

## Segment objective with heterogeneous posterior gains

The Jammer still chooses **one common message mean per observed segment**. It
does not choose an individualized message for each citizen.

For segment `g`, define the one-period objective

```text
U_g(m)
  = mean_i in g [ (mu_i'(m) - theta)^2 ]
    - (m - mu_D)^2,
```

where

- `theta` is the true state;
- `mu_D` is the Jammer's underlying position;
- the first term rewards one-step squared displacement from truth;
- the second term is the documented quadratic cost of moving the message away
  from the Jammer's own position.

Taking the derivative and collecting terms gives

```text
m_g*
  = {
      E_g[ kappa_i ((1-kappa_i) mu_i - theta) ] + mu_D
    }
    / {
      1 - E_g[kappa_i^2]
    }.
```

The Jammer therefore needs segment-level sufficient statistics of current
posterior means and posterior uncertainties. Individual identities are not used
to customize messages inside a segment.

## Concavity and the removed runaway channel

The coefficient on `m^2` is

```text
E_g[kappa_i^2] - 1.
```

Because every finite posterior SD implies `0 <= kappa_i < 1`, the objective is
strictly concave and has a unique finite maximizer.

More importantly, cross-sectional belief disagreement no longer appears in the
curvature term.

In the primary Paper B designs, citizen state posterior SDs start at 5 and
Gaussian state updates weakly reduce them. Hence

```text
max kappa_i = 25 / 26
```

and throughout those designs

```text
1 - E_g[kappa_i^2]
    >= 1 - (25/26)^2
    = 51/676
    ~= 0.07544.
```

This is a model-implied curvature bound, not an imposed clipping rule.

Large differences between citizen belief means can still make a disruptive
message strategically valuable. They can no longer make the objective nearly
linear merely by driving a cross-sectional-dispersion proxy toward infinite
response gain.

## Surveillance resolution K

The surveillance architecture is unchanged.

At a refresh the Jammer:

1. clusters current pre-update citizen belief means into `K` segments;
2. aggregates the current posterior means and posterior SDs within each
   segment;
3. computes the segment-specific sufficient statistics in the closed form
   above;
4. selects one message mean `m_g*` for each segment; and
5. holds those segment-specific means fixed until the next surveillance
   refresh.

Thus `K` continues to represent audience-model granularity. Higher `K`
permits a finer partition of the audience, but the Jammer still sends a common
strategy within each observed segment.

## Timing rule

With the default surveillance interval of five periods, refreshes occur at

```text
t = 0, 5, 10, 15, ...
```

and the selected segment message means are held fixed during

```text
t = 1-4, 6-9, 11-14, ...
```

There is no within-window strategic recurrence.

## Logged diagnostics

Each Jammer strategy record now contains:

- `period`;
- `cluster`;
- `refresh`;
- `segment_mean`;
- `segment_std` — retained as descriptive cross-sectional disagreement only;
- `posterior_sd_mean`;
- `posterior_sd_min`;
- `posterior_sd_max`;
- `response_gain` — mean individual `kappa_i` in the segment;
- `response_gain_max`;
- `response_gain_sq_mean`;
- `objective_denominator = 1 - E[kappa_i^2]`;
- `message_mean`;
- `message_sd`;
- `cluster_size`.

Experiment III and IV gates additionally report the maximum individual response
gain and minimum objective denominator observed across the run.

## Interpretation

The Jammer remains a strategic stress environment rather than the paper's sole
theoretical object.

The revision is intentionally narrow. It corrects the mapping from citizen
uncertainty to one-step responsiveness while preserving:

- the documented quadratic disruption/deviation objective;
- the Jammer's underlying position `mu_D`;
- unit message variance;
- periodic audience surveillance;
- one message strategy per segment;
- fixed within-window strategies;
- the meaning of `K` as surveillance resolution; and
- the absence of arbitrary clipping.

Any Experiment III or IV result generated under the older
cross-sectional-dispersion response gain is retained only as an audit result and
must be revalidated under this posterior-uncertainty formulation before
manuscript use.

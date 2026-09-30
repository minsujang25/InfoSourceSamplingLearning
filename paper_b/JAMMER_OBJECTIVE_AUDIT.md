# Jammer Objective Audit — Paper B Reconstruction Round 2

## Why this audit was necessary

The first 80-run local diagnostic passed topology and finite-state checks but exposed
extreme Jammer-induced MSE in a subset of runs. The corrected citizen SD update remained
well behaved, so the instability was traced to the Jammer message-mean recursion rather
than to the earlier variance/SD bug.

The legacy reconstruction branch still contained the following sender-side logic:

1. surveillance at period 0 and every fifth period;
2. initialization of each segment's message mean at the observed segment centroid;
3. within-window retuning of that message mean every non-surveillance period;
4. a recurrence using an undocumented constant `alpha = 0.95`;
5. a closed-form expression that did not contain the Jammer's own underlying position.

That behavior was inconsistent with the current Paper B working-paper specification.

## Source specification recovered from the working paper

The working paper states that the disruptive provider:

- observes cluster-level summaries of the current audience belief distribution;
- chooses one message mean for each observed segment;
- draws messages around that chosen mean with variance fixed at one;
- seeks to increase citizens' post-communication squared distance from truth;
- pays a quadratic cost for moving the message mean away from the provider's underlying
  position;
- refreshes surveillance at initialization and every fifth step; and
- holds the resulting segment-specific message means fixed until the next surveillance
  update.

The documented objective is

```text
U_D,t = sum_i [
    (mu_i,t - theta)^2
    - (m_D,g(i,t),t-1 - mu_D)^2
].
```

The legacy recurrence therefore had two substantive mismatches:

1. the documented deviation-cost center `mu_D` disappeared from the executable
   message rule;
2. message means were recursively retuned inside the five-period surveillance window
   even though the manuscript says they remain fixed until the next surveillance update.

## Reconstructed one-step segment objective

Paper B gives the Jammer segment-level location and dispersion, not the full network
topology. The reconstruction therefore uses a transparent representative-segment
approximation.

For segment g with current location `mu_g` and dispersion `s_g`, define the
unit-variance Gaussian response gain

```text
kappa_g = s_g^2 / (s_g^2 + 1).
```

For a candidate Jammer message mean `m`, the representative one-step post-message
location is

```text
mu_post,g(m) = (1-kappa_g) mu_g + kappa_g m.
```

The segment objective is then

```text
U_g(m)
  = [mu_post,g(m) - theta]^2
    - [m - mu_D]^2.
```

Because `0 <= kappa_g < 1`, the coefficient on `m^2` is
`kappa_g^2 - 1 < 0`. The objective is therefore strictly concave in `m` and has a
unique finite maximizer:

```text
m_g*
  = { kappa_g[(1-kappa_g)mu_g - theta] + mu_D }
    / (1 - kappa_g^2).
```

No arbitrary clipping or exogenous message bound is used.

## Timing rule

At periods

```text
t = 0, 5, 10, 15, ...
```

for the default five-period surveillance interval, the Jammer:

1. clusters current pre-update citizen beliefs at resolution K;
2. records each segment's current location and dispersion;
3. computes one `m_g*` from the objective above; and
4. uses that message mean for all requests from citizens assigned to segment g.

For periods between surveillance updates, the segment membership and message mean remain
fixed. The model logs the held strategy each period so the empirical trajectory can be
audited directly.

## Why this removes the legacy recurrence problem

The previous code recursively updated an internal expected segment state every period and
fed that state back into a denominator involving `v * alpha - 1`. The resulting mapping
could amplify message means repeatedly even when the observed audience had not been
re-surveilled.

The reconstructed rule has no within-window recurrence. Strategic adaptation occurs only
when the Jammer receives new audience information. The message choice is recalculated from
the documented objective at the scheduled surveillance refresh.

This does not guarantee that every strategically optimal message is numerically small.
A highly dispersed segment can make a disruptive message valuable under the stated
quadratic objective. The important distinction is that any large message now follows from
the explicit objective and observed segment state rather than from an undocumented recursive
feedback rule.

## Diagnostic outputs

The local diagnostic now writes `jammer_strategy_trajectory.csv.gz` with:

- period;
- cluster;
- surveillance-refresh indicator;
- segment mean;
- segment standard deviation;
- response gain `kappa`;
- optimal message mean;
- message standard deviation;
- cluster size.

This makes large-disruption runs traceable back to the actual segment state and the
closed-form strategic message.

## Fixed-horizon implication

The first diagnostic also showed that convergence-based early stopping produced unequal
terminal periods in some matched J=1/J=0 pairs. Paper B's resilience estimand is defined at
a common horizon T, so reconstruction round 2 makes fixed-horizon execution the default.
Convergence is retained only as a diagnostic quantity.

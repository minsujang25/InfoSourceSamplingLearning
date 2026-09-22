# Paper B Scientific Gate — Reconstruction Branch

The Mesa 2.4 -> 3.5 migration gate is complete. This branch begins a new gate:
scientific correspondence between the frozen Paper B theory and the executable model.

## Resolved correctness issues

### Theta uncertainty

The legacy code computed a Gaussian posterior variance but stored it as `sd_theta`.
The reconstruction stores a standard deviation consistently by taking the square root
of posterior variance. Arbitrary clipping is not used.

### Source-displacement uncertainty

The legacy `sd_delta` recurrence mixed SD and variance units. The reconstruction
uses a dimensionally consistent Gaussian posterior and stores its square-root variance.

### Synchronous social learning

The old `Citizen.step()` mutated `mu_theta` and `sd_theta` immediately, allowing
later-executing citizens in the same period to sample already-updated peer states.
The reconstruction snapshots all message states before any citizen update and commits
all posterior states only after every citizen has computed its period-t update.

### Jammer re-surveillance

Legacy re-surveillance read `mu_theta_beliefs[0]`, i.e. initial beliefs. The
reconstruction clusters the current pre-update belief landscape.

### Frozen reliance

The primary frozen counterfactual is now fixed: behavioral source ranking is frozen
after the first credibility audit. Later credibility and substantive learning continue,
but later audits cannot change acquisition ranking.

## Theory-aligned invariants

Before production runs, CI must verify:

- recursive rank-based exploration nests `(1-epsilon, epsilon)`;
- citizen outgoing messages use period-t snapshot states;
- posterior SDs remain positive and finite;
- first-audit-frozen ranking does not drift;
- Jammer refreshes track current beliefs;
- expected reliance shares sum to one;
- no run is silently rescued through clipping.

## Remaining design gate

The only substantive design item still requiring a production freeze is the exact
parameterization of the four accepted-abstract communication environments, especially
localized elite access in the sparse Random and Homophilous conditions.

This is a design-mapping decision, not a reason to alter the reconstructed learning
or timing mechanisms.

## Evidence rule

Legacy figures are historical diagnostics only. Corrected simulations determine which
previous qualitative patterns survive. Theory should not be rewritten merely to
preserve archived output.

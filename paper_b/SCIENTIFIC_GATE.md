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

### Jammer response gain

The earlier reconstruction used cross-sectional belief dispersion as a proxy for
Bayesian responsiveness to a new Jammer message. The 500-seed Experiment III audit
showed that this could collapse the Jammer objective curvature in rare trajectories.

The current rule uses each citizen's state-posterior uncertainty:

```text
kappa_i = sigma_i^2 / (sigma_i^2 + 1).
```

The Jammer still chooses one common segment-specific message mean and retains the same
quadratic disruption/deviation objective. Cross-sectional segment dispersion remains a
descriptive surveillance statistic only. Under the primary initial SD of five, the
objective denominator is bounded below by `51/676`, so audience disagreement alone
cannot drive the curvature to zero.

## Theory-aligned invariants

Before production runs, CI must verify:

- recursive rank-based exploration nests `(1-epsilon, epsilon)`;
- citizen outgoing messages use period-t snapshot states;
- posterior SDs remain positive and finite;
- first-audit-frozen ranking does not drift;
- Jammer refreshes track current beliefs;
- Jammer response gains are based on posterior SD, not cross-sectional disagreement;
- the posterior-gain Jammer objective satisfies its model-implied curvature bound;
- expected reliance shares sum to one;
- no run is silently rescued through clipping.

## Four-environment design gate — cleared

The final design audit maps the manuscript environments as follows:

- isolated elite exposure: Expert + Jammer only;
- Random 2: exactly two local sources from the full citizen+elite pool;
- Group ID: exactly two local sources with a 0.9 same-group / 0.1 cross-group preference, with elite nodes embedded in the same pool;
- Extended: Expert + Jammer + exactly two random citizen peers.

The legacy random/group implementations did not match the current manuscript description because both attached both elites universally and then added a random 0..cap number of citizen peers. They are retained only under explicit `legacy_extended_*` compatibility aliases.

See `NETWORK_ENVIRONMENT_AUDIT.md`.

## Evidence rule

Legacy figures are historical diagnostics only. Corrected simulations determine which
previous qualitative patterns survive. Theory should not be rewritten merely to
preserve archived output.

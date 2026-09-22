# Paper B Scientific Gate

The Mesa framework migration itself has passed a strict 24/24 matched-seed
comparison after normalizing a legacy unordered-neighbor bug.

Two scientific issues remain deliberately separate from that migration gate.

## 1. Non-finite extreme run

The diagnostic condition `group_polarized_k4_n5_delta`, seed 101, reaches
non-finite citizen beliefs in both Mesa 2.4 and Mesa 3.5. This is not a migration
difference: the trajectories are identical across versions.

It *is* a substantive numerical-stability issue. Full Paper B simulations must
not silently accept such runs. The new pipeline therefore includes
`paper_b.validation.assert_finite_state()`, and production runners should fail
fast or mark a run invalid whenever a citizen belief or uncertainty becomes
non-finite.

Before the main simulation grid is launched, the source of the explosive
trajectory should be audited and either:

- corrected with a theoretically justified numerical/model fix, followed by a
  new backtest; or
- explicitly bounded by a defensible parameter-domain restriction.

Do not solve it by silently clipping beliefs after the fact.

## 2. Frozen-reliance semantics

The current model adapts source use through credibility learning and ranked
epsilon-greedy sampling. A frozen-reliance counterfactual can be implemented in
more than one defensible way, for example:

- freeze the initial source allocation at equal/static weights; or
- allow an initialization phase and then freeze the first realized credibility
  ranking.

Those are substantively different treatments. The scaffold therefore records
`reliance_mode` in configuration but does not yet change model behavior.

The exact frozen condition should be frozen in the analysis plan before code is
added.

## Next implementation order

1. Audit the non-finite trajectory.
2. Freeze the behavioral definition of adaptive vs frozen reliance.
3. Add one switch to the shared model; do not fork separate model classes.
4. Add time-indexed source-request/reliance logging.
5. Run a small paired adaptive/frozen milestone.
6. Only after that add redundancy and controlled-homophily experiments.

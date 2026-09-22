# Legacy Seed Reproducibility Audit

## Finding

The public Mesa 2.4 model is not pathwise reproducible from `np.random.seed(...)` alone when peer/network selection is active.

Two independent GitHub Actions executions of the same legacy commit, with the same 8 scenarios and the same three seeds, produced different snapshots in **21 of 24 runs**. All peer-network scenarios changed, including realized edge sets, convergence steps, and belief outcomes.

## Cause

`Citizen.get_neighbor_list()` in the public code returned:

```python
set(list(self.model.grid.get_neighbors(self.pos, include_center=False)))
```

The set removes duplicate agents introduced by the legacy NetworkGrid bookkeeping, but it also destroys stable ordering. Agent objects are hashed by runtime object identity, so set iteration order can differ across fresh Python processes and across Mesa versions.

Downstream code then converts this unordered collection into `citizen_list` / elite-source lists and uses positional NumPy sampling. Consequently, the same NumPy seed can select different counterparts or order the same information sources differently.

This is why direct Mesa 2.4 -> Mesa 3.5 matched-seed equality is not a meaningful migration criterion until neighbor ordering is normalized.

## Resolution used by the migration gate

The backtest harness applies the same deterministic normalization to both versions:

1. read all NetworkGrid neighbors;
2. deduplicate them by stable network position (`agent.pos`), preserving the old set's deduplication role;
3. return neighbors sorted by position.

The Mesa 3.5 source permanently uses this deterministic rule so future simulations can be replayed by seed.

## Interpretation

This normalization is a reproducibility correction, not a substantive change to who is structurally available as a neighbor. It fixes the order in which the same available agents are presented to stochastic selection.

Because archived Mesa 2.4 outputs were generated under process-dependent set ordering, exact reconstruction of an arbitrary archived run from its seed alone may be impossible unless the original process/object ordering was also preserved.

The migration gate therefore compares **deterministically normalized Mesa 2.4** against **deterministically normalized Mesa 3.5**. Historical manuscript patterns should still be rechecked separately against archived outputs at the aggregate/result level.

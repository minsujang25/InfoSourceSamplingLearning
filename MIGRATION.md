# Mesa 3.5 Migration Notes

This branch updates the public ISSL codebase from Mesa 2.4 to Mesa 3.5.1 while preserving the existing substantive model logic as closely as possible.

## Target version

- Python: 3.12
- Mesa: 3.5.1
- Mesa 4.0 is not used because it is still a pre-release.

## API changes

### Scheduler removal

Mesa 2.4 used:

```python
self.schedule = SimultaneousActivation(self)
self.schedule.step()
```

Mesa 3.5 uses the model's `AgentSet`:

```python
self.agents.do("step")
self.agents.do("advance")
```

This is Mesa's documented replacement for `SimultaneousActivation`.

### Automatic agent registration

Agents are automatically registered in `model.agents` when constructed. Explicit calls to `schedule.add()` were therefore removed.

### Automatic agent identifiers

Mesa 3.x assigns `Agent.unique_id` automatically. The identifier supplied by the legacy model is retained in `agent.legacy_id`. Network-space positions continue to use the original node labels through `agent.pos`.

### Step counter

References to `schedule.steps` were replaced by Mesa's built-in `model.steps`.

Mesa increments `model.steps` before entering the user-defined model `step()` body, so first-step initialization now checks:

```python
if self.steps == 1:
    ...
```

### Manuscript time index

The old model maintained its own `time` attribute. Mesa 3.5 reserves and manages simulation time internally, so the manuscript-specific discrete index is now named `period`.

This preserves the old jamming logic in which the disruptive jammer re-surveils citizens every five substantive periods.

### NetworkGrid placement

The legacy code explicitly appended agents to the NetworkX node after calling `NetworkGrid.place_agent()`. Mesa's space implementation already records the agent during placement. The duplicate append was removed.

## What has not been changed

This migration deliberately does **not** redesign the substantive model. In particular, it does not change:

- Bayesian belief updating;
- credibility-learning rules;
- epsilon-greedy source sampling;
- network matching logic;
- disruptive-jammer message optimization;
- stopping thresholds;
- the manuscript batch parameter grid.

Those components should be audited separately before new Paper B experiments are added.

## Validation status

The included smoke test checks that a small configuration can be instantiated and advanced under Mesa 3.5.

It does **not** establish numerical equivalence with archived Mesa 2.4 simulation outputs. Before using the migrated branch for new manuscript simulations, archived-result backtesting should compare old and new runs under controlled seeds and matched parameter settings.

## Recommended next validation step

1. Recreate a small subset of the original batch grid under the archived Mesa 2.4 environment.
2. Run the same configurations under Mesa 3.5.1.
3. Compare:
   - number of steps to convergence,
   - final citizen belief distributions,
   - half-step belief distributions,
   - realized network edges,
   - jammer cluster assignments where deterministic matching is possible.
4. Distinguish differences caused by Mesa API migration from pre-existing stochastic nondeterminism.

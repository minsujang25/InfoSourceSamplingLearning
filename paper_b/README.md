# Paper B Pipeline - Post-Migration Plan

This directory is intentionally documentation-only until the Mesa 2.4 -> 3.5 matched-seed backtest passes. The scientific revision should be a separate branch/PR from the framework migration.

## Intended architecture

```text
paper_b/
|-- README.md
|-- configs/
|   |-- baseline.yaml
|   |-- adaptive_frozen.yaml
|   |-- redundancy_ablation.yaml
|   `-- homophily_priors.yaml
|-- experiments/
|   |-- run_baseline.py
|   |-- run_adaptive_frozen.py
|   |-- run_redundancy_ablation.py
|   `-- run_homophily_priors.py
|-- analysis/
|   |-- effective_influence.py
|   |-- belief_outcomes.py
|   `-- contrasts.py
`-- figures/
    |-- density_rasters.py
    |-- adaptation_heatmap.py
    `-- homophily_prior_heatmap.py
```

The shared substantive model stays under `model/`; Paper B code should configure and measure it rather than fork it into a second model.

## First scientific milestone

After migration validation, run:

```text
4 communication environments
x 3 initial-belief regimes
x 2 reliance modes (adaptive / frozen)
x matched seeds
```

The goal is first to reproduce the old qualitative network-resilience patterns and isolate the incremental effect of adaptive reliance.

## Treatment design

Adaptive and frozen reliance must use the same model class and differ through one explicit configuration switch such as:

```python
adaptive_reliance = True  # or False
```

Do not create separate AdaptiveModel and FrozenModel implementations.

For each seed and structural network realization, paired conditions should share initial beliefs, network, expert/jammer environment, and random stream wherever feasible.

## Output schema

Store simulation outputs before plotting.

Run metadata should include:

- run_id and seed;
- network type and initial-belief condition;
- adaptive/frozen reliance;
- homophily and peer degree;
- expert access and peer retransmission;
- jammer stress condition, including K when it is varied.

Selected effective-influence checkpoints should store:

- ego position;
- source/alter position;
- source type (expert / jammer / peer);
- structural edge indicator;
- realized reliance weight or request share.

Run-level summaries should include belief MAE, belief dispersion, fragmentation/polarization, expert/jammer/peer reliance, effective-influence concentration, and effective homophily/segregation.

## Figure pipeline

The planned main-paper visual grammar is:

1. network structures;
2. structural vs effective influence mechanism;
3. binned belief-density raster replacing the large violin grids;
4. adaptive-vs-frozen heatmap;
5. homophily x prior-segregation heatmap;
6. redundancy/retransmission ablation.

Detailed K/surveillance sweeps remain secondary robustness material so Paper B retains a receiver/network-centered visual identity.

## Branching rule

Once the migration PR passes and is merged:

1. create `paper-b-pipeline` from validated `main`;
2. add the single adaptive/frozen switch and effective-influence outputs;
3. run the small first milestone;
4. only then add redundancy and controlled-homophily experiments.

This keeps framework migration, model correction, and substantive extension separately auditable.

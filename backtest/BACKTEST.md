# Mesa 2.4 -> 3.5 Matched-Seed Backtest

This backtest is the migration gate before any new Paper B experiment is added.

## Diagnostic parameter subset

The first-stage test uses **8 scenarios x 3 matched seeds = 24 runs per Mesa version**. It is deliberately small and is not intended to reproduce the full manuscript grid.

It covers:

- `baseline`, `random`, and `group_id_matching` network modes;
- 0, 2, and 5 maximum peer neighbors;
- flat, polarized, and consensus initial beliefs;
- jammer surveillance/segmentation at K = 1 and K = 4;
- both `delta_comparison` and `z_stat_comparison`.

Each scenario uses N = 42 and at most 40 substantive steps.

## Compared objects

For every scenario/seed pair the harness records:

1. exact number of model steps;
2. exact realized network edge set;
3. canonicalized jammer cluster partition;
4. the full citizen belief trajectory;
5. final citizen beliefs matched by network position;
6. final mean belief, standard deviation, and MAE from truth.

## Acceptance thresholds

The strict migration gate requires all structural objects to match exactly and:

- maximum absolute trajectory difference <= **1e-8**;
- trajectory RMSE <= **1e-10**;
- maximum absolute final-belief difference <= **1e-8**.

These are intentionally strict because this is a framework migration rather than a substantive model revision.

A manual-review band is also reported when structure is exact and:

- maximum absolute trajectory difference <= **1e-6**; and
- each final summary metric differs by <= **1e-6**.

A review-band result is not an automatic pass.

If pathwise equivalence fails materially, do not proceed to Paper B experiments. Locate the earliest divergence first. If the cause is unavoidable numerical-library nondeterminism, run a second distributional backtest with at least 30 matched seeds per affected scenario.

## CI execution

`.github/workflows/mesa_backtest.yml` runs:

- legacy `main` under Python 3.11 + Mesa 2.4;
- the migration branch under Python 3.12 + Mesa 3.5.1;
- identical pinned NumPy/SciPy/scikit-learn/NetworkX versions where possible;
- a comparison job that fails if the strict gate is not met.

Only after this gate passes should substantive Paper B changes be added, in a separate branch/PR.

# Targeted T=400 horizon sensitivity

This run is the calibration follow-up to the 20-seed T=200 pilot. It targets the two sparse/local network environments where the first pilot showed the largest late-horizon movement and seed-to-seed variation.

## Design

```text
20 matched seeds: 1001-1020
x 3 initial-belief regimes: flat, consensus, polarized
x 2 network environments: random_2, group_id
x 2 Jammer states
x 2 reliance modes
= 480 individual simulations
```

Fixed parameters:

```text
K = 1
epsilon = 0.05
credit = 20
citizens = 100
surveillance interval = 5
terminal horizon T = 400
```

The run intentionally omits `elite_only` and `extended` because the T=200 pilot showed little late-horizon movement there. This is a targeted horizon-calibration exercise, not a replacement for the four-environment production design.

## Why rerun to T=400 instead of only extending old shards

The T=400 run starts each matched trajectory from the same seed and design specification and records internal checkpoints at both:

```text
T = 200  -> checkpoint period 199
T = 400  -> checkpoint period 399
```

This makes the T=200 versus T=400 comparison come from the same uninterrupted trajectory.

The runner also records intermediate checkpoints at T=101, 151, 300, and other standard mechanism checkpoints.

## Run locally

Activate the validated environment:

```bash
conda activate paper-b
```

For a 12-core local run:

```bash
PAPER_B_WORKERS=12 bash scripts/run_paper_b_horizon_sensitivity.sh
```

The wrapper constrains BLAS/OpenMP libraries to one thread per worker, so do not set `PAPER_B_WORKERS` above the CPU count you want the pilot to use.

To use another output location:

```bash
PAPER_B_WORKERS=12 \
PAPER_B_OUTPUT_DIR=/path/to/results \
bash scripts/run_paper_b_horizon_sensitivity.sh
```

The wrapper uses `--resume`, so rerunning the identical command after interruption skips valid completed matched blocks.

## New network subset option

The generic matched-pilot runner now accepts:

```bash
--network-environments random_2,group_id
```

Any nonempty unique subset of the four canonical environments is accepted:

```text
elite_only
random_2
group_id
extended
```

The selected subset is part of the design identity and manifest.

## Output

The standard pilot outputs are created, plus two horizon-specific tables when both T=200 and T=400 checkpoints are present:

```text
horizon_sensitivity.csv
adaptive_frozen_horizon_sensitivity.csv
```

`horizon_sensitivity.csv` contains, for each seed x prior x network x reliance mode:

- D_MSE at T=200;
- D_MSE at T=400;
- signed and absolute change in D_MSE;
- corresponding RMSE and MAE disruption changes; and
- J=1 and J=0 MSE at both horizons.

`adaptive_frozen_horizon_sensitivity.csv` contains:

- adaptive-minus-frozen D_MSE at T=200;
- adaptive-minus-frozen D_MSE at T=400; and
- the change between those horizons.

Both files are automatically included in the shareable result ZIP.

## Decision rule after the run

The run is intended to answer one question: whether T=200 is an adequate primary fixed horizon.

The analysis should focus on the distribution of

```text
D_g(400) - D_g(200)
```

and

```text
[D_g^A(400)-D_g^F(400)]
-
[D_g^A(200)-D_g^F(200)].
```

Inspect these overall and separately by prior regime, network environment, and reliance mode, with special attention to heavy tails rather than only means.

If late-horizon movement is negligible for nearly all matched trajectories, T=200 can be frozen as the primary production horizon and T=400 retained as a sensitivity check. If material movement persists systematically, the production horizon should be reconsidered before scaling to hundreds of seeds.

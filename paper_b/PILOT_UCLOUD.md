# Paper B matched calibration pilot on UCloud

This pilot is the computational gate after Reconstruction Round 2. Its purpose is to calibrate the production design, not to produce final manuscript estimates.

## Default design

```text
20 matched seeds
x 3 initial-belief regimes: flat, consensus, polarized
x 4 network environments: elite_only, random_2, group_id, extended
x 2 Jammer states: J=1, J=0
x 2 reliance modes: adaptive, frozen
= 960 individual simulations
```

Fixed calibration parameters:

```text
K = 1
epsilon = 0.05
credit = 20
comparison rule = delta_comparison
surveillance interval = 5
citizens = 100
terminal horizon T = 200
```

K is kept fixed at this stage so the pilot isolates the receiver-side network and adaptive-reliance design.

## Matched-block parallelization

The parallel unit is:

```text
(seed, initial-belief regime, network environment)
```

Each worker executes four conditions sequentially:

```text
adaptive / J=1
adaptive / J=0
frozen   / J=1
frozen   / J=0
```

Before a shard is saved, the runner verifies that the four conditions share the same initial-state fingerprint and structural-network fingerprint.

## Environment

```bash
conda env create -f environment.yml
conda activate paper-b
```

If the environment already exists, only activate it. Python should report version 3.12.x.

## UCloud execution

Set `PAPER_B_WORKERS` equal to the CPU cores allocated to the job.

For example:

```bash
PAPER_B_WORKERS=16 bash scripts/run_paper_b_pilot_ucloud.sh
```

To choose another results directory, set `PAPER_B_OUTPUT_DIR` as well.

Do not set the worker count above the allocated CPU count. The runner uses one process per matched block and does not create nested worker pools.

## Resume

The wrapper uses `--resume`. One atomic compressed JSON shard is written per matched block. If a run is interrupted, rerun the same command. Valid completed shards are skipped, while missing or incomplete blocks are rerun.

The exact design gets a deterministic design ID:

```text
ucloud_results/paper_b_pilot/pilot_<design_id>/
```

The design ID includes both the simulation parameters and a fingerprint of the scientific model/metric/runner code. Changing either the design or the reconstruction code therefore creates a new pilot directory and prevents incompatible shards from being mixed during `--resume`.

## Consolidated outputs

After all shards finish, the runner automatically creates:

```text
pilot_manifest.json
block_manifest.csv
runs.csv
jammer_contrasts.csv
adaptive_frozen_contrasts.csv
terminal_beliefs.csv.gz
lambda_checkpoints.csv
belief_checkpoints.csv
jammer_strategy_trajectory.csv.gz
reliance_checkpoints.csv.gz
pilot_summary.json
pilot_gate.json
shards/
```

The shard directory is the resumable source of truth. Consolidated outputs can be rebuilt with:

```bash
python -m paper_b.experiments.consolidate_pilot <pilot-directory>
```

and validated with:

```bash
python -m paper_b.experiments.check_pilot_output <pilot-directory>
```

## Scientific gate

The pilot passes only if:

1. every expected block exists;
2. every block contains the four matched conditions;
3. initial-state fingerprints match within each quartet;
4. structural-network fingerprints match within each quartet;
5. every citizen state is finite;
6. every run reaches the exact common horizon T;
7. frozen post-audit Lambda dynamics are zero;
8. Jammer message means remain fixed inside each surveillance window; and
9. Jammer refreshes occur at the scheduled periods.

The runner exits non-zero if the gate fails.

After a PASS, it also creates a compact upload bundle next to the pilot directory:

```text
paper_b_pilot_<design_id>_shareable.zip
```

This ZIP contains the consolidated scientific outputs but not the hundreds of resumable shard files. Keep the shard directory on UCloud for recovery, and upload the shareable ZIP for the next result audit.

## What to inspect after PASS

Use the larger pilot to inspect the distribution of Delta MSE by network, prior regime, and reliance mode; adaptive-minus-frozen Delta MSE; seed-to-seed tails in Random 2 and Group ID; J=0 error in sparse networks; convergence-crossing times relative to T=200; late-horizon MSE movement using belief checkpoints; Lambda movement; and Jammer message-mean tails. These diagnostics determine the final T and the production seed count.

## Targeted network subsets and horizon sensitivity

The generic runner accepts a network subset through:

```bash
--network-environments random_2,group_id
```

This is used by the targeted T=400 calibration follow-up. See
`paper_b/HORIZON_SENSITIVITY.md` and run it locally or on UCloud with:

```bash
PAPER_B_WORKERS=12 bash scripts/run_paper_b_horizon_sensitivity.sh
```

That sensitivity design runs 20 seeds x 3 priors x 2 sparse/local networks x 4
matched conditions = 480 simulations at fixed T=400 and automatically produces
T=200 versus T=400 disruption tables.

## Tiny smoke test

```bash
python -m paper_b.experiments.run_matched_pilot \
  --seeds 1 \
  --seed-start 1001 \
  --initial-regimes flat \
  --n-citizens 20 \
  --horizon 8 \
  --workers 2 \
  --output-dir /tmp/paper_b_pilot_smoke
```

This runs 16 individual simulations and exercises sharding, consolidation, and validation.

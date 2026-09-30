# Running the Paper B diagnostic locally

Run all commands from the repository root.

## 1. Create/activate the validated environment

The repository now ships with a canonical `environment.yml` that pins Python 3.12 and
the same scientific stack used in GitHub CI.

```bash
conda env create -f environment.yml
conda activate paper-b
```

If the environment already exists:

```bash
conda activate paper-b
```

Quick check:

```bash
python --version
```

It should report Python 3.12.x.

## 2. Reconstruction-round-2 diagnostic

The wrapper first runs all reconstruction checks and then executes the matched grid:

```bash
bash scripts/run_paper_b_diagnostic.sh
```

Default workload:

```text
4 environments
x 2 jammer states
x 2 reliance modes
x 5 matched seeds
x flat initial beliefs
= 80 runs
```

with 100 citizens and an exact fixed horizon of T=100 periods for every run.

Early convergence is recorded but never used to terminate the primary diagnostic. This
ensures every J=1/J=0 and adaptive/frozen comparison is evaluated at the same T.

## 3. Output

The runner creates a timestamped directory under:

```text
local_results/paper_b_diagnostic/
```

and one ZIP:

```text
paper_b_diagnostic_YYYYMMDD_HHMMSS.zip
```

Upload that ZIP back to ChatGPT.

It contains:

- `manifest.json` — exact run arguments, Git commit, and design metadata;
- `runs.csv` — one row per simulation condition, including fixed-horizon and
  effective-reliance-dynamics metrics;
- `jammer_contrasts.csv` — matched J=1 minus J=0 disruption, including Delta MSE;
- `adaptive_frozen_contrasts.csv` — matched adaptive-minus-frozen disruption contrasts;
- `terminal_beliefs.csv.gz` — citizen-level terminal beliefs;
- `lambda_checkpoints.csv` — compact selected-period Lambda summaries;
- `reliance_checkpoints.csv.gz` — selected-period edge-level Lambda/X records;
- `jammer_strategy_trajectory.csv.gz` — period-by-period Jammer segment state and
  message means;
- `summary.json` — run counts, finite-state status, and fixed-horizon status.

## 4. What changed after the first 80-run diagnostic

Reconstruction round 2 makes three scientific changes:

1. **Fixed terminal horizon.** Every primary run executes exactly T periods.
2. **Jammer objective.** The sender now optimizes the documented one-step disruptive
   objective at surveillance refreshes and holds the chosen segment message mean fixed
   until the next refresh. The undocumented within-window recurrence has been removed.
3. **Dynamic Lambda metrics.** The output now measures change in the identity of
   highly weighted ties after the first credibility audit, not only the static distance
   from equal structural use.

See `paper_b/JAMMER_OBJECTIVE_AUDIT.md` for the Jammer derivation.

## 5. Selected-period Lambda diagnostics

The default diagnostic stores checkpoints at:

```text
t = 0, 1, 5, 10, 25, 50, T
```

when those periods exist.

The compact output reports:

- Expert/Jammer/peer expected reliance;
- Lambda distance from the first post-audit effective network;
- effective homophily;
- total peer reliance.

The run-level output also reports:

- first-audit-to-terminal Lambda TV distance;
- mean period-to-period Lambda turnover;
- cumulative Lambda turnover;
- share of citizens whose top source changed;
- mean number of top-source switches per citizen.

For a correctly implemented frozen condition, these post-audit Lambda-dynamics measures
should be zero.

## 6. Jammer strategy diagnostics

`jammer_strategy_trajectory.csv.gz` records for each active-Jammer segment and period:

- surveillance refresh indicator;
- observed segment mean and dispersion;
- Gaussian response gain;
- optimized message mean;
- cluster size.

This is the first file to inspect if a run produces unusually large disruption.

## 7. Larger follow-up

Do not scale up until the new default 80-run diagnostic has been inspected.

After that gate passes, a useful next local/UCloud pilot is:

```bash
bash scripts/run_paper_b_diagnostic.sh \
  --seeds 20 \
  --initial-regimes flat,consensus,polarized \
  --n-citizens 100 \
  --max-steps 200
```

## Useful options

```text
--seeds 5
--seeds 1001,1002,1003
--initial-regimes flat
--initial-regimes flat,consensus,polarized
--n-citizens 100
--max-steps 100
--k 1
--epsilon 0.05
--credit 20
--comparison-rule delta_comparison
--no-edge-log
--output-dir local_results/paper_b_diagnostic
```

## Pairing rule

Within each seed and initial-belief regime, the same initial belief vector and the same
model seed are reused across:

```text
J=1 / J=0
adaptive / frozen
```

for each structural environment.

The primary matched estimand is therefore evaluated at the common horizon T:

```text
D_g(K; T) = MSE_T(J=1,g,K) - MSE_T(J=0,g).
```

## Production environment names

```text
elite_only
random_2
group_id
extended
```

Do not use the old `mode=random` or `mode=group_id_matching` paths for manuscript
simulations; those remain only for legacy auditing.

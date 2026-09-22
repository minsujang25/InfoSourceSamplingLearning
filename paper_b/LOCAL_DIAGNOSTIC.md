# Running the Paper B diagnostic locally

Run all commands from the repository root.

## 1. Create/activate a clean environment (recommended)

On macOS/Linux:

```bash
python3 -m venv .venv-paper-b
source .venv-paper-b/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "mesa==3.5.1" \
  "numpy==1.26.4" \
  "scipy==1.13.1" \
  "scikit-learn==1.5.2" \
  "networkx==3.3"
```

Python 3.12 is the tested version.

## 2. First diagnostic (recommended)

The wrapper first runs the reconstruction checks and then the matched simulation grid:

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

with 100 citizens and at most 100 model periods.

## 3. Output

The runner creates a timestamped directory under:

```text
local_results/paper_b_diagnostic/
```

and also creates one ZIP:

```text
paper_b_diagnostic_YYYYMMDD_HHMMSS.zip
```

Upload that ZIP back to ChatGPT.

It contains:

- `manifest.json` — exact run arguments and Git commit;
- `runs.csv` — one row per simulation condition;
- `jammer_contrasts.csv` — matched J=1 minus J=0 disruption, including Delta MSE;
- `adaptive_frozen_contrasts.csv` — matched adaptive-minus-frozen disruption contrasts;
- `terminal_beliefs.csv.gz` — citizen-level terminal beliefs;
- `summary.json` — run counts and finite-state status.

## 4. Optional edge-level mechanism log

To save selected-period Lambda/X edge records:

```bash
bash scripts/run_paper_b_diagnostic.sh --save-edge-log
```

This adds `reliance_checkpoints.csv.gz`.

## 5. Larger follow-up after the first ZIP is checked

Do not start here. First run the default 80-run diagnostic.

After it passes inspection, a useful next run is:

```bash
bash scripts/run_paper_b_diagnostic.sh \
  --seeds 20 \
  --initial-regimes flat,consensus,polarized \
  --n-citizens 100 \
  --max-steps 200 \
  --save-edge-log
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
--save-edge-log
--output-dir local_results/paper_b_diagnostic
```

## Pairing rule

Within each seed and initial-belief regime, the same initial belief vector and the same model seed are reused across:

```text
J=1 / J=0
adaptive / frozen
```

for each structural environment.

The run therefore supports the primary disruption estimand

```text
D_g(K) = MSE(J=1,g,K) - MSE(J=0,g)
```

and the adaptive-vs-frozen contrast in D_g(K).

## Important

The four production environment names are now frozen as:

```text
elite_only
random_2
group_id
extended
```

Do not use the old `mode=random` or `mode=group_id_matching` paths for manuscript simulations; those are retained only for legacy auditing.

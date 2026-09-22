# Information Source Sampling and Learning (ISSL)

Open-source research code supporting the manuscript **“Strategic Disinformation, Adaptive Reliance, and Network Resilience: An Agent-Based Model of Collective Belief Formation.”** The model was developed by [Minsu Jang](https://github.com/minsujang25) in Python using the [Mesa](https://mesa.readthedocs.io/) agent-based modelling framework.

## Research scope

The model examines collective belief formation when citizens sample information from providers and other citizens, assess source credibility or bias, and update beliefs across networked interactions. The implementation supports alternative learning rules, network structures, initial belief distributions, and an adversarial information provider with varying surveillance capacity.

## Repository structure

| Path | Purpose |
| --- | --- |
| [`model/InfoSourceSamplingLearning.py`](model/InfoSourceSamplingLearning.py) | Core model, agent classes, network builders, and belief-updating dynamics. |
| [`model/ISSL_MultiProcessor.py`](model/ISSL_MultiProcessor.py) | Multiprocessing helpers for repeated simulation runs. |
| [`scripts/batch_run_jamming.py`](scripts/batch_run_jamming.py) | Batch configuration for the adversarial-jamming experiment. |
| [`InfoSourceSamplingLearning.yml`](InfoSourceSamplingLearning.yml) | Conda environment specification for Python 3.11 and Mesa 2.4. |

## Environment setup

From a terminal:

```bash
git clone https://github.com/minsujang25/InfoSourceSamplingLearning.git
cd InfoSourceSamplingLearning
conda env create -f InfoSourceSamplingLearning.yml
conda activate InfoSourceSamplingLearning
```

The supplied environment now targets **Python 3.12** and **Mesa 3.5.1**, the latest stable Mesa release as of September 2026. Mesa 4.0 remains a pre-release and is intentionally not used here.

## Code entry points

### Import the model

The core model is intended to be imported and configured from Python:

```python
from model.InfoSourceSamplingLearning import (
    Citizen,
    DisruptiveJammer,
    InfoProvider,
    InfoSampleModel,
)
from model.ISSL_MultiProcessor import run_mp
```

`InfoSampleModel` expects a complete `model_attribute` dictionary for the manuscript experiments. The parameter schema and a concrete configuration are documented in [`scripts/batch_run_jamming.py`](scripts/batch_run_jamming.py). This repository does not currently include a separate small-demo runner.

### Run the batch experiment

From the repository root, with the Conda environment active:

```bash
python -m scripts.batch_run_jamming
```

The script creates `batch_result/` and appends serialized results to `batch_result/batch_result(n=500)_jamming.pkl`.

> **Computational note:** the checked-in configuration expands to 720 parameter combinations and 100 repetitions per combination (72,000 model runs), with up to 10,000 steps per run. It is a full batch experiment, not a quick smoke test. Adjust the parameter lists and `iterations` in the script before running on limited hardware. Re-running the script appends another result object to the same output file.

## Connection to the manuscript

The repository provides the computational model, parallel runner, and adversarial-jamming batch definition used to support the manuscript’s agent-based analysis. It does not currently include the manuscript text, precomputed simulation outputs, figure-generation code, or a tagged release. A lightweight smoke test is included to verify that the migrated Mesa 3.5 model can initialize and advance a small simulation; it is not a substantive validation of manuscript results.

## Mesa 3.5 migration

The current codebase has been migrated from the legacy Mesa 2.4 scheduler API to Mesa 3.5's `AgentSet` API. In particular:

- `SimultaneousActivation` was replaced by `model.agents.do("step")` followed by `model.agents.do("advance")`;
- agents are registered automatically with `model.agents`;
- Mesa's built-in `model.steps` counter replaces `schedule.steps`;
- Mesa now assigns agent `unique_id` values automatically, while the pre-migration identifiers are retained as `legacy_id`;
- the model uses a separate `period` counter for the manuscript's substantive time index.

See [`MIGRATION.md`](MIGRATION.md) for details and caveats.

### Quick smoke test

After creating the environment:

```bash
python -m scripts.smoke_test
```

This runs only a tiny configuration and should complete quickly.

## License

This project is released under the [MIT License](LICENSE).

# InfoSourceSamplingLearning

This repository contains a simulation model of **information sampling and belief learning** in a networked environment. The model is built using the [Mesa](https://mesa.readthedocs.io/en/stable/) agent-based modeling framework (version 2.4), and is designed to study how agents selectively sample sources, update beliefs, and interact with one another under varying assumptions about credibility, uncertainty, and network structures.

## 🧠 Overview

The simulation explores how social learning dynamics evolve when agents:
- Encounter multiple sources of information
- Evaluate credibility and consistency
- Choose whether to sample, share, or ignore sources
- Update their beliefs over time through reinforcement or Bayesian learning

This model is particularly useful for examining:
- The spread of misinformation or disinformation
- Polarization dynamics in belief systems
- Influence of network structure on opinion convergence

## 🚀 Getting Started

### 🧪 Environment Setup (via Conda)

To ensure compatibility—especially with `mesa==2.4`, which is required for this simulation—please create the environment using the provided `.yml` file.

```bash
# Step 1: Clone the repository
git clone https://github.com/minsujang25/InfoSourceSamplingLearning.git
cd InfoSourceSamplingLearning

# Step 2: Create the environment from the .yml file
conda env create -f InfoSourceSamplingLearning.yml

# Step 3: Activate the environment
conda activate InfoSourceSamplingLearning
```
ℹ️ Note: This environment uses Python 3.11 and pins mesa==2.4, which allows manual parameter inputs via dictionaries—a feature deprecated in later versions.

## 🏃‍♂️ Running the Simulation

You can run the simulation using command like the following:
```bash
python scripts/run_simulation.py # there is no such file yet 
```
Parameters such as the number of agents, steps, or network topology can be configured inside the script you customize.

### 🔁 Example: Batch Running the Simulation

To run multiple simulations (e.g., varying parameters or simulating adversarial scenarios), use the batch execution script:

```bash
python scripts/batch_run_jamming.py
```

## 📊 Example Use Cases
-	Modeling belief updating under biased information processing
-	Simulating disinformation campaigns
-	Studying trust and source credibility in dynamic information ecosystems

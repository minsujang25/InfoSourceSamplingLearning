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
git clone https://github.com/your-username/InfoSourceSamplingLearning.git
cd InfoSourceSamplingLearning

# Step 2: Create the environment from the .yml file
conda env create -f InfoSourceSamplingLearning.yml

# Step 3: Activate the environment
conda activate InfoSourceSamplingLearning

"""Small Mesa 3.5 smoke test for the ISSL model.

This is intentionally tiny. It verifies initialization and a few model steps;
it is not a reproduction test for the manuscript simulations.
"""

import numpy as np

from model.InfoSourceSamplingLearning import (
    Citizen,
    DisruptiveJammer,
    InfoProvider,
    InfoSampleModel,
)


def build_smoke_config():
    n = 12
    rng = np.random.default_rng(12345)

    mu_theta = [0.0, 4.0] + list(rng.uniform(-1.0, 1.0, n - 2))
    sd_theta = [1.0, 1.0] + [5.0] * (n - 2)

    return {
        "state_of_the_world": 0.0,
        "num_nodes": n,
        "comparison_rule": "delta_comparison",
        "epsilon": 0.05,
        "credit": 20,
        "mu_delta": [[0.0, 0.0, 0.0] for _ in range(n)],
        "sd_delta": [[5.0, 5.0, 5.0] for _ in range(n)],
        "mu_theta": mu_theta,
        "sd_theta": sd_theta,
        "initial_theta_type": "flat",
        "seq_meaningful": True,
        "type_of_agent": [InfoProvider, DisruptiveJammer] + [Citizen] * (n - 2),
        "max_steps": 10,
        "network_type": "fully_connected",
        "mode": "random",
        "network_structure": None,
        "learn_method": "cautious",
        "counterpart_pick_mechanism": "equal",
        "surveil_ability": 1,
        "num_max_citizen_neighbor": 1,
    }


def main():
    np.random.seed(12345)
    model = InfoSampleModel(model_attribute=build_smoke_config(), rng=12345)

    for _ in range(3):
        model.step()

    citizens = [agent for agent in model.agents if agent.type_of_agent == "citizen"]

    assert model.steps == 3
    assert model.period == 3
    assert len(citizens) == 10
    assert len(model.agent_mu_theta_list) == 3
    assert all(len(agent.mu_theta_beliefs) == 4 for agent in citizens)

    print(
        "Mesa 3.5 smoke test passed:",
        f"steps={model.steps}, period={model.period}, citizens={len(citizens)}",
    )


if __name__ == "__main__":
    main()

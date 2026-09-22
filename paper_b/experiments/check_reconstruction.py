"""Deterministic scientific checks for the Paper B theory reconstruction.

These checks are intentionally small.  They validate theory/code invariants,
not manuscript results.
"""

from __future__ import annotations

import math

import numpy as np

from model.InfoSourceSamplingLearning import (
    Citizen,
    DisruptiveJammer,
    InfoProvider,
    InfoSampleModel,
    recursive_rank_probabilities,
)
from paper_b.metrics import reliance_composition, theory_metrics
from paper_b.validation import assert_finite_state


def build_config(
    *,
    reliance_mode="adaptive",
    network_environment="extended",
    jammer_k=1,
    seed=20260922,
):
    n = 14
    rng = np.random.default_rng(seed)
    citizen_mu = list(rng.normal(0.0, 1.0, n - 2))
    return {
        "state_of_the_world": 0.0,
        "num_nodes": n,
        "comparison_rule": "delta_comparison",
        "epsilon": 0.05,
        "credit": 20,
        "mu_delta": [[0.0] * 6 for _ in range(n)],
        "sd_delta": [[5.0] * 6 for _ in range(n)],
        "mu_theta": [0.0, 4.0] + citizen_mu,
        "sd_theta": [1.0, 1.0] + [5.0] * (n - 2),
        "initial_theta_type": "flat",
        "seq_meaningful": True,
        "type_of_agent": [InfoProvider, DisruptiveJammer] + [Citizen] * (n - 2),
        "max_steps": 30,
        "network_type": "fully_connected",
        "mode": "baseline",
        "network_environment": network_environment,
        "reliance_mode": reliance_mode,
        "peer_degree": 2,
        "same_group_probability": 0.9,
        "elite_access_probability": 0.25,
        "jammer_active": True,
        "surveil_ability": jammer_k,
        "surveillance_interval": 5,
        "num_max_citizen_neighbor": 2,
        "learn_method": "cautious",
        "counterpart_pick_mechanism": "equal",
    }


def check_rank_probabilities():
    assert np.allclose(recursive_rank_probabilities(2, 0.05), [0.95, 0.05])
    assert np.allclose(
        recursive_rank_probabilities(4, 0.05),
        [0.95, 0.0475, 0.002375, 0.000125],
    )
    assert math.isclose(float(recursive_rank_probabilities(7, 0.3).sum()), 1.0)


def check_theta_sd_update():
    model = InfoSampleModel(
        model_attribute=build_config(network_environment="elite_only"),
        rng=11,
    )
    citizen = model.citizens[0]
    prior_sd = citizen.sd_theta_beliefs[-1]
    _, posterior_sd = citizen.bayesian_update_theta([0.0, 10.0])
    assert posterior_sd < prior_sd
    assert posterior_sd > 0.0


def check_synchronous_message_snapshot():
    model = InfoSampleModel(
        model_attribute=build_config(network_environment="extended"),
        rng=12,
    )
    sender = model.citizens[0]

    sender.mu_theta = 1.25
    sender.sd_theta = 1e-8
    sender.snapshot_message_state()

    # A within-period posterior mutation must not change the already-snapshotted
    # message distribution used in period t.
    sender.mu_theta = 9.0
    sender.sd_theta = 1e-8
    draws = sender.mu_out(4, model.citizens[1])
    assert np.allclose(draws, [1.25] * 4, atol=1e-6)


def check_first_audit_freeze():
    model = InfoSampleModel(
        model_attribute=build_config(
            network_environment="extended",
            reliance_mode="frozen",
        ),
        rng=13,
    )
    model.step()  # period 0 is a credibility audit for every citizen
    citizen = model.citizens[0]
    frozen = list(citizen.frozen_credibility_ranked_sources)
    assert frozen

    citizen.credibility_ranked_sources = list(reversed(frozen))
    assert citizen._behavioral_ranking() == frozen


def check_jammer_resurveillance_uses_current_beliefs():
    model = InfoSampleModel(
        model_attribute=build_config(
            network_environment="elite_only",
            jammer_k=1,
        ),
        rng=14,
    )
    jammer = model.jammer
    assert jammer is not None

    model._snapshot_message_states()
    jammer.surveil_citizen()
    first = float(jammer.citizen_intel["centroids"][0])

    for citizen in model.citizens:
        citizen.mu_theta = 10.0
        citizen.mu_theta_beliefs[-1] = 10.0

    model._snapshot_message_states()
    jammer.surveil_citizen()
    second = float(jammer.citizen_intel["centroids"][0])

    assert second > first + 5.0
    assert math.isclose(second, 10.0, abs_tol=1e-8)


def check_finite_multiperiod_run():
    model = InfoSampleModel(
        model_attribute=build_config(
            network_environment="homophilous_peer",
            jammer_k=4,
        ),
        rng=15,
    )
    for _ in range(20):
        model.step()
        assert_finite_state(model, context="reconstruction check")
        assert all(c.sd_theta > 0.0 for c in model.citizens)

    composition = reliance_composition(model)
    total = sum(composition.values())
    assert math.isclose(total, 1.0, rel_tol=1e-8, abs_tol=1e-8)

    metrics = theory_metrics(model)
    assert math.isfinite(metrics["mse_truth"])
    assert math.isfinite(metrics["structural_effective_divergence"])


def main():
    checks = [
        check_rank_probabilities,
        check_theta_sd_update,
        check_synchronous_message_snapshot,
        check_first_audit_freeze,
        check_jammer_resurveillance_uses_current_beliefs,
        check_finite_multiperiod_run,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"Paper B reconstruction checks passed: {len(checks)}/{len(checks)}")


if __name__ == "__main__":
    main()

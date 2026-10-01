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
from paper_b.metrics import (
    effective_reliance_dynamics,
    reliance_composition,
    theory_metrics,
)
from paper_b.experiments.consolidate_pilot import _horizon_sensitivity_rows
from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    exp3_redundancy_source_maps,
    exp3b_focal_corrective_connectivity,
    exp3b_path_independence_source_maps,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
    realized_prior_segregation,
    structural_peer_homophily,
    two_step_expert_route_count,
)
from paper_b.validation import assert_finite_state


def build_config(
    *,
    reliance_mode="adaptive",
    network_environment="extended",
    jammer_k=1,
    seed=20260922,
    n=14,
):
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
        "local_degree": 2,
        "peer_degree": 2,
        "same_group_probability": 0.9,
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


def check_fixed_horizon_execution():
    config = build_config(network_environment="elite_only")
    config["max_steps"] = 8
    config["convergence_tolerance"] = 1e9
    config["stop_on_convergence"] = False

    model = InfoSampleModel(model_attribute=config, rng=101)
    model.run_model(print_time=False)

    assert model.steps == 8
    assert model.period == 8


def check_jammer_objective_and_hold_rule():
    means = np.asarray([1.5, -0.5, 2.0], dtype=float)
    sds = np.asarray([1.0, 2.0, 0.5], dtype=float)
    message, diagnostics = DisruptiveJammer.optimal_message_mean(
        citizen_means=means,
        citizen_sds=sds,
        truth=0.0,
        underlying_position=4.0,
    )
    kappas = np.asarray(
        [DisruptiveJammer.posterior_response_gain(sd) for sd in sds],
        dtype=float,
    )

    def utility(m):
        post = (1.0 - kappas) * means + kappas * m
        return float(np.mean(post**2) - (m - 4.0) ** 2)

    assert math.isfinite(message)
    assert utility(message) > utility(message - 1.0)
    assert utility(message) > utility(message + 1.0)
    assert 0.0 < diagnostics["objective_denominator"] <= 1.0
    assert diagnostics["response_gain_max"] < 1.0

    # The documented deviation cost is centered on the Jammer's own position.
    shifted_message, _ = DisruptiveJammer.optimal_message_mean(
        citizen_means=means,
        citizen_sds=sds,
        truth=0.0,
        underlying_position=8.0,
    )
    assert shifted_message > message

    # Cross-sectional disagreement must not determine Bayesian response gain.
    # With the Paper B initial citizen SD of 5, the maximum individual gain is
    # exactly 25/26 no matter how far apart the citizen means are.
    dispersed_message, dispersed = DisruptiveJammer.optimal_message_mean(
        citizen_means=[-1e9, 1e9],
        citizen_sds=[5.0, 5.0],
        truth=0.0,
        underlying_position=4.0,
    )
    max_initial_gain = 25.0 / 26.0
    expected_denom = 1.0 - max_initial_gain**2
    assert math.isfinite(dispersed_message)
    assert math.isclose(
        dispersed["response_gain_max"],
        max_initial_gain,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        dispersed["objective_denominator"],
        expected_denom,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    model = InfoSampleModel(
        model_attribute=build_config(
            network_environment="elite_only",
            jammer_k=1,
        ),
        rng=102,
    )
    jammer = model.jammer
    assert jammer is not None

    model._snapshot_message_states()
    model.period = 0
    first = jammer.prepare_for_period()
    first_mean = first[0]["message_mean"]
    assert first[0]["refresh"] is True
    assert first[0]["response_gain_max"] <= max_initial_gain + 1e-12
    assert first[0]["objective_denominator"] >= expected_denom - 1e-12

    # Change the audience state inside the surveillance window. The strategy
    # must remain fixed until the next scheduled refresh.
    for citizen in model.citizens:
        citizen.mu_theta = 10.0
        citizen.mu_theta_beliefs[-1] = 10.0
    model._snapshot_message_states()

    model.period = 1
    held = jammer.prepare_for_period()
    assert held[0]["refresh"] is False
    assert math.isclose(
        held[0]["message_mean"],
        first_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    model.period = model.surveillance_interval
    refreshed = jammer.prepare_for_period()
    assert refreshed[0]["refresh"] is True
    assert not math.isclose(
        refreshed[0]["message_mean"],
        first_mean,
        rel_tol=0.0,
        abs_tol=1e-8,
    )


def check_effective_reliance_dynamics():
    frozen = InfoSampleModel(
        model_attribute=build_config(
            network_environment="extended",
            reliance_mode="frozen",
        ),
        rng=103,
    )
    for _ in range(10):
        frozen.step()

    frozen_metrics = effective_reliance_dynamics(frozen)
    assert math.isclose(
        frozen_metrics["lambda_first_audit_to_terminal_tv"],
        0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        frozen_metrics["lambda_cumulative_turnover"],
        0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        frozen_metrics["lambda_top_source_changed_share"],
        0.0,
        abs_tol=1e-12,
    )

    adaptive = InfoSampleModel(
        model_attribute=build_config(
            network_environment="extended",
            reliance_mode="adaptive",
        ),
        rng=104,
    )
    for _ in range(10):
        adaptive.step()

    adaptive_metrics = effective_reliance_dynamics(adaptive)
    assert adaptive_metrics["lambda_first_audit_to_terminal_tv"] >= 0.0
    assert adaptive_metrics["lambda_cumulative_turnover"] >= 0.0


def check_horizon_sensitivity_contrasts():
    rows = []
    # Adaptive: D_200 = 3-1 = 2; D_400 = 5-2 = 3.
    # Frozen:   D_200 = 2-1 = 1; D_400 = 3-1 = 2.
    values = {
        ("adaptive", True, 200): 3.0,
        ("adaptive", False, 200): 1.0,
        ("adaptive", True, 400): 5.0,
        ("adaptive", False, 400): 2.0,
        ("frozen", True, 200): 2.0,
        ("frozen", False, 200): 1.0,
        ("frozen", True, 400): 3.0,
        ("frozen", False, 400): 1.0,
    }
    for (reliance, jammer, horizon), mse in values.items():
        rows.append(
            {
                "seed": 1001,
                "initial_regime": "flat",
                "network_environment": "random_2",
                "reliance_mode": reliance,
                "jammer_active": jammer,
                "K": 1,
                "period": horizon - 1,
                "horizon_step": horizon,
                "mse_truth": mse,
                "rmse_truth": math.sqrt(mse),
                "mae_truth": mse / 2.0,
            }
        )

    disruption, adaptive_frozen = _horizon_sensitivity_rows(rows)
    assert len(disruption) == 2
    assert len(adaptive_frozen) == 1

    by_mode = {row["reliance_mode"]: row for row in disruption}
    assert math.isclose(by_mode["adaptive"]["delta_mse_short"], 2.0)
    assert math.isclose(by_mode["adaptive"]["delta_mse_long"], 3.0)
    assert math.isclose(by_mode["adaptive"]["delta_mse_change"], 1.0)
    assert math.isclose(by_mode["frozen"]["delta_mse_short"], 1.0)
    assert math.isclose(by_mode["frozen"]["delta_mse_long"], 2.0)

    af = adaptive_frozen[0]
    assert math.isclose(af["adaptive_minus_frozen_short"], 1.0)
    assert math.isclose(af["adaptive_minus_frozen_long"], 1.0)
    assert math.isclose(af["adaptive_minus_frozen_change"], 0.0)


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


def check_explicit_matched_topology_and_fixed_groups():
    n_citizens = 20
    group_ids = balanced_fixed_group_ids(
        seed=501,
        n_citizens=n_citizens,
    )
    blueprint = exp4_homophily_source_maps(
        seed=501,
        n_citizens=n_citizens,
        group_ids=group_ids,
    )
    mu_theta = exp4_initial_beliefs(
        seed=501,
        n_citizens=n_citizens,
        group_ids=group_ids,
        segregation="low",
    )

    config = build_config(
        network_environment="extended",
        n=n_citizens + 2,
    )
    config["mu_theta"] = mu_theta
    config["fixed_group_ids"] = group_ids
    config["structural_source_map"] = blueprint["low"]
    config["network_environment"] = "exp4_custom_low"

    model = InfoSampleModel(model_attribute=config, rng=501)
    for citizen in model.citizens:
        assert citizen.group_id == group_ids[citizen.pos]
        observed = sorted(source.pos for source in citizen.info_source)
        expected = sorted(blueprint["low"][citizen.pos])
        assert observed == expected


def check_exp3_redundancy_design():
    blueprint = exp3_redundancy_source_maps(
        seed=502,
        n_citizens=20,
    )
    gateways = set(blueprint["expert_gateways"])
    low = two_step_expert_route_count(
        blueprint["low"],
        expert_gateways=gateways,
    )
    high = two_step_expert_route_count(
        blueprint["high"],
        expert_gateways=gateways,
    )

    for ego in low:
        low_elites = [x for x in blueprint["low"][ego] if x < 2]
        high_elites = [x for x in blueprint["high"][ego] if x < 2]
        assert low_elites == high_elites
        assert sum(x >= 2 for x in blueprint["low"][ego]) == 2
        assert sum(x >= 2 for x in blueprint["high"][ego]) == 2

        if ego not in gateways:
            assert low[ego] == 1
            assert high[ego] == 2


def check_exp3b_path_independence_design():
    blueprint = exp3b_path_independence_source_maps(
        seed=504,
        n_citizens=100,
        n_gateways=10,
        n_relays=40,
    )
    shared = blueprint["shared"]
    independent = blueprint["independent"]
    focals = set(blueprint["focals"])
    relays = set(blueprint["relays"])
    gateways = set(blueprint["gateways"])

    # Focal immediate opportunity sets are exactly fixed.
    for focal in focals:
        assert shared[focal] == independent[focal]
        assert JAMMER_POS in shared[focal]
        assert sum(source in relays for source in shared[focal]) == 2

    # Per-node source degree and elite access are exactly matched.
    assert {
        ego: len(sources) for ego, sources in shared.items()
    } == {
        ego: len(sources) for ego, sources in independent.items()
    }
    for ego in shared:
        assert [x for x in shared[ego] if x < 2] == [
            x for x in independent[ego] if x < 2
        ]

    shared_conn = exp3b_focal_corrective_connectivity(
        shared,
        focals=focals,
        relays=relays,
        gateways=gateways,
    )
    independent_conn = exp3b_focal_corrective_connectivity(
        independent,
        focals=focals,
        relays=relays,
        gateways=gateways,
    )
    assert all(value == 1 for value in shared_conn.values())
    assert all(value == 2 for value in independent_conn.values())

    # Gateway indegree from relay nodes is globally identical and balanced.
    shared_indegree = {gateway: 0 for gateway in gateways}
    independent_indegree = {gateway: 0 for gateway in gateways}
    for gateway in blueprint["shared_relay_gateway"].values():
        shared_indegree[gateway] += 1
    for gateway in blueprint["independent_relay_gateway"].values():
        independent_indegree[gateway] += 1
    assert shared_indegree == independent_indegree
    assert len(set(shared_indegree.values())) == 1


def check_exp4_factorial_design():
    n_citizens = 40
    group_ids = balanced_fixed_group_ids(
        seed=503,
        n_citizens=n_citizens,
    )
    blueprint = exp4_homophily_source_maps(
        seed=503,
        n_citizens=n_citizens,
        group_ids=group_ids,
        low_homophily=0.50,
        high_homophily=0.90,
    )

    h_low = structural_peer_homophily(
        blueprint["low"],
        group_ids=group_ids,
    )
    h_high = structural_peer_homophily(
        blueprint["high"],
        group_ids=group_ids,
    )
    assert h_high > h_low + 0.20

    for ego in blueprint["low"]:
        assert [x for x in blueprint["low"][ego] if x < 2] == [
            x for x in blueprint["high"][ego] if x < 2
        ]
        assert [x for x in blueprint["low"][ego] if x < 2] == [0, 1]
        assert sum(x >= 2 for x in blueprint["low"][ego]) == 2
        assert sum(x >= 2 for x in blueprint["high"][ego]) == 2

    low_prior = exp4_initial_beliefs(
        seed=503,
        n_citizens=n_citizens,
        group_ids=group_ids,
        segregation="low",
    )
    high_prior = exp4_initial_beliefs(
        seed=503,
        n_citizens=n_citizens,
        group_ids=group_ids,
        segregation="high",
    )
    s_low = realized_prior_segregation(
        low_prior,
        group_ids=group_ids,
    )
    s_high = realized_prior_segregation(
        high_prior,
        group_ids=group_ids,
    )
    assert s_high > s_low + 3.0


def check_environment_mapping():
    # Isolated: universal access to exactly the two elites.
    isolated = InfoSampleModel(
        model_attribute=build_config(
            network_environment="elite_only",
            n=102,
        ),
        rng=21,
    )
    for citizen in isolated.citizens:
        assert len(citizen.info_source) == 2
        assert {
            source.type_of_agent for source in citizen.info_source
        } == {"infoprovider", "disruptivejammer"}

    # Random 2: exactly two local sources from the full non-ego pool.
    random2 = InfoSampleModel(
        model_attribute=build_config(
            network_environment="random_2",
            n=102,
        ),
        rng=22,
    )
    assert all(len(c.info_source) == 2 for c in random2.citizens)
    # Direct elite access must be localized, not universal.
    assert any(
        all(source.type_of_agent == "citizen" for source in c.info_source)
        for c in random2.citizens
    )

    # Group ID: exact degree two, predominantly same-group but not perfectly
    # segregated under the 0.9/0.1 mixing rule.
    group_id = InfoSampleModel(
        model_attribute=build_config(
            network_environment="group_id",
            n=202,
        ),
        rng=23,
    )
    assert all(len(c.info_source) == 2 for c in group_id.citizens)
    same = 0
    total = 0
    for citizen in group_id.citizens:
        for source in citizen.info_source:
            same += int(source.group_id == citizen.group_id)
            total += 1
    same_share = same / total
    assert 0.80 < same_share < 0.98

    # Extended: both elites plus exactly two citizen peers.
    extended = InfoSampleModel(
        model_attribute=build_config(
            network_environment="extended",
            n=102,
        ),
        rng=24,
    )
    for citizen in extended.citizens:
        assert len(citizen.info_source) == 4
        elite_count = sum(
            source.type_of_agent in {"infoprovider", "disruptivejammer"}
            for source in citizen.info_source
        )
        peer_count = sum(
            source.type_of_agent == "citizen"
            for source in citizen.info_source
        )
        assert elite_count == 2
        assert peer_count == 2


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
        check_fixed_horizon_execution,
        check_jammer_objective_and_hold_rule,
        check_jammer_resurveillance_uses_current_beliefs,
        check_horizon_sensitivity_contrasts,
        check_effective_reliance_dynamics,
        check_explicit_matched_topology_and_fixed_groups,
        check_exp3_redundancy_design,
        check_exp3b_path_independence_design,
        check_exp4_factorial_design,
        check_environment_mapping,
        check_finite_multiperiod_run,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"Paper B reconstruction checks passed: {len(checks)}/{len(checks)}")


if __name__ == "__main__":
    main()

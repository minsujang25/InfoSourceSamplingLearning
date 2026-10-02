"""Deterministic checks for Paper B measurement audit."""

from __future__ import annotations

import math

from model.InfoSourceSamplingLearning import InfoSampleModel
from paper_b.experiments.run_canonical_production import _base_config
from paper_b.measurement import (
    evidence_precision_network_metrics,
    null_last_share,
    structural_uniform_metrics,
)
from paper_b.structural_designs import exp3_redundancy_source_maps


def check_exp3_structural_baselines() -> None:
    design = exp3_redundancy_source_maps(
        seed=6001,
        n_citizens=100,
        peer_degree=2,
        expert_access_share=0.10,
    )
    gateways = set(design["expert_gateways"])
    low = structural_uniform_metrics(
        design["low"],
        gateway_positions=gateways,
    )
    high = structural_uniform_metrics(
        design["high"],
        gateway_positions=gateways,
    )

    assert math.isclose(
        low["A_uniform_gateway_total_share"],
        0.30,
        abs_tol=1e-12,
    )
    assert math.isclose(
        high["A_uniform_gateway_total_share"],
        0.60,
        abs_tol=1e-12,
    )
    assert math.isclose(
        low["A_uniform_gateway_peer_conditional_share"],
        0.45,
        abs_tol=1e-12,
    )
    assert math.isclose(
        high["A_uniform_gateway_peer_conditional_share"],
        0.90,
        abs_tol=1e-12,
    )


def check_null_is_runtime_last() -> None:
    seed = 6001
    cfg = _base_config(
        seed,
        {
            "n_citizens": 20,
            "horizon_T": 8,
            "K": 1,
            "epsilon": 0.05,
            "credit": 20,
            "surveillance_interval": 5,
        },
    )
    cfg.update(
        {
            "network_environment": "extended",
            "reliance_mode": "adaptive",
            "jammer_active": False,
            "jammer_regime": "null",
            "peer_evidence_mode": "source_posterior",
            "tau_social": 1.0,
            "frozen_ranking_mode": "pre_disruption",
        }
    )
    model = InfoSampleModel(model_attribute=cfg, rng=seed)
    assert null_last_share(model) == 1.0
    for _ in range(8):
        model.step()
        assert null_last_share(model) == 1.0


def check_precision_logger_matches_update_terms() -> None:
    seed = 6002
    cfg = _base_config(
        seed,
        {
            "n_citizens": 20,
            "horizon_T": 2,
            "K": 1,
            "epsilon": 0.05,
            "credit": 20,
            "surveillance_interval": 5,
        },
    )
    cfg.update(
        {
            "network_environment": "extended",
            "reliance_mode": "adaptive",
            "jammer_regime": "fixed_biased",
            "peer_evidence_mode": "source_posterior",
            "tau_social": 1.0,
            "frozen_ranking_mode": "pre_disruption",
        }
    )
    model = InfoSampleModel(model_attribute=cfg, rng=seed)
    citizen = model.citizens[0]
    peer = next(
        source for source in citizen.info_source
        if source.type_of_agent == "citizen"
    )
    expert = next(
        source for source in citizen.info_source
        if source.type_of_agent == "infoprovider"
    )

    peer._message_sd = 0.5
    expert._message_sd = 1.0
    citizen._sample_order = [peer, expert]
    citizen.sampled_msgs = [[1.0] * 20, [0.0] * 3]
    citizen._pending_evidence_precision = {}
    citizen.bayesian_update_theta_from_sources()

    peer_q = citizen._pending_evidence_precision[peer]
    expert_q = citizen._pending_evidence_precision[expert]
    assert math.isclose(peer_q, 1.0 / 1.25, abs_tol=1e-12)
    assert math.isclose(expert_q, 3.0, abs_tol=1e-12)

    # Peer repetition count must not change q.
    citizen.sampled_msgs = [[1.0], [0.0] * 3]
    citizen._pending_evidence_precision = {}
    citizen.bayesian_update_theta_from_sources()
    assert math.isclose(
        citizen._pending_evidence_precision[peer],
        peer_q,
        abs_tol=1e-12,
    )


def check_w_metrics_are_finite_after_learning() -> None:
    seed = 6003
    cfg = _base_config(
        seed,
        {
            "n_citizens": 20,
            "horizon_T": 8,
            "K": 1,
            "epsilon": 0.05,
            "credit": 20,
            "surveillance_interval": 5,
        },
    )
    cfg.update(
        {
            "network_environment": "extended",
            "reliance_mode": "adaptive",
            "jammer_regime": "null",
            "peer_evidence_mode": "source_posterior",
            "tau_social": 1.0,
            "frozen_ranking_mode": "pre_disruption",
        }
    )
    model = InfoSampleModel(model_attribute=cfg, rng=seed)
    for _ in range(8):
        model.step()
    metrics = evidence_precision_network_metrics(model)
    for key in (
        "W_expert_precision_share",
        "W_peer_precision_share",
        "W_incoming_hhi",
    ):
        assert math.isfinite(float(metrics[key]))


def main() -> None:
    check_exp3_structural_baselines()
    check_null_is_runtime_last()
    check_precision_logger_matches_update_terms()
    check_w_metrics_are_finite_after_learning()
    print("Paper B measurement-audit checks: PASS")


if __name__ == "__main__":
    main()

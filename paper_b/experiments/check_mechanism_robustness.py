"""Deterministic checks for Expert-rank mechanism robustness."""

from __future__ import annotations

import math

from model.InfoSourceSamplingLearning import recursive_rank_probabilities
from paper_b.experiments.run_canonical_production import _base_config
from paper_b.experiments.run_matched_pilot import run_condition
from paper_b.experiments.run_mechanism_robustness import (
    _nested_degree_check,
    _specs,
)
from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
)


def check_rank_probabilities() -> None:
    p4 = recursive_rank_probabilities(4, 0.05)
    expected = (0.95, 0.0475, 0.002375, 0.000125)
    for observed, target in zip(p4, expected):
        assert math.isclose(float(observed), target, abs_tol=1e-15)


def check_specs() -> None:
    specs = _specs(("epsilon", "degree"))
    labels = [spec["spec_label"] for spec in specs]
    assert labels == [
        "epsilon_0.10_d2",
        "epsilon_0.20_d2",
        "epsilon_0.05_d4",
    ]


def check_degree_nesting() -> None:
    for seed in (6001, 6002, 6010):
        groups = balanced_fixed_group_ids(seed=seed, n_citizens=100)
        assert _nested_degree_check(
            seed=seed,
            n_citizens=100,
            groups=groups,
            low_homophily=0.50,
            high_homophily=0.90,
        )


def _small_config(seed: int) -> tuple[dict, set[int]]:
    n = 20
    groups = balanced_fixed_group_ids(seed=seed, n_citizens=n)
    blueprint = exp4_homophily_source_maps(
        seed=seed,
        n_citizens=n,
        group_ids=groups,
        peer_degree=2,
        low_homophily=0.50,
        high_homophily=0.90,
    )
    initial = exp4_initial_beliefs(
        seed=seed,
        n_citizens=n,
        group_ids=groups,
        segregation="high",
        high_group_shift=3.0,
        residual_sd=1.0,
    )
    task = {
        "n_citizens": n,
        "horizon_T": 8,
        "K": 1,
        "epsilon": 0.05,
        "credit": 20,
        "surveillance_interval": 5,
    }
    cfg = _base_config(seed, task)
    cfg.update(
        {
            "mu_theta": initial,
            "initial_theta_type": "mechanism_smoke",
            "fixed_group_ids": groups,
            "structural_source_map": blueprint["high"],
        }
    )
    return cfg, set(int(x) for x in blueprint["expert_gateways"])


def check_rank_logging_is_passive() -> None:
    seed = 6001
    cfg, gateways = _small_config(seed)

    common = dict(
        base_config=cfg,
        seed=seed,
        regime="segregation_high",
        environment="mechanism_smoke",
        reliance_mode="adaptive",
        jammer_active=False,
        jammer_regime="null",
        peer_evidence_mode="source_posterior",
        frozen_ranking_mode="pre_disruption",
        gateway_positions=gateways,
        k=1,
        design_id="mechanism-smoke",
        block_id="mechanism-smoke",
        save_edge_log=False,
    )

    plain = run_condition(
        **common,
        record_expert_rank_checkpoints=False,
    )
    logged = run_condition(
        **common,
        record_expert_rank_checkpoints=True,
    )

    for key in (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "mean_belief",
        "belief_variance",
        "squared_displacement",
        "expert_reliance",
    ):
        a = float(plain["run"][key])
        b = float(logged["run"][key])
        assert math.isclose(a, b, abs_tol=0.0)

    plain_beliefs = {
        int(row["citizen_pos"]): (
            float(row["terminal_mu_theta"]),
            float(row["terminal_sd_theta"]),
        )
        for row in plain["beliefs"]
    }
    logged_beliefs = {
        int(row["citizen_pos"]): (
            float(row["terminal_mu_theta"]),
            float(row["terminal_sd_theta"]),
        )
        for row in logged["beliefs"]
    }
    assert plain_beliefs == logged_beliefs
    assert logged["expert_rank_checkpoints"]


def main() -> None:
    check_rank_probabilities()
    check_specs()
    check_degree_nesting()
    check_rank_logging_is_passive()
    print("Paper B mechanism-robustness checks: PASS")


if __name__ == "__main__":
    main()

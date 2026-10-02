"""Deterministic checks for exact W-channel decomposition."""

from __future__ import annotations

import math

from paper_b.experiments.run_canonical_production import _base_config
from paper_b.experiments.run_matched_pilot import run_condition
from paper_b.measurement import evidence_channel_decomposition_rows
from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
)


def _config(seed: int, *, n: int = 20, horizon: int = 25) -> dict:
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
        "horizon_T": horizon,
        "K": 1,
        "epsilon": 0.10,
        "credit": 20,
        "surveillance_interval": 5,
    }
    cfg = _base_config(seed, task)
    cfg.update(
        {
            "mu_theta": initial,
            "initial_theta_type": "channel_check_high",
            "fixed_group_ids": groups,
            "structural_source_map": blueprint["high"],
        }
    )
    return cfg


def _run(seed: int, *, record: bool) -> dict:
    return run_condition(
        base_config=_config(seed),
        seed=seed,
        regime="segregation_high",
        environment="channel_check",
        reliance_mode="frozen",
        jammer_active=False,
        jammer_regime="null",
        peer_evidence_mode="source_posterior",
        frozen_ranking_mode="pre_disruption",
        k=1,
        design_id="channel-check",
        block_id="channel-check",
        save_edge_log=False,
        record_channel_decomposition=record,
    )


def check_passive_identity() -> None:
    plain = _run(6001, record=False)
    logged = _run(6001, record=True)

    for key in (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "mean_belief",
        "belief_variance",
        "squared_displacement",
        "effective_homophily",
        "W_effective_homophily",
    ):
        assert float(plain["run"][key]) == float(logged["run"][key])

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


def check_channel_accounting() -> None:
    result = _run(6002, record=True)
    assert result["channel_checkpoints"]
    assert {
        int(row["horizon_step"])
        for row in result["channel_checkpoints"]
    } == {25}

    for row in result["channel_citizens"]:
        w_sum = (
            float(row["W_expert"])
            + float(row["W_same_peer"])
            + float(row["W_other_peer"])
            + float(row["W_jammer"])
        )
        assert math.isclose(w_sum, 1.0, abs_tol=1e-12)
        assert math.isclose(float(row["Q_jammer"]), 0.0, abs_tol=1e-12)
        assert math.isclose(
            float(row["W_corrective_crosscut"]),
            float(row["W_expert"]) + float(row["W_other_peer"]),
            abs_tol=1e-12,
        )
        for name in (
            "I_expert",
            "I_same_peer",
            "I_other_peer",
            "I_jammer",
        ):
            assert 0.0 <= float(row[name]) <= 1.0


def main() -> None:
    check_passive_identity()
    check_channel_accounting()
    print("Paper B W-channel decomposition checks: PASS")


if __name__ == "__main__":
    main()

"""Deterministic smoke checks for the Paper B validity redesign."""

from __future__ import annotations

import math

import numpy as np

from model.InfoSourceSamplingLearning import InfoSampleModel
from paper_b.experiments.run_local_diagnostic import base_model_config
from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
)


def _exp4_config(*, seed: int, reliance_mode: str, jammer_regime: str) -> dict:
    n_citizens = 20
    groups = balanced_fixed_group_ids(seed=seed, n_citizens=n_citizens)
    structure = exp4_homophily_source_maps(
        seed=seed,
        n_citizens=n_citizens,
        group_ids=groups,
        peer_degree=2,
        low_homophily=0.50,
        high_homophily=0.90,
    )
    config = base_model_config(
        regime="flat",
        seed=seed,
        n_citizens=n_citizens,
        max_steps=4,
        k=1,
        epsilon=0.05,
        credit=20,
        comparison_rule="delta_comparison",
        surveillance_interval=5,
    )
    config.update(
        {
            "mu_theta": exp4_initial_beliefs(
                seed=seed,
                n_citizens=n_citizens,
                group_ids=groups,
                segregation="high",
                high_group_shift=3.0,
                residual_sd=1.0,
            ),
            "fixed_group_ids": groups,
            "structural_source_map": structure["high"],
            "network_environment": "validity_smoke",
            "reliance_mode": reliance_mode,
            "jammer_regime": jammer_regime,
            "peer_evidence_mode": "source_posterior",
            "frozen_ranking_mode": "pre_disruption",
        }
    )
    return config


def check_legacy_j0_mapping() -> None:
    cfg = base_model_config(
        regime="flat",
        seed=11,
        n_citizens=10,
        max_steps=1,
        k=1,
        epsilon=0.05,
        credit=20,
        comparison_rule="delta_comparison",
        surveillance_interval=5,
    )
    cfg["jammer_active"] = False
    model = InfoSampleModel(model_attribute=cfg, rng=11)
    assert model.jammer_regime == "truth_clone"


def check_common_pre_disruption_ranking() -> None:
    seed = 9101
    adaptive = InfoSampleModel(
        model_attribute=_exp4_config(
            seed=seed,
            reliance_mode="adaptive",
            jammer_regime="adaptive",
        ),
        rng=seed,
    )
    frozen = InfoSampleModel(
        model_attribute=_exp4_config(
            seed=seed,
            reliance_mode="frozen",
            jammer_regime="adaptive",
        ),
        rng=seed,
    )
    for left, right in zip(adaptive.citizens, frozen.citizens):
        left_order = [int(s.pos) for s in left.credibility_ranked_sources]
        right_order = [int(s.pos) for s in right.frozen_credibility_ranked_sources]
        assert left_order == right_order


def check_null_is_inert_and_last() -> None:
    seed = 9102
    model = InfoSampleModel(
        model_attribute=_exp4_config(
            seed=seed,
            reliance_mode="adaptive",
            jammer_regime="null",
        ),
        rng=seed,
    )
    citizen = model.citizens[0]
    assert citizen._behavioral_ranking()[-1].type_of_agent == "disruptivejammer"
    draws = model.jammer.mu_out(7, citizen)
    assert draws == []


def check_peer_repetition_not_extra_precision() -> None:
    seed = 9103
    model = InfoSampleModel(
        model_attribute=_exp4_config(
            seed=seed,
            reliance_mode="adaptive",
            jammer_regime="null",
        ),
        rng=seed,
    )
    citizen = model.citizens[0]
    peer = next(
        source for source in citizen.info_source
        if source.type_of_agent == "citizen"
    )
    peer._message_sd = 0.75

    citizen._sample_order = [peer]
    citizen.sampled_msgs = [[1.25]]
    one_mu, one_sd = citizen.bayesian_update_theta_from_sources()

    citizen._sample_order = [peer]
    citizen.sampled_msgs = [[1.25] * 20]
    many_mu, many_sd = citizen.bayesian_update_theta_from_sources()

    assert math.isclose(one_mu, many_mu, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(one_sd, many_sd, rel_tol=0.0, abs_tol=1e-12)


def check_sender_regimes_execute() -> None:
    for offset, sender in enumerate(
        ("null", "fixed_biased", "truth_clone", "adaptive")
    ):
        seed = 9200 + offset
        model = InfoSampleModel(
            model_attribute=_exp4_config(
                seed=seed,
                reliance_mode="adaptive",
                jammer_regime=sender,
            ),
            rng=seed,
        )
        for _ in range(3):
            model.step()
        beliefs = np.asarray(model.get_agent_mu_theta(), dtype=float)
        sds = np.asarray(model.get_agent_sd_theta(), dtype=float)
        assert np.isfinite(beliefs).all()
        assert np.isfinite(sds).all()
        assert (sds > 0.0).all()


def main() -> None:
    check_legacy_j0_mapping()
    check_common_pre_disruption_ranking()
    check_null_is_inert_and_last()
    check_peer_repetition_not_extra_precision()
    check_sender_regimes_execute()
    print("Paper B validity-redesign smoke checks: PASS")


if __name__ == "__main__":
    main()

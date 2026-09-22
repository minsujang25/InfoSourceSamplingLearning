"""Diagnose the polarized K=4 / peer-cap=5 numerical explosion.

This script leaves the shared model untouched. It runs the failing matched-seed
scenario twice:

1. the current legacy-compatible theta-uncertainty update;
2. a shadow correction that returns posterior *standard deviation* rather than
   posterior variance.

The second run is diagnostic evidence only, not a committed substantive model
change.
"""

from __future__ import annotations

import math
import numpy as np

from scripts.export_backtest_snapshot import (
    SCENARIOS,
    build_config,
    enforce_deterministic_neighbor_order,
)
import model.InfoSourceSamplingLearning as issl


SCENARIO_ID = "group_polarized_k4_n5_delta"
SEED = 101


def citizens(model):
    return sorted(
        [a for a in model.agents if a.type_of_agent == "citizen"],
        key=lambda a: a.pos,
    )


def finite_abs_max(values):
    vals = [abs(float(x)) for x in values if math.isfinite(float(x))]
    return max(vals) if vals else math.nan


def snapshot(model, label):
    cs = citizens(model)
    mu = [a.mu_theta_beliefs[-1] for a in cs]
    sd = [a.sd_theta_beliefs[-1] for a in cs]
    delta_sd = [
        x
        for a in cs
        for x in a.sd_delta_beliefs[-1]
        if isinstance(x, (int, float, np.floating))
    ]
    bad_mu = [a.pos for a in cs if not math.isfinite(float(a.mu_theta_beliefs[-1]))]
    bad_sd = [a.pos for a in cs if not math.isfinite(float(a.sd_theta_beliefs[-1]))]

    jammer = next(a for a in model.agents if a.type_of_agent == "disruptivejammer")
    params = jammer.msg_param_at_t_per_cluster.get(model.period, {})
    jammer_means = [v["avg"] for v in params.values()] if params else []

    max_mu_agent = max(
        cs,
        key=lambda a: abs(float(a.mu_theta_beliefs[-1]))
        if math.isfinite(float(a.mu_theta_beliefs[-1]))
        else -1,
    )
    max_sd_agent = max(
        cs,
        key=lambda a: abs(float(a.sd_theta_beliefs[-1]))
        if math.isfinite(float(a.sd_theta_beliefs[-1]))
        else -1,
    )

    print(
        f"{label} period={model.period:02d} steps={model.steps:02d} "
        f"max|mu|={finite_abs_max(mu):.6g}@{max_mu_agent.pos} "
        f"max_sd={finite_abs_max(sd):.6g}@{max_sd_agent.pos} "
        f"max_delta_sd={finite_abs_max(delta_sd):.6g} "
        f"max|jammer_mean|={finite_abs_max(jammer_means):.6g} "
        f"bad_mu={bad_mu} bad_sd={bad_sd}"
    )


def make_model():
    scenario = next(s for s in SCENARIOS if s["id"] == SCENARIO_ID)
    np.random.seed(SEED)
    enforce_deterministic_neighbor_order(issl)
    config = build_config(issl, scenario, SEED)
    return issl.InfoSampleModel(model_attribute=config, rng=SEED)


def corrected_sd_update(self, msgs):
    prior_sd = float(self.sd_theta_beliefs[-1])
    std_msgs = max(float(np.std(msgs)), issl.MIN_STD)
    posterior_var = (
        prior_sd**2 * std_msgs**2
        / (prior_sd**2 + std_msgs**2)
    )
    return math.sqrt(max(posterior_var, 0.0))


def run(label, patch_sd=False, n_steps=15):
    original = issl.Citizen.bayesian_update_sd_theta
    if patch_sd:
        issl.Citizen.bayesian_update_sd_theta = corrected_sd_update
    try:
        model = make_model()
        snapshot(model, label)
        for _ in range(n_steps):
            model.step()
            snapshot(model, label)
            if any(
                not math.isfinite(float(a.mu_theta_beliefs[-1]))
                or not math.isfinite(float(a.sd_theta_beliefs[-1]))
                for a in citizens(model)
            ):
                break
        return model
    finally:
        issl.Citizen.bayesian_update_sd_theta = original


def main():
    print("=== CURRENT LEGACY-COMPATIBLE UPDATE ===")
    run("legacy_formula", patch_sd=False)

    print("\n=== SHADOW STANDARD-DEVIATION CORRECTION ===")
    run("sqrt_formula", patch_sd=True)

    # A direct dimensional sanity check.
    prior_sd = 5.0
    likelihood_sd = 100.0
    legacy_value = (
        prior_sd**2 * likelihood_sd**2
        / (prior_sd**2 + likelihood_sd**2)
    )
    corrected_value = math.sqrt(legacy_value)
    print(
        "\nSanity check: prior_sd=5, likelihood_sd=100 -> "
        f"legacy stored value={legacy_value:.6g}; "
        f"posterior_sd={corrected_value:.6g}"
    )


if __name__ == "__main__":
    main()

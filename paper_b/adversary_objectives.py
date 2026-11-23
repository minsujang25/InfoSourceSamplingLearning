"""Prespecified one-step credibility-aware attack objective for Paper B Gate 0C.

This is a *proxy* for the dynamic attacker problem, not a finite-horizon
Bellman solution or an exact forecast of next-period rank transitions.

It explicitly weights immediate learning damage by actual policy-based
probability of at least one Jammer message, and penalizes expected
squared posterior source-displacement after a credibility audit.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _inputs(
    *,
    citizen_means: Iterable[float],
    citizen_sds: Iterable[float],
    prior_delta_means: Iterable[float],
    prior_delta_sds: Iterable[float],
    theta_inclusion: Iterable[float],
    audit_inclusion: Iterable[float],
    expected_audit_counts: Iterable[float],
) -> tuple[np.ndarray, ...]:
    values = tuple(
        np.asarray(list(v), dtype=float)
        for v in (
            citizen_means,
            citizen_sds,
            prior_delta_means,
            prior_delta_sds,
            theta_inclusion,
            audit_inclusion,
            expected_audit_counts,
        )
    )
    if values[0].size == 0 or any(v.shape != values[0].shape for v in values):
        raise ValueError("Nonempty, equally sized 1-D audience inputs are required.")
    if any(not np.isfinite(v).all() for v in values):
        raise ValueError("Non-finite audience input.")
    means, sd_theta, delta_mu, delta_sd, q_theta, q_audit, counts = values
    if (sd_theta < 0).any() or (delta_sd < 0).any():
        raise ValueError("Posterior standard deviations cannot be negative.")
    if ((q_theta < 0) | (q_theta > 1)).any():
        raise ValueError("Theta inclusion probabilities must lie in [0,1].")
    if ((q_audit < 0) | (q_audit > 1)).any():
        raise ValueError("Audit inclusion probabilities must lie in [0,1].")
    if (counts < 0).any() or ((q_audit > 0) & (counts <= 0)).any():
        raise ValueError("Positive audit inclusion requires positive audit count.")
    return values


def credibility_aware_components(
    *,
    citizen_means: Iterable[float],
    citizen_sds: Iterable[float],
    prior_delta_means: Iterable[float],
    prior_delta_sds: Iterable[float],
    theta_inclusion: Iterable[float],
    audit_inclusion: Iterable[float],
    expected_audit_counts: Iterable[float],
    single_message_variance: float = 1.0,
) -> dict:
    """Return estimated state-update and delta-audit response gains.

    State gain kappa uses the same one-message approximation as the archived
    myopic objective.  The inclusion probabilities account for the actual
    source-ranking acquisition policy, but the gain is not a multi-source
    posterior calculation. Audit gain eta follows Citizen.learn_delta() with
    the expected (possibly fractional) allocation count C/degree.
    """
    mu, sigma, b, delta_sd, q, qa, n = _inputs(
        citizen_means=citizen_means,
        citizen_sds=citizen_sds,
        prior_delta_means=prior_delta_means,
        prior_delta_sds=prior_delta_sds,
        theta_inclusion=theta_inclusion,
        audit_inclusion=audit_inclusion,
        expected_audit_counts=expected_audit_counts,
    )
    if not math.isfinite(single_message_variance) or single_message_variance <= 0:
        raise ValueError("single_message_variance must be positive and finite.")
    theta_var = sigma * sigma
    delta_var = delta_sd * delta_sd
    kappa = theta_var / (theta_var + single_message_variance)
    obs_var = theta_var + single_message_variance / np.maximum(n, 1.0)
    eta = delta_var / (delta_var + obs_var)
    return {
        "mu": mu, "b": b, "kappa": kappa, "eta": eta,
        "q_theta": q, "q_audit": qa,
    }


def credibility_aware_utility(
    message_mean: float, *,
    truth: float,
    underlying_position: float,
    audit_probability: float,
    credibility_penalty: float = 1.0,
    **components: np.ndarray,
) -> float:
    """Evaluate the one-period objective (up to message-variance constants)."""
    if not 0 <= audit_probability <= 1:
        raise ValueError("audit_probability must be in [0,1]")
    if not math.isfinite(credibility_penalty) or credibility_penalty < 0:
        raise ValueError("credibility_penalty must be finite and nonnegative")
    mu = components["mu"]
    b = components["b"]
    k = components["kappa"]
    eta = components["eta"]
    q = components["q_theta"]
    qa = components["q_audit"]
    err = (1-k)*mu + k*message_mean - truth
    expected_delta = (1-eta)*b + eta*(message_mean-mu)
    return (
        (1-audit_probability) * float(np.mean(q*err*err))
        - (message_mean-underlying_position)**2
        - credibility_penalty*audit_probability
          * float(np.mean(qa*expected_delta*expected_delta))
    )


def optimize_credibility_aware_message(
    *,
    citizen_means: Iterable[float],
    citizen_sds: Iterable[float],
    prior_delta_means: Iterable[float],
    prior_delta_sds: Iterable[float],
    theta_inclusion: Iterable[float],
    audit_inclusion: Iterable[float],
    expected_audit_counts: Iterable[float],
    truth: float,
    underlying_position: float,
    audit_probability: float,
    credibility_penalty: float = 1.0,
    single_message_variance: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Optimize the quadratic one-step proxy analytically.

    Let p=P(audit), q_i=P(at least one Jammer request | theta-learning),
    a_i=P(at least one Jammer audit request | audit), k_i=theta gain,
    e_i=delta gain, b_i=current prior source-displacement mean.

    U(m) = (1-p) E[q_i ((1-k_i) mu_i + k_i m - theta)^2]
           - (m-mu_D)^2
           - lambda p E[a_i ((1-e_i)b_i + e_i(m-mu_i))^2].

    An attack mean is common within the observed audience cluster. This does
    not model peer recirculation, strategic multi-period continuation values,
    or ranking changes caused by the audit. Those are outcomes of the ABM,
    not claims of this local approximation.
    """
    c = credibility_aware_components(
        citizen_means=citizen_means,
        citizen_sds=citizen_sds,
        prior_delta_means=prior_delta_means,
        prior_delta_sds=prior_delta_sds,
        theta_inclusion=theta_inclusion,
        audit_inclusion=audit_inclusion,
        expected_audit_counts=expected_audit_counts,
        single_message_variance=single_message_variance,
    )
    p, lam = float(audit_probability), float(credibility_penalty)
    theta, mu_d = float(truth), float(underlying_position)
    if not 0.0 <= p <= 1.0 or not math.isfinite(p):
        raise ValueError("audit_probability must lie in [0, 1]")
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("credibility_penalty must be finite and nonnegative")
    if not math.isfinite(theta) or not math.isfinite(mu_d):
        raise ValueError("Underlying positions and truth must be finite")

    mu, b, k, eta = c["mu"], c["b"], c["kappa"], c["eta"]
    q, qa = c["q_theta"], c["q_audit"]
    benefit_curvature = (1-p)*float(np.mean(q*k*k))
    benefit_linear = (1-p)*float(np.mean(q*k*((1-k)*mu-theta)))
    audit_curvature = lam*p*float(np.mean(qa*eta*eta))
    audit_linear = lam*p*float(
        np.mean(qa*eta*((1-eta)*b-eta*mu))
    )
    denom = 1.0 - benefit_curvature + audit_curvature
    if not math.isfinite(denom) or denom <= 0:
        raise FloatingPointError("Attacker objective is not strictly concave")
    message = (mu_d + benefit_linear - audit_linear)/denom
    if not math.isfinite(message):
        raise FloatingPointError("Non-finite credibility-aware message")
    diagnostics = {
        "audit_probability": p,
        "credibility_penalty": lam,
        "mean_theta_inclusion": float(np.mean(q)),
        "mean_audit_inclusion": float(np.mean(qa)),
        "mean_theta_response_gain": float(np.mean(k)),
        "mean_delta_audit_response_gain": float(np.mean(eta)),
        "objective_denominator": float(denom),
        "objective_theta_curvature": float(benefit_curvature),
        "objective_audit_curvature": float(audit_curvature),
        "objective_value": float(credibility_aware_utility(
            message, truth=theta, underlying_position=mu_d,
            audit_probability=p, credibility_penalty=lam, **c
        )),
    }
    return float(message), diagnostics

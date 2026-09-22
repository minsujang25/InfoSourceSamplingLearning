"""Theory-aligned run-level and effective-reliance metrics for Paper B.

The main-text measurement architecture is deliberately parsimonious:
1. truth-centered MSE/RMSE/MAE and the MSE displacement-dispersion decomposition;
2. Expert/Jammer/peer reliance composition;
3. structural-to-effective reliance divergence;
4. effective homophily and structural-to-effective homophily divergence.

Realized-flow and concentration diagnostics remain available as secondary checks.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def _agents(model):
    if hasattr(model, "agents"):
        return list(model.agents)
    return list(model.schedule.agents)


def citizens(model):
    return sorted(
        [
            agent
            for agent in _agents(model)
            if getattr(agent, "type_of_agent", None) == "citizen"
        ],
        key=lambda agent: agent.pos,
    )


def belief_vector(model) -> np.ndarray:
    return np.asarray(
        [float(agent.mu_theta_beliefs[-1]) for agent in citizens(model)],
        dtype=float,
    )


def belief_summary(model) -> dict:
    values = belief_vector(model)
    truth = float(model.state_of_the_world)
    finite = np.isfinite(values)

    if values.size == 0 or not finite.all():
        return {
            "n_citizens": int(values.size),
            "n_nonfinite": int((~finite).sum()) if values.size else 0,
            "mean_belief": math.nan,
            "belief_sd": math.nan,
            "mse_truth": math.nan,
            "rmse_truth": math.nan,
            "mae_truth": math.nan,
            "squared_displacement": math.nan,
            "belief_variance": math.nan,
        }

    errors = values - truth
    mean_belief = float(values.mean())
    variance = float(values.var(ddof=0))
    squared_displacement = float((mean_belief - truth) ** 2)
    mse = float(np.mean(errors**2))

    return {
        "n_citizens": int(values.size),
        "n_nonfinite": 0,
        "mean_belief": mean_belief,
        "belief_sd": float(math.sqrt(variance)),
        "mse_truth": mse,
        "rmse_truth": float(math.sqrt(mse)),
        "mae_truth": float(np.abs(errors).mean()),
        "squared_displacement": squared_displacement,
        "belief_variance": variance,
    }


def disruption(mse_jammer: float, mse_no_jammer: float) -> float:
    """Primary Paper B disruption estimand: jammer-induced Delta MSE."""
    return float(mse_jammer) - float(mse_no_jammer)


def structural_source_counts(model) -> dict:
    """Count structural opportunity edges by source type."""
    counts = Counter()
    for citizen in citizens(model):
        for source in getattr(citizen, "info_source", []):
            counts[getattr(source, "type_of_agent", "unknown")] += 1
    return dict(counts)


def _source_class(source) -> str:
    source_type = getattr(source, "type_of_agent", "unknown")
    if source_type == "infoprovider":
        return "expert"
    if source_type == "disruptivejammer":
        return "jammer"
    if source_type == "citizen":
        return "peer"
    return source_type


def reliance_composition(model) -> dict:
    """Population mean expected reliance on Expert, Jammer, and peers.

    These are Lambda_t acquisition/reliance weights, not causal influence
    estimates and not realized request shares.
    """
    cs = citizens(model)
    if not cs:
        return {
            "expert_reliance": math.nan,
            "jammer_reliance": math.nan,
            "peer_reliance": math.nan,
        }

    totals = Counter()
    for citizen in cs:
        for source, weight in getattr(citizen, "reliance_probabilities", {}).items():
            totals[_source_class(source)] += float(weight)

    n = len(cs)
    return {
        "expert_reliance": float(totals["expert"] / n),
        "jammer_reliance": float(totals["jammer"] / n),
        "peer_reliance": float(totals["peer"] / n),
    }


def realized_flow_composition(model) -> dict:
    """Population mean realized request shares X_t by source class."""
    cs = citizens(model)
    if not cs:
        return {
            "expert_realized": math.nan,
            "jammer_realized": math.nan,
            "peer_realized": math.nan,
        }

    totals = Counter()
    for citizen in cs:
        for source, weight in getattr(citizen, "realized_reliance", {}).items():
            totals[_source_class(source)] += float(weight)

    n = len(cs)
    return {
        "expert_realized": float(totals["expert"] / n),
        "jammer_realized": float(totals["jammer"] / n),
        "peer_realized": float(totals["peer"] / n),
    }


def structural_effective_divergence(model) -> float:
    """Mean total-variation distance from equal use of available ties.

    For citizen i:
        D_i = .5 * sum_j |lambda_ij - A_ij / degree_i|.

    This measures magnitude of reweighting, not whether reweighting is good.
    """
    values = []
    for citizen in citizens(model):
        sources = list(getattr(citizen, "info_source", []))
        if not sources:
            continue
        uniform = 1.0 / len(sources)
        weights = getattr(citizen, "reliance_probabilities", {})
        values.append(
            0.5
            * sum(abs(float(weights.get(source, 0.0)) - uniform) for source in sources)
        )
    return float(np.mean(values)) if values else math.nan


def structural_homophily(model) -> float:
    """Mean same-initial-group share among structurally available citizen peers."""
    values = []
    for citizen in citizens(model):
        peers = [
            source
            for source in getattr(citizen, "info_source", [])
            if getattr(source, "type_of_agent", None) == "citizen"
        ]
        if not peers:
            continue
        values.append(
            sum(source.group_id == citizen.group_id for source in peers) / len(peers)
        )
    return float(np.mean(values)) if values else math.nan


def effective_homophily(model) -> dict:
    """Effective same-group peer reliance, conditional on peer reliance.

    Returns H^Lambda, mean peer reliance mass, H^A, and Delta H = H^Lambda-H^A.
    Citizens with zero peer reliance are excluded from H^Lambda but remain
    represented in the separately reported peer-reliance mass.
    """
    effective_values = []
    peer_mass = []

    for citizen in citizens(model):
        weights = getattr(citizen, "reliance_probabilities", {})
        peers = [
            source
            for source in getattr(citizen, "info_source", [])
            if getattr(source, "type_of_agent", None) == "citizen"
        ]
        mass = float(sum(float(weights.get(source, 0.0)) for source in peers))
        peer_mass.append(mass)

        if mass <= 0.0:
            continue
        same_mass = sum(
            float(weights.get(source, 0.0))
            for source in peers
            if source.group_id == citizen.group_id
        )
        effective_values.append(same_mass / mass)

    h_struct = structural_homophily(model)
    h_effective = (
        float(np.mean(effective_values)) if effective_values else math.nan
    )
    delta_h = (
        h_effective - h_struct
        if math.isfinite(h_effective) and math.isfinite(h_struct)
        else math.nan
    )
    return {
        "structural_homophily": h_struct,
        "effective_homophily": h_effective,
        "homophily_divergence": delta_h,
        "peer_reliance_mass": float(np.mean(peer_mass)) if peer_mass else math.nan,
    }


def reliance_hhi(model) -> float:
    """Secondary diagnostic: normalized reliance concentration.

    The normalization maps equal use of d structurally available sources to 0
    and exclusive reliance to 1 for each citizen.
    """
    values = []
    for citizen in citizens(model):
        sources = list(getattr(citizen, "info_source", []))
        d = len(sources)
        if d <= 1:
            values.append(1.0 if d == 1 else math.nan)
            continue
        weights = getattr(citizen, "reliance_probabilities", {})
        hhi = sum(float(weights.get(source, 0.0)) ** 2 for source in sources)
        normalized = (hhi - 1.0 / d) / (1.0 - 1.0 / d)
        values.append(float(normalized))
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


def theory_metrics(model) -> dict:
    """One compact run-level snapshot aligned with the manuscript theory."""
    out = {}
    out.update(belief_summary(model))
    out.update(reliance_composition(model))
    out["structural_effective_divergence"] = structural_effective_divergence(model)
    out.update(effective_homophily(model))
    return out

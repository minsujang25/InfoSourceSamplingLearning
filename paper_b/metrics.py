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
from collections import Counter, defaultdict

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


def _period_expected_weights(model, period: int) -> dict[int, dict[int, float]]:
    """Return expected reliance weights keyed by ego and source for one period."""
    if period < 0 or period >= len(getattr(model, "reliance_history", [])):
        return {}

    out: dict[int, dict[int, float]] = defaultdict(dict)
    for record in model.reliance_history[period]:
        out[int(record["ego"])][int(record["source"])] = float(
            record["expected_reliance"]
        )
    return dict(out)


def _mean_tv_distance(
    left: dict[int, dict[int, float]],
    right: dict[int, dict[int, float]],
) -> float:
    values = []
    for ego in sorted(set(left) | set(right)):
        l = left.get(ego, {})
        r = right.get(ego, {})
        sources = set(l) | set(r)
        if not sources:
            continue
        values.append(
            0.5
            * sum(abs(float(l.get(s, 0.0)) - float(r.get(s, 0.0))) for s in sources)
        )
    return float(np.mean(values)) if values else math.nan


def _top_source(weight_map: dict[int, float]) -> int | None:
    if not weight_map:
        return None
    return min(
        weight_map,
        key=lambda source: (-float(weight_map[source]), int(source)),
    )


def effective_reliance_dynamics(model, baseline_period: int = 1) -> dict:
    """Measure endogenous movement of Lambda_t rather than rank concentration.

    The structural-to-effective divergence is constant when epsilon and source
    degree are fixed because rank changes only permute a fixed probability
    vector.  These diagnostics instead track how the *identity* of highly
    weighted ties changes after the first credibility audit.

    baseline_period=1 corresponds to the first state-learning period after the
    period-0 credibility audit in the Paper B schedule.
    """
    history = getattr(model, "reliance_history", [])
    if not history:
        return {
            "lambda_baseline_period": math.nan,
            "lambda_first_audit_to_terminal_tv": math.nan,
            "lambda_mean_period_turnover": math.nan,
            "lambda_cumulative_turnover": math.nan,
            "lambda_top_source_changed_share": math.nan,
            "lambda_top_source_switches_per_citizen": math.nan,
        }

    base = min(max(int(baseline_period), 0), len(history) - 1)
    terminal = len(history) - 1
    base_weights = _period_expected_weights(model, base)
    terminal_weights = _period_expected_weights(model, terminal)

    terminal_tv = _mean_tv_distance(base_weights, terminal_weights)

    turnovers = []
    switch_counts = Counter()
    previous = base_weights
    previous_top = {ego: _top_source(weights) for ego, weights in previous.items()}

    for period in range(base + 1, terminal + 1):
        current = _period_expected_weights(model, period)
        turnovers.append(_mean_tv_distance(previous, current))

        current_top = {ego: _top_source(weights) for ego, weights in current.items()}
        for ego in set(previous_top) | set(current_top):
            if previous_top.get(ego) != current_top.get(ego):
                switch_counts[ego] += 1

        previous = current
        previous_top = current_top

    base_top = {ego: _top_source(weights) for ego, weights in base_weights.items()}
    terminal_top = {
        ego: _top_source(weights) for ego, weights in terminal_weights.items()
    }
    egos = sorted(set(base_top) | set(terminal_top))
    changed_share = (
        float(
            np.mean(
                [
                    base_top.get(ego) != terminal_top.get(ego)
                    for ego in egos
                ]
            )
        )
        if egos
        else math.nan
    )
    switches_per_citizen = (
        float(np.mean([switch_counts.get(ego, 0) for ego in egos]))
        if egos
        else math.nan
    )

    finite_turnovers = [x for x in turnovers if math.isfinite(x)]
    return {
        "lambda_baseline_period": int(base),
        "lambda_first_audit_to_terminal_tv": terminal_tv,
        "lambda_mean_period_turnover": (
            float(np.mean(finite_turnovers)) if finite_turnovers else 0.0
        ),
        "lambda_cumulative_turnover": (
            float(np.sum(finite_turnovers)) if finite_turnovers else 0.0
        ),
        "lambda_top_source_changed_share": changed_share,
        "lambda_top_source_switches_per_citizen": switches_per_citizen,
    }


def reliance_checkpoint_metrics(
    model,
    period: int,
    baseline_period: int = 1,
) -> dict:
    """Compact Lambda_t summary for a selected period."""
    weights = _period_expected_weights(model, period)
    if not weights:
        return {
            "period": int(period),
            "expert_reliance": math.nan,
            "jammer_reliance": math.nan,
            "peer_reliance": math.nan,
            "lambda_from_first_audit_tv": math.nan,
            "effective_homophily": math.nan,
            "peer_reliance_mass": math.nan,
        }

    source_meta = {}
    for citizen in citizens(model):
        for source in citizen.info_source:
            source_meta[int(source.pos)] = {
                "type": getattr(source, "type_of_agent", "unknown"),
                "group": getattr(source, "group_id", None),
            }

    citizen_group = {int(c.pos): c.group_id for c in citizens(model)}
    totals = Counter()
    homophily_values = []
    peer_mass_values = []

    for ego, ego_weights in weights.items():
        peer_mass = 0.0
        same_peer_mass = 0.0
        for source, weight in ego_weights.items():
            meta = source_meta.get(source, {"type": "unknown", "group": None})
            source_type = meta["type"]
            if source_type == "infoprovider":
                totals["expert"] += weight
            elif source_type == "disruptivejammer":
                totals["jammer"] += weight
            elif source_type == "citizen":
                totals["peer"] += weight
                peer_mass += weight
                if meta["group"] == citizen_group.get(ego):
                    same_peer_mass += weight

        peer_mass_values.append(peer_mass)
        if peer_mass > 0.0:
            homophily_values.append(same_peer_mass / peer_mass)

    n = max(len(weights), 1)
    base = min(max(int(baseline_period), 0), len(model.reliance_history) - 1)
    base_weights = _period_expected_weights(model, base)

    return {
        "period": int(period),
        "expert_reliance": float(totals["expert"] / n),
        "jammer_reliance": float(totals["jammer"] / n),
        "peer_reliance": float(totals["peer"] / n),
        "lambda_from_first_audit_tv": _mean_tv_distance(base_weights, weights),
        "effective_homophily": (
            float(np.mean(homophily_values)) if homophily_values else math.nan
        ),
        "peer_reliance_mass": (
            float(np.mean(peer_mass_values)) if peer_mass_values else math.nan
        ),
    }


def posterior_precision_checkpoint(model, period: int) -> dict:
    """Posterior-SD diagnostics after the requested zero-indexed period."""
    cs = citizens(model)
    if not cs:
        return {
            "period": int(period),
            "posterior_sd_mean": math.nan,
            "posterior_sd_median": math.nan,
            "posterior_sd_min": math.nan,
            "posterior_sd_max": math.nan,
            "posterior_sd_floor_share": math.nan,
        }

    history_index = int(period) + 1
    values = []
    for citizen in cs:
        history = getattr(citizen, "sd_theta_beliefs", [])
        if history_index >= len(history):
            continue
        values.append(float(history[history_index]))

    if not values:
        return {
            "period": int(period),
            "posterior_sd_mean": math.nan,
            "posterior_sd_median": math.nan,
            "posterior_sd_min": math.nan,
            "posterior_sd_max": math.nan,
            "posterior_sd_floor_share": math.nan,
        }

    arr = np.asarray(values, dtype=float)
    floor = float(getattr(model, "numerical_min_sd", 1e-8))
    return {
        "period": int(period),
        "posterior_sd_mean": float(arr.mean()),
        "posterior_sd_median": float(np.median(arr)),
        "posterior_sd_min": float(arr.min()),
        "posterior_sd_max": float(arr.max()),
        "posterior_sd_floor_share": float(
            np.mean(arr <= floor * (1.0 + 1e-12))
        ),
    }


def dominant_reliance_skeleton_metrics(
    model,
    period: int | None = None,
    gateway_positions: set[int] | None = None,
) -> dict:
    """Summarize the top-ranked effective-reliance skeleton.

    Each citizen contributes one outgoing edge to the structurally available
    source with the largest expected-reliance weight. Following those edges
    ends at the Expert, the Jammer, or a citizen-only directed cycle. This is a
    behavioral dependence skeleton, not a causal influence graph.
    """
    history = getattr(model, "reliance_history", [])
    if not history:
        return {
            "dominant_expert_reach_share": math.nan,
            "dominant_jammer_reach_share": math.nan,
            "dominant_citizen_cycle_share": math.nan,
            "dominant_same_group_cycle_share": math.nan,
            "dominant_cycle_mean_size": math.nan,
            "dominant_cycle_max_size": math.nan,
            "dominant_two_cycle_attractor_share": math.nan,
            "dominant_threeplus_cycle_attractor_share": math.nan,
            "effective_incoming_hhi": math.nan,
            "effective_incoming_max_share": math.nan,
            "effective_incoming_top5_share": math.nan,
            "gateway_incoming_reliance_share": math.nan,
        }

    p = len(history) - 1 if period is None else int(period)
    weights = _period_expected_weights(model, p)
    if not weights:
        return {
            "dominant_expert_reach_share": math.nan,
            "dominant_jammer_reach_share": math.nan,
            "dominant_citizen_cycle_share": math.nan,
            "dominant_same_group_cycle_share": math.nan,
            "dominant_cycle_mean_size": math.nan,
            "dominant_cycle_max_size": math.nan,
            "dominant_two_cycle_attractor_share": math.nan,
            "dominant_threeplus_cycle_attractor_share": math.nan,
            "effective_incoming_hhi": math.nan,
            "effective_incoming_max_share": math.nan,
            "effective_incoming_top5_share": math.nan,
            "gateway_incoming_reliance_share": math.nan,
        }

    agents_by_pos = {
        int(agent.pos): agent for agent in _agents(model)
    }
    citizen_positions = {int(c.pos) for c in citizens(model)}
    top = {
        int(ego): _top_source(ego_weights)
        for ego, ego_weights in weights.items()
    }

    fate_counts = Counter()
    unique_cycles: set[frozenset[int]] = set()
    same_group_cycle_egos = 0
    two_cycle_egos = 0
    threeplus_cycle_egos = 0

    for ego in sorted(citizen_positions):
        current = ego
        path: list[int] = []
        index: dict[int, int] = {}
        fate = "unresolved"
        cycle_nodes: list[int] = []

        while True:
            if current in index:
                cycle_nodes = path[index[current]:]
                fate = "cycle"
                unique_cycles.add(frozenset(cycle_nodes))
                break
            if current not in citizen_positions:
                agent = agents_by_pos.get(current)
                source_type = getattr(agent, "type_of_agent", None)
                if source_type == "infoprovider":
                    fate = "expert"
                elif source_type == "disruptivejammer":
                    fate = "jammer"
                break
            nxt = top.get(current)
            if nxt is None:
                break
            index[current] = len(path)
            path.append(current)
            next_agent = agents_by_pos.get(int(nxt))
            next_type = getattr(next_agent, "type_of_agent", None)
            if next_type == "infoprovider":
                fate = "expert"
                break
            if next_type == "disruptivejammer":
                fate = "jammer"
                break
            current = int(nxt)

        fate_counts[fate] += 1
        if fate == "cycle" and cycle_nodes:
            ego_group = getattr(agents_by_pos.get(ego), "group_id", None)
            cycle_groups = {
                getattr(agents_by_pos.get(node), "group_id", None)
                for node in cycle_nodes
            }
            if len(cycle_groups) == 1 and ego_group in cycle_groups:
                same_group_cycle_egos += 1
            if len(cycle_nodes) == 2:
                two_cycle_egos += 1
            if len(cycle_nodes) >= 3:
                threeplus_cycle_egos += 1

    n = max(len(citizen_positions), 1)
    cycle_sizes = [len(cycle) for cycle in unique_cycles]

    incoming = Counter()
    for ego_weights in weights.values():
        for source, weight in ego_weights.items():
            incoming[int(source)] += float(weight)
    total_incoming = float(sum(incoming.values()))
    shares = (
        sorted(
            [float(value / total_incoming) for value in incoming.values()],
            reverse=True,
        )
        if total_incoming > 0.0
        else []
    )
    incoming_hhi = float(sum(share * share for share in shares)) if shares else math.nan
    gateway_share = math.nan
    if gateway_positions is not None and total_incoming > 0.0:
        gateways = {int(x) for x in gateway_positions}
        gateway_share = float(
            sum(incoming.get(node, 0.0) for node in gateways) / total_incoming
        )

    return {
        "dominant_expert_reach_share": float(fate_counts["expert"] / n),
        "dominant_jammer_reach_share": float(fate_counts["jammer"] / n),
        "dominant_citizen_cycle_share": float(fate_counts["cycle"] / n),
        "dominant_same_group_cycle_share": float(same_group_cycle_egos / n),
        "dominant_cycle_mean_size": (
            float(np.mean(cycle_sizes)) if cycle_sizes else 0.0
        ),
        "dominant_cycle_max_size": (
            int(max(cycle_sizes)) if cycle_sizes else 0
        ),
        "dominant_two_cycle_attractor_share": float(two_cycle_egos / n),
        "dominant_threeplus_cycle_attractor_share": float(
            threeplus_cycle_egos / n
        ),
        "effective_incoming_hhi": incoming_hhi,
        "effective_incoming_max_share": (
            float(shares[0]) if shares else math.nan
        ),
        "effective_incoming_top5_share": (
            float(sum(shares[:5])) if shares else math.nan
        ),
        "gateway_incoming_reliance_share": gateway_share,
    }


def theory_metrics(model) -> dict:
    """One compact run-level snapshot aligned with the manuscript theory."""
    out = {}
    out.update(belief_summary(model))
    out.update(reliance_composition(model))
    out["structural_effective_divergence"] = structural_effective_divergence(model)
    out.update(effective_homophily(model))
    out.update(effective_reliance_dynamics(model))
    return out

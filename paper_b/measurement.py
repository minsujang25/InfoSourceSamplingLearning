"""Measurement-audit helpers for Paper B.

This module distinguishes:
    A   structural opportunity,
    Lambda behavioral acquisition/reliance,
    W   cumulative source precision actually added to Gaussian theta updates.

None of these helpers changes model behavior.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


EXPERT_POS = 0
JAMMER_POS = 1
CITIZEN_START = 2


def structural_uniform_metrics(
    source_map: dict[int, list[int]],
    *,
    group_ids: dict[int, int] | None = None,
    gateway_positions: set[int] | None = None,
) -> dict:
    """Metrics under equal use of every structurally available source.

    U^A_ij = A_ij / degree_i.

    gateway_uniform_total_share uses all incoming source mass as its
    denominator. gateway_uniform_peer_conditional_share first renormalizes
    within each ego's citizen-peer opportunities, then averages across egos
    with at least one citizen peer.
    """
    incoming = Counter()
    homophily_values = []
    gateway_peer_values = []

    gateways = (
        {int(x) for x in gateway_positions}
        if gateway_positions is not None
        else None
    )

    for ego, raw_sources in source_map.items():
        sources = [int(x) for x in raw_sources]
        if not sources:
            continue
        weight = 1.0 / len(sources)
        for source in sources:
            incoming[source] += weight

        peers = [source for source in sources if source >= CITIZEN_START]
        if peers and group_ids is not None:
            same = sum(
                int(group_ids[int(ego)] == group_ids[int(source)])
                for source in peers
            )
            homophily_values.append(same / len(peers))

        if peers and gateways is not None:
            gateway_peer_values.append(
                sum(int(source in gateways) for source in peers) / len(peers)
            )

    total = float(sum(incoming.values()))
    shares = (
        sorted(
            (float(value / total) for value in incoming.values()),
            reverse=True,
        )
        if total > 0.0
        else []
    )

    gateway_total_share = math.nan
    if gateways is not None and total > 0.0:
        gateway_total_share = float(
            sum(incoming.get(node, 0.0) for node in gateways) / total
        )

    return {
        "A_uniform_structural_homophily": (
            float(np.mean(homophily_values))
            if homophily_values else math.nan
        ),
        "A_uniform_incoming_hhi": (
            float(sum(share * share for share in shares))
            if shares else math.nan
        ),
        "A_uniform_incoming_top5_share": (
            float(sum(shares[:5])) if shares else math.nan
        ),
        "A_uniform_gateway_total_share": gateway_total_share,
        "A_uniform_gateway_peer_conditional_share": (
            float(np.mean(gateway_peer_values))
            if gateway_peer_values else math.nan
        ),
    }


def null_last_share(model) -> float:
    """Fraction of citizens whose null source is last in behavioral ranking."""
    if getattr(model, "jammer_regime", None) != "null":
        return math.nan

    values = []
    for citizen in model.citizens:
        ranking = list(citizen._behavioral_ranking())
        null_positions = [
            idx
            for idx, source in enumerate(ranking)
            if source.type_of_agent == "disruptivejammer"
        ]
        if len(null_positions) != 1:
            values.append(False)
            continue
        values.append(null_positions[0] == len(ranking) - 1)
    return float(np.mean(values)) if values else math.nan


def _source_class(source) -> str:
    source_type = getattr(source, "type_of_agent", "unknown")
    if source_type == "infoprovider":
        return "expert"
    if source_type == "disruptivejammer":
        return "jammer"
    if source_type == "citizen":
        return "peer"
    return str(source_type)


def evidence_precision_weights(model) -> dict[int, dict[int, float]]:
    """Return ego-normalized cumulative source-precision weights W_T."""
    out: dict[int, dict[int, float]] = {}
    for citizen in model.citizens:
        cumulative = getattr(citizen, "cumulative_evidence_precision", {})
        values = {
            int(source.pos): float(value)
            for source, value in cumulative.items()
            if float(value) > 0.0
        }
        total = float(sum(values.values()))
        if total <= 0.0:
            out[int(citizen.pos)] = {
                int(source.pos): 0.0 for source in citizen.info_source
            }
        else:
            out[int(citizen.pos)] = {
                int(source.pos): float(cumulative.get(source, 0.0)) / total
                for source in citizen.info_source
            }
    return out


def _top_source(weight_map: dict[int, float]) -> int | None:
    positive = {
        int(source): float(weight)
        for source, weight in weight_map.items()
        if float(weight) > 0.0
    }
    if not positive:
        return None
    return min(
        positive,
        key=lambda source: (-positive[source], int(source)),
    )


def evidence_precision_network_metrics(
    model,
    *,
    gateway_positions: set[int] | None = None,
) -> dict:
    """Summarize cumulative evidence-precision network W.

    W is not a causal influence graph. It records the share of source precision
    actually added to each citizen's Gaussian state updates.
    """
    weights = evidence_precision_weights(model)
    citizens = list(model.citizens)
    if not citizens:
        return {}

    agents_by_pos = {int(agent.pos): agent for agent in model.agents}
    citizen_positions = {int(c.pos) for c in citizens}
    citizen_group = {int(c.pos): c.group_id for c in citizens}

    totals = Counter()
    homophily_values = []
    peer_mass_values = []
    incoming = Counter()

    for ego, ego_weights in weights.items():
        peer_mass = 0.0
        same_peer_mass = 0.0
        for source_pos, weight in ego_weights.items():
            source = agents_by_pos[int(source_pos)]
            source_class = _source_class(source)
            totals[source_class] += float(weight)
            incoming[int(source_pos)] += float(weight)

            if source_class == "peer":
                peer_mass += float(weight)
                if getattr(source, "group_id", None) == citizen_group.get(ego):
                    same_peer_mass += float(weight)

        peer_mass_values.append(peer_mass)
        if peer_mass > 0.0:
            homophily_values.append(same_peer_mass / peer_mass)

    n = max(len(citizens), 1)
    total_incoming = float(sum(incoming.values()))
    shares = (
        sorted(
            (float(v / total_incoming) for v in incoming.values()),
            reverse=True,
        )
        if total_incoming > 0.0
        else []
    )

    gateway_share = math.nan
    if gateway_positions is not None and total_incoming > 0.0:
        gateways = {int(x) for x in gateway_positions}
        gateway_share = float(
            sum(incoming.get(node, 0.0) for node in gateways)
            / total_incoming
        )

    top = {ego: _top_source(ego_weights) for ego, ego_weights in weights.items()}
    fate_counts = Counter()
    unique_cycles: set[frozenset[int]] = set()
    same_group_cycle_egos = 0

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
                source = agents_by_pos.get(current)
                source_type = getattr(source, "type_of_agent", None)
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

            source = agents_by_pos.get(int(nxt))
            source_type = getattr(source, "type_of_agent", None)
            if source_type == "infoprovider":
                fate = "expert"
                break
            if source_type == "disruptivejammer":
                fate = "jammer"
                break
            current = int(nxt)

        fate_counts[fate] += 1
        if fate == "cycle" and cycle_nodes:
            ego_group = citizen_group.get(ego)
            cycle_groups = {
                citizen_group.get(node)
                for node in cycle_nodes
            }
            if len(cycle_groups) == 1 and ego_group in cycle_groups:
                same_group_cycle_egos += 1

    return {
        "W_expert_precision_share": float(totals["expert"] / n),
        "W_jammer_precision_share": float(totals["jammer"] / n),
        "W_peer_precision_share": float(totals["peer"] / n),
        "W_effective_homophily": (
            float(np.mean(homophily_values))
            if homophily_values else math.nan
        ),
        "W_peer_precision_mass": (
            float(np.mean(peer_mass_values))
            if peer_mass_values else math.nan
        ),
        "W_incoming_hhi": (
            float(sum(share * share for share in shares))
            if shares else math.nan
        ),
        "W_incoming_top5_share": (
            float(sum(shares[:5])) if shares else math.nan
        ),
        "W_gateway_incoming_share": gateway_share,
        "W_dominant_expert_reach_share": float(
            fate_counts["expert"] / n
        ),
        "W_dominant_jammer_reach_share": float(
            fate_counts["jammer"] / n
        ),
        "W_dominant_citizen_cycle_share": float(
            fate_counts["cycle"] / n
        ),
        "W_dominant_same_group_cycle_share": float(
            same_group_cycle_egos / n
        ),
        "W_dominant_cycle_count": int(len(unique_cycles)),
    }

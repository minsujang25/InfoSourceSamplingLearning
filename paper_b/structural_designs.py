"""Clean matched structural designs for Paper B Experiments III and IV.

These helpers generate explicit source maps before a Mesa model is created.
The model then consumes those maps verbatim through `structural_source_map`.
This makes it possible to hold selected structural components exactly fixed
across counterfactual conditions instead of relying on coincident RNG draws.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EXPERT_POS = 0
JAMMER_POS = 1
CITIZEN_START = 2


def citizen_positions(n_citizens: int) -> list[int]:
    if n_citizens < 4:
        raise ValueError("n_citizens must be at least 4.")
    return list(range(CITIZEN_START, CITIZEN_START + int(n_citizens)))


def _rng(seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), *(int(p) for p in parts)])
    )


def balanced_fixed_group_ids(
    *,
    seed: int,
    n_citizens: int,
) -> dict[int, int]:
    """Return balanced citizen labels that do not depend on initial beliefs.

    Expert is assigned to the non-positive group (-1) and Jammer to +1, matching
    the source-group convention used elsewhere in the model. Citizen labels are
    shuffled once per matched seed and reused across all Experiment IV cells.
    """
    positions = citizen_positions(n_citizens)
    n_left = len(positions) // 2
    labels = np.asarray(
        [-1] * n_left + [1] * (len(positions) - n_left),
        dtype=int,
    )
    _rng(seed, 4101).shuffle(labels)
    mapping = {EXPERT_POS: -1, JAMMER_POS: 1}
    mapping.update(
        {pos: int(label) for pos, label in zip(positions, labels)}
    )
    return mapping


def select_expert_gateways(
    *,
    seed: int,
    n_citizens: int,
    expert_access_share: float = 0.10,
    min_count: int = 2,
) -> tuple[int, ...]:
    """Choose an exact fixed subset with direct Expert access."""
    if not 0.0 < expert_access_share <= 1.0:
        raise ValueError("expert_access_share must lie in (0,1].")
    positions = citizen_positions(n_citizens)
    requested = int(round(expert_access_share * len(positions)))
    count = min(len(positions), max(int(min_count), requested))
    chosen = _rng(seed, 4201).choice(
        np.asarray(positions, dtype=int),
        size=count,
        replace=False,
    )
    return tuple(sorted(int(x) for x in chosen))


def _ranked(
    *,
    seed: int,
    ego: int,
    salt: int,
    candidates: Iterable[int],
) -> list[int]:
    values = np.asarray(sorted(set(int(x) for x in candidates)), dtype=int)
    if values.size == 0:
        return []
    order = _rng(seed, salt, ego).permutation(values.size)
    return [int(values[int(i)]) for i in order]


def _with_fixed_elite_access(
    *,
    ego: int,
    peers: list[int],
    expert_gateways: set[int],
    jammer_universal: bool = True,
) -> list[int]:
    sources = []
    if jammer_universal:
        sources.append(JAMMER_POS)
    if ego in expert_gateways:
        sources.append(EXPERT_POS)
    sources.extend(peers)
    if ego in sources:
        raise ValueError(f"Self source generated for citizen {ego}.")
    if len(sources) != len(set(sources)):
        raise ValueError(f"Duplicate source generated for citizen {ego}.")
    return sorted(sources)


def exp3_redundancy_source_maps(
    *,
    seed: int,
    n_citizens: int,
    peer_degree: int = 2,
    expert_access_share: float = 0.10,
) -> dict:
    """Generate low/high local corrective-route redundancy.

    Direct elite access is exactly identical in both conditions:
      * Jammer access is universal;
      * Expert access is restricted to one fixed gateway subset.

    Peer degree is exactly two in both conditions. For citizens without direct
    Expert access:
      * LOW: one peer is an Expert gateway and one is a non-gateway;
      * HIGH: both peers are distinct Expert gateways.

    Thus the manipulation changes the number of distinct length-2 corrective
    routes while holding direct elite access and peer degree fixed.

    Citizens with direct Expert access receive the same two non-gateway peers in
    both conditions, so their local structure is unchanged.
    """
    if peer_degree != 2:
        raise ValueError(
            "Experiment III primary design is defined for peer_degree=2."
        )

    positions = citizen_positions(n_citizens)
    gateways = set(
        select_expert_gateways(
            seed=seed,
            n_citizens=n_citizens,
            expert_access_share=expert_access_share,
            min_count=2,
        )
    )
    nongateways = set(positions) - gateways
    if len(gateways) < 2:
        raise ValueError("High redundancy requires at least two Expert gateways.")
    if len(nongateways) < 2:
        raise ValueError("Low redundancy requires at least two non-gateways.")

    low: dict[int, list[int]] = {}
    high: dict[int, list[int]] = {}

    for ego in positions:
        gateway_order = _ranked(
            seed=seed,
            ego=ego,
            salt=4301,
            candidates=(g for g in gateways if g != ego),
        )
        nongateway_order = _ranked(
            seed=seed,
            ego=ego,
            salt=4302,
            candidates=(g for g in nongateways if g != ego),
        )

        if ego in gateways:
            # Keep gateway citizens exactly identical across the manipulation.
            peers = nongateway_order[:peer_degree]
            if len(peers) < peer_degree:
                fallback = [
                    x for x in gateway_order
                    if x not in peers
                ]
                peers.extend(fallback[: peer_degree - len(peers)])
            low_peers = list(peers)
            high_peers = list(peers)
        else:
            if not gateway_order or not nongateway_order:
                raise ValueError(
                    "Insufficient gateway/non-gateway candidates for Exp III."
                )
            low_peers = [gateway_order[0], nongateway_order[0]]
            high_peers = gateway_order[:peer_degree]
            if len(high_peers) < peer_degree:
                raise ValueError(
                    "Insufficient distinct Expert gateways for high redundancy."
                )

        low[ego] = _with_fixed_elite_access(
            ego=ego,
            peers=low_peers,
            expert_gateways=gateways,
        )
        high[ego] = _with_fixed_elite_access(
            ego=ego,
            peers=high_peers,
            expert_gateways=gateways,
        )

    return {
        "expert_gateways": tuple(sorted(gateways)),
        "jammer_universal": True,
        "peer_degree": peer_degree,
        "expert_access_share_realized": len(gateways) / len(positions),
        "low": low,
        "high": high,
    }


def two_step_expert_route_count(
    source_map: dict[int, list[int]],
    *,
    expert_gateways: Iterable[int],
) -> dict[int, int]:
    """Count distinct length-2 routes ego -> gateway -> Expert."""
    gateways = set(int(x) for x in expert_gateways)
    return {
        int(ego): sum(
            1
            for source in sources
            if int(source) in gateways and int(source) >= CITIZEN_START
        )
        for ego, sources in source_map.items()
    }



def exp3b_path_independence_source_maps(
    *,
    seed: int,
    n_citizens: int = 100,
    n_gateways: int = 10,
    n_relays: int = 40,
) -> dict:
    """Generate a clean shared-bottleneck vs independent-path counterfactual.

    The design is deliberately layered. Citizens are partitioned once per seed
    into three fixed roles:

      * gateways: direct Expert + Jammer access;
      * relays: Jammer + exactly one gateway source;
      * focal citizens: Jammer + exactly two fixed relay sources.

    The focal citizen -> relay edges are identical in both conditions. Only the
    relay -> gateway assignment is rewired.

    Relay pairs are grouped into four-relay blocks. For each block with
    gateway pair (g1, g2):

      SHARED:
          r1,r2 -> g1
          r3,r4 -> g2

      INDEPENDENT:
          r1,r3 -> g1
          r2,r4 -> g2

    Thus each focal citizen always observes the same two immediate relays. In
    the shared condition those relays converge on one gateway, whereas in the
    independent condition they connect to distinct gateways. Gateway indegree
    from relays is identical in both conditions, as are all per-node source
    counts and all elite opportunities.

    With the default N=100 design:
      10 gateways + 40 relays + 50 focal citizens = 100 citizens.
    """
    positions = citizen_positions(n_citizens)
    if n_gateways < 2 or n_gateways % 2 != 0:
        raise ValueError("n_gateways must be an even integer >= 2.")
    if n_relays < 4 or n_relays % 4 != 0:
        raise ValueError("n_relays must be a positive multiple of 4.")
    n_focals = len(positions) - int(n_gateways) - int(n_relays)
    if n_focals <= 0:
        raise ValueError("Exp IIIb requires at least one focal citizen.")

    role_order = list(
        _rng(seed, 4501).permutation(np.asarray(positions, dtype=int))
    )
    gateways = [int(x) for x in role_order[:n_gateways]]
    relays = [
        int(x)
        for x in role_order[n_gateways : n_gateways + n_relays]
    ]
    focals = [
        int(x)
        for x in role_order[n_gateways + n_relays :]
    ]

    gateway_order = list(
        _rng(seed, 4502).permutation(np.asarray(gateways, dtype=int))
    )
    relay_order = list(
        _rng(seed, 4503).permutation(np.asarray(relays, dtype=int))
    )
    focal_order = list(
        _rng(seed, 4504).permutation(np.asarray(focals, dtype=int))
    )

    relay_pairs = [
        (relay_order[i], relay_order[i + 1])
        for i in range(0, len(relay_order), 2)
    ]
    n_pair_blocks = len(relay_pairs) // 2
    if n_pair_blocks <= 0:
        raise ValueError("Exp IIIb requires at least two relay pairs.")

    # Assign each four-relay block a pair of distinct gateways while keeping
    # gateway use balanced. With the default 10 gateways / 10 blocks, each
    # gateway appears in exactly two blocks.
    block_gateways = []
    for block in range(n_pair_blocks):
        g1 = gateway_order[block % len(gateway_order)]
        offset = max(1, len(gateway_order) // 2)
        g2 = gateway_order[(block + offset) % len(gateway_order)]
        if g1 == g2:
            raise RuntimeError("Exp IIIb generated identical block gateways.")
        block_gateways.append((int(g1), int(g2)))

    shared_relay_gateway: dict[int, int] = {}
    independent_relay_gateway: dict[int, int] = {}
    for block in range(n_pair_blocks):
        pair_a = relay_pairs[2 * block]
        pair_b = relay_pairs[2 * block + 1]
        g1, g2 = block_gateways[block]
        r1, r2 = pair_a
        r3, r4 = pair_b

        shared_relay_gateway[r1] = g1
        shared_relay_gateway[r2] = g1
        shared_relay_gateway[r3] = g2
        shared_relay_gateway[r4] = g2

        independent_relay_gateway[r1] = g1
        independent_relay_gateway[r2] = g2
        independent_relay_gateway[r3] = g1
        independent_relay_gateway[r4] = g2

    # Reuse relay pairs across focal citizens, but keep every focal's immediate
    # sources exactly identical across the two path-overlap conditions.
    focal_to_relays: dict[int, tuple[int, int]] = {}
    pair_cycle = list(
        _rng(seed, 4505).permutation(np.arange(len(relay_pairs), dtype=int))
    )
    for idx, focal in enumerate(focal_order):
        pair_idx = int(pair_cycle[idx % len(pair_cycle)])
        focal_to_relays[int(focal)] = tuple(
            int(x) for x in relay_pairs[pair_idx]
        )

    def build(relay_gateway: dict[int, int]) -> dict[int, list[int]]:
        source_map: dict[int, list[int]] = {}
        for gateway in gateways:
            source_map[int(gateway)] = sorted([EXPERT_POS, JAMMER_POS])
        for relay in relays:
            source_map[int(relay)] = sorted(
                [JAMMER_POS, int(relay_gateway[int(relay)])]
            )
        for focal in focals:
            r1, r2 = focal_to_relays[int(focal)]
            source_map[int(focal)] = sorted([JAMMER_POS, r1, r2])
        return source_map

    shared = build(shared_relay_gateway)
    independent = build(independent_relay_gateway)

    roles = {
        **{int(x): "gateway" for x in gateways},
        **{int(x): "relay" for x in relays},
        **{int(x): "focal" for x in focals},
    }

    return {
        "shared": shared,
        "independent": independent,
        "roles": roles,
        "gateways": tuple(sorted(gateways)),
        "relays": tuple(sorted(relays)),
        "focals": tuple(sorted(focals)),
        "focal_to_relays": {
            int(k): tuple(int(x) for x in v)
            for k, v in focal_to_relays.items()
        },
        "shared_relay_gateway": {
            int(k): int(v) for k, v in shared_relay_gateway.items()
        },
        "independent_relay_gateway": {
            int(k): int(v) for k, v in independent_relay_gateway.items()
        },
        "n_gateways": int(n_gateways),
        "n_relays": int(n_relays),
        "n_focals": int(n_focals),
    }


def exp3b_focal_corrective_connectivity(
    source_map: dict[int, list[int]],
    *,
    focals: Iterable[int],
    relays: Iterable[int],
    gateways: Iterable[int],
) -> dict[int, int]:
    """Count internally vertex-disjoint designed focal->Expert paths.

    In the layered Exp IIIb graph each focal has two relay sources. Each relay
    has exactly one gateway source, and each gateway has direct Expert access.
    The maximum number of internally vertex-disjoint length-3 corrective paths
    is therefore the number of distinct gateways reached by the focal's two
    relay sources: one under the shared-bottleneck treatment and two under the
    independent-path treatment.
    """
    relay_set = set(int(x) for x in relays)
    gateway_set = set(int(x) for x in gateways)
    out: dict[int, int] = {}
    for focal in focals:
        relay_sources = [
            int(x)
            for x in source_map[int(focal)]
            if int(x) in relay_set
        ]
        reached = set()
        for relay in relay_sources:
            reached.update(
                int(x)
                for x in source_map[int(relay)]
                if int(x) in gateway_set
            )
        out[int(focal)] = len(reached)
    return out


def exp3b_focal_nominal_corrective_route_count(
    source_map: dict[int, list[int]],
    *,
    focals: Iterable[int],
    relays: Iterable[int],
    gateways: Iterable[int],
) -> dict[int, int]:
    """Count designed focal->relay->gateway->Expert routes, overlap allowed.

    Unlike exp3b_focal_corrective_connectivity, this quantity counts both
    nominal corrective routes even when they share the same gateway. The clean
    IIIb counterfactual requires this count to equal two in both treatments.
    """
    relay_set = set(int(x) for x in relays)
    gateway_set = set(int(x) for x in gateways)
    out: dict[int, int] = {}
    for focal in focals:
        count = 0
        relay_sources = [
            int(x)
            for x in source_map[int(focal)]
            if int(x) in relay_set
        ]
        for relay in relay_sources:
            count += sum(
                int(source) in gateway_set
                for source in source_map[int(relay)]
            )
        out[int(focal)] = int(count)
    return out


def exp3b_focal_shared_bottleneck_indicator(
    source_map: dict[int, list[int]],
    *,
    focals: Iterable[int],
    relays: Iterable[int],
    gateways: Iterable[int],
) -> dict[int, int]:
    """Return one when a focal's nominal corrective routes share a gateway."""
    nominal = exp3b_focal_nominal_corrective_route_count(
        source_map,
        focals=focals,
        relays=relays,
        gateways=gateways,
    )
    disjoint = exp3b_focal_corrective_connectivity(
        source_map,
        focals=focals,
        relays=relays,
        gateways=gateways,
    )
    return {
        int(focal): int(nominal[int(focal)] > disjoint[int(focal)])
        for focal in focals
    }


def _homophilous_peers_for_ego(
    *,
    seed: int,
    ego: int,
    group_ids: dict[int, int],
    peer_degree: int,
    same_group_probability: float,
) -> list[int]:
    peers = [
        pos
        for pos in group_ids
        if pos >= CITIZEN_START and pos != ego
    ]
    same = [pos for pos in peers if group_ids[pos] == group_ids[ego]]
    other = [pos for pos in peers if group_ids[pos] != group_ids[ego]]

    same_order = _ranked(
        seed=seed,
        ego=ego,
        salt=4401,
        candidates=same,
    )
    other_order = _ranked(
        seed=seed,
        ego=ego,
        salt=4402,
        candidates=other,
    )
    draws = _rng(seed, 4403, ego).random(peer_degree)

    selected: list[int] = []
    same_idx = 0
    other_idx = 0
    for draw in draws:
        want_same = bool(draw < same_group_probability)
        preferred = same_order if want_same else other_order
        fallback = other_order if want_same else same_order

        chosen = None
        if want_same:
            while same_idx < len(same_order):
                candidate = same_order[same_idx]
                same_idx += 1
                if candidate not in selected:
                    chosen = candidate
                    break
        else:
            while other_idx < len(other_order):
                candidate = other_order[other_idx]
                other_idx += 1
                if candidate not in selected:
                    chosen = candidate
                    break

        if chosen is None:
            for candidate in fallback:
                if candidate not in selected:
                    chosen = candidate
                    break

        if chosen is None:
            raise ValueError(f"Unable to assign {peer_degree} peers to {ego}.")
        selected.append(int(chosen))

    return selected


def exp4_homophily_source_maps(
    *,
    seed: int,
    n_citizens: int,
    group_ids: dict[int, int] | None = None,
    peer_degree: int = 2,
    low_homophily: float = 0.50,
    high_homophily: float = 0.90,
) -> dict:
    """Generate paired low/high homophily structures with universal elite access.

    Experiment IV is a clean manipulation of citizen-peer mixing and prior
    segregation. Every citizen therefore has the same two elite opportunities
    (Expert and Jammer) plus exactly peer_degree citizen peers in every
    factorial cell. This avoids the asymmetric topology in which the Jammer
    was universal but the Expert was available only to a small subset.

    Low/high homophily use common random draws and common candidate rankings,
    making them paired structural counterfactuals.
    """
    if not (0.0 <= low_homophily <= high_homophily <= 1.0):
        raise ValueError("Require 0 <= low_homophily <= high_homophily <= 1.")
    if peer_degree <= 0:
        raise ValueError("peer_degree must be positive.")

    positions = citizen_positions(n_citizens)
    if group_ids is None:
        group_ids = balanced_fixed_group_ids(
            seed=seed,
            n_citizens=n_citizens,
        )
    else:
        group_ids = {int(k): int(v) for k, v in group_ids.items()}

    # Both elite sources are universal in Experiment IV. This matches the
    # extended-network baseline and makes peer homophily the only structural
    # quantity manipulated by the H treatment.
    gateways = set(positions)

    maps = {}
    for label, probability in (
        ("low", low_homophily),
        ("high", high_homophily),
    ):
        source_map = {}
        for ego in positions:
            peers = _homophilous_peers_for_ego(
                seed=seed,
                ego=ego,
                group_ids=group_ids,
                peer_degree=peer_degree,
                same_group_probability=probability,
            )
            source_map[ego] = _with_fixed_elite_access(
                ego=ego,
                peers=peers,
                expert_gateways=gateways,
            )
        maps[label] = source_map

    return {
        "fixed_group_ids": group_ids,
        "expert_gateways": tuple(sorted(gateways)),
        "jammer_universal": True,
        "peer_degree": peer_degree,
        "expert_access_share_realized": 1.0,
        "jammer_access_share_realized": 1.0,
        "elite_access_mode": "universal_expert_and_jammer",
        "low_homophily_probability": float(low_homophily),
        "high_homophily_probability": float(high_homophily),
        **maps,
    }


def structural_peer_homophily(
    source_map: dict[int, list[int]],
    *,
    group_ids: dict[int, int],
) -> float:
    same = 0
    total = 0
    for ego, sources in source_map.items():
        for source in sources:
            if source < CITIZEN_START:
                continue
            same += int(group_ids[int(ego)] == group_ids[int(source)])
            total += 1
    return float(same / total) if total else math.nan


def exp4_initial_beliefs(
    *,
    seed: int,
    n_citizens: int,
    group_ids: dict[int, int],
    segregation: str,
    high_group_shift: float = 3.0,
    residual_sd: float = 1.0,
) -> list[float]:
    """Generate paired low/high prior segregation from common residual draws.

    LOW: both fixed groups share the same N(0, residual_sd^2) distribution.
    HIGH: the exact same individual residual is shifted by +/- high_group_shift
          according to the fixed group label.

    Expected between-group mean separation is therefore 0 under LOW and
    2*high_group_shift under HIGH, while individual stochastic residuals are
    held constant across the segregation counterfactual.
    """
    segregation = str(segregation).lower()
    if segregation not in {"low", "high"}:
        raise ValueError("segregation must be 'low' or 'high'.")

    positions = citizen_positions(n_citizens)
    residuals = _rng(seed, 4501).normal(
        loc=0.0,
        scale=float(residual_sd),
        size=len(positions),
    )
    shift = 0.0 if segregation == "low" else float(high_group_shift)

    citizens = [
        float(residual + shift * group_ids[pos])
        for pos, residual in zip(positions, residuals)
    ]
    return [0.0, 4.0] + citizens


def realized_prior_segregation(
    mu_theta: list[float],
    *,
    group_ids: dict[int, int],
) -> float:
    """Absolute realized citizen group-mean gap."""
    left = [
        float(mu_theta[pos])
        for pos in group_ids
        if pos >= CITIZEN_START and group_ids[pos] == -1
    ]
    right = [
        float(mu_theta[pos])
        for pos in group_ids
        if pos >= CITIZEN_START and group_ids[pos] == 1
    ]
    if not left or not right:
        return math.nan
    return float(abs(np.mean(right) - np.mean(left)))

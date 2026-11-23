"""Consolidate a Gate 0B sender-grid pilot without reusing J1/J0 estimands.

Every seed/prior/network/receiver matched block contains at least a null sender
and the selected sender regimes. The 'null' comparator retains its structural
source slot and the same period-0 centroid audit as adversarial senders.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from paper_b.experiments.consolidate_pilot import (
    _read_shard,
    _write_csv,
    _write_gzip_csv,
)


def _key(row: dict, *fields: str) -> tuple:
    return tuple(row[field] for field in fields)


def sender_damage(rows: list[dict], *, by_period: bool = False) -> list[dict]:
    """Within-seed sender minus null MSE, never sender minus truth clone."""
    fields = ["seed", "initial_regime", "network_environment", "reliance_mode"]
    if by_period:
        fields.append("horizon_step")
    index = {}
    for row in rows:
        index[_key(row, *fields) + (row["jammer_regime"],)] = row
    output = []
    bases = sorted({_key(row, *fields) for row in rows})
    for base in bases:
        null = index.get(base + ("null",))
        if null is None:
            raise ValueError(f"Missing matched null sender for {base}")
        mse_null = float(null["mse_truth"])
        for sender in sorted({
            row["jammer_regime"]
            for row in rows
            if _key(row, *fields) == base and row["jammer_regime"] != "null"
        }):
            candidate = index[base + (sender,)]
            mse = float(candidate["mse_truth"])
            record = dict(zip(fields, base))
            record.update(
                sender_regime=sender,
                baseline_regime="null",
                mse_sender=mse,
                mse_null=mse_null,
                delta_mse=mse - mse_null,
            )
            output.append(record)
    return output


def peer_damage_differences(
    damage: list[dict],
    *,
    elite: str = "elite_only",
    peer: str = "extended",
) -> list[dict]:
    """Peer-minus-elite contrast in excess damage D for each matched seed."""
    keys = ("seed", "initial_regime", "reliance_mode", "sender_regime")
    idx = {_key(row, *keys) + (row["network_environment"],): row
           for row in damage}
    output = []
    for base in sorted({_key(row, *keys) for row in damage}):
        first = idx.get(base + (elite,))
        second = idx.get(base + (peer,))
        if first is None or second is None:
            continue
        row = dict(zip(keys, base))
        row.update(
            network_without_peers=elite,
            network_with_peers=peer,
            d_elite=float(first["delta_mse"]),
            d_peers=float(second["delta_mse"]),
            peer_minus_elite_damage=(
                float(second["delta_mse"]) - float(first["delta_mse"])
            ),
            peer_minus_elite_total_mse=(
                float(second["mse_sender"]) - float(first["mse_sender"])
            ),
            peer_minus_elite_null_mse=(
                float(second["mse_null"]) - float(first["mse_null"])
            ),
        )
        output.append(row)
    return output


def adaptation_network_interactions(
    damage: list[dict],
    *,
    elite: str = "elite_only",
    peer: str = "extended",
) -> list[dict]:
    """(adaptive - fixed) difference in D, compared across peer environments."""
    idx = {
        _key(row, "seed", "initial_regime", "reliance_mode",
             "network_environment", "sender_regime"): float(row["delta_mse"])
        for row in damage
    }
    out = []
    bases = sorted({_key(row, "seed", "initial_regime", "reliance_mode")
                    for row in damage})
    for base in bases:
        try:
            e_a = idx[base + (elite, "adaptive")]
            e_f = idx[base + (elite, "fixed_biased")]
            p_a = idx[base + (peer, "adaptive")]
            p_f = idx[base + (peer, "fixed_biased")]
        except KeyError:
            continue
        out.append({
            "seed": base[0],
            "initial_regime": base[1],
            "reliance_mode": base[2],
            "adaptive_minus_fixed_elite": e_a - e_f,
            "adaptive_minus_fixed_peer": p_a - p_f,
            "adaptive_by_peer_interaction": (p_a - p_f) - (e_a - e_f),
        })
    return out


def receiver_differences(damage: list[dict]) -> list[dict]:
    """Adaptive minus frozen excess damage (do not conflate sender adaptation)."""
    fields = ("seed", "initial_regime", "network_environment", "sender_regime")
    idx = {_key(row, *fields) + (row["reliance_mode"],): row
           for row in damage}
    out = []
    for base in sorted({_key(row, *fields) for row in damage}):
        a = idx.get(base + ("adaptive",))
        f = idx.get(base + ("frozen",))
        if a is None or f is None:
            continue
        record = dict(zip(fields, base))
        record.update(
            adaptive_minus_frozen_damage=(
                float(a["delta_mse"]) - float(f["delta_mse"])
            ),
            adaptive_minus_frozen_total_mse=(
                float(a["mse_sender"]) - float(f["mse_sender"])
            ),
        )
        out.append(record)
    return out


def consolidate_gate0b(run_root: str | Path) -> dict:
    root = Path(run_root)
    manifest = json.loads((root / "pilot_manifest.json").read_text())
    if manifest.get("analysis_mode") != "sender_grid":
        raise ValueError("Gate 0B consolidation requires analysis_mode=sender_grid")
    paths = sorted((root / "shards").glob("*.json.gz"))
    shards = [_read_shard(p) for p in paths]
    runs = [x for p in shards for x in p.get("runs", [])]
    beliefs = [x for p in shards for x in p.get("terminal_beliefs", [])]
    trajectories = [x for p in shards for x in p.get("belief_checkpoints", [])]
    reliance = [x for p in shards for x in p.get("lambda_checkpoints", [])]
    strategies = [x for p in shards for x in p.get("jammer_strategy", [])]
    edges = [x for p in shards for x in p.get("reliance_edges", [])]

    # Enforce matched initial-state identity across network environments.
    initial = defaultdict(set)
    for p in shards:
        key = (int(p["seed"]), p["initial_regime"])
        initial[key].add(p["initial_state_fingerprint"])
    bad = [key for key, hashes in initial.items() if len(hashes) != 1]
    if bad:
        raise ValueError(f"Cross-network initial states differ for {bad[:5]}")

    damage = sender_damage(runs)
    peer_diffs = peer_damage_differences(damage)
    interactions = adaptation_network_interactions(damage)
    receiver = receiver_differences(damage)
    time_damage = sender_damage(trajectories, by_period=True)
    time_peer = peer_damage_differences(time_damage)

    _write_csv(root / "runs.csv", runs)
    _write_csv(root / "belief_checkpoints.csv", trajectories)
    _write_csv(root / "lambda_checkpoints.csv", reliance)
    _write_gzip_csv(root / "terminal_beliefs.csv.gz", beliefs)
    _write_gzip_csv(root / "jammer_strategy_trajectory.csv.gz", strategies)
    if edges:
        _write_gzip_csv(root / "reliance_checkpoints.csv.gz", edges)
    _write_csv(root / "gate0b_sender_damage.csv", damage)
    _write_csv(root / "gate0b_peer_damage.csv", peer_diffs)
    _write_csv(root / "gate0b_adaptation_network_interaction.csv", interactions)
    _write_csv(root / "gate0b_receiver_differences.csv", receiver)
    _write_csv(root / "gate0b_checkpoint_sender_damage.csv", time_damage)
    _write_csv(root / "gate0b_checkpoint_peer_damage.csv", time_peer)

    blocks = [{
        "seed": int(p["seed"]),
        "initial_regime": p["initial_regime"],
        "network_environment": p["network_environment"],
        "block_id": p["block_id"],
        "initial_state_fingerprint": p["initial_state_fingerprint"],
        "structural_fingerprint": p["structural_fingerprint"],
    } for p in shards]
    _write_csv(root / "block_manifest.csv", blocks)

    summaries = defaultdict(list)
    for row in damage:
        summaries[
            (row["network_environment"], row["reliance_mode"],
             row["sender_regime"])
        ].append(float(row["delta_mse"]))
    def bounds(values: list[float]) -> dict:
        if not values or not all(math.isfinite(x) for x in values):
            raise ValueError("Missing/nonfinite sender damage")
        return {
            "n": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    summary = {
        "design_id": manifest["design_id"],
        "analysis_mode": "sender_grid",
        "baseline_regime": "null",
        "comparison": "sender minus structural-slot null; never truth clone",
        "n_blocks": len(shards),
        "n_runs": len(runs),
        "n_sender_damage": len(damage),
        "n_peer_differences": len(peer_diffs),
        "n_adaptation_interactions": len(interactions),
        "n_receiver_differences": len(receiver),
        "cross_network_initial_state_matched": True,
        "sender_damage_by_condition": [
            {
                "network_environment": key[0],
                "reliance_mode": key[1],
                "sender_regime": key[2],
                **bounds(vals),
            }
            for key, vals in sorted(summaries.items())
        ],
        "warning": (
            "Small-seed Gate 0B pilot is a mechanism/implementation diagnostic. "
            "Peer-minus-elite is a total addition-of-sources effect, not "
            "isolated causal peer-transmission mediation."
        ),
    }
    (root / "pilot_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_dir")
    args = parser.parse_args()
    print(json.dumps(consolidate_gate0b(args.pilot_dir), indent=2))

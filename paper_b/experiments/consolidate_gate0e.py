"""Gate 0E: source-preserving peer-channel contrasts and invariants.

The E/M/S/L decomposition is algebraically exact for matched D contrasts,
NOT a causal mediation decomposition or provenance attribution.
"""
from __future__ import annotations

from collections import defaultdict
import math

from paper_b.experiments.consolidate_pilot import _write_csv

CHANNEL_MODES = (
    "elite_only", "extended_muted", "extended_stale", "extended_live"
)
EXTENDED_MODES = CHANNEL_MODES[1:]


def channel_damage_contrasts(damage: list[dict]) -> list[dict]:
    """Paired comparisons anchored on null-relative sender damage."""
    keys = ("seed", "initial_regime", "reliance_mode", "sender_regime")
    indexed = {
        tuple(row[x] for x in keys) + (row["network_environment"],): row
        for row in damage
    }
    out = []
    for base in sorted({tuple(x[k] for k in keys) for x in damage}):
        parts = [indexed.get(base + (mode,)) for mode in CHANNEL_MODES]
        if any(x is None for x in parts):
            raise ValueError(f"Missing matched E/M/S/L cells for {base}")
        e, m, s, l = parts
        d = [float(x["delta_mse"]) for x in parts]
        total = [float(x["mse_sender"]) for x in parts]
        nulls = [float(x["mse_null"]) for x in parts]
        if not all(math.isfinite(x) for x in d + total + nulls):
            raise ValueError(f"Nonfinite channel estimand for {base}")
        slots, signal, update, overall = (
            d[1] - d[0], d[2] - d[1], d[3] - d[2], d[3] - d[0],
        )
        if not math.isclose(slots+signal+update,overall,rel_tol=0,abs_tol=1e-8):
            raise AssertionError(f"Channel contrast telescoping failed {base}")
        out.append({
            **dict(zip(keys,base)),
            "d_elite":d[0],"d_muted":d[1],
            "d_stale":d[2],"d_live":d[3],
            "slot_attention_contrast":slots,
            "static_peer_signal_contrast":signal,
            "endogenous_peer_update_contrast":update,
            "total_peer_contrast":overall,
            "mse_elite":total[0],"mse_muted":total[1],
            "mse_stale":total[2],"mse_live":total[3],
            "null_mse_elite":nulls[0],
            "null_mse_muted":nulls[1],
            "null_mse_stale":nulls[2],
            "null_mse_live":nulls[3],
        })
    return out


def validate_channel_blocks(shards: list[dict], events: list[dict],
                            *, event_period_limit: int) -> dict:
    """Exact within-extended topology and t0 pre-treatment audit equality."""
    grouped = defaultdict(dict)
    for shard in shards:
        key = (int(shard["seed"]),shard["initial_regime"])
        label = shard["network_environment"]
        if label in grouped[key]:
            raise ValueError(f"Duplicated channel {label} in {key}")
        grouped[key][label] = shard
    for key, channels in grouped.items():
        if set(channels) != set(CHANNEL_MODES):
            raise ValueError(f"Missing channel cells for {key}: {set(channels)}")
        structural = {
            channels[c]["structural_fingerprint"] for c in EXTENDED_MODES
        }
        initial = {
            channels[c]["initial_state_fingerprint"] for c in CHANNEL_MODES
        }
        if len(structural) != 1:
            raise ValueError(f"Extended peer edge map differs across channels {key}")
        if len(initial) != 1:
            raise ValueError(f"Initial beliefs differ across channel modes {key}")

    same_audit = None
    event_count = len(events)
    if event_period_limit:
        t0 = defaultdict(dict)
        for row in events:
            if int(row["period"]) != 0:
                continue
            if row["network_environment"] not in EXTENDED_MODES:
                continue
            key = (int(row["seed"]),row["initial_regime"],
                   row["reliance_mode"],row["jammer_regime"],
                   int(row["citizen_pos"]))
            chan = row["network_environment"]
            if chan in t0[key]:
                raise ValueError(f"Duplicate initial event {key}/{chan}")
            signature = (
                row["learning_target"],
                row["peer_audit_signature"],
                row["peer_initial_means"],
                row["peer_outgoing_sds"],
                int(row["source_count"]),
                int(row["peer_request_count"]),
                int(row["peer_delivered_count"]),
                float(row["theta_mu_pre"]),
                float(row["theta_mu_post"]),
                float(row["jammer_delta_mu_post"]),
            )
            t0[key][chan] = signature
        if not t0 or len(t0)*3 != sum(
            1 for row in events
            if int(row["period"])==0
            and row["network_environment"] in EXTENDED_MODES
        ):
            raise ValueError("Missing or duplicated matched t0 peer audit records")
        for key, signatures in t0.items():
            if set(signatures) != set(EXTENDED_MODES):
                raise ValueError(f"Missing t0 channel counterparts {key}")
            if len(set(signatures.values())) != 1:
                raise ValueError(f"t0 audit differs across M/S/L for {key}")
        for row in events:
            if row["network_environment"] == "extended_muted" and int(row["period"])>=1:
                if int(row["peer_delivered_count"]) != 0:
                    raise ValueError("Muted peer delivered post-audit message")
            if row["network_environment"] in EXTENDED_MODES:
                if int(row["peer_delivered_count"]) > int(row["peer_request_count"]):
                    raise ValueError("Peer delivery count exceeds requested credits")
        same_audit = True
    return {
        "n_matched_channel_seed_regime_blocks":len(grouped),
        "n_sender_events":event_count,
        "extended_edge_maps_equal":True,
        "initial_beliefs_equal":True,
        "matched_t0_audits_equal":same_audit,
    }


def summarize_peer_events(events: list[dict]) -> list[dict]:
    fields = ("seed","initial_regime","network_environment",
              "reliance_mode","jammer_regime","period")
    groups = defaultdict(list)
    for row in events:
        groups[tuple(row[x] for x in fields)].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        output.append({
            **dict(zip(fields,key)),
            "n_citizens":len(rows),
            "audit_share":sum(r["learning_target"]=="audit" for r in rows)/len(rows),
            "peer_requests":sum(int(r["peer_request_count"]) for r in rows),
            "peer_delivered":sum(int(r["peer_delivered_count"]) for r in rows),
            "jammer_requests":sum(int(r["jammer_request_count"]) for r in rows),
            "jammer_delivered":sum(int(r["jammer_delivered_count"]) for r in rows),
            "expert_requests":sum(int(r["expert_request_count"]) for r in rows),
            "expert_delivered":sum(int(r["expert_delivered_count"]) for r in rows),
            "share_receiving_peer":sum(int(r["peer_delivered_count"])>0 for r in rows)/len(rows),
            "mean_peer_precision_post":sum(float(r["peer_precision_post"]) for r in rows)/len(rows),
            "mean_jammer_rank_pre":sum(float(r["jammer_rank_pre"]) for r in rows)/len(rows),
        })
    return output


def augment_gate0e(root, manifest, shards, damage, events) -> dict:
    if not set(CHANNEL_MODES).issubset(set(manifest["network_environments"])):
        return {}
    if set(manifest["network_environments"]) != set(CHANNEL_MODES):
        raise ValueError("Gate 0E requires exactly the four declared modes")
    stats = validate_channel_blocks(
        shards, events,
        event_period_limit=int(manifest.get("sender_event_period_limit",0)),
    )
    contrasts = channel_damage_contrasts(damage)
    _write_csv(root/"gate0e_channel_contrasts.csv",contrasts)
    if events:
        _write_csv(root/"gate0e_peer_event_period_summary.csv",
                   summarize_peer_events(events))
    return {
        **stats,
        "n_gate0e_channel_contrasts":len(contrasts),
        "channel_criterion":"Paired algebraic E/M/S/L damage contrasts; no mediation claim",
    }

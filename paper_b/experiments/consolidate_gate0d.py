"""Gate 0D staged-sender mechanism contrasts and audited event summaries.

No causal claim of isolated social transmission: all contrasts compare entire
sender regimes or peer opportunity sets and preserve total/null MSE.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

STAGES = (
    "adaptive",
    "timing_only",
    "exposure_uniform",
    "exposure_ranked",
    "credibility_aware",
)

STAGE_LABELS = (
    "timing_discount",
    "uniform_exposure",
    "rank_specific_exposure",
    "credibility_penalty",
)


def staged_contrasts(damage: list[dict]) -> list[dict]:
    """Matched adjacent-stage damage contrasts, no non-matched pooling."""
    fields = ("seed", "initial_regime", "network_environment", "reliance_mode")
    idx = {
        tuple(r[k] for k in fields) + (r["sender_regime"],): r
        for r in damage
    }
    result = []
    for base in sorted({tuple(r[k] for k in fields) for r in damage}):
        for label, earlier, later in zip(
            STAGE_LABELS, STAGES[:-1], STAGES[1:]
        ):
            a = idx.get(base + (earlier,))
            b = idx.get(base + (later,))
            if a is None or b is None:
                raise ValueError(f"Missing staged sender for {base}: {earlier}->{later}")
            record = dict(zip(fields, base))
            record.update({
                "mechanism": label,
                "base_sender": earlier,
                "next_sender": later,
                "delta_damage": float(b["delta_mse"])-float(a["delta_mse"]),
                "delta_total_mse": (
                    float(b["mse_sender"])-float(a["mse_sender"])
                ),
                "base_damage": float(a["delta_mse"]),
                "next_damage": float(b["delta_mse"]),
            })
            result.append(record)
    return result


def summarize_sender_events(events: list[dict]) -> list[dict]:
    """Period/condition summary of *requested* versus *delivered* Jammer data."""
    fields = (
        "seed", "initial_regime", "network_environment",
        "reliance_mode", "jammer_regime", "period",
    )
    groups = defaultdict(list)
    for row in events:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        n = len(rows)
        audit = [x for x in rows if x["learning_target"] == "audit"]
        theta = [x for x in rows if x["learning_target"] == "theta"]
        total_requests = sum(int(x["jammer_request_count"]) for x in rows)
        total_messages = sum(int(x["jammer_delivered_count"]) for x in rows)
        audit_delivered = sum(
            int(x["jammer_delivered_count"]) for x in audit
        )
        theta_delivered = sum(
            int(x["jammer_delivered_count"]) for x in theta
        )
        audit_receivers = [
            x for x in audit if int(x["jammer_delivered_count"]) > 0
        ]
        first = dict(zip(fields, key))
        first.update({
            "n_citizens": n,
            "n_audit": len(audit),
            "n_theta": len(theta),
            "audit_share": len(audit)/n,
            "requested_messages": total_requests,
            "delivered_messages": total_messages,
            "audit_delivered_messages": audit_delivered,
            "theta_delivered_messages": theta_delivered,
            "share_with_jammer_messages": sum(
                int(x["jammer_delivered_count"] > 0) for x in rows
            ) / n,
            "mean_jammer_substantive_prob": mean(
                float(x["jammer_substantive_prob_pre"]) for x in rows
            ),
            "mean_jammer_rank_pre": mean(
                float(x["jammer_rank_pre"]) for x in rows
            ),
            "mean_abs_delta_pre_audit_receivers": (
                mean(abs(float(x["jammer_delta_mu_pre"]))
                     for x in audit_receivers)
                if audit_receivers else None
            ),
            "mean_abs_delta_post_audit_receivers": (
                mean(abs(float(x["jammer_delta_mu_post"]))
                     for x in audit_receivers)
                if audit_receivers else None
            ),
            "mean_abs_delta_change_audit_receivers": (
                mean(
                    abs(float(x["jammer_delta_mu_post"]))
                    - abs(float(x["jammer_delta_mu_pre"]))
                    for x in audit_receivers
                ) if audit_receivers else None
            ),
            "mean_theta_shift_theta_receivers": (
                mean(float(x["theta_mu_post"])-float(x["theta_mu_pre"])
                     for x in theta) if theta else None
            ),
            "mean_jammer_precision_increment_theta_receivers": (
                mean(float(x["jammer_precision_increment"])
                     for x in theta) if theta else None
            ),
        })
        if len(audit) + len(theta) != n:
            raise AssertionError(f"Invalid audit/theta classification for {key}")
        if total_messages > total_requests:
            raise AssertionError("Delivered Jammer messages exceed requests")
        if theta_delivered + audit_delivered != total_messages:
            raise AssertionError("Event message accounting mismatch")
        result.append(first)
    return result

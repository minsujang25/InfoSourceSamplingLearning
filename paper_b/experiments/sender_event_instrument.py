"""Optional, RNG-neutral per-citizen Jammer acquisition/audit event trace.

Instrumentation wraps Citizen.sample_messages *before* the model clears
sampled_msgs in commit_pending_update. It does not alter Mesa model code,
source behavior, random draws, or citizen posterior arithmetic.

Use only for Gate 0D diagnostic horizons; aggregate recording is disabled
by default for other callers and production experiments.
"""
from __future__ import annotations

import math


def attach_sender_event_capture(model, *, limit: int = 26) -> list[dict]:
    if limit < 0:
        raise ValueError("Event capture limit cannot be negative.")
    rows: list[dict] = []
    if limit == 0:
        return rows
    jammer = model.jammer
    if jammer is None:
        raise RuntimeError("Sender event capture requires a Jammer source")

    for citizen in model.citizens:
        original_sample = citizen.sample_messages

        def instrumented_sample(*, _citizen=citizen, _original=original_sample):
            _original()
            if model.period >= limit:
                return
            source = jammer
            ordered_sources = _citizen._sample_order
            if source not in ordered_sources:
                raise AssertionError("Jammer missing from structural source set")
            index = ordered_sources.index(source)
            requested = int(_citizen.num_request[index])
            delivered_messages = _citizen.sampled_msgs[index]
            delivered = len(delivered_messages)
            if delivered > requested:
                raise AssertionError("Delivered messages exceed allocation")
            pre_ranking = list(_citizen._behavioral_ranking())
            rank = pre_ranking.index(source) + 1
            row = {
                "period": int(model.period),
                "citizen_pos": int(_citizen.pos),
                "citizen_group": int(_citizen.group_id),
                "learning_target": (
                    "audit" if _citizen.theta_or_delta == 1 else "theta"
                ),
                "source_count": len(_citizen.info_source),
                "jammer_rank_pre": int(rank),
                "jammer_substantive_prob_pre": float(
                    _citizen.substantive_reliance_probabilities[source]
                ),
                "jammer_request_count": requested,
                "jammer_delivered_count": delivered,
                "jammer_message_mean": (
                    float(sum(delivered_messages) / delivered)
                    if delivered > 0 else None
                ),
                "theta_mu_pre": float(_citizen.mu_theta),
                "theta_sd_pre": float(_citizen.sd_theta),
                "jammer_delta_mu_pre": float(_citizen.mu_delta[source]),
                "jammer_delta_sd_pre": float(_citizen.sd_delta[source]),
                "jammer_precision_pre": float(
                    _citizen.cumulative_evidence_precision[source]
                ),
            }
            rows.append(row)

        citizen.sample_messages = instrumented_sample

    return rows


def complete_sender_events(model, rows: list[dict], *, limit: int = 26) -> None:
    """Join post-commit states after each period without touching simulation RNG."""
    period_completed = int(model.period) - 1
    if period_completed < 0 or period_completed >= limit:
        return
    citizens = {c.pos: c for c in model.citizens}
    count = 0
    for row in reversed(rows):
        if row["period"] != period_completed:
            break
        citizen = citizens[row["citizen_pos"]]
        jammer = model.jammer
        row.update({
            "theta_mu_post": float(citizen.mu_theta),
            "theta_sd_post": float(citizen.sd_theta),
            "jammer_delta_mu_post": float(citizen.mu_delta[jammer]),
            "jammer_delta_sd_post": float(citizen.sd_delta[jammer]),
            "jammer_precision_post": float(
                citizen.cumulative_evidence_precision[jammer]
            ),
            "jammer_rank_post": (
                citizen._behavioral_ranking().index(jammer) + 1
            ),
        })
        row["jammer_precision_increment"] = (
            row["jammer_precision_post"] - row["jammer_precision_pre"]
        )
        count += 1
    if count != len(model.citizens):
        raise AssertionError(
            f"Period {period_completed} event rows: {count} rather than "
            f"{len(model.citizens)} citizens"
        )
    # Structural audit never changes theta beliefs.
    if period_completed == 0:
        for row in rows[-count:]:
            if row["learning_target"] != "audit":
                raise AssertionError("t=0 must have all-audit behavior")
            if not math.isclose(row["theta_mu_pre"], row["theta_mu_post"],
                                rel_tol=0, abs_tol=1e-12):
                raise AssertionError("t=0 audit changed a theta belief")

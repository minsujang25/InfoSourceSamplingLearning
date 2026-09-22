"""Run-level metrics for Paper B.

These functions are intentionally read-only: they extract outcomes from the
shared model without changing simulation behavior.
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
    return [
        agent
        for agent in _agents(model)
        if getattr(agent, "type_of_agent", None) == "citizen"
    ]


def belief_vector(model) -> np.ndarray:
    return np.asarray(
        [float(agent.mu_theta_beliefs[-1]) for agent in citizens(model)],
        dtype=float,
    )


def belief_summary(model) -> dict:
    values = belief_vector(model)
    truth = float(model.state_of_the_world)
    finite = np.isfinite(values)

    if not finite.all():
        return {
            "n_citizens": int(values.size),
            "n_nonfinite": int((~finite).sum()),
            "mean_belief": math.nan,
            "belief_sd": math.nan,
            "mae_truth": math.nan,
            "rmse_truth": math.nan,
        }

    errors = values - truth
    return {
        "n_citizens": int(values.size),
        "n_nonfinite": 0,
        "mean_belief": float(values.mean()),
        "belief_sd": float(values.std()),
        "mae_truth": float(np.abs(errors).mean()),
        "rmse_truth": float(np.sqrt(np.mean(errors**2))),
    }


def structural_source_counts(model) -> dict:
    """Count final structural counterpart edges by source type.

    This is not yet an effective-influence metric. Effective influence requires
    time-indexed request/reliance logging, which will be added as a separate
    instrumentation change before the adaptive-vs-frozen experiment.
    """
    counts = Counter()
    for citizen in citizens(model):
        for source in getattr(citizen, "info_source", []):
            counts[getattr(source, "type_of_agent", "unknown")] += 1
    return dict(counts)

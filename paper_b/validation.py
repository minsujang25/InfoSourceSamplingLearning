"""Validation helpers for Paper B simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FiniteStateReport:
    ok: bool
    step: int
    nonfinite_agents: tuple[int, ...]


class NonFiniteStateError(RuntimeError):
    """Raised when a Paper B run produces NaN or infinite citizen beliefs."""


def _agents(model):
    if hasattr(model, "agents"):
        return list(model.agents)
    return list(model.schedule.agents)


def finite_state_report(model) -> FiniteStateReport:
    bad = []
    for agent in _agents(model):
        if getattr(agent, "type_of_agent", None) != "citizen":
            continue
        mu = float(agent.mu_theta_beliefs[-1])
        sd = float(agent.sd_theta_beliefs[-1])
        if not (math.isfinite(mu) and math.isfinite(sd)):
            bad.append(int(agent.pos))

    step = int(getattr(model, "steps", getattr(getattr(model, "schedule", None), "steps", 0)))
    return FiniteStateReport(
        ok=not bad,
        step=step,
        nonfinite_agents=tuple(sorted(bad)),
    )


def assert_finite_state(model, *, context: str = "") -> None:
    report = finite_state_report(model)
    if report.ok:
        return
    prefix = f"{context}: " if context else ""
    raise NonFiniteStateError(
        f"{prefix}non-finite citizen state at step {report.step}; "
        f"positions={list(report.nonfinite_agents)}"
    )

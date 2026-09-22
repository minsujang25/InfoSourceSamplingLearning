"""Configuration objects for the Paper B experiment layer.

The current scaffold does not yet modify the shared model. In particular,
'adaptive' versus 'frozen' reliance is represented as a planned treatment field
but is not injected until its exact behavioral definition is frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PaperBCondition:
    condition_id: str
    network_environment: str
    initial_belief_regime: str
    reliance_mode: str
    seed: int
    jammer_k: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


def first_milestone_conditions(
    *,
    network_environments: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[PaperBCondition]:
    """Generate the planned 3 x 2 receiver-side core design.

    The caller must supply the exact network-environment labels after the
    extended-network mapping to implementation is audited.
    """
    initial_regimes = ("flat", "consensus", "polarized")
    reliance_modes = ("adaptive", "frozen")

    conditions = []
    for network in network_environments:
        for prior in initial_regimes:
            for reliance in reliance_modes:
                for seed in seeds:
                    conditions.append(
                        PaperBCondition(
                            condition_id=f"{network}__{prior}__{reliance}__s{seed}",
                            network_environment=network,
                            initial_belief_regime=prior,
                            reliance_mode=reliance,
                            seed=seed,
                        )
                    )
    return conditions

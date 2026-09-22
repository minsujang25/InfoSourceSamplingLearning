"""Configuration objects for the theory-aligned Paper B experiment layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass


VALID_NETWORK_ENVIRONMENTS = (
    "elite_only",
    "random_2",
    "group_id",
    "extended",
)
VALID_RELIANCE_MODES = ("adaptive", "frozen")


@dataclass(frozen=True)
class PaperBCondition:
    condition_id: str
    network_environment: str
    initial_belief_regime: str
    reliance_mode: str
    seed: int
    jammer_k: int = 1
    jammer_active: bool = True
    local_degree: int = 2
    peer_degree: int = 2
    same_group_probability: float = 0.9
    surveillance_interval: int = 5

    def __post_init__(self):
        if self.network_environment not in VALID_NETWORK_ENVIRONMENTS:
            raise ValueError(
                f"network_environment must be one of {VALID_NETWORK_ENVIRONMENTS}"
            )
        if self.reliance_mode not in VALID_RELIANCE_MODES:
            raise ValueError(
                f"reliance_mode must be one of {VALID_RELIANCE_MODES}"
            )
        if self.local_degree < 0 or self.peer_degree < 0:
            raise ValueError("local_degree and peer_degree must be nonnegative.")
        if not 0.0 <= self.same_group_probability <= 1.0:
            raise ValueError("same_group_probability must lie in [0,1].")

    def as_dict(self) -> dict:
        return asdict(self)


def receiver_side_conditions(
    *,
    seeds: tuple[int, ...],
    initial_regimes: tuple[str, ...] = ("flat", "consensus", "polarized"),
    network_environments: tuple[str, ...] = VALID_NETWORK_ENVIRONMENTS,
    reliance_modes: tuple[str, ...] = VALID_RELIANCE_MODES,
    jammer_k_values: tuple[int, ...] = (1,),
    include_no_jammer_counterfactual: bool = True,
    local_degree: int = 2,
    peer_degree: int = 2,
    same_group_probability: float = 0.9,
) -> list[PaperBCondition]:
    """Generate the theory-aligned receiver-side experiment grid.

    Jammer/no-jammer conditions retain the same structural source slot.  The
    no-jammer condition neutralizes the Jammer's content rather than removing
    its node, preserving the matched opportunity network.
    """
    jammer_states = (True, False) if include_no_jammer_counterfactual else (True,)
    conditions = []

    for network in network_environments:
        for prior in initial_regimes:
            for reliance in reliance_modes:
                for k in jammer_k_values:
                    for jammer_active in jammer_states:
                        for seed in seeds:
                            j_label = "J1" if jammer_active else "J0"
                            condition_id = (
                                f"{network}__{prior}__{reliance}"
                                f"__k{k}__{j_label}__s{seed}"
                            )
                            conditions.append(
                                PaperBCondition(
                                    condition_id=condition_id,
                                    network_environment=network,
                                    initial_belief_regime=prior,
                                    reliance_mode=reliance,
                                    seed=seed,
                                    jammer_k=k,
                                    jammer_active=jammer_active,
                                    local_degree=local_degree,
                                    peer_degree=peer_degree,
                                    same_group_probability=same_group_probability,
                                )
                            )
    return conditions


# Backwards-compatible alias used by the original scaffold.
def first_milestone_conditions(
    *,
    network_environments: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[PaperBCondition]:
    return receiver_side_conditions(
        network_environments=network_environments,
        seeds=seeds,
        jammer_k_values=(1,),
        include_no_jammer_counterfactual=False,
    )

"""Theory-aligned reconstruction of the Info Source Sampling Learning model.

This module reconstructs the legacy Mesa model around the Social Networks
Paper B theory freeze (v1.0).  The reconstruction intentionally separates:

    structural opportunity A
        -> credibility ranking R_t
        -> acquisition policy Pi_t
        -> effective reliance Lambda_t
        -> realized flow X_t
        -> belief outcomes.

Scientific changes relative to the legacy implementation
---------------------------------------------------------
1. Citizen-to-citizen communication is synchronous.  Period-t messages are
   generated from sender states snapshotted at the beginning of period t.
   Posterior states become outgoing-message parameters only at t+1.
2. Citizen messages are sincere draws from the sender's current posterior:
       M_{j->i,t} ~ Normal(mu_{j,t}, sigma_{j,t}^2).
3. Multi-source theta acquisition uses recursive rank-based exploration, which
   exactly nests the two-source (1-epsilon, epsilon) rule.
4. Adaptive and first-audit-frozen reliance use the same model class.
5. Theta uncertainty stores a standard deviation consistently.  Gaussian
   updates return sqrt(posterior variance); no arbitrary clipping is used.
6. Delta/source-displacement uncertainty is updated dimensionally as a
   standard deviation.
7. Periodic Jammer re-surveillance observes current pre-update citizen beliefs,
   not initial beliefs.
8. Reliance probabilities and realized request shares are logged separately.

The old public API is retained where practical so archived scripts can still be
used for auditing.  Theory-aligned Paper B experiments should pass the explicit
network_environment and reliance_mode configuration fields.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from mesa import Agent, Model
from mesa.space import NetworkGrid


PERIODS = 10
NUM_NODES = 100
TYPES = ["citizen", "infoprovider"]
AGENT_TYPE_LIST = ["Citizen", "InfoProvider"]
CHOICE_OPTION = [0, 1]  # 0 = theta/state learning, 1 = delta/credibility audit

MIN_SD = 1e-8
MIN_VAR = MIN_SD**2
DEFAULT_SINGLE_MESSAGE_VAR = 1.0

warnings.filterwarnings("ignore")


def make_fully_connected_digraph(num_nodes: int = NUM_NODES) -> nx.DiGraph:
    """Return a fully connected directed graph without self loops."""
    graph = nx.DiGraph(name="fully_connected")
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(
        (i, j)
        for i in range(num_nodes)
        for j in range(num_nodes)
        if i != j
    )
    return graph


def make_snm_digraph(num_nodes: int = NUM_NODES, num_ips: int = 2) -> nx.DiGraph:
    """Return the legacy elite-to-citizen opportunity graph."""
    graph = nx.DiGraph(name="social_network_model")
    graph.add_nodes_from(range(num_nodes))
    citizens = range(num_ips, num_nodes)
    graph.add_edges_from((citizen, ip) for citizen in citizens for ip in range(num_ips))
    return graph


def random_state_assignment(min_val: float = -1.0, max_val: float = 1.0, rng=None) -> float:
    rng = np.random.default_rng() if rng is None else rng
    return float(rng.uniform(min_val, max_val))


def recursive_rank_probabilities(num_sources: int, epsilon: float) -> np.ndarray:
    """Return the theory-freeze recursive rank-based acquisition probabilities.

    For d sources:
      p_1 = 1-epsilon
      p_r = (1-epsilon) epsilon^(r-1), r=2,...,d-1
      p_d = epsilon^(d-1)

    For d=2 this is exactly (1-epsilon, epsilon).
    """
    if num_sources < 1:
        raise ValueError("At least one source is required.")
    if not 0.0 <= epsilon <= 0.5:
        raise ValueError("epsilon must lie in [0, 0.5] so rank weights remain non-increasing.")
    if num_sources == 1:
        return np.asarray([1.0], dtype=float)

    probs = [(1.0 - epsilon) * (epsilon**r) for r in range(num_sources - 1)]
    probs.append(epsilon ** (num_sources - 1))
    probs = np.asarray(probs, dtype=float)
    # Protect only against floating-point accumulation, not substantive values.
    probs = probs / probs.sum()
    return probs


def _observation_mean_variance(
    messages: Iterable[float],
    *,
    single_message_variance: float = DEFAULT_SINGLE_MESSAGE_VAR,
) -> tuple[float, float]:
    """Return sample mean and estimated variance of that sample mean.

    With >=2 observations, use s^2/n.  With a single observation the empirical
    variance is unidentified, so use an explicit fallback observation variance
    rather than treating one message as perfectly precise.
    """
    values = np.asarray(list(messages), dtype=float)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty message batch.")
    if not np.isfinite(values).all():
        raise ValueError("Message batch contains non-finite values.")

    mean = float(values.mean())
    if values.size == 1:
        mean_var = float(single_message_variance)
    else:
        sample_var = float(values.var(ddof=1))
        mean_var = sample_var / values.size

    return mean, max(mean_var, MIN_VAR)


class InfoSampleModel(Model):
    """Mesa model for credibility learning, networked reliance, and jamming."""

    def __init__(
        self,
        num_nodes: int = NUM_NODES,
        network: nx.DiGraph | None = None,
        mode: str = "incidental_learning_allowed",
        comparison_rule: str = "delta_comparison",
        epsilon: float = 0.1,
        credit: int = 20,
        network_type: str = "fully_connected",
        network_structure: nx.DiGraph | None = None,
        learn_method: str = "cautious",
        counterpart_pick_mechanism: str = "equal",
        num_max_citizen_neighbor: int = 0,
        max_steps: int = 1000,
        model_attribute: dict | None = None,
        state_of_the_world: float | None = None,
        rng=None,
        *,
        network_environment: str | None = None,
        reliance_mode: str = "adaptive",
        local_degree: int = 2,
        peer_degree: int = 2,
        same_group_probability: float = 0.9,
        elite_access_probability: float = 0.10,
        jammer_active: bool = True,
        surveillance_interval: int = 5,
        convergence_tolerance: float = 1e-3,
        single_message_variance: float = DEFAULT_SINGLE_MESSAGE_VAR,
    ):
        super().__init__(rng=rng)

        cfg = {} if model_attribute is None else dict(model_attribute)
        self.model_attribute = cfg or None

        def take(name, default):
            return cfg[name] if name in cfg else default

        if "state_of_the_world" in cfg:
            initial_state = cfg["state_of_the_world"]
        elif state_of_the_world is not None:
            initial_state = state_of_the_world
        else:
            initial_state = random_state_assignment(rng=self.rng)
        self.state_of_the_world = float(initial_state)
        self.num_nodes = int(take("num_nodes", num_nodes))
        self.comparison_rule = take("comparison_rule", comparison_rule)
        self.epsilon = float(take("epsilon", epsilon))
        self.credit = int(take("credit", credit))
        self.initial_theta_type = take("initial_theta_type", "random")
        self.network_type = take("network_type", network_type)
        self.learn_method = take("learn_method", learn_method)
        self.counterpart_pick_mechanism = take(
            "counterpart_pick_mechanism", counterpart_pick_mechanism
        )
        self.mode = take("mode", mode)
        self.surveillance_ability = int(take("surveil_ability", take("jammer_k", 1)))
        self.num_max_citizen_neighbor = int(
            take("num_max_citizen_neighbor", num_max_citizen_neighbor)
        )
        self.max_steps = int(take("max_steps", max_steps))
        self.reliance_mode = str(take("reliance_mode", reliance_mode)).lower()
        if self.reliance_mode not in {"adaptive", "frozen"}:
            raise ValueError("reliance_mode must be 'adaptive' or 'frozen'.")

        raw_environment = str(
            take(
                "network_environment",
                network_environment or self._legacy_environment_alias(self.mode),
            )
        ).lower()
        self.network_environment = self._normalize_environment(raw_environment)

        self.local_degree = int(take("local_degree", local_degree))
        self.peer_degree = int(take("peer_degree", peer_degree))
        if self.local_degree < 0 or self.peer_degree < 0:
            raise ValueError("local_degree and peer_degree must be nonnegative.")

        self.same_group_probability = float(
            take("same_group_probability", same_group_probability)
        )
        self.elite_access_probability = float(
            take("elite_access_probability", elite_access_probability)
        )
        self.jammer_active = bool(take("jammer_active", jammer_active))
        self.surveillance_interval = max(
            1, int(take("surveillance_interval", surveillance_interval))
        )
        self.convergence_tolerance = float(
            take("convergence_tolerance", convergence_tolerance)
        )
        self.single_message_variance = float(
            take("single_message_variance", single_message_variance)
        )

        self.period = 0
        self.p = 1.0
        self.p_pair = [0.0, 1.0]

        self.agent_mu_theta_list: list[list[float]] = []
        self.agent_num_request_list: list[list[list[int]]] = []
        self.reliance_history: list[list[dict]] = []
        self.avg_agent_mu_theta_diff = math.inf
        self.avg_agent_mu_theta_diff_rate = math.inf
        self.avg_agent_sd_theta = math.inf
        self.running = True

        placement_graph = self._placement_graph(
            supplied_network=network,
            network_structure=take("network_structure", network_structure),
        )
        self.grid = NetworkGrid(placement_graph)

        self._create_agents(cfg)
        self._build_structural_sources()
        self._initialize_citizen_source_priors()
        self.update_network()

    @staticmethod
    def _legacy_environment_alias(mode: str) -> str:
        """Map legacy mode labels to explicit compatibility environments.

        The old random/group modes always included both elite sources and then
        added 0..num_max_citizen_neighbor peers. They therefore do not
        implement the manuscript's Random 2 or Group ID environments.
        """
        mapping = {
            "baseline": "elite_only",
            "random": "legacy_extended_random",
            "group_id_matching": "legacy_extended_homophilous",
            "clustered": "legacy_extended_clustered",
            "incidental_learning_allowed": "legacy_extended_random",
        }
        return mapping.get(mode, mode)

    @staticmethod
    def _normalize_environment(environment: str) -> str:
        """Normalize public aliases to the four manuscript environment names."""
        aliases = {
            "isolated": "elite_only",
            "isolated_elite": "elite_only",
            "isolated_elite_exposure": "elite_only",
            "random_peer": "random_2",
            "random2": "random_2",
            "random_2_matching": "random_2",
            "homophilous_peer": "group_id",
            "group_id_matching": "group_id",
            "extended_network": "extended",
        }
        return aliases.get(environment, environment)

    def _placement_graph(
        self,
        *,
        supplied_network: nx.DiGraph | None,
        network_structure: nx.DiGraph | None,
    ) -> nx.DiGraph:
        if self.network_type == "manually_defined" and network_structure is not None:
            graph = network_structure.copy()
            graph.add_nodes_from(range(self.num_nodes))
            return graph
        if supplied_network is not None:
            graph = supplied_network.copy()
            graph.add_nodes_from(range(self.num_nodes))
            return graph
        if self.network_type == "social_network_model":
            return make_snm_digraph(self.num_nodes)
        return make_fully_connected_digraph(self.num_nodes)

    def _create_agents(self, cfg: dict) -> None:
        types = cfg.get("type_of_agent")
        mu_theta = cfg.get("mu_theta")
        sd_theta = cfg.get("sd_theta")
        mu_delta = cfg.get("mu_delta")
        sd_delta = cfg.get("sd_delta")

        if types is None:
            types = [InfoProvider, DisruptiveJammer] + [Citizen] * (self.num_nodes - 2)
        if len(types) != self.num_nodes:
            raise ValueError("type_of_agent length must equal num_nodes.")

        if mu_theta is None:
            mu_theta = [
                self.state_of_the_world,
                self.state_of_the_world + 4.0,
            ] + list(self.rng.uniform(-1.0, 1.0, self.num_nodes - 2))
        if sd_theta is None:
            sd_theta = [1.0, 1.0] + [5.0] * (self.num_nodes - 2)
        if mu_delta is None:
            mu_delta = [[0.0] for _ in range(self.num_nodes)]
        if sd_delta is None:
            sd_delta = [[5.0] for _ in range(self.num_nodes)]

        if not (len(mu_theta) == len(sd_theta) == self.num_nodes):
            raise ValueError("mu_theta and sd_theta lengths must equal num_nodes.")

        seq_meaningful = bool(cfg.get("seq_meaningful", True))
        positions = sorted(range(self.num_nodes)) if seq_meaningful else list(range(self.num_nodes))

        for pos in positions:
            agent_cls = types[pos]
            agent = agent_cls(
                pos,
                pos,
                self,
                mu_delta=mu_delta[pos] if pos < len(mu_delta) else [0.0],
                sd_delta=sd_delta[pos] if pos < len(sd_delta) else [5.0],
                mu_theta=float(mu_theta[pos]),
                sd_theta=float(sd_theta[pos]),
            )
            self.grid.place_agent(agent, pos)

    def _agents_of_type(self, agent_type: str) -> list[Agent]:
        return [
            a for a in self.agents
            if getattr(a, "type_of_agent", None) == agent_type
        ]

    @property
    def citizens(self) -> list["Citizen"]:
        return sorted(self._agents_of_type("citizen"), key=lambda a: a.pos)

    @property
    def elite_sources(self) -> list["InfoAgents"]:
        return sorted(
            [
                a for a in self.agents
                if getattr(a, "type_of_agent", None)
                in {"infoprovider", "disruptivejammer"}
            ],
            key=lambda a: a.pos,
        )

    @property
    def jammer(self) -> "DisruptiveJammer | None":
        jammers = self._agents_of_type("disruptivejammer")
        return jammers[0] if jammers else None

    def _sample_without_replacement(self, pool: list, n: int) -> list:
        n = min(max(int(n), 0), len(pool))
        if n == 0:
            return []
        idx = self.rng.choice(len(pool), size=n, replace=False)
        return [pool[int(i)] for i in np.atleast_1d(idx)]

    def _random_peer_sources(self, citizen: "Citizen", degree: int) -> list["Citizen"]:
        peers = [c for c in self.citizens if c is not citizen]
        return self._sample_without_replacement(peers, degree)

    def _random_local_sources(self, citizen: "Citizen", degree: int) -> list["InfoAgents"]:
        """Random 2: sample exactly the requested number of local sources.

        The opportunity pool contains citizens, the Expert, and the Jammer.
        Direct elite access is therefore localized rather than guaranteed.
        """
        pool = [agent for agent in self.agents if agent is not citizen]
        return self._sample_without_replacement(pool, degree)

    def _group_id_local_sources(self, citizen: "Citizen", degree: int) -> list["InfoAgents"]:
        """Group ID: exact local degree with predominantly same-group sources.

        Group membership is fixed from the initial/source location sign. The
        Expert at mu=0 belongs to the non-positive group, while the Jammer's
        positive underlying position places it in the positive group. Elite
        nodes are therefore embedded in the same group-based opportunity pool
        as citizens rather than attached universally.
        """
        pool = [agent for agent in self.agents if agent is not citizen]
        same = [agent for agent in pool if agent.group_id == citizen.group_id]
        other = [agent for agent in pool if agent.group_id != citizen.group_id]

        selected: list[InfoAgents] = []
        for _ in range(min(degree, len(pool))):
            want_same = bool(self.rng.random() < self.same_group_probability)
            preferred = [a for a in (same if want_same else other) if a not in selected]
            fallback = [a for a in (other if want_same else same) if a not in selected]
            candidates = preferred or fallback
            if not candidates:
                break
            chosen = candidates[int(self.rng.integers(0, len(candidates)))]
            selected.append(chosen)
        return selected

    def _homophilous_peer_sources(self, citizen: "Citizen", degree: int) -> list["Citizen"]:
        peers = [c for c in self.citizens if c is not citizen]
        same = [c for c in peers if c.group_id == citizen.group_id]
        other = [c for c in peers if c.group_id != citizen.group_id]

        selected: list[Citizen] = []
        while len(selected) < min(degree, len(peers)):
            want_same = bool(self.rng.random() < self.same_group_probability)
            preferred = same if want_same else other
            fallback = other if want_same else same
            preferred = [c for c in preferred if c not in selected]
            fallback = [c for c in fallback if c not in selected]
            pool = preferred or fallback
            if not pool:
                break
            selected.append(pool[int(self.rng.integers(0, len(pool)))])
        return selected

    def _clustered_peer_sources(self, citizen: "Citizen", degree: int) -> list["Citizen"]:
        peers = [c for c in self.citizens if c is not citizen]
        if not peers or degree <= 0:
            return []
        ordering = list(self.rng.permutation(len(peers)))
        selected = []
        for idx in ordering:
            candidate = peers[int(idx)]
            distance = abs(citizen.mu_theta - candidate.mu_theta)
            power = 0.5 if candidate.group_id == citizen.group_id else 1.0
            p_connect = math.exp(-distance * power)
            if self.rng.random() < p_connect:
                selected.append(candidate)
                if len(selected) >= degree:
                    break
        return selected

    def _localized_elites(self) -> list["InfoAgents"]:
        selected = []
        for source in self.elite_sources:
            if self.rng.random() < self.elite_access_probability:
                selected.append(source)
        return selected

    def _build_structural_sources(self) -> None:
        """Construct fixed source sets for the theory-aligned environments."""
        env = self.network_environment

        for citizen in self.citizens:
            if env == "elite_only":
                sources = list(self.elite_sources)

            elif env == "random_2":
                sources = self._random_local_sources(
                    citizen,
                    self.local_degree,
                )

            elif env == "group_id":
                sources = self._group_id_local_sources(
                    citizen,
                    self.local_degree,
                )

            elif env == "extended":
                sources = (
                    list(self.elite_sources)
                    + self._random_peer_sources(citizen, self.peer_degree)
                )

            elif env == "legacy_extended_random":
                sources = (
                    list(self.elite_sources)
                    + self._random_peer_sources(citizen, self.peer_degree)
                )

            elif env == "legacy_extended_homophilous":
                sources = (
                    list(self.elite_sources)
                    + self._homophilous_peer_sources(citizen, self.peer_degree)
                )

            elif env == "legacy_extended_clustered":
                sources = (
                    list(self.elite_sources)
                    + self._clustered_peer_sources(citizen, self.peer_degree)
                )

            else:
                raise ValueError(
                    f"Unknown network_environment={env!r}. "
                    "Use elite_only, random_2, group_id, extended, "
                    "or an explicit legacy_extended_* compatibility environment."
                )

            # Stable order is important for matched-seed reproducibility.
            unique = {source.pos: source for source in sources if source is not citizen}
            citizen.info_source = [unique[pos] for pos in sorted(unique)]
            if not citizen.info_source:
                raise ValueError(
                    f"Citizen {citizen.pos} has no information sources under {env}."
                )

    def _initialize_citizen_source_priors(self) -> None:
        for citizen in self.citizens:
            citizen.initialize_source_priors(citizen.info_source)

    def update_network(self) -> None:
        """Store the fixed structural opportunity network A."""
        graph = nx.DiGraph(name="structural_opportunity")
        for agent in self.agents:
            graph.add_node(agent.pos, agent_type=agent.type_of_agent)
        for citizen in self.citizens:
            for source in citizen.info_source:
                graph.add_edge(citizen.pos, source.pos)
        self.network = graph

    def get_agent_mu_theta(self) -> list[float]:
        return [float(c.mu_theta_beliefs[-1]) for c in self.citizens]

    def get_agent_sd_theta(self) -> list[float]:
        return [float(c.sd_theta_beliefs[-1]) for c in self.citizens]

    def get_agent_avg_sd_theta(self) -> float:
        values = self.get_agent_sd_theta()
        return float(np.mean(values)) if values else math.nan

    def get_agent_num_request(self) -> list[list[int]]:
        return [list(c.num_request) for c in self.citizens]

    def _theta_update_change(self) -> float:
        changes = []
        for citizen in self.citizens:
            if citizen.theta_or_delta != 0 or len(citizen.mu_theta_beliefs) < 2:
                continue
            old = float(citizen.mu_theta_beliefs[-2])
            new = float(citizen.mu_theta_beliefs[-1])
            denom = max(abs(old), 1e-8)
            changes.append(abs(new - old) / denom)
        return float(np.mean(changes)) if changes else math.inf

    def _snapshot_message_states(self) -> None:
        for agent in self.agents:
            agent.snapshot_message_state()

    def _prepare_adversary(self) -> None:
        if self.jammer is not None:
            self.jammer.prepare_for_period()

    def _record_reliance(self) -> None:
        records = []
        for citizen in self.citizens:
            for source in citizen.info_source:
                records.append(
                    {
                        "period": int(self.period),
                        "ego": int(citizen.pos),
                        "source": int(source.pos),
                        "source_type": source.type_of_agent,
                        "source_group": getattr(source, "group_id", None),
                        "ego_group": citizen.group_id,
                        "structural_edge": 1,
                        "expected_reliance": float(
                            citizen.reliance_probabilities.get(source, 0.0)
                        ),
                        "sampling_probability": float(
                            citizen.sampling_probabilities.get(source, 0.0)
                        ),
                        "learning_target": (
                            "credibility" if citizen.theta_or_delta == 1 else "theta"
                        ),
                        "realized_reliance": float(
                            citizen.realized_reliance.get(source, 0.0)
                        ),
                    }
                )
        self.reliance_history.append(records)

    def step(self) -> None:
        """Execute one synchronous theory-aligned simulation period."""
        self.p = 1.0 / (self.period + 1.0)
        self.p_pair = [1.0 - self.p, self.p]

        # Freeze all sender states for period t before any citizen updates.
        self._snapshot_message_states()
        self._prepare_adversary()

        # Stage 0-1: choose learning target, then sample/read from t states.
        for citizen in self.citizens:
            citizen.stage_period()

        # Stage 2: compute posteriors/rankings without exposing them as messages.
        for citizen in self.citizens:
            citizen.compute_pending_update()

        # Stage 3: commit all t+1 citizen states together.
        for citizen in self.citizens:
            citizen.commit_pending_update()

        self._record_reliance()

        self.agent_mu_theta_list.append(self.get_agent_mu_theta())
        self.agent_num_request_list.append(self.get_agent_num_request())
        self.avg_agent_mu_theta_diff_rate = self._theta_update_change()
        self.avg_agent_sd_theta = self.get_agent_avg_sd_theta()

        self.period += 1

    def run_model(self, print_time: bool = True) -> None:
        while self.running:
            if print_time:
                print(f"Running step :{self.period}")
            self.step()

            reached_limit = self.steps >= self.max_steps
            converged = (
                self.steps >= 3
                and math.isfinite(self.avg_agent_mu_theta_diff_rate)
                and self.avg_agent_mu_theta_diff_rate < self.convergence_tolerance
            )
            if reached_limit or converged:
                self.running = False


class InfoAgents(Agent):
    """Base information-source agent."""

    def __init__(
        self,
        legacy_id,
        pos,
        model,
        mu_delta,
        sd_delta,
        mu_theta,
        sd_theta,
        type_of_agent="genericagent",
    ):
        super().__init__(model)
        self.legacy_id = legacy_id
        self.pos = pos
        self.mu_theta = float(mu_theta)
        self.sd_theta = max(float(sd_theta), MIN_SD)
        self._initial_mu_delta_input = list(mu_delta) if isinstance(mu_delta, (list, tuple, np.ndarray)) else [float(mu_delta)]
        self._initial_sd_delta_input = list(sd_delta) if isinstance(sd_delta, (list, tuple, np.ndarray)) else [float(sd_delta)]
        self.type_of_agent = type_of_agent
        self.epsilon = model.epsilon
        self.group_id = -1 if self.mu_theta <= 0 else 1

        self._message_mu = self.mu_theta
        self._message_sd = self.sd_theta

    def snapshot_message_state(self) -> None:
        self._message_mu = float(self.mu_theta)
        self._message_sd = max(float(self.sd_theta), MIN_SD)

    def mu_out(self, n_req: int, requester) -> list[float]:
        if n_req <= 0:
            return []
        return list(
            self.model.rng.normal(
                self._message_mu,
                self._message_sd,
                int(n_req),
            )
        )


class InfoProvider(InfoAgents):
    """Non-updating elite information provider."""

    def __init__(
        self,
        legacy_id,
        pos,
        model,
        mu_delta,
        sd_delta,
        mu_theta,
        sd_theta,
        type_of_agent="infoprovider",
    ):
        super().__init__(
            legacy_id,
            pos,
            model,
            mu_delta,
            sd_delta,
            mu_theta,
            sd_theta,
            type_of_agent,
        )
        # Source displacement follows the baseline-paper convention:
        # source location minus the truth.
        self.delta = self.mu_theta - model.state_of_the_world


class DisruptiveJammer(InfoAgents):
    """Audience-adaptive disruptive information provider."""

    def __init__(
        self,
        legacy_id,
        pos,
        model,
        mu_delta,
        sd_delta,
        mu_theta,
        sd_theta,
        type_of_agent="disruptivejammer",
        surveillance_ability=5,
    ):
        super().__init__(
            legacy_id,
            pos,
            model,
            mu_delta,
            sd_delta,
            mu_theta,
            sd_theta,
            type_of_agent,
        )
        self.surveillance_ability = int(model.surveillance_ability)
        self.citizen_intel: dict = {}
        self.citizen_per_cl: dict[int, list[Citizen]] = {}
        self.msg_param_at_t_per_cluster: dict[int, dict] = {}
        self.expected_cluster_theta_beliefs: dict[int, dict] = {}
        self._current_msg_param: dict[int, dict] = {}

    def surveil_citizen(self) -> None:
        """Cluster current pre-update citizen beliefs at surveillance time."""
        citizens = self.model.citizens
        if not citizens:
            self.citizen_intel = {"centroids": np.asarray([]), "membership": {}}
            self.citizen_per_cl = {}
            return

        beliefs = np.asarray([c._message_mu for c in citizens], dtype=float).reshape(-1, 1)
        ids = [c.unique_id for c in citizens]
        k = max(1, min(self.surveillance_ability, len(citizens)))
        # Use a deterministic clustering seed that does not consume the
        # citizen/model RNG stream.  This preserves matched stochastic streams
        # across J=1/J=0 and K counterfactuals as far as the behavioral paths
        # themselves permit.
        random_state = int(17 + 1009 * k + 7919 * self.model.period)
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(beliefs)
        centroids = kmeans.cluster_centers_.reshape(-1)

        membership = {cid: int(label) for cid, label in zip(ids, labels)}
        clusters = {
            int(label): [
                c for c in citizens if membership[c.unique_id] == int(label)
            ]
            for label in np.unique(labels)
        }

        self.citizen_intel = {
            "centroids": centroids,
            "membership": membership,
        }
        self.citizen_per_cl = clusters

    @staticmethod
    def calculate_v(prior_std: float, alpha: float) -> float:
        prior_var = max(float(prior_std) ** 2, MIN_VAR)
        denom = prior_var + 2.0 * alpha - 1.0
        if abs(denom) < MIN_SD:
            denom = MIN_SD if denom >= 0 else -MIN_SD
        return prior_var / denom

    def expected_cluster_mean(
        self,
        prior_mu: float,
        prior_std: float,
        alpha: float,
        message_mean: float,
    ) -> float:
        v = self.calculate_v(prior_std, alpha)
        return (1.0 - v) * prior_mu + v * alpha * message_mean

    @staticmethod
    def expected_cluster_std(prior_std: float) -> float:
        """Legacy jammer belief forecast retained, returned as an SD."""
        prior_var = max(float(prior_std) ** 2, MIN_VAR)
        posterior_var = prior_var / (prior_var + 1.0)
        return math.sqrt(max(posterior_var, MIN_VAR))

    def _initialize_from_current_surveillance(self) -> None:
        params = {}
        expected = {}
        centroids = self.citizen_intel["centroids"]

        for cluster, citizens in self.citizen_per_cl.items():
            current = np.asarray([c._message_mu for c in citizens], dtype=float)
            centroid = float(centroids[cluster])
            spread = float(current.std(ddof=0)) if current.size > 1 else 1.0
            spread = max(spread, MIN_SD)
            params[cluster] = {"avg": centroid, "std": 1.0}
            expected[cluster] = {"avg": centroid, "std": spread}

        self._current_msg_param = params
        self.expected_cluster_theta_beliefs = expected

    def _tune_current_message_param(self) -> None:
        if not self.expected_cluster_theta_beliefs:
            self._initialize_from_current_surveillance()
            return

        alpha = 0.95
        theta = self.model.state_of_the_world
        tuned = {}
        next_expected = {}

        for cluster, prior in self.expected_cluster_theta_beliefs.items():
            prior_mu = float(prior["avg"])
            prior_std = max(float(prior["std"]), MIN_SD)
            v = self.calculate_v(prior_std, alpha)
            denom = v * alpha - 1.0
            if abs(denom) < 1e-8:
                msg_mean = prior_mu
            else:
                msg_mean = (
                    v * alpha * theta
                    - prior_mu
                    - v * alpha * (1.0 - v) * prior_mu
                ) / denom

            tuned[cluster] = {"avg": float(msg_mean), "std": 1.0}
            next_expected[cluster] = {
                "avg": float(
                    self.expected_cluster_mean(
                        prior_mu, prior_std, alpha, msg_mean
                    )
                ),
                "std": float(self.expected_cluster_std(prior_std)),
            }

        self._current_msg_param = tuned
        self.expected_cluster_theta_beliefs = next_expected

    def prepare_for_period(self) -> None:
        """Prepare one common period-t message policy before citizens sample."""
        if not self.model.jammer_active:
            # Matched J=0 counterfactual keeps the node and structural source slot
            # but neutralizes adversarial content.
            self._current_msg_param = {}
            return

        refresh = (
            not self.citizen_intel
            or self.model.period % self.model.surveillance_interval == 0
        )
        if refresh:
            self.surveil_citizen()
            self._initialize_from_current_surveillance()
        else:
            self._tune_current_message_param()

    def mu_out(self, n_req: int, requester) -> list[float]:
        if n_req <= 0:
            return []

        if not self.model.jammer_active:
            return list(
                self.model.rng.normal(
                    self.model.state_of_the_world,
                    1.0,
                    int(n_req),
                )
            )

        if not self.citizen_intel:
            self.prepare_for_period()

        cluster = self.citizen_intel["membership"].get(requester.unique_id)
        if cluster is None:
            # A structural source should always have a current cluster mapping;
            # this fallback keeps failure explicit but finite.
            mean = self.model.state_of_the_world
            sd = 1.0
        else:
            param = self._current_msg_param[cluster]
            mean = float(param["avg"])
            sd = float(param.get("std", 1.0))

        return list(self.model.rng.normal(mean, sd, int(n_req)))


class Citizen(InfoAgents):
    """Citizen who learns state and source credibility."""

    def __init__(
        self,
        legacy_id,
        pos,
        model,
        mu_delta,
        sd_delta,
        mu_theta,
        sd_theta,
        type_of_agent="citizen",
    ):
        super().__init__(
            legacy_id,
            pos,
            model,
            mu_delta,
            sd_delta,
            mu_theta,
            sd_theta,
            type_of_agent,
        )

        self.mu_theta_beliefs = [float(mu_theta)]
        self.sd_theta_beliefs = [max(float(sd_theta), MIN_SD)]
        self.mu_delta_beliefs: list[list[float]] = []
        self.sd_delta_beliefs: list[list[float]] = []

        self.mu_delta: dict[InfoAgents, float] = {}
        self.sd_delta: dict[InfoAgents, float] = {}

        self.info_source: list[InfoAgents] = []
        self.credibility_ranked_sources: list[InfoAgents] = []
        self.frozen_credibility_ranked_sources: list[InfoAgents] | None = None

        self.theta_or_delta_history = [math.nan]
        self.optimal_arm_id_history = [math.nan]
        self.num_request_history = [math.nan]
        self.theta_or_delta = math.nan
        self.num_request: list[int] = []

        self.sampled_msgs: list[list[float]] = []
        # substantive_reliance_probabilities is the theory object Lambda_t:
        # the rank-based policy that would govern state-learning acquisition.
        # sampling_probabilities is the policy actually used in the current
        # round; credibility audits intentionally sample sources evenly.
        self.substantive_reliance_probabilities: dict[InfoAgents, float] = {}
        self.sampling_probabilities: dict[InfoAgents, float] = {}
        self.reliance_probabilities: dict[InfoAgents, float] = {}
        self.realized_reliance: dict[InfoAgents, float] = {}

        self._pending_mu_theta = self.mu_theta
        self._pending_sd_theta = self.sd_theta
        self._pending_mu_delta: dict[InfoAgents, float] | None = None
        self._pending_sd_delta: dict[InfoAgents, float] | None = None
        self._pending_ranking: list[InfoAgents] | None = None

    def initialize_source_priors(self, sources: list[InfoAgents]) -> None:
        self.info_source = list(sources)
        n = len(sources)

        def expand(values, default):
            values = [float(v) for v in values] if values else [float(default)]
            if len(values) >= n:
                return values[:n]
            return values + [values[0]] * (n - len(values))

        mu_values = expand(self._initial_mu_delta_input, 0.0)
        sd_values = [max(v, MIN_SD) for v in expand(self._initial_sd_delta_input, 5.0)]

        self.mu_delta = {s: mu_values[i] for i, s in enumerate(sources)}
        self.sd_delta = {s: sd_values[i] for i, s in enumerate(sources)}
        self.mu_delta_beliefs = [mu_values.copy()]
        self.sd_delta_beliefs = [sd_values.copy()]

        # Prior ties have no substantive ranking.  Stable seeded shuffling avoids
        # node-order privilege until the first audit produces evidence.
        order = list(sources)
        self.model.rng.shuffle(order)
        self.credibility_ranked_sources = order

    def learn_theta_or_delta(self) -> int:
        return int(self.model.rng.choice(CHOICE_OPTION, p=self.model.p_pair))

    def _behavioral_ranking(self) -> list[InfoAgents]:
        if (
            self.model.reliance_mode == "frozen"
            and self.frozen_credibility_ranked_sources is not None
        ):
            return list(self.frozen_credibility_ranked_sources)
        return list(self.credibility_ranked_sources)

    def _equal_audit_requests(self, sources: list[InfoAgents]) -> tuple[list[int], np.ndarray]:
        n = len(sources)
        if n == 0:
            raise ValueError("Citizen has no sources.")
        base, remainder = divmod(self.model.credit, n)
        counts = np.full(n, base, dtype=int)
        if remainder:
            idx = self.model.rng.choice(n, size=remainder, replace=False)
            counts[np.asarray(idx, dtype=int)] += 1
        probs = counts / max(int(counts.sum()), 1)
        return counts.tolist(), probs.astype(float)

    def sample_messages(self) -> None:
        """Sample period-t messages while keeping reliance and audit policy distinct.

        Lambda_t is the rank-based substantive acquisition policy implied by
        the pre-update credibility ranking.  Credibility-audit rounds use an
        equal diagnostic sampling policy, but they do not redefine Lambda_t.
        X_t records the messages actually sampled in the current round.
        """
        behavioral_order = self._behavioral_ranking()
        substantive_policy = recursive_rank_probabilities(
            len(behavioral_order),
            self.model.epsilon,
        )
        self.substantive_reliance_probabilities = {
            source: float(substantive_policy[i])
            for i, source in enumerate(behavioral_order)
        }
        # Backwards-facing alias used by the Paper B metrics.
        self.reliance_probabilities = dict(
            self.substantive_reliance_probabilities
        )

        if self.theta_or_delta == 1:
            ordered_sources = list(self.info_source)
            counts, sampling_policy = self._equal_audit_requests(ordered_sources)
        else:
            ordered_sources = behavioral_order
            sampling_policy = substantive_policy
            choices = self.model.rng.choice(
                len(ordered_sources),
                p=sampling_policy,
                size=self.model.credit,
            )
            counts = [
                int(np.count_nonzero(choices == idx))
                for idx in range(len(ordered_sources))
            ]

        messages = [
            ordered_sources[i].mu_out(counts[i], self)
            for i in range(len(ordered_sources))
        ]

        self._sample_order = ordered_sources
        self.num_request = counts
        self.sampled_msgs = messages
        self.sampling_probabilities = {
            source: float(sampling_policy[i])
            for i, source in enumerate(ordered_sources)
        }

        total = max(sum(counts), 1)
        self.realized_reliance = {
            source: counts[i] / total
            for i, source in enumerate(ordered_sources)
        }

    def bayesian_update_theta(self, msgs: Iterable[float]) -> tuple[float, float]:
        prior_mu = float(self.mu_theta_beliefs[-1])
        prior_var = max(float(self.sd_theta_beliefs[-1]) ** 2, MIN_VAR)
        obs_mu, obs_mean_var = _observation_mean_variance(
            msgs,
            single_message_variance=self.model.single_message_variance,
        )

        post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_mean_var)
        post_mu = post_var * (
            prior_mu / prior_var + obs_mu / obs_mean_var
        )
        return float(post_mu), math.sqrt(max(float(post_var), MIN_VAR))

    # Compatibility wrappers used by archived diagnostics.
    def bayesian_update_mu_theta(self, msgs) -> float:
        return self.bayesian_update_theta(msgs)[0]

    def bayesian_update_sd_theta(self, msgs) -> float:
        return self.bayesian_update_theta(msgs)[1]

    def learn_delta(self) -> tuple[dict, dict]:
        """Bayesian source-displacement update with dimensionally consistent SDs.

        Message model for source s:
            mean(message_s) ~= theta + delta_s + noise.

        Marginalizing over citizen uncertainty in theta gives a belief-relative
        observation of source displacement,
        d_s = mean(message_s) - mu_theta, with variance equal to the
        citizen's current theta variance plus the sampling variance of the
        source-message mean.
        """
        prior_mu_theta = float(self.mu_theta_beliefs[-1])
        prior_var_theta = max(float(self.sd_theta_beliefs[-1]) ** 2, MIN_VAR)

        new_mu = dict(self.mu_delta)
        new_sd = dict(self.sd_delta)

        by_source = {
            source: self.sampled_msgs[self._sample_order.index(source)]
            if source in self._sample_order
            else []
            for source in self.info_source
        }

        for source in self.info_source:
            msgs = by_source[source]
            if not msgs:
                continue

            msg_mean, msg_mean_var = _observation_mean_variance(
                msgs,
                single_message_variance=self.model.single_message_variance,
            )
            observed_delta = msg_mean - prior_mu_theta
            observation_var = prior_var_theta + msg_mean_var

            prior_mu_delta = float(self.mu_delta[source])
            prior_var_delta = max(float(self.sd_delta[source]) ** 2, MIN_VAR)

            post_var = 1.0 / (
                1.0 / prior_var_delta + 1.0 / observation_var
            )
            post_mu = post_var * (
                prior_mu_delta / prior_var_delta
                + observed_delta / observation_var
            )

            new_mu[source] = float(post_mu)
            new_sd[source] = math.sqrt(max(float(post_var), MIN_VAR))

        return new_mu, new_sd

    def calculate_z_stat(self) -> dict:
        """Return a standardized discrepancy score for each sampled source."""
        prior_mu = float(self.mu_theta_beliefs[-1])
        prior_var = max(float(self.sd_theta_beliefs[-1]) ** 2, MIN_VAR)
        z = {}

        for source in self.info_source:
            if source not in self._sample_order:
                z[source] = math.inf
                continue
            msgs = self.sampled_msgs[self._sample_order.index(source)]
            if not msgs:
                z[source] = math.inf
                continue
            msg_mean, mean_var = _observation_mean_variance(
                msgs,
                single_message_variance=self.model.single_message_variance,
            )
            z[source] = abs(msg_mean - prior_mu) / math.sqrt(
                max(prior_var + mean_var, MIN_VAR)
            )
        return z

    def _rank_from_scores(self, scores: dict[InfoAgents, float]) -> list[InfoAgents]:
        items = list(scores.items())
        self.model.rng.shuffle(items)
        items.sort(key=lambda item: item[1])
        return [source for source, _ in items]

    def decide_optimal_arm(self) -> tuple[dict | None, dict | None, list[InfoAgents]]:
        if self.model.comparison_rule == "delta_comparison":
            new_mu, new_sd = self.learn_delta()
            scores = {source: abs(new_mu[source]) for source in self.info_source}
            ranking = self._rank_from_scores(scores)
            return new_mu, new_sd, ranking

        scores = self.calculate_z_stat()
        ranking = self._rank_from_scores(scores)
        return None, None, ranking

    def stage_period(self) -> None:
        self.theta_or_delta = self.learn_theta_or_delta()
        self.sample_messages()

    def compute_pending_update(self) -> None:
        self._pending_mu_theta = float(self.mu_theta_beliefs[-1])
        self._pending_sd_theta = float(self.sd_theta_beliefs[-1])
        self._pending_mu_delta = None
        self._pending_sd_delta = None
        self._pending_ranking = None

        if self.theta_or_delta == 1:
            new_mu_delta, new_sd_delta, ranking = self.decide_optimal_arm()
            self._pending_mu_delta = new_mu_delta
            self._pending_sd_delta = new_sd_delta
            self._pending_ranking = ranking
        else:
            flat_messages = [
                message
                for source_messages in self.sampled_msgs
                for message in source_messages
            ]
            if flat_messages:
                (
                    self._pending_mu_theta,
                    self._pending_sd_theta,
                ) = self.bayesian_update_theta(flat_messages)

    def commit_pending_update(self) -> None:
        self.mu_theta = float(self._pending_mu_theta)
        self.sd_theta = max(float(self._pending_sd_theta), MIN_SD)
        self.mu_theta_beliefs.append(self.mu_theta)
        self.sd_theta_beliefs.append(self.sd_theta)

        if self._pending_mu_delta is not None:
            self.mu_delta = dict(self._pending_mu_delta)
        if self._pending_sd_delta is not None:
            self.sd_delta = dict(self._pending_sd_delta)

        self.mu_delta_beliefs.append(
            [float(self.mu_delta[s]) for s in self.info_source]
        )
        self.sd_delta_beliefs.append(
            [float(self.sd_delta[s]) for s in self.info_source]
        )

        if self._pending_ranking is not None:
            self.credibility_ranked_sources = list(self._pending_ranking)
            if (
                self.model.reliance_mode == "frozen"
                and self.frozen_credibility_ranked_sources is None
            ):
                self.frozen_credibility_ranked_sources = list(
                    self._pending_ranking
                )

        self.theta_or_delta_history.append(self.theta_or_delta)
        self.optimal_arm_id_history.append(
            [int(s.pos) for s in self._behavioral_ranking()]
        )
        self.num_request_history.append(list(self.num_request))

        self.sampled_msgs = []
        self._sample_order = []


if __name__ == "__main__":
    model = InfoSampleModel(rng=12345)
    model.run_model()

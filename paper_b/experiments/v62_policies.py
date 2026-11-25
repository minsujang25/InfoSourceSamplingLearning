"""V6.2 process-local policy interventions. Archived files remain unchanged."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import numpy as np
import model.InfoSourceSamplingLearning as m
from paper_b.experiments import run_matched_pilot as up, run_v6_pilot as base
from paper_b.experiments import run_v6_behavioral as behavioral
from paper_b.experiments.v61_assumption_interventions import intervention

VARIANTS = ("baseline", "neutral", "RA", "RB", "uniform_tail", "constant_audit", "epsilon35")
SEEDS = tuple(range(31001,31101))
PARITY_SEEDS = (31001,31025,31050,31075,31100)
ORIGINAL_CONFIG = base.config_for
ORIGINAL_SAMPLE = m.Citizen.sample_messages

def uniform_probabilities(n, epsilon):
    if n < 1:
        raise ValueError("empty source set")
    return np.full(n, 1.0/n, dtype=float)

def canonical_sources(sources):
    return sorted(sources, key=lambda s: int(s.pos))

def neutral_sample(self):
    # Native audit sampling uses info_source, not behavioral_order. During
    # substantive sampling use ID order, not latent rank order; restore before
    # any update or diagnostic. No change to source generation or audit draws.
    with patch.object(self, "_behavioral_ranking",
                      lambda: canonical_sources(self.info_source)):
        ORIGINAL_SAMPLE(self)

def intermediate_config(*args, **kwargs):
    cfg, env, regime, gateways = ORIGINAL_CONFIG(*args, **kwargs)
    cfg["epsilon"] = .35
    return cfg, env, regime, gateways

def cells():
    grid=[]
    for variant in VARIANTS:
        suites = ("H",) if variant in ("uniform_tail","constant_audit","epsilon35") else ("H","G")
        modes = ("adaptive","frozen") if variant in ("baseline","RA","RB") else ("adaptive",)
        for suite in suites:
            for level in ("low","high"):
                for mode in modes:
                    grid.append((variant,suite,level,mode))
    return tuple(grid)

def native_cell(suite, level, mode):
    return (f"H_{level}_S_high" if suite=="H" else f"G_{level}",
            level, "high" if suite=="H" else "flat", "fixed_biased", mode)

@contextmanager
def scope(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    with ExitStack() as stack:
        if variant in ("RA","RB"):
            stack.enter_context(intervention(variant))
        elif variant in ("uniform_tail","constant_audit"):
            stack.enter_context(behavioral.variant_scope(variant))
        elif variant == "epsilon35":
            stack.enter_context(patch.object(base,"config_for",intermediate_config))
        elif variant == "neutral":
            stack.enter_context(patch.object(m,"recursive_rank_probabilities",uniform_probabilities))
            stack.enter_context(patch.object(up,"recursive_rank_probabilities",uniform_probabilities))
            stack.enter_context(patch.object(m.Citizen,"sample_messages",neutral_sample))
        yield

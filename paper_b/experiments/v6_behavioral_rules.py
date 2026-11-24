"""Isolated, one-factor receiver behavior sensitivities for H/G only."""
import math
import numpy as np
import model.InfoSourceSamplingLearning as m

REFERENCE_HORIZON = 400
AUDIT_P = math.fsum(1.0/(t+1) for t in range(1,REFERENCE_HORIZON))/(REFERENCE_HORIZON-1)


def uniform_tail_rank_probabilities(num_sources, epsilon):
    """Preserve top mass 1-epsilon; split epsilon among the remaining slots."""
    if num_sources < 1 or not 0.0 <= epsilon <= 0.5:
        raise ValueError("Invalid source count or epsilon")
    if num_sources == 1:
        return np.asarray([1.0],dtype=float)
    p=np.asarray([1.0-epsilon]+[epsilon/(num_sources-1)]*(num_sources-1),dtype=float)
    return p/p.sum()


def audit_probability(period):
    if period < 0:
        raise ValueError("Period cannot be negative")
    return 1.0 if period==0 else AUDIT_P


def constant_audit_choice(self):
    """Receiver learning-target choice; no other model rule is replaced.

    Original model.step sets harmonic p before sender preparation. This
    intervention is deliberately restricted to null/fixed senders, for
    which preparation does not optimize against p. Set p and p_pair
    before each citizen choice, using the same RNG method and options.
    """
    if self.model.jammer_regime not in ("null","fixed_biased"):
        raise RuntimeError("Constant audit sensitivity only supports null/fixed_biased")
    p=audit_probability(int(self.model.period))
    self.model.p=p
    self.model.p_pair=[1.0-p,p]
    return int(self.model.rng.choice(m.CHOICE_OPTION,p=self.model.p_pair))

"""Isolated V6.1 assumption interventions; no edits to the archived model."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import math

import model.InfoSourceSamplingLearning as m
from paper_b.experiments import v6_precision_cap as cap

BOUND = 5.0
VARIANTS = ("RA", "RB", "RC")
_ORIG_OUT = m.DisruptiveJammer.mu_out
_ORIG_PREPARE = m.DisruptiveJammer.prepare_for_period
_ORIG_MYOPIC = m.DisruptiveJammer.optimal_message_mean
_ORIG_AWARE = m.optimize_credibility_aware_message


def constrain(mean, diagnostics):
    """Exact box optimum for either archived strictly concave quadratic."""
    denom = float(diagnostics["objective_denominator"])
    mean = float(mean)
    if not math.isfinite(mean) or not math.isfinite(denom) or denom <= 0:
        raise FloatingPointError("Invalid unconstrained optimum/curvature")
    bounded = max(-BOUND, min(BOUND, mean))
    d = dict(diagnostics, unconstrained_message_mean=mean,
             bound_active=bounded != mean, message_mean_bound=BOUND)
    if "objective_value" in d:
        d["unconstrained_objective_value"] = d["objective_value"]
        # U(m) = U(m*) - denom * (m-m*)^2.
        d["objective_value"] = float(d["objective_value"]) - denom*(bounded-mean)**2
    return bounded, d


def bounded_myopic(cls, **kwargs):
    return constrain(*_ORIG_MYOPIC(**kwargs))


def bounded_aware(**kwargs):
    return constrain(*_ORIG_AWARE(**kwargs))


def biased_initial(self):
    return (self.model.period == 0 and self.model.initial_audit_mode == "centroid"
            and self.model.jammer_regime == "fixed_biased")


def biased_prepare(self):
    rows = _ORIG_PREPARE(self)
    if biased_initial(self):
        if float(self.mu_theta) != 4.0:
            raise RuntimeError("RA requires underlying sender position 4")
        # Keep native clustering/initialization and RNG path. Only change
        # emitted audit mean, not the segment centroid or the null policy.
        for row in rows:
            self._current_msg_param[int(row["cluster"])]["avg"] = 4.0
            row["message_mean"] = 4.0
            row["strategy_phase"] = "initial_audit_fixed_biased"
    return rows


def biased_mu_out(self, n_req, requester):
    if n_req > 0 and biased_initial(self):
        return list(self.model.rng.normal(4.0, 1.0, int(n_req)))
    return _ORIG_OUT(self, n_req, requester)


@contextmanager
def intervention(variant):
    """Patch in process-local scope only; restore even after failure."""
    if variant not in (*VARIANTS, "baseline"):
        raise ValueError(variant)
    with ExitStack() as stack:
        if variant == "RA":
            stack.enter_context(patch.object(m.DisruptiveJammer, "prepare_for_period", biased_prepare))
            stack.enter_context(patch.object(m.DisruptiveJammer, "mu_out", biased_mu_out))
        elif variant == "RB":
            # Reuse the archived cap-5 implementation with a temporary cap=1.
            # All delivered draws still enter the sample mean; audit unchanged.
            stack.enter_context(patch.object(cap, "CAP", 1))
            stack.enter_context(patch.object(m.Citizen, "bayesian_update_theta_from_sources",
                                             cap.capped_theta_update))
        elif variant == "RC":
            stack.enter_context(patch.object(m.DisruptiveJammer, "optimal_message_mean",
                                             classmethod(bounded_myopic)))
            stack.enter_context(patch.object(m, "optimize_credibility_aware_message", bounded_aware))
        yield

"""Gate 0C prespecified credibility-aware sender regression tests.

Run: python -m unittest paper_b.test_gate0c_credibility_aware -v
"""

import math
import unittest

import numpy as np

from model.InfoSourceSamplingLearning import DisruptiveJammer, InfoSampleModel
from paper_b.adversary_objectives import (
    credibility_aware_components,
    credibility_aware_utility,
    optimize_credibility_aware_message,
)
from paper_b.experiments.consolidate_gate0b import (
    credibility_aware_peer_interaction,
    credibility_aware_vs_myopic,
    sender_damage,
)
from paper_b.experiments.run_local_diagnostic import base_model_config
from paper_b.experiments.run_matched_pilot import (
    condition_specs, parse_sender_regimes,
)


def basic_inputs():
    return {
        "citizen_means": [0.0, 0.0, 0.0, 0.0],
        "citizen_sds": [5.0] * 4,
        "prior_delta_means": [0.0] * 4,
        "prior_delta_sds": [5.0] * 4,
        "theta_inclusion": [1.0] * 4,
        "audit_inclusion": [1.0] * 4,
        "expected_audit_counts": [10.0] * 4,
        "truth": 0.0,
        "underlying_position": 4.0,
        "audit_probability": 0.5,
        "credibility_penalty": 1.0,
    }


def make_model(regime: str, *, seed: int = 6901, lam: float = 1.0):
    cfg = base_model_config(
        regime="flat", seed=seed, n_citizens=20,
        max_steps=8, k=1, epsilon=0.05, credit=20,
        comparison_rule="delta_comparison",
        surveillance_interval=5,
    )
    cfg.update({
        "network_environment": "extended",
        "jammer_regime": regime,
        "jammer_credibility_penalty": lam,
        "reliance_mode": "adaptive",
        "frozen_ranking_mode": "first_audit",
        "initial_audit_mode": "centroid",
        "peer_evidence_mode": "source_posterior",
        "tau_social": 1.0,
    })
    return InfoSampleModel(model_attribute=cfg, rng=seed)


class CredibilityAwareObjectiveTests(unittest.TestCase):
    def test_numeric_optimum_of_quadratic(self):
        inp = basic_inputs()
        inp["citizen_means"] = [-1.1, 0.4, 0.8, 2.3]
        inp["citizen_sds"] = [3.1, 4.0, 2.3, 5.2]
        inp["prior_delta_means"] = [-1.0, 1.1, 0.3, -0.2]
        inp["prior_delta_sds"] = [2.0, 3.0, 1.5, 4.0]
        inp["theta_inclusion"] = [0.65, 0.05, 0.98, 0.1]
        inp["audit_inclusion"] = [1.0, 1.0, 0.0, 1.0]
        inp["expected_audit_counts"] = [10.0, 10.0, 0.0, 5.0]
        optimum, stats = optimize_credibility_aware_message(**inp)
        self.assertGreater(stats["objective_denominator"], 0)
        args = {
            k: inp[k] for k in (
                "citizen_means", "citizen_sds", "prior_delta_means",
                "prior_delta_sds", "theta_inclusion", "audit_inclusion",
                "expected_audit_counts",
            )
        }
        components = credibility_aware_components(**args)
        def utility(m):
            return credibility_aware_utility(
                m, truth=inp["truth"],
                underlying_position=inp["underlying_position"],
                audit_probability=inp["audit_probability"],
                credibility_penalty=inp["credibility_penalty"],
                **components,
            )
        self.assertGreater(utility(optimum), utility(optimum - 0.05))
        self.assertGreater(utility(optimum), utility(optimum + 0.05))
        self.assertAlmostEqual(
            (utility(optimum + 1e-4) - utility(optimum - 1e-4)) / (2e-4),
            0.0, delta=1e-6,
        )

    def test_no_exposure_reverts_to_underlying_position(self):
        inp = basic_inputs()
        inp["theta_inclusion"] = [0] * 4
        inp["audit_inclusion"] = [0] * 4
        msg, _ = optimize_credibility_aware_message(**inp)
        self.assertAlmostEqual(msg, 4.0, places=12)

    def test_credible_penalty_reduces_extreme_first_attack_in_symmetric_case(self):
        inp = basic_inputs()
        aware, _ = optimize_credibility_aware_message(**inp)
        no_penalty, _ = optimize_credibility_aware_message(
            **{**inp, "credibility_penalty": 0.0}
        )
        myopic, _ = DisruptiveJammer.optimal_message_mean(
            citizen_means=inp["citizen_means"],
            citizen_sds=inp["citizen_sds"],
            truth=inp["truth"],
            underlying_position=inp["underlying_position"],
        )
        self.assertGreater(myopic, 50.0)
        self.assertLess(abs(aware), abs(no_penalty))
        self.assertLess(abs(no_penalty), abs(myopic))
        self.assertGreater(aware, 0.0)

    def test_invalid_penalty_rejected(self):
        with self.assertRaises(ValueError):
            optimize_credibility_aware_message(
                **{**basic_inputs(), "credibility_penalty": -1}
            )
        with self.assertRaises(ValueError):
            make_model("credibility_aware", lam=-0.5)

    def test_preserved_initial_audit_and_separate_strategic_message(self):
        aware = make_model("credibility_aware")
        myopic = make_model("adaptive")
        aware.step()
        myopic.step()
        self.assertEqual(aware.period, 1)
        self.assertEqual(myopic.period, 1)
        x0 = aware.jammer_strategy_history
        y0 = myopic.jammer_strategy_history
        self.assertEqual(x0[0]["strategy_phase"], "initial_audit")
        self.assertEqual(x0[0]["message_mean"], y0[0]["message_mean"])
        self.assertTrue(all(x.theta_or_delta == 1 for x in aware.citizens))
        aware.step()
        myopic.step()
        x1 = [x for x in aware.jammer_strategy_history if x["period"] == 1]
        y1 = [x for x in myopic.jammer_strategy_history if x["period"] == 1]
        self.assertEqual(len(x1), 1)
        self.assertEqual(len(y1), 1)
        self.assertTrue(x1[0]["refresh"])
        self.assertEqual(x1[0]["optimization_rule"], "credibility_aware")
        self.assertEqual(y1[0]["optimization_rule"], "adaptive")
        self.assertGreater(abs(x1[0]["message_mean"] -
                               y1[0]["message_mean"]), 1.0)
        self.assertLess(abs(x1[0]["message_mean"]),
                        abs(y1[0]["message_mean"]))
        for _ in range(5):
            aware.step()
        self.assertEqual(
            sorted({x["period"] for x in aware.jammer_strategy_history
                    if x["refresh"]}),
            [0, 1, 6],
        )

    def test_sender_grid_contains_aware_as_active_sender(self):
        sender = parse_sender_regimes(
            "null,fixed_biased,adaptive,credibility_aware"
        )
        specs = condition_specs(sender)
        self.assertEqual(len(specs), 8)
        aware = [x for x in specs if x[1] == "credibility_aware"]
        self.assertEqual(len(aware), 2)
        self.assertTrue(all(x[2] for x in aware))

    def test_aware_minus_myopic_contrasts(self):
        rows = []
        for net, n0, m0, a0 in (
            ("elite_only", 1.0, 8.0, 3.0),
            ("extended", 2.0, 10.0, 7.0),
        ):
            for reliance in ("adaptive", "frozen"):
                for regime, mse in (
                    ("null", n0),
                    ("adaptive", m0),
                    ("credibility_aware", a0),
                ):
                    rows.append({
                        "seed": 6901,
                        "initial_regime": "flat",
                        "network_environment": net,
                        "reliance_mode": reliance,
                        "jammer_regime": regime,
                        "mse_truth": mse,
                    })
        d = sender_damage(rows)
        paired = credibility_aware_vs_myopic(d)
        self.assertEqual(len(paired), 4)
        interaction = credibility_aware_peer_interaction(paired)
        self.assertEqual(len(interaction), 2)
        for row in interaction:
            self.assertAlmostEqual(row["aware_by_peer_interaction"], 2.0)


if __name__ == "__main__":
    unittest.main()

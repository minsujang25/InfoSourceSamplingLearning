"""Gate 0A regression tests: initial centroid audit and strategic onset.

Run from repository root:
    python -m unittest paper_b.test_gate0a_initial_audit -v

These tests intentionally use the unchanged Mesa model with small populations.
They check message distributions, not a production treatment effect.
"""

from __future__ import annotations

import unittest

import numpy as np

from model.InfoSourceSamplingLearning import InfoSampleModel


def model_for(
    *,
    seed: int = 6901,
    regime: str = "adaptive",
    reliance: str = "adaptive",
    k: int = 2,
    initial_audit_mode: str = "centroid",
    ranking: str = "first_audit",
) -> InfoSampleModel:
    # 12 citizens in two separated belief clusters; elite positions at 0 and 1.
    citizen_means = [-3.2, -3.0, -2.8, -2.6, -3.1, -2.9,
                     2.7, 2.9, 3.1, 3.3, 2.8, 3.2]
    cfg = {
        "num_nodes": 14,
        "state_of_the_world": 0.0,
        "mu_theta": [0.0, 4.0] + citizen_means,
        "sd_theta": [1.0, 1.0] + [5.0] * len(citizen_means),
        "network_environment": "elite_only",
        "jammer_regime": regime,
        "reliance_mode": reliance,
        "frozen_ranking_mode": ranking,
        "initial_audit_mode": initial_audit_mode,
        "peer_evidence_mode": "source_posterior",
        "tau_social": 1.0,
        "surveil_ability": k,
        "surveillance_interval": 5,
        "epsilon": 0.05,
        "credit": 20,
        "comparison_rule": "delta_comparison",
        "max_steps": 8,
        "stop_on_convergence": False,
    }
    return InfoSampleModel(model_attribute=cfg, rng=seed)


class InitialAuditCorrectionTests(unittest.TestCase):

    def test_t0_centroid_message_params_and_distribution(self):
        for k in (1, 2, 4):
            model = model_for(k=k)
            model._snapshot_message_states()
            model._prepare_adversary()
            jammer = model.jammer
            self.assertEqual(len(jammer.citizen_per_cl), k)
            for cluster, citizens in jammer.citizen_per_cl.items():
                centroid = float(np.mean([c._message_mu for c in citizens]))
                self.assertAlmostEqual(
                    jammer._current_msg_param[cluster]["avg"], centroid, places=12
                )
                receiver = citizens[0]
                msgs = jammer.mu_out(3000, receiver)
                self.assertAlmostEqual(np.mean(msgs), centroid, delta=0.07)
                self.assertAlmostEqual(np.std(msgs), 1.0, delta=0.07)
            self.assertTrue(all(
                rec["strategy_phase"] == "initial_audit"
                for rec in model.jammer_strategy_history
            ))
            self.assertEqual(len(model.jammer_strategy_history), k)

    def test_t0_has_credibility_audits_without_theta_updates(self):
        for regime in ("adaptive", "fixed_biased", "truth_clone", "null"):
            model = model_for(regime=regime)
            prior = np.array([c.mu_theta for c in model.citizens])
            prior_sd = np.array([c.sd_theta for c in model.citizens])
            model.step()
            self.assertEqual(model.period, 1)
            self.assertTrue(all(c.theta_or_delta == 1 for c in model.citizens))
            np.testing.assert_array_equal(
                [c.mu_theta for c in model.citizens], prior
            )
            np.testing.assert_array_equal(
                [c.sd_theta for c in model.citizens], prior_sd
            )
            self.assertTrue(all(
                rec["strategy_phase"] == "initial_audit"
                for rec in model.jammer_strategy_history
            ))

    def test_t1_adaptive_attack_and_refresh_schedule(self):
        model = model_for(k=1)
        model.step()
        p0 = list(model.jammer_strategy_history)
        self.assertEqual(len(p0), 1)
        self.assertEqual(p0[0]["strategy_phase"], "initial_audit")
        model.step()
        p1 = [r for r in model.jammer_strategy_history if r["period"] == 1]
        self.assertEqual(len(p1), 1)
        self.assertTrue(p1[0]["refresh"])
        self.assertEqual(p1[0]["strategy_phase"], "adaptive_disruption")
        self.assertNotAlmostEqual(
            p1[0]["message_mean"], p0[0]["message_mean"], delta=0.1
        )
        for _ in range(5):
            model.step()  # advance to period 7; refresh at period 6
        refresh_times = sorted({
            rec["period"]
            for rec in model.jammer_strategy_history
            if rec["refresh"]
        })
        self.assertEqual(refresh_times, [0, 1, 6])

    def test_frozen_first_audit_starts_with_same_rank_as_adaptive(self):
        adaptive = model_for(seed=6911, reliance="adaptive")
        frozen = model_for(seed=6911, reliance="frozen")
        adaptive.step()
        frozen.step()
        for ac, fc in zip(adaptive.citizens, frozen.citizens):
            a_rank = [x.pos for x in ac._behavioral_ranking()]
            f_rank = [x.pos for x in fc._behavioral_ranking()]
            self.assertEqual(a_rank, f_rank)
            self.assertEqual(
                f_rank,
                [x.pos for x in fc.frozen_credibility_ranked_sources],
            )

    def test_legacy_mode_remains_available_for_old_production(self):
        model = model_for(k=1, initial_audit_mode="legacy")
        model.step()
        self.assertEqual(model.period, 1)
        first = [r for r in model.jammer_strategy_history if r["period"] == 0]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["strategy_phase"], "adaptive_disruption")
        self.assertGreater(
            abs(first[0]["message_mean"] - first[0]["segment_mean"]), 1.0
        )

    def test_invalid_audit_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            model_for(initial_audit_mode="unknown")


if __name__ == "__main__":
    unittest.main()

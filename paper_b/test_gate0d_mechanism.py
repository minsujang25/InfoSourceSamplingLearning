"""Gate 0D objective ablations and RNG-neutral sender event capture."""
from __future__ import annotations

import unittest

from paper_b.experiments.run_local_diagnostic import base_model_config
from paper_b.experiments.run_matched_pilot import (
    condition_specs, parse_sender_regimes, run_condition,
)
from paper_b.experiments.consolidate_gate0d import (
    STAGES, STAGE_LABELS, staged_contrasts, summarize_sender_events,
)


def config(seed=6901, n=12, steps=4):
    return base_model_config(
        regime="flat", seed=seed, n_citizens=n,
        max_steps=steps, k=1, epsilon=0.05,
        credit=20, comparison_rule="delta_comparison",
        surveillance_interval=5,
    )


def run(
    sender, *, seed=6901, event_limit=0, n=12, steps=4,
    peer="extended", lam=1.0,
):
    return run_condition(
        base_config=config(seed=seed, n=n, steps=steps),
        seed=seed, regime="flat", environment=peer,
        reliance_mode="adaptive",
        jammer_active=sender not in {"null", "fixed_biased"},
        jammer_regime=sender,
        peer_evidence_mode="source_posterior",
        tau_social=1.0, jammer_credibility_penalty=lam,
        initial_audit_mode="centroid",
        frozen_ranking_mode="first_audit", k=1,
        design_id="gate0d_test", block_id="gate0d_seed",
        save_edge_log=False,
        sender_event_period_limit=event_limit,
    )


class Gate0DTests(unittest.TestCase):
    def test_explicit_matrix(self):
        sender_grid = ("null", "fixed_biased") + STAGES
        self.assertEqual(len(sender_grid), 7)
        self.assertEqual(
            len(condition_specs(parse_sender_regimes(",".join(sender_grid)))),
            14,
        )

    def test_expected_stage_order_and_initial_centroid(self):
        messages = {}
        for regime in STAGES:
            out = run(regime, steps=2)
            history = out["jammer_strategy"]
            t0 = [x for x in history if x["period"] == 0]
            t1 = [x for x in history if x["period"] == 1]
            self.assertEqual(len(t0), 1)
            self.assertEqual(len(t1), 1)
            self.assertEqual(t0[0]["strategy_phase"], "initial_audit")
            self.assertAlmostEqual(
                t0[0]["segment_mean"], t0[0]["message_mean"], places=10,
            )
            self.assertEqual(t1[0]["strategy_phase"], "adaptive_disruption")
            self.assertEqual(t1[0]["optimization_rule"], regime)
            messages[regime] = float(t1[0]["message_mean"])
        self.assertGreater(abs(messages["adaptive"]), 10.0)
        self.assertLess(abs(messages["timing_only"]),
                        abs(messages["adaptive"]))
        # Rank-specific exposure and credibility penalty need not be
        # monotone in message mean for arbitrary actual belief states.
        self.assertEqual(len(messages), 5)

    def test_aware_lambda_zero_is_exact_ranked_exposure(self):
        ranked = run("exposure_ranked", lam=1.0, steps=5)
        aware_zero = run("credibility_aware", lam=0.0, steps=5)
        self.assertAlmostEqual(
            ranked["run"]["mse_truth"], aware_zero["run"]["mse_truth"],
            places=12
        )
        ranked_means = [x["message_mean"] for x in ranked["jammer_strategy"]]
        aware_means = [x["message_mean"] for x in aware_zero["jammer_strategy"]]
        self.assertEqual(ranked_means, aware_means)

    def test_capture_on_off_outcome_parity_and_real_delivery(self):
        for sender in ("null", "adaptive", "exposure_uniform",
                       "exposure_ranked", "credibility_aware"):
            original = run(sender, n=12, steps=4)
            instrumented = run(sender, event_limit=4, n=12, steps=4)
            self.assertEqual(
                original["run"]["mse_truth"],
                instrumented["run"]["mse_truth"],
            )
            self.assertEqual(
                original["run"]["W_jammer_precision_share"],
                instrumented["run"]["W_jammer_precision_share"],
            )
            self.assertEqual(
                [x["message_mean"] for x in original["jammer_strategy"]],
                [x["message_mean"] for x in instrumented["jammer_strategy"]],
            )
            events = instrumented["sender_events"]
            self.assertEqual(len(events), 4 * 12)
            self.assertTrue(all(x["theta_mu_post"] is not None for x in events))
            self.assertTrue(all(
                x["learning_target"] == "audit"
                for x in events if x["period"] == 0
            ))
            self.assertTrue(all(
                x["jammer_delivered_count"] <= x["jammer_request_count"]
                for x in events
            ))
            if sender == "null":
                self.assertTrue(all(
                    x["jammer_delivered_count"] == 0
                    for x in events if x["period"] >= 1
                ))
            else:
                self.assertGreater(sum(x["jammer_delivered_count"]
                                       for x in events), 0)
            summary = summarize_sender_events(events)
            self.assertEqual(len(summary), 4)
            self.assertEqual(summary[0]["n_audit"], 12)

    def test_stage_paired_differences_reconcile(self):
        rows = []
        for sender, val in zip(STAGES, (42.0, 10.0, 9.0, 8.0, 7.0)):
            rows.append({
                "seed": 1,
                "initial_regime": "flat",
                "network_environment": "elite_only",
                "reliance_mode": "adaptive",
                "sender_regime": sender,
                "mse_sender": val + 2,
                "delta_mse": val,
            })
        differences = staged_contrasts(rows)
        self.assertEqual(len(differences), 4)
        self.assertEqual([r["mechanism"] for r in differences],
                         list(STAGE_LABELS))
        self.assertAlmostEqual(
            sum(r["delta_damage"] for r in differences), 7.0 - 42.0
        )


if __name__ == "__main__":
    unittest.main()

"""Unit and small-model checks for the Gate 0B matched sender design.

Run: python -m unittest paper_b.test_gate0b_canonical_pilot -v
"""

from __future__ import annotations

import unittest

from paper_b.experiments.consolidate_gate0b import (
    adaptation_network_interactions,
    peer_damage_differences,
    receiver_differences,
    sender_damage,
)
from paper_b.experiments.run_matched_pilot import (
    condition_specs,
    parse_sender_regimes,
    run_condition,
)
from paper_b.experiments.run_local_diagnostic import base_model_config


def synthetic_runs(*, with_time: bool = False) -> list[dict]:
    """Known sender and peer effects: peer interaction = +3."""
    rows = []
    for network, null_mse, fixed_d, adaptive_d in (
        ("elite_only", 1.0, 3.0, 5.0),
        ("extended", 2.0, 1.0, 6.0),
    ):
        for reliance in ("adaptive", "frozen"):
            for sender, mse in (
                ("null", null_mse),
                ("fixed_biased", null_mse + fixed_d),
                ("adaptive", null_mse + adaptive_d),
            ):
                row = {
                    "seed": 6901,
                    "initial_regime": "flat",
                    "network_environment": network,
                    "reliance_mode": reliance,
                    "jammer_regime": sender,
                    "mse_truth": mse,
                }
                rows.append(row)
                if with_time:
                    rows.append({**row, "horizon_step": 1})
                    rows.append({**row, "horizon_step": 26})
    if with_time:
        return [r for r in rows if "horizon_step" in r]
    return rows


class Gate0BDesignTests(unittest.TestCase):
    def test_sender_grid_parser_and_legacy_default(self):
        self.assertIsNone(parse_sender_regimes(""))
        self.assertEqual(
            parse_sender_regimes("null,fixed_biased,adaptive"),
            ("null", "fixed_biased", "adaptive"),
        )
        self.assertEqual(len(condition_specs(None)), 4)
        self.assertEqual(
            len(condition_specs(parse_sender_regimes(
                "null,fixed_biased,adaptive"
            ))), 6
        )
        with self.assertRaises(ValueError):
            parse_sender_regimes("null,null")
        with self.assertRaises(ValueError):
            parse_sender_regimes("unknown")

    def test_explicit_config_propagates_to_each_sender_model(self):
        cfg = base_model_config(
            regime="flat", seed=6901, n_citizens=12,
            max_steps=3, k=1, epsilon=0.05, credit=20,
            comparison_rule="delta_comparison",
            surveillance_interval=5,
        )
        for sender in ("null", "fixed_biased", "adaptive"):
            for reliance in ("adaptive", "frozen"):
                result = run_condition(
                    base_config=cfg,
                    seed=6901,
                    regime="flat",
                    environment="extended",
                    reliance_mode=reliance,
                    jammer_active=sender == "adaptive",
                    jammer_regime=sender,
                    peer_evidence_mode="source_posterior",
                    tau_social=1.0,
                    frozen_ranking_mode="first_audit",
                    initial_audit_mode="centroid",
                    k=1,
                    design_id="gate0btest",
                    block_id="gate0btest_s6901",
                    save_edge_log=False,
                )
                row = result["run"]
                self.assertEqual(row["jammer_regime"], sender)
                self.assertEqual(row["peer_evidence_mode"], "source_posterior")
                self.assertEqual(row["frozen_ranking_mode"], "first_audit")
                self.assertEqual(row["initial_audit_mode"], "centroid")
                self.assertEqual(float(row["tau_social"]), 1.0)
                self.assertEqual(row["steps_run"], 3)
                self.assertEqual(row["condition"], f"{reliance}__{sender}")
                t0 = [x for x in result["jammer_strategy"]
                      if int(x["period"]) == 0]
                self.assertTrue(t0)
                self.assertTrue(all(
                    x["strategy_phase"] == "initial_audit"
                    for x in t0
                ))

    def test_null_differences_and_network_interaction(self):
        damage = sender_damage(synthetic_runs())
        self.assertEqual(len(damage), 8)
        peer = peer_damage_differences(damage)
        fixed = [x for x in peer
                 if x["reliance_mode"] == "adaptive"
                 and x["sender_regime"] == "fixed_biased"]
        self.assertEqual(len(fixed), 1)
        self.assertAlmostEqual(fixed[0]["peer_minus_elite_damage"], -2.0)
        interactions = adaptation_network_interactions(damage)
        self.assertEqual(len(interactions), 2)
        for row in interactions:
            self.assertAlmostEqual(row["adaptive_by_peer_interaction"], 3.0)
        receiver = receiver_differences(damage)
        self.assertEqual(len(receiver), 4)
        self.assertTrue(all(
            abs(r["adaptive_minus_frozen_damage"]) < 1e-12
            for r in receiver
        ))

    def test_peer_time_series_keeps_checkpoint_key(self):
        time_damage = sender_damage(synthetic_runs(with_time=True),
                                    by_period=True)
        peers = peer_damage_differences(time_damage)
        self.assertEqual(len(peers), 8)
        self.assertEqual({row["horizon_step"] for row in peers}, {1, 26})

    def test_missing_null_is_rejected(self):
        rows = [r for r in synthetic_runs() if r["jammer_regime"] != "null"]
        with self.assertRaisesRegex(ValueError, "Missing matched null"):
            sender_damage(rows)


if __name__ == "__main__":
    unittest.main()

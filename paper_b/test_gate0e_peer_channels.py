"""Gate 0E regression tests for source-preserving peer-message treatments."""
from __future__ import annotations

import unittest

from model.InfoSourceSamplingLearning import InfoSampleModel
from paper_b.experiments.run_local_diagnostic import base_model_config
from paper_b.experiments.run_matched_pilot import run_condition
from paper_b.experiments.consolidate_gate0e import (
    CHANNEL_MODES, channel_damage_contrasts,
    validate_channel_blocks,
)


def run_channel(mode: str, *, seed: int = 6901, n: int = 24,
                t: int = 6, capture: int = 6, sender="adaptive"):
    base = base_model_config(
        regime="flat", seed=seed, n_citizens=n, max_steps=t,
        k=1, epsilon=0.05, credit=20,
        comparison_rule="delta_comparison", surveillance_interval=5,
    )
    return run_condition(
        base_config=base, seed=seed, regime="flat",
        environment=mode, reliance_mode="adaptive",
        jammer_active=sender == "adaptive",
        jammer_regime=sender,
        peer_evidence_mode="source_posterior",
        tau_social=1.0, initial_audit_mode="centroid",
        frozen_ranking_mode="first_audit",
        jammer_credibility_penalty=1.0,
        k=1, design_id="gate0e_test", block_id="gate0e_test",
        save_edge_log=False,
        sender_event_period_limit=capture,
    )


class Gate0EPeerChannelTests(unittest.TestCase):
    def test_live_default_bitwise_parity(self):
        cfg = base_model_config(
            regime="flat", seed=6901, n_citizens=24, max_steps=6,
            k=1, epsilon=0.05, credit=20,
            comparison_rule="delta_comparison", surveillance_interval=5,
        )
        cfg.update(network_environment="extended",
                   jammer_regime="adaptive",
                   peer_evidence_mode="source_posterior", tau_social=1.0,
                   initial_audit_mode="centroid")
        implicit = InfoSampleModel(model_attribute=cfg, rng=6901)
        explicit = InfoSampleModel(
            model_attribute={**cfg,"peer_signal_mode":"live"}, rng=6901
        )
        for _ in range(6):
            implicit.step()
            explicit.step()
        self.assertEqual(implicit.agent_mu_theta_list,
                         explicit.agent_mu_theta_list)
        self.assertEqual(implicit.jammer_strategy_history,
                         explicit.jammer_strategy_history)
        self.assertEqual(implicit.agent_num_request_list,
                         explicit.agent_num_request_list)

    def test_initial_peer_audit_identical_across_three_modes(self):
        outputs = {
            mode: run_channel(mode, capture=2, t=2)
            for mode in ("extended_muted","extended_stale","extended_live")
        }
        peer_signature = []
        for mode, out in outputs.items():
            self.assertEqual(out["run"]["peer_signal_mode"],
                             mode.split("_")[1])
            rows = sorted(
                (r for r in out["sender_events"] if int(r["period"])==0),
                key=lambda r: r["citizen_pos"]
            )
            self.assertEqual(len(rows),24)
            self.assertTrue(all(x["learning_target"]=="audit" for x in rows))
            peer_signature.append([
                (r["peer_audit_signature"],r["peer_request_count"],
                 r["peer_delivered_count"],r["jammer_delta_mu_post"],
                 r["theta_mu_post"])
                for r in rows
            ])
            self.assertTrue(all(r["peer_delivered_count"]>0 for r in rows))
        self.assertEqual(peer_signature[0],peer_signature[1])
        self.assertEqual(peer_signature[1],peer_signature[2])

    def test_muted_consumes_credits_but_delivers_nothing(self):
        out=run_channel("extended_muted",capture=6)
        rows=[r for r in out["sender_events"] if r["period"]>=1]
        self.assertGreater(sum(r["peer_request_count"] for r in rows),0)
        self.assertEqual(sum(r["peer_delivered_count"] for r in rows),0)
        self.assertTrue(all(r["peer_request_count"]>=r["peer_delivered_count"]
                            for r in rows))
        self.assertTrue(all(r["source_count"]==4 for r in rows))

    def test_stale_posterior_uses_fixed_channel_sd(self):
        cfg=base_model_config(
            regime="flat",seed=6901,n_citizens=24,max_steps=6,
            k=1,epsilon=.05,credit=20,
            comparison_rule="delta_comparison",surveillance_interval=5,
        )
        cfg.update(
            network_environment="extended",peer_signal_mode="stale",
            jammer_regime="adaptive",peer_evidence_mode="source_posterior",
            tau_social=1.0,initial_audit_mode="centroid"
        )
        m=InfoSampleModel(model_attribute=cfg,rng=6901)
        for _ in range(4):
            m.step()
        found_evolving_source=False
        for citizen in m.citizens:
            for source in citizen.info_source:
                if source.type_of_agent=="citizen":
                    delivered_sd=citizen._delivered_peer_signal_sds[source]
                    self.assertEqual(
                        delivered_sd,source._initial_peer_signal_sd
                    )
                    if source._message_sd!=source._initial_peer_signal_sd:
                        found_evolving_source=True
        self.assertTrue(found_evolving_source)
        with self.assertRaises(ValueError):
            InfoSampleModel(
                model_attribute={**cfg,"peer_evidence_mode":"legacy_batch"},
                rng=6901,
            )

    def test_event_capture_parity_all_modes(self):
        for mode in CHANNEL_MODES:
            base=run_channel(mode,capture=0)
            traced=run_channel(mode,capture=6)
            self.assertEqual(base["run"]["mse_truth"],
                             traced["run"]["mse_truth"])
            self.assertEqual(
                base["jammer_strategy"],traced["jammer_strategy"]
            )
            self.assertEqual(
                base["belief_checkpoints"],traced["belief_checkpoints"]
            )
            self.assertEqual(len(traced["sender_events"]),24*6)

    def test_telescope_with_null_referenced_damage(self):
        damage=[]
        for network,d in zip(CHANNEL_MODES,(2.0,3.0,6.0,7.0)):
            damage.append(dict(
                seed=1,initial_regime="flat",reliance_mode="adaptive",
                sender_regime="adaptive",network_environment=network,
                mse_sender=d+1.0,mse_null=1.0,delta_mse=d
            ))
        records=channel_damage_contrasts(damage)
        self.assertEqual(len(records),1)
        r=records[0]
        self.assertEqual(r["slot_attention_contrast"],1.0)
        self.assertEqual(r["static_peer_signal_contrast"],3.0)
        self.assertEqual(r["endogenous_peer_update_contrast"],1.0)
        self.assertEqual(r["total_peer_contrast"],5.0)


if __name__=="__main__":
    unittest.main()

"""Deterministic design and validation tests, no ABM simulation required."""
import unittest
from unittest.mock import patch
from paper_b.experiments import run_v6_strategic_scope as s


class StrategicScopeTests(unittest.TestCase):
    def test_grid(self):
        self.assertEqual(len(s.cells()),12)
        self.assertEqual(s.SETTINGS["segmented4"],(4,5))
        self.assertEqual(s.SETTINGS["refresh1"],(1,1))
        self.assertEqual(s.PHASES["strategic"][0],tuple(range(7501,7521)))

    def test_native_k4_initial_audit(self):
        def output(k):
            return {"run":{"K":k,"variant":"segmented4","initial_audit_K":k,
                           "strategic_K":k,"surveillance_interval":5,"sender_regime":"null"},
                    "jammer_strategy":[dict(period=0,cluster=i,cluster_size=25,
                        strategy_phase="initial_audit",refresh=True,message_sd=1.,
                        segment_mean=float(i),message_mean=float(i)) for i in range(4)]}
        s.validate_strategy(output(4),26,"segmented4")
        invalid=output(4)
        invalid["jammer_strategy"]=[dict(invalid["jammer_strategy"][0],cluster_size=100)]
        with self.assertRaises(RuntimeError):
            s.validate_strategy(invalid,26,"segmented4")

    def test_archive_check_only_k1_null(self):
        archive={"runs":{(7501,"elite_only","null","adaptive"):
                          {"initial_state_fingerprint":"i","structural_fingerprint":"g"}}}
        def output(k):
            return {"run":{"seed":7501,"network_environment":"elite_only","reliance_mode":"adaptive",
                           "sender_regime":"null","initial_audit_K":k,
                           "initial_state_fingerprint":"i","structural_fingerprint":"g"}}
        with patch.object(s.archive_api,"compare_archive") as check:
            s.compare_block([output(4)],archive,400)
            check.assert_not_called()
            s.compare_block([output(1)],archive,400)
            check.assert_called_once()

    def test_scope_and_restore(self):
        original=s.base.run_condition
        surveil=s.m.DisruptiveJammer.surveil_citizen
        with patch.object(s,"ORIGINAL_RUN",lambda **kw:kw):
            with s.variant_scope("segmented4"):
                cfg=s.base.run_condition(k=1,base_config={"surveillance_interval":5})
                self.assertEqual(cfg["k"],4)
                self.assertEqual(cfg["base_config"]["surveillance_interval"],5)
                self.assertIs(s.m.DisruptiveJammer.surveil_citizen,surveil)
            with s.variant_scope("refresh1"):
                cfg=s.base.run_condition(k=1,base_config={"surveillance_interval":5})
                self.assertEqual(cfg["k"],1)
                self.assertEqual(cfg["base_config"]["surveillance_interval"],1)
        self.assertIs(s.base.run_condition,original)
        self.assertIs(s.m.DisruptiveJammer.surveil_citizen,surveil)

    def test_contrast_and_change(self):
        paths={(7501,c,j,r):{400:float(i+v)} for i,c in enumerate(s.CHANNELS)
               for j,v in (("null",0),("adaptive",2),("credibility_aware",3))
               for r in s.RECEIVERS}
        old=s.contrasts(7501,"archived_baseline",paths)
        self.assertEqual(len(old),4)
        self.assertEqual(old[0]["D_L_minus_E"],0.)
        self.assertEqual(old[0]["sender_MSE_L_minus_E"],1.)
        paths[(7501,"extended_live","adaptive","adaptive")][400]+=2
        new=s.contrasts(7501,"segmented4",paths)
        changes=s.differences(new,old)
        self.assertEqual(changes[0]["D_L_minus_E"],2.)
        self.assertEqual(changes[0]["sender_MSE_L_minus_E"],2.)


if __name__=="__main__":
    unittest.main()

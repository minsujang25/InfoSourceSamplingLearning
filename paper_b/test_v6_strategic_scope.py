"""Deterministic design and validation tests, no ABM simulation required."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from paper_b.experiments import run_v6_strategic_scope as s


class StrategicScopeTests(unittest.TestCase):
    def test_grid(self):
        self.assertEqual(len(s.cells()),12)
        self.assertEqual(s.SETTINGS["segmented4"],(4,5))
        self.assertEqual(s.SETTINGS["refresh1"],(1,1))
        self.assertEqual(s.PHASES["strategic"][0],tuple(range(7501,7521)))

    def test_common_audit_only_changes_t0(self):
        seen=[]
        def original(obj):
            seen.append(obj.surveillance_ability)
        j=SimpleNamespace(model=SimpleNamespace(period=0),surveillance_ability=4)
        with patch.object(s,"ORIGINAL_SURVEIL",original):
            s.common_audit_surveil(j)
            self.assertEqual(j.surveillance_ability,4)
            j.model.period=1
            s.common_audit_surveil(j)
        self.assertEqual(seen,[1,4])

    def test_restore_on_exception(self):
        def fail(obj):
            raise ValueError("test")
        j=SimpleNamespace(model=SimpleNamespace(period=0),surveillance_ability=4)
        with patch.object(s,"ORIGINAL_SURVEIL",fail):
            with self.assertRaises(ValueError):
                s.common_audit_surveil(j)
        self.assertEqual(j.surveillance_ability,4)

    def test_scope_and_restore(self):
        original=s.base.run_condition
        surveil=s.m.DisruptiveJammer.surveil_citizen
        with patch.object(s,"ORIGINAL_RUN",lambda **kw:kw):
            with s.variant_scope("segmented4"):
                cfg=s.base.run_condition(k=1,base_config={"surveillance_interval":5})
                self.assertEqual(cfg["k"],4)
                self.assertEqual(cfg["base_config"]["surveillance_interval"],5)
                self.assertIs(s.m.DisruptiveJammer.surveil_citizen,s.common_audit_surveil)
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

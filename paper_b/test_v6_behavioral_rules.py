"""Deterministic tests of acquisition and audit interventions."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
import model.InfoSourceSamplingLearning as m
from paper_b.experiments import run_matched_pilot as up
from paper_b.experiments import run_v6_behavioral as r
from paper_b.experiments.v6_behavioral_rules import (
    AUDIT_P, audit_probability, uniform_tail_rank_probabilities, constant_audit_choice,
)


class BehavioralTests(unittest.TestCase):
    def test_rank_probabilities(self):
        for d in (1,2,3,4,7):
            p=uniform_tail_rank_probabilities(d,.05)
            self.assertAlmostEqual(float(p.sum()),1.)
            self.assertAlmostEqual(p[0],1. if d==1 else .95)
            if d>1:
                np.testing.assert_allclose(p[1:],.05/(d-1),rtol=0,atol=1e-15)
        np.testing.assert_allclose(uniform_tail_rank_probabilities(2,.05),m.recursive_rank_probabilities(2,.05))

    def test_invalid_rank(self):
        for args in ((0,.05),(4,-.1),(4,.6)):
            with self.assertRaises(ValueError):
                uniform_tail_rank_probabilities(*args)

    def test_matched_expected_audits(self):
        self.assertEqual(audit_probability(0),1.)
        self.assertEqual(audit_probability(1),audit_probability(399))
        self.assertAlmostEqual(1+399*AUDIT_P,math.fsum(1/i for i in range(1,401)),places=13)
        self.assertAlmostEqual(r.expected_audits("constant_audit",26),1+25*AUDIT_P)

    def test_actual_choice_uses_new_probabilities(self):
        calls=[]
        class RNG:
            def choice(self,options,p):
                calls.append((list(options),list(p)))
                return options[1]
        model=SimpleNamespace(period=0,jammer_regime="fixed_biased",rng=RNG(),p=1.,p_pair=[0.,1.])
        c=SimpleNamespace(model=model)
        for t in (0,1,200,399):
            model.period=t
            constant_audit_choice(c)
            self.assertEqual(calls[-1][1],[1-audit_probability(t),audit_probability(t)])
            self.assertEqual(model.p,audit_probability(t))
        model.jammer_regime="adaptive"
        with self.assertRaises(RuntimeError):
            constant_audit_choice(c)

    def test_scope_restores_both_aliases_and_choice(self):
        orig=m.recursive_rank_probabilities
        diag=up.recursive_rank_probabilities
        choice=m.Citizen.learn_theta_or_delta
        with r.variant_scope("uniform_tail"):
            self.assertIs(m.recursive_rank_probabilities,uniform_tail_rank_probabilities)
            self.assertIs(up.recursive_rank_probabilities,uniform_tail_rank_probabilities)
            self.assertIs(m.Citizen.learn_theta_or_delta,choice)
        self.assertIs(m.recursive_rank_probabilities,orig)
        self.assertIs(up.recursive_rank_probabilities,diag)
        with r.variant_scope("constant_audit"):
            self.assertIs(m.Citizen.learn_theta_or_delta,constant_audit_choice)
            self.assertIs(m.recursive_rank_probabilities,orig)
        self.assertIs(m.Citizen.learn_theta_or_delta,choice)

    def test_grid(self):
        self.assertEqual(len(r.cells("H")),8)
        self.assertEqual(len(r.cells("G")),8)
        self.assertTrue(all(c[2]=="high" for c in r.cells("H")))
        self.assertEqual(r.VARIANTS,("uniform_tail","constant_audit"))


if __name__=="__main__":
    unittest.main()

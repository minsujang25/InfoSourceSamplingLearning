"""Unit tests for V6.1 diagnostics; full ABM validation is the local preflight."""
import math
import unittest

import numpy as np
import model.InfoSourceSamplingLearning as m
from paper_b.adversary_objectives import credibility_aware_components, credibility_aware_utility
from paper_b.experiments import v6_precision_cap as cap
from paper_b.experiments.v61_assumption_interventions import constrain, intervention
from paper_b.experiments.run_v61_assumption_robustness import cells, summarize


class AssumptionTests(unittest.TestCase):
    def test_grid(self):
        for v in ("RA","RB"):
            self.assertEqual(len(cells(v,"H")),8)
            self.assertEqual(len(cells(v,"G")),8)
            self.assertTrue(all(c[2]=="high" for c in cells(v,"H")))
        self.assertEqual(len(cells("RC","C")),16)
        self.assertTrue(all(c[3]!="null" for c in cells("RC","C")))

    def test_exact_quadratic_bound(self):
        for mean in (-53.,-5.,0.,5.,53.):
            b,d=constrain(mean,{"objective_denominator":.2,"objective_value":1000.})
            self.assertEqual(b,max(-5,min(5,mean)))
            self.assertAlmostEqual(d["objective_value"],1000.-.2*(b-mean)**2)
            self.assertEqual(d["bound_active"],b!=mean)
        with self.assertRaises(FloatingPointError):
            constrain(2,{"objective_denominator":0})

    def test_myopic(self):
        args=dict(citizen_means=[0.,0.],citizen_sds=[5.,5.],truth=0.,underlying_position=4.)
        orig=m.DisruptiveJammer.optimal_message_mean
        self.assertGreater(orig(**args)[0],50.)
        with intervention("RC"):
            value,d=m.DisruptiveJammer.optimal_message_mean(**args)
            self.assertEqual(value,5.)
            self.assertTrue(d["bound_active"])
        self.assertEqual(m.DisruptiveJammer.optimal_message_mean(**args),orig(**args))

    def test_aware(self):
        args=dict(citizen_means=[0.,0.],citizen_sds=[5.,5.],
                  prior_delta_means=[0.,0.],prior_delta_sds=[1.,1.],
                  theta_inclusion=[1.,1.],audit_inclusion=[1.,1.],
                  expected_audit_counts=[5.,5.],truth=0.,underlying_position=4.,
                  audit_probability=.01,credibility_penalty=1.,single_message_variance=1.)
        c=credibility_aware_components(**{k:v for k,v in args.items()
          if k not in ("truth","underlying_position","audit_probability","credibility_penalty")})
        util=lambda v:credibility_aware_utility(v,truth=0.,underlying_position=4.,
                                               audit_probability=.01,credibility_penalty=1.,**c)
        with intervention("RC"):
            val,d=m.optimize_credibility_aware_message(**args)
            self.assertLessEqual(abs(val),5.)
            self.assertAlmostEqual(d["objective_value"],util(val),places=9)
            self.assertGreaterEqual(util(val)+1e-10,max(util(x) for x in np.linspace(-5,5,101)))

    def test_patch_restoration(self):
        original=m.Citizen.bayesian_update_theta_from_sources
        old=cap.CAP
        with self.assertRaisesRegex(RuntimeError,"test"):
            with intervention("RB"):
                self.assertEqual(cap.CAP,1)
                self.assertIs(m.Citizen.bayesian_update_theta_from_sources,cap.capped_theta_update)
                raise RuntimeError("test")
        self.assertEqual(cap.CAP,old)
        self.assertIs(m.Citizen.bayesian_update_theta_from_sources,original)

    def test_bootstrap(self):
        rows=[dict(seed=1,variant="RA",delta_mse_truth=1.),dict(seed=2,variant="RA",delta_mse_truth=3.)]
        s=summarize(rows,[1,2])
        self.assertEqual(len(s),1)
        self.assertEqual(s[0]["mean"],2.)
        self.assertEqual(s[0]["n"],2)
        with self.assertRaises(RuntimeError):
            summarize(rows[:1],[1,2])


if __name__=="__main__":
    unittest.main()

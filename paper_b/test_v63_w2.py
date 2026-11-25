"""Fast tests for W2's separate parameter and reference contract."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper_b.experiments import run_v63_w2 as w
from paper_b.experiments import run_v6_pilot as base
import model.InfoSourceSamplingLearning as m

class W2Tests(unittest.TestCase):
    def test_fixed_grid(self):
        self.assertEqual(len(w.GRID),8)
        self.assertEqual(len(set(w.GRID)),8)
        self.assertEqual(len(w.SEEDS),100)
        self.assertEqual(len(w.GRID)*len(w.SEEDS),800)
        self.assertEqual({c[0] for c in w.GRID},{"baseline","neutral"})
        self.assertEqual({c[3] for c in w.GRID},{"adaptive"})

    def test_invalid_tau(self):
        for value in (0.,3.,-1.,math.inf,math.nan,None,"2",True):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                w.checked_tau(value)

    def test_actual_peer_precision_rule(self):
        class Source:
            pass
        peer=Source(); peer.type_of_agent="citizen"
        expert=Source(); expert.type_of_agent="infoprovider"; expert._message_sd=1.
        for tau in (1.,2.):
            for sd in (0.,3.):
                c=SimpleNamespace(mu_theta_beliefs=[0.],sd_theta_beliefs=[1.],
                                  _sample_order=[peer,expert],sampled_msgs=[[2.,2.,2.],[0.,0.]],
                                  _delivered_peer_signal_sds={peer:sd},
                                  model=SimpleNamespace(tau_social=tau),_pending_evidence_precision={})
                mu,posterior_sd=m.Citizen.bayesian_update_theta_from_sources(c)
                q=1/(sd**2+tau**2)
                self.assertAlmostEqual(c._pending_evidence_precision[peer],q)
                self.assertAlmostEqual(c._pending_evidence_precision[expert],2.)
                self.assertAlmostEqual(1/posterior_sd**2,3.+q)
                self.assertAlmostEqual(mu,2*q/(3.+q))

    def test_scope_only_changes_tau_and_restores(self):
        calls=[]
        def native(*args,**kwargs):
            calls.append(kwargs.copy())
            return {"run":{"tau_social":kwargs["tau_social"]}}
        def validate(out,horizon):
            self.assertEqual(out["run"]["tau_social"],1.)
            self.assertEqual(horizon,26)
        with patch.object(base,"run_condition",native),patch.object(base,"check_run",validate):
            with w.tau_scope(2.):
                out=base.run_condition(tau_social=1.,seed=31001,reliance_mode="adaptive")
                base.check_run(out,26)
                self.assertEqual(out["run"]["tau_social"],2.)
                self.assertEqual(calls,[{"tau_social":2.,"seed":31001,"reliance_mode":"adaptive"}])
            self.assertIs(base.run_condition,native)
            self.assertIs(base.check_run,validate)

    def test_replay_contract(self):
        w.compare_tree({"x":[1.,None,True]},{"x":[1.+1e-13,None,True]})
        with self.assertRaises(RuntimeError):
            w.compare_tree({"x":[1.,None]},{"x":[1.1,None]})

    def test_seed_interaction(self):
        # Synthetic arithmetic check; not simulation output.
        outputs={}
        for tau in ("tau1","tau2"):
            for c in w.GRID:
                v,s,l,_=c
                x=0. if l=="low" else (3. if v=="baseline" else 1.)*(1 if tau=="tau1" else 2)
                outputs[tau,c]={"all":{"mse":x}}
        with patch.object(w,"level_metrics",lambda obj:obj):
            rows=w.seed_rows(31001,outputs)
        def pick(kind,tau,policy):
            r=[x["value"] for x in rows if x["suite"]=="H" and x["kind"]==kind
               and x["variant"]==tau and x["mode"]==policy]
            self.assertEqual(len(r),1)
            return r[0]
        self.assertEqual(pick("high_minus_low","tau2","baseline"),6.)
        self.assertEqual(pick("tau2_minus_tau1","tau2","baseline"),3.)
        self.assertEqual(pick("adaptive_minus_neutral","tau1","baseline"),2.)
        self.assertEqual(pick("adaptive_minus_neutral","tau2","baseline"),4.)
        self.assertEqual(pick("AN_tau2_minus_tau1","tau2","baseline"),2.)

if __name__=="__main__":
    unittest.main()

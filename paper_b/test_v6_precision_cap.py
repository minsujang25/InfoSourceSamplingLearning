"""Deterministic unit checks for the isolated v6 precision intervention."""
import math
import unittest
from types import SimpleNamespace
import model.InfoSourceSamplingLearning as m
from paper_b.experiments.v6_precision_cap import capped_theta_update
from paper_b.experiments import run_v6_robustness as r


class Source:
    def __init__(self,kind,sd=1.):
        self.type_of_agent=kind
        self._message_sd=sd


def citizen(sources,messages,peer_sd=None):
    return SimpleNamespace(model=SimpleNamespace(peer_evidence_mode="source_posterior",tau_social=1.),
        theta_or_delta=0,mu_theta_beliefs=[0.],sd_theta_beliefs=[1.],
        _sample_order=sources,sampled_msgs=messages,_delivered_peer_signal_sds=peer_sd or {},
        _pending_evidence_precision={})


class PrecisionCapTests(unittest.TestCase):
    def test_both_external_sources_capped_and_all_draws_used(self):
        e,j=Source("infoprovider"),Source("disruptivejammer")
        c=citizen([e,j],[[0.]*5+[2.]*5,[4.]*20])
        mu,sd=capped_theta_update(c)
        self.assertEqual(c._pending_evidence_precision,{e:5.,j:5.})
        self.assertAlmostEqual(mu,25./11.)
        self.assertAlmostEqual(sd,math.sqrt(1./11.))

    def test_nonbinding_matches_original(self):
        e,j=Source("infoprovider"),Source("disruptivejammer")
        p=Source("citizen")
        a=citizen([e,j,p],[[1.,2.],[4.]*5,[2.]*10],{p:2.})
        b=citizen([e,j,p],[[1.,2.],[4.]*5,[2.]*10],{p:2.})
        self.assertEqual(capped_theta_update(a),m.Citizen.bayesian_update_theta_from_sources(b))
        self.assertEqual(a._pending_evidence_precision,b._pending_evidence_precision)

    def test_peer_precision_unchanged(self):
        p=Source("citizen")
        c=citizen([p],[[3.]*20],{p:2.})
        mu,sd=capped_theta_update(c)
        self.assertAlmostEqual(c._pending_evidence_precision[p],.2)
        self.assertAlmostEqual(mu,.5)
        self.assertAlmostEqual(sd,math.sqrt(1/1.2))

    def test_empty_delivery_no_update(self):
        e=Source("infoprovider")
        c=citizen([e],[[]])
        self.assertEqual(capped_theta_update(c),(0.,1.))
        self.assertEqual(c._pending_evidence_precision,{})

    def test_audit_not_in_scope(self):
        c=citizen([],[])
        c.theta_or_delta=1
        with self.assertRaises(RuntimeError):
            capped_theta_update(c)

    def test_variant_scope_restores_original(self):
        old=m.Citizen.bayesian_update_theta_from_sources
        original=r.base.config_for
        with r.variant_scope("precision_cap5"):
            self.assertIs(m.Citizen.bayesian_update_theta_from_sources,capped_theta_update)
            self.assertEqual(r.base.config_for(6001,"H","high","high",26)[0]["epsilon"],.05)
        self.assertIs(m.Citizen.bayesian_update_theta_from_sources,old)
        self.assertIs(r.base.config_for,original)
        with r.variant_scope("epsilon20"):
            self.assertEqual(r.base.config_for(6001,"H","high","high",26)[0]["epsilon"],.20)
            self.assertIs(m.Citizen.bayesian_update_theta_from_sources,old)
        self.assertIs(r.base.config_for,original)

    def test_frozen_grid(self):
        self.assertEqual(len(r.cells("H")),8)
        self.assertTrue(all(c[2]=="high" for c in r.cells("H")))
        self.assertEqual(len(r.cells("G")),8)


if __name__=="__main__":
    unittest.main()

"""Small deterministic unit tests; no ABM run required."""
import unittest
from types import SimpleNamespace
from paper_b.experiments.v6_scientific_measurement import (
    belief_rows, groups_for, request_rows, summarize_model,
)


class V6ScientificMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.citizens = [
            SimpleNamespace(pos=2, group_id=1, mu_theta_beliefs=[2.,2.,1.],
                            sd_theta_beliefs=[5.,5.,.5]),
            SimpleNamespace(pos=3, group_id=-1, mu_theta_beliefs=[-2.,-2.,0.],
                            sd_theta_beliefs=[5.,5.,.25]),
        ]
        self.groups=groups_for(self.citizens,{3})

    def edge(self,t,ego,group,src,srcgroup,stype,count):
        return dict(period=t,ego=ego,ego_group=group,source=src,source_group=srcgroup,
                    source_type=stype,realized_reliance=count/20,
                    learning_target="credibility" if t==0 else "theta")

    def history(self):
        return [
            [self.edge(t,2,1,1,1,"disruptivejammer",j),
             self.edge(t,2,1,3,-1,"citizen",20-j),
             self.edge(t,3,-1,0,-1,"infoprovider",20)]
            for t,j in [(0,10),(1,5)]
        ]

    def test_belief_indexing_and_decomposition(self):
        rows=belief_rows(self.citizens,self.groups,[0,1,2],0.)
        pop=[r for r in rows if r["group_type"]=="all"]
        self.assertEqual([r["period"] for r in pop],[-1,0,1])
        self.assertEqual([r["mse_truth"] for r in pop],[4.,4.,.5])
        self.assertEqual(pop[-1]["squared_bias"],.25)
        self.assertEqual(pop[-1]["belief_variance"],.25)

    def test_requests_are_integer_and_audit_split(self):
        rows=request_rows(self.history(),self.groups,[0,1,2],20,{3})
        pop={r["learning_target"]:r for r in rows
             if r["group_type"]=="all" and r["horizon_step"]==2}
        self.assertEqual(pop["all"]["total_requests"],80)
        self.assertEqual(pop["all"]["requests_jammer"],15)
        self.assertEqual(pop["all"]["requests_gateway_peer"],25)
        self.assertEqual(pop["all"]["requests_other_peer"],25)
        self.assertEqual(pop["theta"]["requests_jammer"],5)
        self.assertEqual(pop["credibility"]["requests_jammer"],10)
        self.assertEqual(pop["all"]["requests_expert"],40)
        self.assertEqual(pop["theta"]["citizen_periods"],2)
        zero=next(r for r in rows if r["group_type"]=="all" and r["horizon_step"]==0)
        self.assertIsNone(zero["request_share_jammer"])

    def test_invalid_fraction_rejected(self):
        h=self.history()
        h[0][0]["realized_reliance"]=.123
        with self.assertRaises(RuntimeError):
            request_rows(h,self.groups,[2],20,{3})

    def test_missing_citizen_rejected(self):
        h=self.history()
        h[1].pop()
        with self.assertRaises(RuntimeError):
            request_rows(h,self.groups,[2],20,{3})

    def test_observer_does_not_write_histories(self):
        h=self.history()
        m=SimpleNamespace(citizens=self.citizens,max_steps=2,reliance_history=h,state_of_the_world=0.)
        before=repr(m)
        summarize_model(m,{3},[0,2],20)
        self.assertEqual(repr(m),before)


if __name__=="__main__":
    unittest.main()

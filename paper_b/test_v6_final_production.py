"""Deterministic tests for final production analysis and grid."""
import unittest
from paper_b.experiments import run_v6_final_production as p
from paper_b.experiments import v6_final_production_analysis as a


class FinalProductionTests(unittest.TestCase):
    def test_seed_grid(self):
        self.assertEqual(len(p.SEEDS),500)
        self.assertEqual(p.SEEDS[0],20001)
        self.assertEqual(p.SEEDS[-1],20500)
        self.assertFalse(set(p.SEEDS)&set(p.PREFLIGHT_SEEDS))
        self.assertEqual(p.HORIZON,400)
        self.assertEqual(len(p.base.cells("H")),16)
        self.assertEqual(len(p.base.cells("G")),8)

    def test_peer_only_topology(self):
        citizens=[dict(citizen_pos=i,citizen_group=(-1 if i<50 else 1)) for i in range(100)]
        edges=[dict(period=0,ego=i,source=(i+j)%100,source_type="citizen")
               for i in range(100) for j in (1,99)]
        edges += [dict(period=0,ego=i,source=101,source_type="infoprovider") for i in range(100)]
        outputs=[dict(run=dict(seed=1,suite="H",level=level,structural_fingerprint=level),
                      beliefs=citizens,edges=edges) for level in ("low","high")]
        rows=a.topology_rows(outputs)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["peer_edges"],200)
        self.assertEqual(rows[0]["peer_indegree_mean"],2.)
        self.assertEqual(rows[0]["peer_indegree_sd"],0.)
        self.assertEqual(rows[0]["directed_reciprocity"],1.)
        self.assertAlmostEqual(rows[0]["same_group_peer_share"],.98)
        self.assertAlmostEqual(rows[0]["ei_index"],-.96)

    def test_gap_interaction(self):
        rows=[dict(seed=1,level=h,prior=s,sender_regime=j,reliance_mode=r,horizon_step=400,
                   gap=(2 if h=="high" and s=="high" else 0))
              for h in ("low","high") for s in ("low","high")
              for j in ("null","fixed_biased") for r in ("adaptive","frozen")]
        result=a.gap_contrasts(rows,400)
        self.assertEqual(len(result),12)
        self.assertTrue(all(r["delta_group_gap"]==2 for r in result if r["contrast"]=="H_by_S"))

    def test_gap_absolute(self):
        out={"run":dict(run_id="x",seed=1,suite="H",cell="H_low_S_high",prior="high",level="low",
                        sender_regime="null",reliance_mode="adaptive"),
             "group_belief_checkpoints":[dict(group_type="citizen_group",group="1",horizon_step=400,mean_belief=-2.),
                                        dict(group_type="citizen_group",group="-1",horizon_step=400,mean_belief=1.)]}
        self.assertEqual(a.group_gap_rows([out])[0]["gap"],3.)

    def test_extra_metrics_and_receiver_contrast(self):
        outputs=[]
        for cell,level,prior,sender,receiver in p.base.cells("H"):
            values={k:0. for k in (*a.sci.METRICS,"I_expert","I_same_peer","I_other_peer")}
            values["mse_truth"]=float(level=="high" and receiver=="adaptive")
            values["Q_jammer"]=float(level=="low")
            values["W_jammer"]=float(level=="low")
            group=dict(cell=cell,sender_regime=sender,reliance_mode=receiver,group_type="all",group="all",**values)
            q=dict(group_type="all",group="all",horizon_step=400,learning_target="theta",total_requests=20,
                   **{k:0. for k in a.REQUEST_METRICS})
            q["requests_jammer"]=float(level=="low")
            outputs.append(dict(run=dict(suite="H",seed=1),group_terminal=[group],group_request_checkpoints=[q]))
        rows=a.extended_contrasts(outputs,400)
        self.assertEqual(len(rows),24)
        r=next(r for r in rows if r["contrast"]=="high_minus_low" and r["prior"]=="high"
               and r["sender_regime"]=="fixed_biased" and r["reliance_mode"]=="adaptive")
        self.assertEqual(r["divergence_X"],1)
        r=next(r for r in rows if r["contrast"]=="high_minus_low" and r["prior"]=="high"
               and r["sender_regime"]=="fixed_biased" and r["reliance_mode"]=="adaptive_minus_frozen")
        self.assertEqual(r["delta_mse_truth"],1.)

if __name__=="__main__":
    unittest.main()

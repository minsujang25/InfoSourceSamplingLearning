"""Deterministic bridge estimand tests; no production simulation."""
import unittest
from paper_b.experiments import v6_channel_bridge_analysis as a
from paper_b.experiments import run_v6_channel_bridge as b


class BridgeTests(unittest.TestCase):
    def test_grid(self):
        cells=b.cells()
        self.assertEqual(len(cells),16)
        self.assertEqual({c[0] for c in cells},set(a.CHANNELS))
        self.assertEqual({c[3] for c in cells},{"null","fixed_biased"})

    def test_damage_and_total_mse_are_distinct(self):
        r=a.channel_contrasts(7501,"frozen","fixed_biased",400,
                             [0.,1.,2.,0.],[4.,6.,5.,4.5],"new_bridge")
        self.assertEqual(r["D_L_minus_S"],1.5)
        self.assertEqual(r["sender_MSE_L_minus_S"],-.5)
        self.assertEqual(r["null_MSE_L_minus_S"],-2.)
        self.assertEqual(r["D_L_minus_E"],.5)
        self.assertEqual(r["D_M_minus_E"]+r["D_S_minus_M"]+r["D_L_minus_S"],.5)

    def test_sender_and_receiver_interactions(self):
        rows=[]
        for receiver,x in (("adaptive",2.),("frozen",1.)):
            for sender,y in (("fixed_biased",0.),("adaptive",3.),("credibility_aware",1.)):
                rows.append(a.channel_contrasts(7501,receiver,sender,400,
                             [0.,0.,0.,0.],[1.,1.,1.,1.+x+y],"test"))
        r=a.interaction_rows(rows)
        self.assertEqual(len(r),7)
        for x in r:
            if x["dimension"]=="receiver":
                self.assertEqual(x["D_L_minus_E"],1.)
            elif x["sender_regime"]=="adaptive":
                self.assertEqual(x["D_L_minus_E"],3.)
            else:
                self.assertEqual(x["D_L_minus_E"],1.)

    def test_bootstrap_signs_and_pairing(self):
        rows=[a.channel_contrasts(s,"adaptive","fixed_biased",400,
                                  [0.,0.,0.,0.],[1.,1.,1.,1.+x],"new_bridge")
              for s,x in ((7501,1.),(7502,-1.))]
        summary=a.summarize(rows,[7501,7502],("sender_regime","reliance_mode","horizon_step"))
        r=next(r for r in summary if r["metric"]=="D_L_minus_E")
        self.assertEqual(r["mean"],0.)
        self.assertEqual(r["mcse"],1.)
        self.assertEqual(r["positive_fraction"],.5)
        with self.assertRaises(RuntimeError):
            a.summarize(rows,[7501,7503],("sender_regime","reliance_mode","horizon_step"))

    def test_archive_constants(self):
        self.assertEqual(len(a.SEEDS),100)
        self.assertEqual(a.SEEDS[0],7501)
        self.assertEqual(a.SEEDS[-1],7600)
        self.assertEqual(len(a.STEPS),13)


if __name__=="__main__":
    unittest.main()

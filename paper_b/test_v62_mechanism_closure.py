"""Fast V6.2 gates; runtime checks follow in the master runner."""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import model.InfoSourceSamplingLearning as m
from paper_b.experiments import v62_policies as p, v62_reporting as r
from paper_b.experiments import v62_observer as o
from paper_b.experiments import run_v62_mechanism_closure as runner
from paper_b.experiments.v6_behavioral_rules import AUDIT_P, uniform_tail_rank_probabilities

class ProtocolTests(unittest.TestCase):
    def test_grid(self):
        grid=p.cells()
        self.assertEqual(len(grid),34)
        self.assertEqual(len(set(grid)),34)
        self.assertEqual(len(p.SEEDS),100)
        self.assertEqual(len(grid)*len(p.SEEDS),3400)
        self.assertEqual(sum(runner.parity_required("production",seed,c)
                             for seed in p.SEEDS for c in grid),55)
        self.assertEqual(sum(c[0]=="baseline" and c[1]=="H" for c in grid)*len(p.SEEDS),400)
        self.assertEqual(sum(c[1]=="H" for c in grid),20)
        self.assertEqual(sum(c[1]=="G" for c in grid),14)

    def test_uniform_and_intermediate(self):
        for n in (3,4):
            np.testing.assert_array_equal(p.uniform_probabilities(n,.05),np.full(n,1/n))
        np.testing.assert_allclose(m.recursive_rank_probabilities(4,.35),[.65,.2275,.079625,.042875])
        np.testing.assert_allclose(uniform_tail_rank_probabilities(4,.05),[.95,.05/3,.05/3,.05/3])
        self.assertAlmostEqual(1+399*AUDIT_P,math.fsum(1/i for i in range(1,401)),places=12)

    def test_neutral_order(self):
        sources=[SimpleNamespace(pos=i) for i in (7,0,3,1)]
        c=SimpleNamespace(info_source=sources,_behavioral_ranking=lambda:list(reversed(sources)))
        def sample(x):
            self.assertEqual([a.pos for a in x._behavioral_ranking()],[0,1,3,7])
        with patch.object(p,"ORIGINAL_SAMPLE",sample):
            p.neutral_sample(c)
        self.assertEqual([a.pos for a in c._behavioral_ranking()],[1,3,0,7])

    def test_scope_restores(self):
        orig=m.recursive_rank_probabilities
        with p.scope("neutral"):
            np.testing.assert_array_equal(m.recursive_rank_probabilities(4,.05),[.25]*4)
        self.assertIs(m.recursive_rank_probabilities,orig)

    def test_signed_accounting(self):
        parts=dict(prior=2.,expert=0.,jammer=0.,peer=0.)
        increments=dict(expert=-1.,jammer=4.,peer=1.)
        x=o.account_step(parts,2.,increments,6.,6.)
        self.assertAlmostEqual(sum(x.values())/6,1.)
        clipped=o.account_step(parts,2.,increments,6.,3.)
        self.assertAlmostEqual(sum(clipped.values())/3,1.)
        self.assertAlmostEqual(clipped["prior"],1.)
        self.assertAlmostEqual(sum(v/3 for v in clipped.values()),1.)

    def test_sequence_primitives(self):
        self.assertTrue(o.drift_condition(2,3,.5,0))
        self.assertTrue(o.drift_condition(6,5,-.5,0))
        self.assertFalse(o.drift_condition(2,3,.5,1))
        self.assertFalse(o.drift_condition(2,1,.5,0))
        self.assertFalse(o.drift_condition(2,3,-.5,0))
        self.assertEqual([o.window(i) for i in (1,10,11,100,101,399)],
                         ["early","early","middle","middle","late","late"])

    def test_categories(self):
        self.assertEqual(r.sign_category(-1,1),"less_Q_higher_error")
        self.assertEqual(r.sign_category(1,-1),"more_Q_lower_error")
        self.assertEqual(r.sign_category(0,1),"zero")
        self.assertIsNone(r.ratio(0,0))
        self.assertEqual(r.ratio(-2,4),-.5)

    def test_seed_summary(self):
        rows=[dict(seed=s,kind="high_minus_low",variant="baseline",suite="H",
                   mode="adaptive",group="group_1",metric="mse",value=v)
              for s,v in [(1,1.),(2,3.),(3,None)]]
        out=r.summarize(rows,[1,2,3])[0]
        self.assertEqual(out["n_valid"],2)
        self.assertEqual(out["n_expected"],3)
        self.assertEqual(out["mean"],2.)
        self.assertEqual(out["positive"],1.)

    def test_packaging(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            paths=[root/f"part_{i}.bin" for i in range(3)]
            for i,f in enumerate(paths): f.write_bytes(bytes([i])*300000)
            with patch.object(r,"LIMIT",1024**2):
                idx=r.package(root,paths,"test","id")
                self.assertEqual(len(idx["parts"]),3)
                self.assertEqual(r.verify_package(root/"shareable"/"test"),3)
                f=root/"shareable"/"test"/idx["parts"][0]["file"]
                with f.open("ab") as h: h.write(b"bad")
                with self.assertRaises(RuntimeError):
                    r.verify_package(root/"shareable"/"test")

    def test_replay_missingness(self):
        source=dict(beliefs=[dict(terminal_mu_theta=1.)],
                    group_terminal=[dict(mse_truth=1.)],
                    group_belief_checkpoints=[dict(mse_truth=1.)],
                    channel_citizens=[dict(Q_jammer=2.,I_corrective_crosscut_any=math.nan),
                                      dict(Q_jammer=3.,I_corrective_crosscut_any=.5)])
        normalized=r.replay_tables(source)
        self.assertIsNone(normalized["channel_citizens"][0]["I_corrective_crosscut_any"])
        self.assertEqual(normalized["channel_citizens"][1]["I_corrective_crosscut_any"],.5)
        self.assertEqual(normalized["replay_missingness"],{"channel_citizens.I_corrective_crosscut_any":1})
        self.assertTrue(math.isnan(source["channel_citizens"][0]["I_corrective_crosscut_any"]))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"replay.json.gz"
            r.write_gz(path,normalized)
            self.assertEqual(r.read_gz(path),normalized)
        source["channel_citizens"][0]["Q_jammer"]=math.nan
        with self.assertRaises(ValueError):
            r.replay_tables(source)
        source["channel_citizens"][0]["Q_jammer"]=2.
        source["channel_citizens"][0]["I_corrective_crosscut_any"]=math.inf
        with self.assertRaises(ValueError):
            r.replay_tables(source)
        source["channel_citizens"][0]["I_corrective_crosscut_any"]=math.nan
        source["beliefs"][0]["terminal_mu_theta"]=math.nan
        with self.assertRaises(ValueError):
            r.replay_tables(source)

    def test_resume_requires_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/"preflight").mkdir()
            cell=p.cells()[0]
            path,_,_=runner.paths(root,"preflight",31001,cell)
            self.assertIsNone(runner.saved(root,"preflight",31001,cell,26,"id"))
            path.touch()
            with self.assertRaises(RuntimeError):
                runner.saved(root,"preflight",31001,cell,26,"id")

if __name__=="__main__":
    unittest.main()

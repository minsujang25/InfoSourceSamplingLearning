"""Read-only compact-archive audit; no model imports or simulation runs."""
import argparse, gzip, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

SPECS = {
    'study1': ('e86f6ff3be12', 20001, 500, '0fd01381ddc86f6828fda5aab2e7b3a9d83b358508bb901d21acae76ecf27ce0'),
    'pilot': ('85faac503d4a', 6001, 20, '084f1558b9f5c360861c75e1889467be9439b9bc84505d0f4a9fe3ce577925a3'),
}
KEY = ['seed','suite','cell','prior','sender_regime','reliance_mode']
METRICS = ['mse_truth','squared_bias','belief_variance','mean_posterior_sd']

def read(z, name, **kwargs):
    with gzip.GzipFile(fileobj=z.open(name)) as f:
        return pd.read_csv(f, keep_default_na=False, **kwargs)

def inspect(path, cohort, out):
    did, first, n, expected_hash = SPECS[cohort]
    with open(path,'rb') as f: sha = hashlib.file_digest(f,'sha256').hexdigest()
    assert sha == expected_hash, 'Archive SHA mismatch'
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        m = json.loads(z.read('v6_manifest.json')); gate = json.loads(z.read('v6_gate.json'))
        assert m['design_id'] == gate['design_id'] == did and m['seeds'] == list(range(first,first+n))
        assert gate['pass'] and m['horizon'] == 400 and m['N'] == 100
        runs = read(z,'runs.csv.gz',usecols=KEY+['design_id','run_id','level','terminal_horizon_T','steps_run','n_citizens'])
        expected = {(s,suite,c[0],c[2],c[3],c[4]) for s in m['seeds'] for suite,cells in m['cells'].items() for c in cells}
        assert len(runs)==len(expected)==gate['grid_runs'] and set(runs[KEY].itertuples(index=False,name=None))==expected
        assert runs.run_id.is_unique and runs.design_id.eq(did).all() and runs.terminal_horizon_T.eq(400).all() and runs.steps_run.eq(400).all() and runs.n_citizens.eq(100).all()
        g = read(z,'group_belief_checkpoints.csv.gz'); t = read(z,'group_terminal.csv.gz')
        assert len(g)==gate['row_counts']['group_belief_checkpoints'] and len(t)==gate['row_counts']['group_terminal']
        ids=['run_id','group_type','group']; assert not g.duplicated(ids+['horizon_step']).any() and not t.duplicated(ids).any()
        assert set(g.run_id)==set(t.run_id)==set(runs.run_id)
        assert g.groupby(ids).size().eq(14).all() and set(g.horizon_step)=={0,1,2,3,6,11,26,51,100,101,151,200,300,400}
        assert g.period.eq(g.horizon_step-1).all()
        end=g[g.horizon_step==400].merge(t,on=ids,validate='one_to_one',suffixes=('_path','_terminal'))
        assert len(end)==len(t)
        errors={v:float(np.max(np.abs(end[v+'_path']-end[v+'_terminal']))) for v in ['mse_truth','squared_bias','belief_variance','mean_belief']}
        assert max(errors.values())<1e-10
        assert np.max(np.abs(g.mse_truth-g.squared_bias-g.belief_variance))<1e-10
        # Focal H high-segregation, fixed-biased; both receiver policies, all groups.
        h=g[(g.suite=='H')&(g.prior=='high')&(g.sender_regime=='fixed_biased')].copy()
        h['level']=h.cell.str.split('_').str[1]
        assert h.n_citizens.eq(np.where(h.group=='all',100,50)).all()
        # G absolute active MSE/SD by population and role; no ANCILLARY prior-group strata.
        gg=g[(g.suite=='G')&(g.sender_regime=='fixed_biased')&g.group_type.isin(['all','gateway_role'])].copy()
        gg['level']=gg.cell.str.split('_').str[1]
        focal=pd.concat([h,gg]); focal.to_csv(out/f'{cohort}_focal_levels.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        # Shared bootstrap indices across contrasts, seed is independent unit.
        bidx=np.random.default_rng(20261125).integers(n,size=(10000,n))
        summaries=[]; paired=[]
        for (suite,mode,group,step), q in focal.groupby(['suite','reliance_mode','group','horizon_step']):
            low=q[q.level=='low'].set_index('seed').sort_index(); high=q[q.level=='high'].set_index('seed').sort_index()
            assert low.index.tolist()==high.index.tolist()==m['seeds']
            for metric in METRICS:
                a=low[metric].to_numpy(float); b=high[metric].to_numpy(float); d=b-a
                vals=d[bidx].mean(axis=1); lo,hi=np.quantile(vals,[.025,.975])
                key=dict(cohort=cohort,suite=suite,mode=mode,group=group,horizon_step=int(step),metric=metric)
                summaries.append(dict(**key,n_seeds=n,low_mean=a.mean(),high_mean=b.mean(),delta=d.mean(),lo=lo,hi=hi))
                paired.extend(dict(**key,seed=int(s),low=x,high=y,delta=dd) for s,x,y,dd in zip(low.index,a,b,d))
        pd.DataFrame(summaries).to_csv(out/f'{cohort}_checkpoint_summary.csv',index=False)
        pd.DataFrame(paired).to_csv(out/f'{cohort}_checkpoint_pairs.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        subset=runs[(runs.suite=='G')|((runs.suite=='H')&(runs.prior=='high'))]
        return dict(cohort=cohort,sha256=sha,bytes=path.stat().st_size,design_id=did,zip_members=len(z.namelist()),runs=len(runs),checkpoint_rows=len(g),terminal_rows=len(t),terminal_max_abs_differences=errors,high_segregation_H_and_G_runs=len(subset),excluded_low_segregation_H_runs=len(runs)-len(subset),bootstrap_seed=20261125,bootstrap_draws=10000)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study1',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    result=[inspect(a.study1,'study1',a.out),inspect(a.pilot,'pilot',a.out)]
    (a.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()

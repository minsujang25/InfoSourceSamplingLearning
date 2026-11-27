"""Re-estimate W2 from its exact archived seed table transfer; no simulator."""
import argparse, gzip, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['kind','variant','suite','mode','group','metric']
SHA='b3be02e7cb942c5f20fa71fc4239cd08f33914fa47339b9b411824a8ff77a14e'

def main(source,reference,out):
    assert hashlib.sha256(source.read_bytes()).hexdigest()==SHA
    with zipfile.ZipFile(source) as z:
        m=json.loads(z.read('MANIFEST.json')); p=json.loads(z.read('PART_MANIFEST.json'))
        assert p['design_id']==m['design_id']=='b8f139d546be' and len(p['members'])==105
        assert len(z.namelist())==106 and set(z.namelist())=={'PART_MANIFEST.json'}|{r['path'] for r in p['members']}
        for r in p['members']:
            b=z.read(r['path']); assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
        assert m['seeds']==list(range(31001,31101)) and m['tau_social']==2 and m['reference_tau']==1 and m['expected_runs']==800 and m['horizon']==400
        rows=[]
        for s in m['seeds']:
            d=json.loads(gzip.decompress(z.read(f'seed_tables/seed_{s}.json.gz')))
            assert len(d)==5304 and all(r['seed']==s for r in d);rows.extend(d)
        archived=pd.read_csv(z.open('contrast_summary.csv.gz'),compression='gzip',float_precision='round_trip')
    a=pd.DataFrame(rows).pivot(index='seed',columns=KEY,values='value').sort_index().sort_index(axis=1)
    assert a.shape==(100,5304) and a.index.tolist()==m['seeds']
    n_diff=0; max_diff=0.
    for key in a.columns:
        kind,t,s,p,g,x=key
        def get(k,t=t,p=p):return a[(k,t,s,p,g,x)].to_numpy()
        if kind=='high_minus_low':pred=get('level_high')-get('level_low')
        elif kind=='tau2_minus_tau1':pred=get('high_minus_low')-get('high_minus_low','tau1')
        elif kind=='adaptive_minus_neutral':pred=get('high_minus_low',p='baseline')-get('high_minus_low',p='neutral')
        elif kind=='AN_tau2_minus_tau1':pred=get('adaptive_minus_neutral')-get('adaptive_minus_neutral','tau1')
        else:continue
        actual=a[key].to_numpy();np.testing.assert_allclose(actual,pred,rtol=1e-12,atol=1e-12,equal_nan=True)
        n_diff+=100; ok=np.isfinite(actual-pred);max_diff=max(max_diff,float(np.max(abs(actual[ok]-pred[ok]))) if ok.any() else 0.)
    draws=np.random.default_rng(m['bootstrap_seed']).integers(0,100,(m['bootstrap_draws'],100)); counts=np.zeros((len(draws),100));np.add.at(counts,(np.arange(len(draws))[:,None],draws),1)
    x=a.to_numpy();results=[]
    for i in range(0,x.shape[1],128):
        y=x[:,i:i+128]; valid=np.isfinite(y);n=counts@np.where(valid,y,0.);den=counts@valid.astype(float);bs=np.divide(n,den,out=np.full_like(n,np.nan),where=den>0)
        for j,k in enumerate(a.columns[i:i+128]):
            u=y[valid[:,j],j];b=bs[np.isfinite(bs[:,j]),j]
            results.append(dict(zip(KEY,k),n_valid=len(u),mean=u.mean() if len(u) else np.nan,lo=np.quantile(b,.025) if len(b) else np.nan,hi=np.quantile(b,.975) if len(b) else np.nan,mcse=u.std(ddof=1)/np.sqrt(len(u)) if len(u)>1 else np.nan,positive=np.mean(u>0) if len(u) else np.nan,negative=np.mean(u<0) if len(u) else np.nan,zero=np.mean(u==0) if len(u) else np.nan))
    got=pd.DataFrame(results).sort_values(KEY).reset_index(drop=True); checks={}
    for name,ref in [('transfer',archived),('legacy',pd.read_csv(reference,float_precision='round_trip'))]:
        ref=ref.sort_values(KEY).reset_index(drop=True);assert got[KEY].equals(ref[KEY]);errs={}
        for col in ['n_valid','mean','lo','hi','mcse','positive','negative','zero']:
            u,v=got[col].to_numpy(),ref[col].to_numpy();np.testing.assert_allclose(u,v,rtol=1e-12,atol=1e-10,equal_nan=True);ok=np.isfinite(u)&np.isfinite(v);errs[col]=float(np.max(abs(u[ok]-v[ok]))) if ok.any() else 0.
        checks[name]=errs
    out.mkdir(parents=True,exist_ok=False);got.to_csv(out/'recomputed_summary.csv.gz',index=False,compression={'method':'gzip','mtime':0}); report=dict(status='PASS',transfer_sha256=SHA,member_checks=105,seed_values=len(rows),summary_keys=len(got),derived_identity_values=n_diff,max_identity_residual=max_diff,bootstrap_seed=m['bootstrap_seed'],bootstrap_draws=m['bootstrap_draws'],comparisons=checks,new_model_executions=0,limitations='Seed-level W2 regeneration; no raw W2 trajectories or model execution.')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('reference',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.source,a.reference,a.out)

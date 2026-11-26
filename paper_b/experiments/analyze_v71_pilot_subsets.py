"""V7.1 finite-archive diagnostic of 20-seed percentile intervals; not true coverage."""
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def main(source, out):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as z:
        m=json.loads(z.read('MANIFEST.json'))
        b=z.read('beliefs.csv.gz')
        f=next(f for f in m['members'] if f['name']=='beliefs.csv.gz')
        assert hashlib.sha256(b).hexdigest()==f['sha256'] and len(b)==f['bytes']
        d=pd.read_csv(io.BytesIO(b),compression='gzip',float_precision='round_trip')
        assert len(d)==f['rows'] and list(d)==f['columns']
    d=d[(d.suite=='H')&(d.prior=='high')&(d.sender_regime=='fixed_biased')&(d.citizen_group==1)].copy()
    assert not d.duplicated(['seed','reliance_mode','level','citizen_pos']).any()
    assert d.groupby(['seed','reliance_mode','level']).size().eq(50).all()
    d['mse']=d.terminal_mu_theta**2
    x=d.groupby(['seed','reliance_mode','level']).mse.mean().unstack('level')
    x['contrast']=x.high-x.low
    rows=[]
    rng=np.random.default_rng(20261126)
    for mode,g in x.groupby('reliance_mode'):
        g=g.reset_index().sort_values('seed')
        assert g.seed.tolist()==list(range(20001,20501))
        v=g.contrast.to_numpy()
        ref=v.mean()
        means=[]; hits=[]; widths=[]
        for _ in range(2000):
            sub=rng.choice(v,20,replace=False)
            lo,hi=np.quantile(sub[rng.integers(0,20,(2000,20))].mean(axis=1),[.025,.975])
            means.append(sub.mean()); hits.append(lo<=ref<=hi); widths.append(hi-lo)
        lo,hi=np.quantile(means,[.025,.975])
        rows.append(dict(mode=mode,reference_500=ref,subset_mean_lo=lo,subset_mean_hi=hi,reference_in_interval=np.mean(hits),mean_interval_width=np.mean(widths)))
    x.reset_index().to_csv(out/'study1_H_aligned_seed.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'pilot_subset_diagnostic.csv',index=False)
    (out/'pilot_validation.json').write_text(json.dumps(dict(status='PASS',input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),subsets=2000,subset_size=20,bootstrap_resamples=2000,rng_seed=20261126,new_simulations=0,interpretation='Coverage of fixed 500-seed archive mean, not true population coverage or pilot-variant verification.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('source',type=Path); p.add_argument('out',type=Path)
    a=p.parse_args(); main(a.source,a.out)

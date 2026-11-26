"""V7.1 passive diagnostics from frozen Study 2 transfers. No simulator imports."""
import argparse, gzip, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

S = list(range(31001, 31101))
K = ['seed','suite','level','mode','citizen_pos']
P = {'recursive': np.array([.95,.0475,.002375,.000125]),
     'uniform_tail': np.array([.95,.05/3,.05/3,.05/3])}
BOOT = np.random.default_rng(20261125).integers(0,100,(10000,100))

def main(transfer, replay, out):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(transfer) as z:
        m = json.loads(z.read('TRANSFER_V652B.json'))
        assert m['design_id']=='7e2e4f6724e3' and m['source_commit']=='55264824e35c02284174e206d86955269d2b0bbc'
        assert m['seeds']==S and m['runs']==400 and m['floor_events']==0
        frames = {}
        for f in m['files']:
            b=z.read(f['name'])
            assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256']
            d=pd.read_csv(io.BytesIO(b),compression='gzip',float_precision='round_trip')
            assert len(d)==f['rows'] and list(d)==f['fields']
            frames[f['name']]=d
    a=frames['baseline_HG_initial_audit_all_sources.csv.gz']
    c=frames['baseline_HG_checkpoints.csv.gz']
    w=frames['baseline_HG_peer_type_windows.csv.gz']
    assert not a.duplicated(K+['source_pos']).any() and not c.duplicated(K+['h']).any()
    assert not w.duplicated(K+['window','source_class']).any()
    assert a.initial_sd.eq(5).all() and a.variant.eq('baseline').all() and a['mode'].eq('adaptive').all()
    labels=a[K+['group','role']].drop_duplicates()
    assert len(labels)==40000 and not labels.duplicated(K).any()
    h=a[a.suite.eq('H')].copy()
    assert len(h)==80000 and h.groupby(K).rank_post.apply(lambda x: sorted(x)==[1,2,3,4]).all()
    assert h.groupby(['seed','level','group']).size().eq(200).all()
    h['type']=h.source_class.where(~h.source_class.str.endswith('_peer'),'peer')
    # Conditional on a substantive period, apply both policies to the same audit ranks.
    rank=h.groupby(['seed','level','group','rank_post','source_class']).size().div(50).rename('share').reset_index()
    rank.to_csv(out/'rank_seed.csv',index=False)
    parts=[]
    for policy,p in P.items():
        assert np.isclose(p.sum(),1)
        x=h.copy()
        x['p']=p[x.rank_post.to_numpy(int)-1]
        x['requests']=20*x.p
        x['inclusion']=1-(1-x.p)**20
        x['Q']=np.where(x.type.eq('peer'),x.inclusion/26,x.requests)
        assert np.allclose(x.groupby(K).requests.sum(),20)
        for cls in ['expert','jammer','same_peer','other_peer','peer','all']:
            y=x if cls=='all' else x[x.type.eq('peer') if cls=='peer' else x.source_class.eq(cls)]
            v=y.groupby(['seed','level','group'])[['requests','inclusion','Q']].sum().div(50)
            v=v.reindex(pd.MultiIndex.from_product([S,['low','high'],[-1,1]],names=['seed','level','group']),fill_value=0).reset_index()
            v['policy']=policy; v['source_class']=cls; parts.append(v)
    q=pd.concat(parts,ignore_index=True)
    q.to_csv(out/'tail_seed.csv',index=False)
    # Sender-state potential precision if consulted at the next period, not realized Q.
    c=c.merge(labels,on=K,validate='many_to_one')
    assert c.sd.gt(0).all()
    c['potential_Q']=1/(c.sd**2+1)
    c['stratum']=np.where(c.suite.eq('H'),np.where(c.group.eq(1),'aligned','opposite'),c.role)
    cs=c.groupby(['seed','suite','level','stratum','h'])[['sd','potential_Q']].mean().reset_index()
    cs.to_csv(out/'checkpoint_seed.csv',index=False)
    w=w[w.source_class.str.endswith('_peer')].merge(labels,on=K,validate='many_to_one')
    w['stratum']=np.where(w.suite.eq('H'),np.where(w.group.eq(1),'aligned','opposite'),w.role)
    ws=w.groupby(['seed','suite','level','stratum','window'])[['Q','observations']].sum().reset_index()
    assert ws.observations.gt(0).all()
    ws['Q_per_observation']=ws.Q/ws.observations
    ws.to_csv(out/'peer_weight_seed.csv',index=False)
    # Independent history-derived period-1 audit markers (H adaptive and frozen).
    with zipfile.ZipFile(replay) as z:
        r=pd.DataFrame(json.loads(gzip.decompress(z.read('replayed_citizens.json.gz'))))
    assert len(r)==40000 and r.early_audit.isin([0,1]).all()
    ids=['seed','mode','citizen_pos']
    l=r[r.level.eq('low')]; u=r[r.level.eq('high')]
    j=l.merge(u,on=ids,suffixes=('_low','_high'),validate='one_to_one')
    assert len(j)==20000 and j.group_low.eq(j.group_high).all() and j.initial_mu_low.eq(j.initial_mu_high).all()
    j['low_learner']=1-j.early_audit_low; j['high_learner']=1-j.early_audit_high
    j['both']=j.low_learner*j.high_learner; j['mismatch']=abs(j.low_learner-j.high_learner)
    ls=j.groupby(['seed','mode','group_low'])[['low_learner','high_learner','both','mismatch']].sum().reset_index()
    ls.to_csv(out/'learner_identity_seed.csv',index=False)
    # Validate adaptive history markers using checkpoint precision change.
    cc=c[(c.suite=='H')&(c.h==2)].merge(r[r['mode']=='adaptive'][K[:1]+['level','mode','citizen_pos','early_audit']],on=['seed','level','mode','citizen_pos'],validate='one_to_one')
    assert len(cc)==20000 and np.array_equal(cc.sd.lt(5).to_numpy(),cc.early_audit.eq(0).to_numpy())
    def summarize(d,keys,metrics,name):
        rows=[]
        for key,g in d.groupby(keys):
            assert sorted(g.seed)==S
            g=g.sort_values('seed')
            for metric in metrics:
                x=g[metric].to_numpy(float)
                lo,hi=np.quantile(x[BOOT].mean(axis=1),[.025,.975])
                rows.append(dict(zip(keys,key if isinstance(key,tuple) else [key]),metric=metric,mean=x.mean(),lo=lo,hi=hi))
        pd.DataFrame(rows).to_csv(out/name,index=False)
    summarize(q,['policy','level','group','source_class'],['requests','inclusion','Q'],'tail_levels.csv')
    wide=q.pivot(index=['seed','group','source_class'],columns=['policy','level'],values=['requests','inclusion','Q'])
    contrasts=[]
    for metric in ['requests','inclusion','Q']:
        for policy in P:
            v=(wide[metric,policy,'high']-wide[metric,policy,'low']).rename('value').reset_index()
            v['metric_name']=metric; v['contrast']=policy+'_high_minus_low'; contrasts.append(v)
        v=((wide[metric,'uniform_tail','high']-wide[metric,'uniform_tail','low'])-(wide[metric,'recursive','high']-wide[metric,'recursive','low'])).rename('value').reset_index()
        v['metric_name']=metric; v['contrast']='tail_minus_recursive_interaction'; contrasts.append(v)
    contrasts=pd.concat(contrasts,ignore_index=True)
    contrasts.to_csv(out/'tail_contrast_seed.csv',index=False)
    summarize(contrasts,['group','source_class','metric_name','contrast'],['value'],'tail_contrasts.csv')
    summarize(cs,['suite','level','stratum','h'],['sd','potential_Q'],'checkpoints.csv')
    summarize(ws,['suite','level','stratum','window'],['Q_per_observation'],'peer_weight.csv')
    validation=dict(status='PASS',new_simulations=0,source_commit=m['source_commit'],transfer_sha256=hashlib.sha256(transfer.read_bytes()).hexdigest(),replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),audit_rows=len(a),checkpoint_rows=len(c),replay_rows=len(r),learner_mismatch_by_mode=j.groupby('mode').mismatch.sum().to_dict(),adaptive_marker_checkpoint_agreement=len(cc),bootstrap_resamples=10000,bootstrap_seed=20261125)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('transfer',type=Path); p.add_argument('replay',type=Path); p.add_argument('out',type=Path)
    a=p.parse_args(); main(a.transfer,a.replay,a.out)

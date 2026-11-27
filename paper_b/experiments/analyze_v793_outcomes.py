"""Passive analysis of archived Study 2 outcomes. No simulation or causal rank effects."""
import argparse, gzip, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
SEEDS=list(range(31001,31101)); B=10000; BOOT=20261127

def run(inputs, companions, transfer, out):
    out.mkdir(parents=True,exist_ok=False)
    z=zipfile.ZipFile(inputs); records=[]; digests={}
    for name in ['baseline_H_individuals.json(1).gz','baseline_G_adaptive_terminal.json(1).gz','baseline_G_frozen_terminal.json(1).gz']:
        data=z.read(next(n for n in z.namelist() if n.endswith('/'+name)))
        digests[name]=hashlib.sha256(data).hexdigest(); records.extend(json.loads(gzip.decompress(data)))
    d=pd.DataFrame(records); assert len(d)==80000 and not d.duplicated(['suite','level','mode','seed','citizen_pos']).any()
    assert sorted(d.seed.unique())==SEEDS and (d.groupby(['suite','mode','level','seed']).size()==100).all()
    assert (d.variant=='baseline').all(); np.testing.assert_allclose(d.mu**2,d.terminal_squared_error,atol=1e-10,rtol=1e-10)
    d['side']=(d.mu>2).astype(int);d['near']=(abs(d.mu)<=.5).astype(int)
    rng=np.random.default_rng(BOOT); draws=rng.integers(0,100,(B,100)); weights=np.zeros((B,100));np.add.at(weights,(np.arange(B)[:,None],draws),1)
    def ci(a,b=None):
        if b is None: sim=weights@a/100; est=a.mean()
        else: sim=np.divide(weights@a,weights@b,out=np.full(B,np.nan),where=weights@b>0);est=a.sum()/b.sum()
        assert np.isfinite(sim).all(); return [float(est),*map(float,np.quantile(sim,[.025,.975]))]
    rank=[]; decomps=[]; rank_seed=[]; scopes=[]
    for mode in ['adaptive','frozen']:
        x=d[(d.suite=='H')&(d.group==1)&(d['mode']==mode)];assert (x.groupby(['seed','level']).size()==50).all();arrays={}
        for lev in ['low','high']:
            y=x[x.level==lev]; N=[]; J=[]
            for r in range(1,5):
                g=y[y.initial_rank==r].groupby('seed').side.agg(['size','sum']).reindex(SEEDS,fill_value=0)
                n=g['size'].to_numpy();j=g['sum'].to_numpy();N.append(n);J.append(j)
                rank_seed.extend(dict(seed=s,mode=mode,level=lev,rank=r,n=int(a),side=int(b)) for s,a,b in zip(SEEDS,n,j))
                rank.append(dict(mode=mode,level=lev,rank=r,n=int(n.sum()),side=int(j.sum()),**dict(zip(['mean','lo','hi'],ci(j,n)))))
            arrays[lev]=(np.stack(N,axis=1),np.stack(J,axis=1))
        def calc(w):
            counts=[(w@arrays[l][0],w@arrays[l][1]) for l in ['low','high']]; p0,p1=[n/n.sum(axis=1,keepdims=True) for n,j in counts];c0,c1=[j/n for n,j in counts]; comp=((p1-p0)*(c1+c0)/2).sum(axis=1);within=((c1-c0)*(p1+p0)/2).sum(axis=1); net=(p1*c1-p0*c0).sum(axis=1);np.testing.assert_allclose(comp+within,net,atol=1e-12);return np.stack([comp,within,net],axis=1)
        point=calc(np.ones((1,100)))[0];bs=calc(weights);decomps.append(dict(mode=mode,**{k:dict(zip(['mean','lo','hi'],[float(point[i]),*map(float,np.quantile(bs[:,i],[.025,.975]))])) for i,k in enumerate(['composition','within','net'])}))
    data=[]; seen=set(); input_members={}
    with zipfile.ZipFile(companions[0]) as zz:
        mf=next(n for n in zz.namelist() if n.endswith('/SHA256SUMS.txt'));root=mf.rsplit('/',1)[0]+'/'
        hashes={v.split(maxsplit=1)[1].strip():v.split(maxsplit=1)[0] for v in zz.read(mf).decode().splitlines() if v.strip()}
    for p in companions:
        with zipfile.ZipFile(p) as zz:
            for n in sorted(zz.namelist()):
                if '/seed_tables/' not in n or not n.endswith('.json.gz'):continue
                raw=zz.read(n);h=hashlib.sha256(raw).hexdigest();assert h==hashes[n[len(root):]];input_members[n]=h
                rows=json.loads(gzip.decompress(raw));s=rows[0]['seed'];assert s not in seen and len(rows)==136524 and all(r['seed']==s for r in rows);seen.add(s)
                data.extend(r for r in rows if r['kind'] in ['level_low','level_high'] and r['metric'] in ['mse','jammer_side','near_truth'])
    assert sorted(seen)==SEEDS
    # W2 has a separate exact transfer, with matched tau=1 values.
    w2=[]
    with zipfile.ZipFile(transfer) as zz:
        raw=zz.read(next(n for n in zz.namelist() if n.endswith('v62_b8f139d546be_w2_primary_001_of_001.zip')))
    assert hashlib.sha256(raw).hexdigest()=='b3be02e7cb942c5f20fa71fc4239cd08f33914fa47339b9b411824a8ff77a14e'
    with zipfile.ZipFile(io.BytesIO(raw)) as zz:
        members=json.loads(zz.read('PART_MANIFEST.json'))['members'];assert len(members)==105
        for r in members:
            b=zz.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
        for seed in SEEDS:
            rows=json.loads(gzip.decompress(zz.read(f'seed_tables/seed_{seed}.json.gz')));assert len(rows)==5304 and all(r['seed']==seed for r in rows)
            for r in rows:
                if r['kind'] not in ['level_low','level_high'] or r['metric'] not in ['mse','jammer_side','near_truth']:continue
                w2.append(r.copy())
    a=pd.DataFrame(data);assert not a.duplicated(['seed','kind','variant','suite','mode','group','metric']).any()
    w=pd.DataFrame(w2); assert set(w['variant'])=={'tau1','tau2'} and set(w['mode'])=={'baseline','neutral'}
    w['policy']=w['mode']; w['mode']='adaptive'
    k=['seed','kind','suite','mode','group','metric','variant']
    w1=w[w.variant=='tau1'].copy();w1['variant']=w1.policy
    j=w1.merge(a,on=k,suffixes=('_w2','_study2'),validate='one_to_one');assert len(j)==len(w1)
    np.testing.assert_allclose(j.value_w2,j.value_study2,atol=1e-10,rtol=1e-10)
    w=w[w.variant=='tau2'].copy();w['variant']='W2_'+w.policy;w=w.drop(columns='policy');a=pd.concat([a,w],ignore_index=True)
    for key,g in a.groupby(['variant','suite','mode','group','metric']):
        p=g.pivot(index='seed',columns='kind',values='value').reindex(SEEDS);assert p.notna().all().all();lo=p.level_low.to_numpy();hi=p.level_high.to_numpy();scopes.append(dict(zip(['variant','suite','mode','group','metric'],key),low=float(lo.mean()),high=float(hi.mean()),**dict(zip(['delta','lo','hi'],ci(hi-lo)))))
    checked=0
    for (suite,mode,level),g in d.groupby(['suite','mode','level']):
        groups={'all':g};groups.update({('group_'+str(k) if suite=='H' else k):v for k,v in g.groupby('group' if suite=='H' else 'role')})
        for label,gg in groups.items():
            for metric,col in [('mse','terminal_squared_error'),('jammer_side','side'),('near_truth','near')]:
                sub=a[(a.variant=='baseline')&(a.suite==suite)&(a['mode']==mode)&(a.kind=='level_'+level)&(a.group==label)&(a.metric==metric)].sort_values('seed')
                if sub.empty:continue
                np.testing.assert_allclose(gg.groupby('seed')[col].mean().reindex(SEEDS),sub.value,atol=1e-10,rtol=1e-10);checked+=100
    pd.DataFrame(rank_seed).to_csv(out/'rank_seed_counts.csv',index=False);pd.DataFrame(rank).to_csv(out/'rank_conditional.csv',index=False);a.to_csv(out/'threshold_seed_values.csv.gz',index=False,compression={'method':'gzip','mtime':0});pd.DataFrame(scopes).to_csv(out/'threshold_summary.csv',index=False)
    report=dict(seeds=SEEDS,bootstrap_seed=BOOT,draws=B,records=len(d),rank_conditionals=rank,decompositions=decomps,threshold_summary_rows=len(scopes),threshold_seed_rows=len(a),baseline_seed_values_crosschecked=checked,w2_tau1_crosschecks=len(j),w2_members=105,terminal_hashes=digests,companion_member_hashes=input_members,new_model_executions=0,limitations='Pooled conditional ratios with whole-seed bootstrap. Rank is post-structure; no causal mediation. Study2 selected threshold summaries, not full raw trajectory regeneration.')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['seeds','companion_member_hashes','rank_conditionals']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('companion1',type=Path);p.add_argument('companion2',type=Path);p.add_argument('transfer',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.inputs,[a.companion1,a.companion2],a.transfer,a.out)

"""Verify transferred history replay against frozen terminal and checkpoint records."""
import argparse, gzip, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

K=['seed','level','mode','citizen_pos']; R=K[:-1]
def digest(b): return hashlib.sha256(b).hexdigest()
def load(z,n): return json.loads(gzip.decompress(z.read(n)))
def csv(z,n): return pd.read_csv(z.open(n),compression='gzip',float_precision='round_trip')
def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    hashes={}; residuals={}; counts={}
    with zipfile.ZipFile(root/'PaperB_V6.4.1_Final_Analytical_Closure.zip') as z:
        old=json.loads(z.read('primary/validation.json'))['input_hashes']
    for n in ['baseline_H_individuals.json(1).gz','shard_inventory.csv(1).gz']:
        hashes[n]=digest((root/n).read_bytes()); assert hashes[n]==old[n]
    with zipfile.ZipFile(root/'PaperB_V6.5.2C_History_Replay_Release.zip') as z:
        script_hash=digest(z.read('replay_v652c.py')); inv_hash=digest(z.read('expected_H_shards.csv.gz')); expected=csv(z,'expected_H_shards.csv.gz')
    with zipfile.ZipFile(root/'PaperB_V6.5.2C_Transfer.zip') as z:
        assert z.testzip() is None
        m=json.loads(z.read('TRANSFER_V652C.json'))
        assert m['script_sha256']==script_hash and m['inventory_sha256']==inv_hash and m['design_id']=='7e2e4f6724e3' and m['manifest_sha256']=='3a83e3b7ff4af3ceeeaeda34bbd47633c9df07228089e25f6d57a56394cfff61' and m['source_commit']=='55264824e35c02284174e206d86955269d2b0bbc'
        assert (m['runs'],m['citizens'],m['history_rows'],m['new_simulations'])==(400,40000,16000000,0)
        for f in m['files']:
            b=z.read(f['name']); assert len(b)==f['bytes'] and digest(b)==f['sha256']
        a=pd.DataFrame(load(z,'replayed_citizens.json.gz')).sort_values(K).reset_index(drop=True)
        inv=pd.DataFrame(json.loads(z.read('history_inventory.json'))); ex=csv(z,'example_histories.csv.gz')
    assert len(a)==40000 and not a.duplicated(K).any() and len(inv)==400 and not inv.duplicated(R).any()
    assert set(inv[R].itertuples(index=False,name=None))=={(s,l,p) for s in range(31001,31101) for l in ['low','high'] for p in ['adaptive','frozen']}
    assert inv.rows.eq(40000).all() and inv.history_sha256.str.fullmatch('[0-9a-f]{64}').all() and inv.history_bytes.gt(0).all()
    prior=pd.read_csv(root/'shard_inventory.csv(1).gz'); prior=prior[(prior.suite=='H')&(prior.variant=='baseline')]
    for ref in [prior,expected]:
        assert inv[R+['shard_sha256']].rename(columns={'shard_sha256':'sha256'}).sort_values(R).reset_index(drop=True).equals(ref[R+['sha256']].sort_values(R).reset_index(drop=True))
    t=pd.DataFrame(json.loads(gzip.decompress((root/'baseline_H_individuals.json(1).gz').read_bytes()))).sort_values(K).reset_index(drop=True)
    assert len(t)==len(a) and a[K].equals(t[K])
    def ck(name,x,y):
        x,y=np.asarray(x),np.asarray(y); assert np.array_equal(pd.isna(x),pd.isna(y)),name
        mask=~pd.isna(x); x,y=x[mask].astype(float),y[mask].astype(float)
        residuals[name]=max(residuals.get(name,0.),float(np.max(np.abs(x-y))) if len(x) else 0.)
        np.testing.assert_allclose(x,y,atol=1e-8,rtol=1e-10,err_msg=name)
    common=[k for k in a if k in t and k not in K+['Q']]
    for k in common: ck('terminal_fields',a[k],t[k])
    for k in ['expert','jammer','peer']:
        ck('terminal_QB',a.Q.map(lambda d:d[k]),t.Q.map(lambda d:d[k])); ck('terminal_QB',a.B.map(lambda d:d[k]),t.B_raw.map(lambda d:d[k]))
    counts['terminal_scalar_fields_per_citizen']=len(common); counts['terminal_QB_fields_per_citizen']=6
    assert a.groupby(R+['group']).size().eq(50).all() and a.audit_count.add(a.theta_count).eq(400).all()
    assert a[a['mode']=='frozen'][['sequence','strict_sequence']].eq(0).all().all() and a[a['mode']=='frozen'].promotion.isna().all()
    assert (a.strict_sequence<=a.sequence).all()
    bpath=root/'PaperB_V6.5.2B_Transfer.zip'; dup=root/'PaperB_V6.5.2B_Transfer(1).zip'
    if dup.exists(): assert digest(bpath.read_bytes())==digest(dup.read_bytes())
    with zipfile.ZipFile(bpath) as z:
        bm=json.loads(z.read('TRANSFER_V652B.json'))
        for f in bm['files']: assert digest(z.read(f['name']))==f['sha256']
        bi=csv(z,'shard_inventory.csv.gz'); c=csv(z,'baseline_HG_checkpoints.csv.gz')
    bi=bi[bi.suite=='H']; assert len(bi)==200
    assert inv[inv['mode']=='adaptive'][R+['history_sha256']].sort_values(R).reset_index(drop=True).equals(bi[R+['history_sha256']].sort_values(R).reset_index(drop=True))
    h=a[a['mode']=='adaptive'].copy(); c=c[(c.suite=='H')&(c.h==2)].sort_values(K).reset_index(drop=True); assert h[K].reset_index(drop=True).equals(c[K]) and len(c)==20000
    ck('h2_checkpoint',h.early_mu,c.mu); ck('h2_checkpoint',h.early_sd,c.sd); ck('h2_precision',h.early_precision,1/c.sd**2); ck('h2_prior_weight',h.early_prior_weight,.04*c.sd**2)
    assert np.array_equal(h.early_audit.eq(0).to_numpy(),(1/c.sd**2>.04+1e-12).to_numpy())
    # Reconstruct the five exported example trajectories using vectorized masks, not observer event labels.
    assert len(ex)==m['examples']==2000 and ex.groupby(K).size().eq(400).all()
    for key,g in ex.groupby(K):
        g=g.sort_values('period'); assert g.period.tolist()==list(range(400))
        s=a.set_index(K).loc[key]; assert g.audit.iloc[0]==1
        sub=g[g.audit==0]; initial=g.rank_post.iloc[0]; promos=g[(g.period>0)&(g.audit==1)&(g.rank_pre!=1)&(g.rank_post==1)] if initial>=3 else g.iloc[:0]
        p=None if promos.empty else int(promos.period.iloc[0]); assert (p is None and pd.isna(s.promotion)) or p==s.promotion
        pre=sub[sub.period<(p if p is not None else 400)]
        drift=pre[(pre.Q_jammer_step==0)&((4-pre.mu_pre)*pre.peer_signed_update>0)&((pre.mu_post-4).abs()<(pre.mu_pre-4).abs())] if initial>=3 else pre.iloc[:0]
        strict=drift[drift.Q_jammer_cumulative==0]; follow=sub[sub.period>p].head(5) if p is not None else sub.iloc[:0]; uptake=follow[follow.Q_jammer_step>0]
        seq=int(p is not None and not drift.empty and not uptake.empty); strict_seq=int(seq and not strict.empty and promos.Q_jammer_cumulative.iloc[0]==0)
        assert seq==s.sequence and strict_seq==s.strict_sequence and len(follow)==s.postpromo_opportunities
        ck('example_terminal',[g.mu_post.iloc[-1],g.sd_post.iloc[-1],g.Q_jammer_step.sum()],[s.mu,s.sd,s.Q['jammer']]); ck('example_early',[g.mu_post.iloc[1],g.sd_post.iloc[1],g.audit.iloc[1]],[s.early_mu,s.early_sd,s.early_audit])
    # Whole-seed summaries; equal weight across seeds, not pooled learners.
    rng=np.random.default_rng(20261125); boot=rng.integers(0,100,(10000,100)); rows=[]; seedrows=[]
    def emit(block,mode,group,metric,p):
        p=p.reindex(index=range(31001,31101),columns=['low','high']); assert p.notna().all().all()
        for level,v in [('low',p.low),('high',p.high),('high_minus_low',p.high-p.low)]:
            x=v.to_numpy(float); lo,hi=np.quantile(x[boot].mean(1),[.025,.975]); rows.append(dict(block=block,mode=mode,group=group,metric=metric,contrast=level,mean=x.mean(),lo=lo,hi=hi)); seedrows.extend(dict(block=block,mode=mode,group=group,metric=metric,contrast=level,seed=s,value=y) for s,y in zip(p.index,x))
    a['eligible']=a.initial_rank.ge(3); a['promoted']=a.promotion.notna(); a['learner_fraction']=a.early_audit.eq(0); a['mse']=a.early_mu**2; a['prior_weight']=a.early_prior_weight; a['precision']=a.early_precision
    for mode,d in a.groupby('mode'):
        for group,g in [('all',d),*[(f'group_{k}',v) for k,v in d.groupby('group')]]:
            for metric in ['eligible','promoted','sequence','strict_sequence','censored','learner_fraction','prior_weight','precision','mse']:
                emit('all',mode,group,metric,g.groupby(['seed','level'])[metric].mean().unstack())
            for metric in ['prior_weight','precision','mse']:
                emit('learners',mode,group,metric,g[g.learner_fraction].groupby(['seed','level'])[metric].mean().unstack())
    result=pd.DataFrame(rows)
    with zipfile.ZipFile(root/'PaperB_V6.5.2B_Partition_Sensitivity_Package.zip') as z: prev=pd.read_csv(z.open('summary.csv'))
    prev=prev[(prev.suite=='H')&prev.block.isin(['early','early_learners'])].copy(); prev['block']=prev.block.map({'early':'all','early_learners':'learners'}); prev['mode']='adaptive'
    cols=['block','mode','group','metric','contrast']; match=prev.merge(result,on=cols,suffixes=('_old','_new'),validate='one_to_one'); assert len(match)==len(prev)==36
    for k in ['mean','lo','hi']: ck('B_early_estimates',match[k+'_old'],match[k+'_new'])
    result.to_csv(out/'replay_summary.csv',index=False); pd.DataFrame(seedrows).to_csv(out/'replay_seed_statistics.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    v=dict(status='PASS',transfer_sha256=digest((root/'PaperB_V6.5.2C_Transfer.zip').read_bytes()),input_hashes=hashes,script_sha256=script_hash,reference_inventory_sha256=inv_hash,local_replay_report=m,matched_shards=400,matched_history_hashes_across_transfers=200,matched_citizens=len(a),matched_h2_checkpoints=len(c),matched_B_estimates=len(match),examples_independently_checked=len(ex)//400,counts=counts,residuals=residuals,raw_histories_available_here=False,new_simulations=0)
    (out/'validation.json').write_text(json.dumps(v,indent=2)+'\n'); print(json.dumps(v,indent=2)); print(result[(result['mode']=='adaptive')&(result.group=='group_1')&result.metric.isin(['eligible','promoted','sequence','strict_sequence','censored'])].to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); main(a.root,a.out)

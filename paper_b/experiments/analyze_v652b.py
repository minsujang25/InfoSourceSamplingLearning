"""Frozen V6.5.2B passive analysis; no model imports or simulation."""
import argparse, gzip, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

K=['seed','variant','suite','level','mode','citizen_pos']
R=K[:-1]
S=list(range(31001,31101))
BOOT=np.array([np.bincount(x,minlength=100) for x in np.random.default_rng(20261125).integers(0,100,(10000,100))],float)/100

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    checks={}; rows=[]; seeds=[]
    def ck(k,a,b,atol=1e-8):
        a,b=np.asarray(a),np.asarray(b); checks[k]=max(checks.get(k,0),float(np.max(np.abs(a-b))))
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=atol)
    def emit(block,suite,group,metric,x,**extra):
        x=np.asarray(x,float); assert x.shape==(100,) and np.isfinite(x).all()
        lo,hi=np.quantile(BOOT@x,[.025,.975]); meta=dict(block=block,suite=suite,group=group,metric=metric,**extra)
        rows.append(dict(**meta,mean=x.mean(),lo=lo,hi=hi)); seeds.extend(dict(**meta,seed=s,value=v) for s,v in zip(S,x))
    def paired(block,suite,group,metric,l,high,**extra):
        for label,x in [('low',l),('high',high),('high_minus_low',high-l)]: emit(block,suite,group,metric,x,contrast=label,**extra)
    def read(z,n): return pd.read_csv(z.open(n),compression='gzip',float_precision='round_trip',keep_default_na=False,na_values=[''])
    transfer=root/'PaperB_V6.5.2B_Transfer.zip'
    with zipfile.ZipFile(transfer) as z:
        assert z.testzip() is None
        m=json.loads(z.read('TRANSFER_V652B.json')); assert m['design_id']=='7e2e4f6724e3' and m['source_manifest_sha256']=='3a83e3b7ff4af3ceeeaeda34bbd47633c9df07228089e25f6d57a56394cfff61' and m['extractor_sha256']=='7a1398a516dc51abd4291543ba559fb6d54996811dc1f167664416d7d109089a'
        assert m['seeds']==S and m['runs']==400 and m['citizens']==40000 and m['floor_events']==0
        frames=[]
        for f in m['files']:
            data=z.read(f['name']); assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256']
            d=read(z,f['name']); assert len(d)==f['rows'] and list(d.columns)==f['fields']; frames.append(d)
        w,a,c,inv=frames
    assert (len(w),len(a),len(c),len(inv))==(353637,142000,440000,400)
    with zipfile.ZipFile(root/'PaperB_V6.4.1_Final_Analytical_Closure.zip') as z:
        oldv=json.loads(z.read('primary/validation.json')); old=pd.read_csv(z.open('primary/accounting.csv'))
    inputs=['shard_inventory.csv(1).gz','first_audit_external.csv(1).gz','baseline_H_individuals.json(1).gz','baseline_G_adaptive_terminal.json(1).gz']
    hashes={n:sha(root/n) for n in inputs}; assert all(h==oldv['input_hashes'][n] for n,h in hashes.items())
    prior=pd.read_csv(root/inputs[0]); prior=prior[(prior.variant=='baseline')&(prior['mode']=='adaptive')]
    assert len(prior)==400 and inv[R+['sha256','file']].sort_values(R).reset_index(drop=True).equals(prior[R+['sha256','file']].sort_values(R).reset_index(drop=True))
    for d,keys in [(w,K+['window','source_class']),(a,K+['source_pos']),(c,K+['h'])]: assert not d.duplicated(keys).any()
    ext=pd.read_csv(root/inputs[1],float_precision='round_trip'); ext=ext[(ext.variant=='baseline')&(ext['mode']=='adaptive')].sort_values(K+['source_pos']).reset_index(drop=True)
    obs=a[a.source_class.isin(['expert','jammer'])].sort_values(K+['source_pos']).reset_index(drop=True); assert len(ext)==len(obs)==62000
    for col in ext:
        if pd.api.types.is_numeric_dtype(ext[col]):
            assert ext[col].isna().equals(obs[col].isna()); mask=ext[col].notna();
            if mask.any(): ck('external_audit',ext.loc[mask,col],obs.loc[mask,col])
        else: assert ext[col].fillna('').equals(obs[col].fillna(''))
    t=pd.concat([pd.DataFrame(json.load(gzip.open(root/n,'rt'))) for n in inputs[2:]],ignore_index=True); t=t[t['mode']=='adaptive'].copy().sort_values(K).reset_index(drop=True)
    assert len(t)==40000 and not t.duplicated(K).any() and t.floor_events.eq(0).all()
    for d in (a,c,w): assert d[K].drop_duplicates().sort_values(K).reset_index(drop=True).equals(t[K])
    # Match checkpoints and audit states to the independently supplied terminal extract.
    end=c[c.h==400].sort_values(K).reset_index(drop=True); init=c[c.h==1].sort_values(K).reset_index(drop=True)
    for k in ['mu','sd','rank']: ck('terminal_checkpoint',end[k],t[k])
    for k in ['mu','sd','rank']: ck('initial_checkpoint',init[k],t['initial_'+k])
    assert c.groupby(K).size().eq(11).all() and set(c.h)=={0,1,2,3,6,11,26,51,100,200,400}
    agg=w.groupby(K+['source_class'])[['Q','B']].sum().unstack(fill_value=0).reindex(pd.MultiIndex.from_frame(t[K]),fill_value=0)
    for k in ['expert','jammer','peer']:
        for v in ['Q','B']:
            x=agg[v]; z=x[k] if k!='peer' else x[[u for u in x if u.endswith('_peer')]].sum(axis=1)
            ck('window_reconciliation',z,t['Q' if v=='Q' else 'B_raw'].map(lambda d:d[k]))
    parts=[]; focal=[]
    def account(q,b,phi,mu):
        n=len(q)//2; ql,qh=q[:n],q[n:]; bl,bh=b[:n],b[n:]; wl,wh=ql/phi[:n,None],qh/phi[n:,None]
        ml=np.divide(bl,ql,out=np.zeros_like(bl),where=ql>0); mh=np.divide(bh,qh,out=np.zeros_like(bh),where=qh>0)
        both=(ql>0)&(qh>0); f=(mu[:n]+mu[n:])[:,None]
        ct=np.where(both,(wl+wh)/2*(mh-ml)*f,0); wt=np.where(both,(ml+mh)/2*(wh-wl)*f,0); ee=np.where(~both,(bh/phi[n:,None]-bl/phi[:n,None])*f,0)
        de=mu[n:]**2-mu[:n]**2; ck('individual_additivity',(ct+wt+ee).sum(1),de)
        return {'content':ct.sum(1),'weight':wt.sum(1),'entry_exit':ee.sum(1),'weight_entry_exit':(wt+ee).sum(1),'total_change':de}
    def group_masks(l,suite):
        return {'all':np.ones(len(l),bool),**({f'group_{g}':l.group.eq(g).to_numpy() for g in [-1,1]} if suite=='H' else {g:l.role.eq(g).to_numpy() for g in ['gateway','non_gateway']})}
    for suite in ['H','G']:
        d=t[t.suite==suite].sort_values(['level','seed','citizen_pos'],ascending=[False,True,True]); l=d[d.level=='low']; h=d[d.level=='high']; assert len(l)==len(h)==10000
        for k in ['seed','citizen_pos','group','role','initial_mu','initial_sd','initial_phi']: assert np.array_equal(l[k],h[k])
        d=pd.concat([l,h]); phi=d.phi.to_numpy(); mu=d.mu.to_numpy(); p=agg.reindex(pd.MultiIndex.from_frame(d[K])); types=['same_peer','other_peer'] if suite=='H' else ['gateway_peer','non_gateway_peer']
        q5=np.column_stack([d.initial_phi,p['Q'][['expert','jammer']+types]]); b5=np.column_stack([d.B_raw.map(lambda v:v['prior']),p['B'][['expert','jammer']+types]])
        ck('precision',q5.sum(1),phi); ck('posterior',b5.sum(1)/phi,mu)
        q4=np.column_stack([q5[:,:3],q5[:,3:].sum(1)]); b4=np.column_stack([b5[:,:3],b5[:,3:].sum(1)])
        vals={'four':account(q4,b4,phi,mu),'five':account(q5,b5,phi,mu)}
        dq=h.Q.map(lambda v:v['jammer']).to_numpy()-l.Q.map(lambda v:v['jammer']).to_numpy(); de=vals['four']['total_change']; dw=h.W_jammer.to_numpy()-l.W_jammer.to_numpy()
        for group,mask in group_masks(l,suite).items():
            n=mask.sum()//100; assert np.all(mask.reshape(100,100).sum(1)==n)
            sm=lambda x:x[mask].reshape(100,n).mean(1)
            for k in vals['four']:
                u,v=sm(vals['four'][k]),sm(vals['five'][k]); o=old[(old.suite==suite)&(old.variant=='baseline')&(old['mode']=='adaptive')&(old.group==group)&(old.source=='all_sources')&(old.metric==k)]; assert len(o)==1
                ck('archived_four_class',[u.mean(),*np.quantile(BOOT@u,[.025,.975])],o[['mean','lo','hi']].iloc[0]);
                for partition,x in [('four',u),('five',v),('five_minus_four',v-u)]: emit('partition',suite,group,k,x,partition=partition)
            for measure,delta in [('Q',dq),('W',dw)]:
                cats=np.select([(delta<0)&(de>0),(delta>0)&(de<0),(delta>0)&(de>0),(delta<0)&(de<0)],['less_higher','more_lower','more_higher','less_lower'],default='zero')
                for cat in ['less_higher','more_lower','more_higher','less_lower','zero']:
                    cm=cats==cat; emit('joint',suite,group,'prevalence',sm(cm),measure=measure,category=cat,partition='both')
                    for k in ['content','weight_entry_exit','total_change']:
                        u,v=[sm(vals[z][k]*cm) for z in ['four','five']]
                        for partition,x in [('four',u),('five',v),('five_minus_four',v-u)]: emit('joint',suite,group,k,x,measure=measure,category=cat,partition=partition)
    # Same-observation rank rescore: common variance means absolute discrepancy ordering.
    a=a.merge(t[K+['initial_mu','initial_sd','group','role']],on=K,suffixes=('','_t'),validate='many_to_one'); ck('audit_initial',a.initial_mu,a.initial_mu_t); ck('audit_initial',a.initial_sd,a.initial_sd_t)
    assert a.group.eq(a.group_t).all() and a.role.fillna('').eq(a.role_t.fillna('')).all()
    assert a.delta_pre.eq(0).all() and a.delta_sd_pre.eq(5).all() and a.initial_sd.eq(5).all()
    a['abs_discrepancy']=(a.message_mean-a.initial_mu).abs(); a['shrinkage']=25/(50+a.sample_mean_variance); a['score']=a.delta_post.abs(); ck('score',a.score,a.abs_discrepancy*a.shrinkage)
    a['d']=a.groupby(K).source_pos.transform('size'); assert a.groupby(K).requested.sum().eq(20).all()
    ties={}
    for name,col in [('actual','score'),('common','abs_discrepancy')]:
        ties[name]=int(a.duplicated(K+[col],keep=False).sum()); a=a.sort_values(K+[col,'rank_post']); a[name+'_rank']=a.groupby(K).cumcount()+1
        r=a[name+'_rank']; a[name+'_prob']=np.where(r==a.d,.05**(r-1),.95*.05**(r-1)); ck('policy_mass',a.groupby(K)[name+'_prob'].sum(),1)
    ck('rank_replay',a.actual_rank,a.rank_post)
    jammer=a[a.source_class=='jammer'].sort_values(K); ck('initial_policy',jammer.actual_prob,t.initial_policy)
    for suite in ['H','G']:
        d=a[a.suite==suite].copy(); d['stratum']=d.group.map(lambda x:f'group_{x}') if suite=='H' else d.role
        for group,g in [('all',d),*list(d.groupby('stratum'))]:
            for cls,p in g.groupby('source_class'):
                for metric in ['abs_discrepancy','shrinkage','score','sample_mean_variance','actual_rank','common_rank','actual_prob','common_prob']:
                    x=p.groupby(['level','seed'])[metric].mean().unstack(0).reindex(index=S,columns=['low','high'])
                    if x.notna().all().all(): paired('audit_type',suite,group,metric,x.low.to_numpy(),x.high.to_numpy(),source_class=cls)
            x=g[g.source_class.isin(['expert','jammer'])].copy(); l=x[x.level=='low']; h=x[x.level=='high']; j=l.merge(h,on=['seed','citizen_pos','source_pos'],suffixes=('_l','_h'),validate='one_to_one'); assert len(j)==len(l)==len(h)
            ck('matched_external_scores',j.score_l,j.score_h); ck('matched_external_discrepancy',j.abs_discrepancy_l,j.abs_discrepancy_h)
            for cls,p in [('external',j),*list(j.groupby('source_class_l'))]:
                for method in ['actual','common']:
                    rl,rh=p[method+'_rank_l'],p[method+'_rank_h']; pp=p.assign(changed=rl.ne(rh),rank1_l=rl.eq(1),rank1_h=rh.eq(1),rank_delta=rh-rl,prob_delta=p[method+'_prob_h']-p[method+'_prob_l'])
                    for metric in ['changed','rank_delta','prob_delta']:
                        emit('audit_displacement',suite,group,metric,pp.groupby('seed')[metric].mean().reindex(S),source_class=cls,method=method)
                    l1,h1=[pp.groupby('seed')[k].mean().reindex(S).to_numpy() for k in ['rank1_l','rank1_h']]; paired('audit_displacement',suite,group,'rank1',l1,h1,source_class=cls,method=method)
    # First scheduled post-audit period; infer substantive update only from precision gain.
    c=c.merge(t[K+['group','role','initial_phi','initial_sd']],on=K,validate='many_to_one'); c['precision']=1/c.sd**2; c['prior_weight']=c.initial_phi/c.precision; c['mse']=c.mu**2
    for suite in ['H','G']:
        d=c[c.suite==suite].copy(); d['stratum']=d.group.map(lambda x:f'group_{x}') if suite=='H' else d.role
        for group,g in [('all',d),*list(d.groupby('stratum'))]:
            for h,p in g.groupby('h'):
                for metric in ['precision','prior_weight','mse','sd']:
                    x=p.groupby(['seed','level'])[metric].mean().unstack().reindex(S); paired('checkpoint',suite,group,metric,x.low.to_numpy(),x.high.to_numpy(),h=h)
                if h==2:
                    p=p.copy(); p['learner']=p.precision>p.initial_phi+1e-12; assert np.all(p.precision>=p.initial_phi-1e-12)
                    x=p.groupby(['seed','level']).learner.mean().unstack().reindex(S); paired('early',suite,group,'learner_fraction',x.low.to_numpy(),x.high.to_numpy())
                    for metric in ['prior_weight','precision','mse']:
                        x=p[p.learner].groupby(['seed','level'])[metric].mean().unstack().reindex(S); assert x.notna().all().all(); paired('early_learners',suite,group,metric,x.low.to_numpy(),x.high.to_numpy())
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False); pd.DataFrame(seeds).to_csv(out/'seed_statistics.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    (out/'validation.json').write_text(json.dumps(dict(transfer_sha256=sha(transfer),inputs=hashes,rows={'windows':len(w),'audit':len(a),'checkpoints':len(c),'citizens':len(t),'shards':len(inv),'summary':len(rows),'seed_statistics':len(seeds)},max_residual=checks,ties=ties,new_simulations=0,bootstrap_seed=20261125,bootstrap_draws=10000),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); main(a.root,a.out)

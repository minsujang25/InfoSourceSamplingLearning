"""Passive V6.5.2A: four-class joint signs and terminal confidence. No model runs."""
import argparse, gzip, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ['prior','expert','jammer','peer']
CATS = ['less_Q_higher_error','more_Q_lower_error','more_Q_higher_error','less_Q_lower_error','zero']
SEEDS = np.arange(31001,31101)
FILES = ['baseline_H_individuals.json(1).gz','baseline_G_adaptive_terminal.json(1).gz','baseline_G_frozen_terminal.json(1).gz']

def main(root, out, closure):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(closure) as z:
        validation = json.loads(z.read('primary/validation.json'))
        old = pd.read_csv(z.open('primary/accounting.csv'))
        signs = pd.read_csv(z.open('primary/sign_shares.csv'))
        parts = pd.read_csv(z.open('primary/partitions.csv'))
    frames=[]; hashes={}
    for name in FILES:
        p=root/name
        with p.open('rb') as f: hashes[name]=hashlib.file_digest(f,'sha256').hexdigest()
        assert hashes[name]==validation['input_hashes'][name]
        with gzip.open(p,'rt') as f: frames.append(pd.DataFrame(json.load(f)))
    d=pd.concat(frames,ignore_index=True); del frames
    keys=['suite','mode','level','seed','citizen_pos']
    assert len(d)==80000 and not d.duplicated(keys).any() and d.variant.eq('baseline').all()
    assert d.floor_events.eq(0).all() and d.groupby(keys[:-1]).size().eq(100).all()
    assert len(d.groupby(keys[:-1]))==800 and np.array_equal(np.sort(d.seed.unique()),SEEDS)
    rng=np.random.default_rng(20261125)
    boot=np.array([np.bincount(x,minlength=100) for x in rng.integers(0,100,(10000,100))],dtype=float)
    rows=[]; sr=[]; tr=[]; ts=[]; checks={k:0. for k in ['precision','posterior','individual_additivity','category_additivity','archived_accounting','archived_signs','archived_H_partitions']}; counts={k:0 for k in ['archived_accounting','archived_signs','archived_H_partitions']}
    def ck(k,a,b,atol=1e-8):
        a,b=np.asarray(a),np.asarray(b); checks[k]=max(checks[k],float(np.max(np.abs(a-b))))
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=atol)
    def summary(x):
        draws=boot@x/100
        return dict(mean=float(x.mean()),lo=float(np.quantile(draws,.025)),hi=float(np.quantile(draws,.975)))
    def seedmean(x): return x.reshape(100,-1).mean(1)
    def arrays(a):
        q=np.column_stack([a.initial_phi,*[a.Q.map(lambda v:v[c]) for c in CLASSES[1:]]]); b=np.column_stack([a.B_raw.map(lambda v:v[c]) for c in CLASSES]); phi=a.phi.to_numpy()[:,None]
        w=q/phi; z=b/phi; m=np.divide(b,q,out=np.zeros_like(b),where=q>0)
        ck('precision',q.sum(1),phi[:,0]); ck('posterior',z.sum(1),a.mu); ck('precision',1/a.sd**2,a.phi)
        return q,w,z,m,a.mu.to_numpy()
    for (suite,mode),x in d.groupby(['suite','mode']):
        l,h=[x[x.level.eq(lev)].sort_values(['seed','citizen_pos']).reset_index(drop=True) for lev in ['low','high']]
        for c in ['seed','citizen_pos','group','role','initial_mu','initial_phi','initial_sd']: assert l[c].equals(h[c]),c
        assert len(l)==len(h)==10000
        ql,wl,zl,ml,al=arrays(l); qh,wh,zh,mh,ah=arrays(h)
        both=(ql>0)&(qh>0); f=(al+ah)[:,None]; dz=(zh-zl)*f
        ct=np.where(both,(wl+wh)/2*(mh-ml)*f,0).sum(1)
        wt=np.where(both,(ml+mh)/2*(wh-wl)*f,0).sum(1)
        ee=np.where(~both,dz,0).sum(1); de=ah**2-al**2; dq=qh[:,2]-ql[:,2]; dw=h.W_jammer.to_numpy()-l.W_jammer.to_numpy()
        ck('individual_additivity',ct+wt+ee,de)
        masks={'all':np.ones(len(l),bool)}
        masks.update({f'group_{g}':l.group.eq(g).to_numpy() for g in [-1,1]} if suite=='H' else {g:l.role.eq(g).to_numpy() for g in ['gateway','non_gateway']})
        for group,mask in masks.items():
            n=int(mask.sum()//100); assert n in (10,50,90,100) and np.all(mask.reshape(100,100).sum(1)==n)
            meta=dict(suite=suite,mode=mode,group=group,n_per_seed=n)
            term={}
            for lev,a in [('low',l),('high',h)]:
                mu=a.mu.to_numpy()[mask].reshape(100,n); sd=a.sd.to_numpy()[mask].reshape(100,n)
                term[lev]={'mse':(mu**2).mean(1),'squared_mean_error':mu.mean(1)**2,'belief_variance':mu.var(1),'mean_subjective_sd':sd.mean(1),'mean_prior_weight':(a.initial_phi.to_numpy()[mask]/a.phi.to_numpy()[mask]).reshape(100,n).mean(1)}
                ck('individual_additivity',term[lev]['mse'],term[lev]['squared_mean_error']+term[lev]['belief_variance'])
            for metric in term['low']:
                a,b=term['low'][metric],term['high'][metric]; ts.append(dict(**meta,metric=metric,low_mean=a.mean(),high_mean=b.mean(),**summary(b-a)))
                tr.extend(dict(**meta,metric=metric,seed=int(s),low=u,high=v,delta=v-u) for s,u,v in zip(SEEDS,a,b))
            if mode!='adaptive': continue
            base={'content':ct,'weight':wt,'entry_exit':ee,'weight_entry_exit':wt+ee,'delta_MSE':de}
            for k,v in base.items():
                metric='total_change' if k=='delta_MSE' else k
                o=old[(old.suite==suite)&(old.variant=='baseline')&(old['mode']==mode)&(old.group==group)&(old.source=='all_sources')&(old.metric==metric)]
                assert len(o)==1; ck('archived_accounting',list(summary(seedmean(v[mask])).values()),o[['mean','lo','hi']].iloc[0].to_numpy(float)); counts['archived_accounting']+=1
            for measure,delta in [('Q',dq),('W',dw)]:
                cat=np.select([(delta<0)&(de>0),(delta>0)&(de<0),(delta>0)&(de>0),(delta<0)&(de<0)],CATS[:4],default='zero')
                metrics={'prevalence':np.ones(len(l)), 'delta_dependence':delta,**base}
                sums={k:np.zeros(100) for k in metrics}
                for c in CATS:
                    cm=(cat==c); m=dict(**meta,measure=measure,category=c)
                    for k,v in metrics.items():
                        a=seedmean((v*cm)[mask]); sums[k]+=a; rows.append(dict(**m,metric=k,**summary(a)))
                        sr.extend(dict(**m,metric=k,seed=int(s),value=u) for s,u in zip(SEEDS,a))
                    o=signs[(signs.suite==suite)&(signs.variant=='baseline')&(signs['mode']==mode)&(signs.group==group)&(signs.measure==measure)&(signs.category==c)]
                    assert len(o)==1; ck('archived_signs',[cm[mask].mean(),(delta*cm)[mask].mean(),(de*cm)[mask].mean()],o[['prevalence','dependence_contribution','mse_contribution']].iloc[0].to_numpy(float)); counts['archived_signs']+=1
                    if suite=='H' and measure=='Q':
                        for k,v in metrics.items():
                            metric='delta_Q' if k=='delta_dependence' else k
                            o=parts[(parts.suite==suite)&(parts.variant=='baseline')&(parts['mode']==mode)&(parts.group==group)&(parts.partition=='sign')&(parts.category==c)&(parts.metric==metric)]
                            assert len(o)==1; ck('archived_H_partitions',list(summary(seedmean((v*cm)[mask])).values()),o[['mean','lo','hi']].iloc[0].to_numpy(float)); counts['archived_H_partitions']+=1
                for k,v in metrics.items(): ck('category_additivity',sums[k],seedmean(v[mask]))
    for name,data in [('joint_sign_accounting',rows),('joint_sign_seed',sr),('terminal_summary',ts),('terminal_seed',tr)]:
        pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
    pd.DataFrame([dict(level=lev,structural_gateway_share=sg,uniform_expected_gateway_request_share=ug,study1_adaptive_printed=ad) for lev,sg,ug,ad in [('low',.45,30/65,.363),('high',.90,60/65,.926)]]).to_csv(out/'G_expected_request_benchmark.csv',index=False)
    (out/'validation.json').write_text(json.dumps(dict(input_hashes=hashes,closure_sha256=hashlib.sha256(closure.read_bytes()).hexdigest(),terminal_records=len(d),runs=800,adaptive_records=40000,bootstrap_seed=20261125,bootstrap_draws=10000,max_abs_residual=checks,archive_comparisons=counts,new_simulations=0),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--closure',type=Path,required=True); a=p.parse_args(); main(a.root,a.out,a.closure)

"""V6.4 supplementary paired differences; no simulation."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import pandas as pd

def run(root,out):
    names=['baseline_H_individuals','baseline_G_adaptive_terminal','neutral_G_adaptive_terminal','neutral_H_adaptive_terminal','uniform_tail_H_adaptive_terminal','baseline_G_frozen_terminal']
    def locate(n):
        p=root/(n+'.json.gz')
        return p if p.exists() else root/(n+'.json(1).gz')
    d=pd.DataFrame([r for n in names for r in json.load(gzip.open(locate(n),'rt'))])
    B=np.array([np.bincount(x,minlength=100) for x in np.random.default_rng(20261125).integers(0,100,(10000,100))],dtype=float)
    def seedmeans(x,s):
        return np.bincount(s-31001,weights=x,minlength=100)/np.bincount(s-31001,minlength=100)
    def stat(x):
        return [x.mean(),*np.quantile(B@x/100,[.025,.975])]
    rows=[];g={}
    for (suite,variant,mode),b in d.groupby(['suite','variant','mode']):
        l,h=[b[b.level.eq(lev)].sort_values(['seed','citizen_pos']).reset_index(drop=True) for lev in ['low','high']]
        for c in ['seed','citizen_pos','group','role','initial_mu']:assert l[c].equals(h[c])
        if suite=='G':g[variant,mode]=(l,h)
        ql,qh=[x.Q.map(lambda q:q['peer']).to_numpy() for x in [l,h]]
        ml,mh=[np.divide(x.B_raw.map(lambda q:q['peer']).to_numpy(),q,out=np.zeros(len(x)),where=q>0) for x,q in [(l,ql),(h,qh)]]
        wl,wh=ql/l.phi.to_numpy(),qh/h.phi.to_numpy()
        f=l.mu.to_numpy()+h.mu.to_numpy();both=(ql>0)&(qh>0)
        ct=np.where(both,f*(wl+wh)/2*(mh-ml),0);wt=np.where(both,f*(ml+mh)/2*(wh-wl),0)
        masks={'all':np.ones(len(l),bool)}
        masks.update({'group_1':l.group.eq(1).to_numpy(),'group_-1':l.group.eq(-1).to_numpy()} if suite=='H' else {r:l.role.eq(r).to_numpy() for r in sorted(l.role.unique())})
        for group,m in masks.items():
            s=l.seed.to_numpy()[m];a,bw=seedmeans(ct[m],s),seedmeans(wt[m],s)
            rows.append([suite,variant,mode,group,'peer_content_minus_weight',*stat(a-bw)])
            bs=(abs(B@a)-abs(B@bw))/100
            rows.append([suite,variant,mode,group,'abs_peer_content_minus_abs_peer_weight',abs(a.mean())-abs(bw.mean()),*np.quantile(bs,[.025,.975])])
    ar,ne=g['baseline','adaptive'],g['neutral','adaptive']
    for c in ['seed','citizen_pos','role','initial_mu']:assert all(ar[0][c].equals(x[c]) for x in [ar[1],*ne])
    roles=[]
    for role in ['all','gateway','non_gateway']:
        m=np.ones(len(ar[0]),bool) if role=='all' else ar[0].role.eq(role).to_numpy()
        s=ar[0].seed.to_numpy()[m]
        a=(ar[1].mu**2-ar[0].mu**2).to_numpy()[m];n=(ne[1].mu**2-ne[0].mu**2).to_numpy()[m]
        for label,x in [('adaptive_structural',a),('neutral_structural',n),('policy_interaction',a-n)]:
            roles.append([role,label,*stat(seedmeans(x,s))])
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=['suite','variant','mode','group','metric','mean','lo','hi']).to_csv(out/'peer_paired_differences.csv',index=False)
    pd.DataFrame(roles,columns=['role','metric','mean','lo','hi']).to_csv(out/'G_role_interactions.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)

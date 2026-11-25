"""V6.4.1 passive accounting; no simulations. Run with --root containing original extracts."""
import argparse,gzip,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def run(d,out):
    out.mkdir(parents=True,exist_ok=True)
    keys=['seed','suite','variant','mode','level','citizen_pos']
    assert len(d)==140000 and not d.duplicated(keys).any()
    assert (d.groupby(keys[:-1]).size()==100).all() and d.floor_events.eq(0).all()
    assert np.array_equal(np.sort(d.seed.unique()),np.arange(31001,31101))
    C=['prior','expert','jammer','peer']
    for c in C:
        d['q_'+c]=d.initial_phi if c=='prior' else d.Q.map(lambda x:x[c])
        d['b_'+c]=d.B_raw.map(lambda x:x[c])
    Q=d[['q_'+c for c in C]].to_numpy(); Z=d[['b_'+c for c in C]].to_numpy()/d.phi.to_numpy()[:,None]
    np.testing.assert_allclose(Q.sum(1),d.phi,atol=1e-9,rtol=1e-10)
    np.testing.assert_allclose(Z.sum(1),d.mu,atol=1e-9,rtol=1e-10)
    np.testing.assert_allclose(np.array(d.signed_mse.map(lambda x:sum(x.values()))),d.mu**2,atol=1e-9,rtol=1e-10)
    rng=np.random.default_rng(20261125)
    B=np.array([np.bincount(x,minlength=100) for x in rng.integers(0,100,(10000,100))],dtype=float)
    def stats(x,s,den=None):
        n=np.bincount(s-31001,weights=x,minlength=100)
        q=np.bincount(s-31001,weights=np.ones(len(s)) if den is None else den,minlength=100)
        with np.errstate(divide='ignore',invalid='ignore'):
            a=(B@n)/(B@q)
        return [n.sum()/q.sum() if q.sum() else np.nan,*np.nanquantile(a,[.025,.975])] if np.isfinite(a).any() else [np.nan]*3
    pairs={};blocks=[]
    def sorted_level(b,lev):return b[b.level.eq(lev)].sort_values(['seed','citizen_pos']).reset_index(drop=True)
    for suite,b in d[(d.variant=='baseline')&(d['mode']=='adaptive')].groupby('suite'):
        l=sorted_level(b,'low');pairs[suite]=np.tile(np.arange(len(l)),(101,1))
        for (s,g),x in l.groupby(['seed','group' if suite=='H' else 'role'],sort=True):
            j=x.sort_values(['initial_mu','citizen_pos']).index.to_numpy()
            for k in range(0,len(j),5):blocks.append((s,suite,g,k,j[k:k+5]))
    prng=np.random.default_rng(20261125)
    for r in range(1,101):
        for s,suite,g,k,j in sorted(blocks,key=lambda x:x[:4]):pairs[suite][r,j]=prng.permutation(j)
    def arrays(x):
        q=x[['q_'+c for c in C]].to_numpy();b=x[['b_'+c for c in C]].to_numpy()
        w=q/x.phi.to_numpy()[:,None];z=b/x.phi.to_numpy()[:,None]
        m=np.divide(b,q,out=np.zeros_like(b),where=q>0)
        return q,w,z,m,x.mu.to_numpy()
    def terms(L,H,shift=0):
        ql,wl,zl,ml,al=L;qh,wh,zh,mh,ah=H
        f=(al+ah)[:,None];both=(ql>0)&(qh>0);en=(ql==0)&(qh>0);ex=(ql>0)&(qh==0)
        dz=zh-zl+shift*(wh-wl)
        ct=np.where(both,(wl+wh)/2*(mh-ml),0)*f
        wt=np.where(both,(ml+mh+2*shift)/2*(wh-wl),0)*f
        ent=np.where(en,dz,0)*f;ext=np.where(ex,dz,0)*f
        np.testing.assert_allclose(ct+wt+ent+ext,dz*f,atol=1e-9,rtol=1e-10)
        np.testing.assert_allclose((ct+wt+ent+ext).sum(1),ah**2-al**2,atol=1e-9,rtol=1e-10)
        return dict(content=ct,weight=wt,entry=ent,exit=ext,total_change=dz*f,old_signed_mse=ah[:,None]*zh-al[:,None]*zl)
    def category(q,e):
        return np.select([(q<0)&(e>0),(q>0)&(e<0),(q>0)&(e>0),(q<0)&(e<0)],['less_Q_higher_error','more_Q_lower_error','more_Q_higher_error','less_Q_lower_error'],default='zero')
    estimates=[];rep_rows=[];refs=[];parts=[];signs=[];s59=[];roles=[];cache={}
    for (suite,variant,mode),b in d.groupby(['suite','variant','mode']):
        l,h=sorted_level(b,'low'),sorted_level(b,'high')
        for c in ['seed','citizen_pos','group','role','initial_mu','initial_phi']:assert l[c].equals(h[c]),c
        s=l.seed.to_numpy();L,H=arrays(l),arrays(h);T=terms(L,H)
        de=H[-1]**2-L[-1]**2;dq=H[0][:,2]-L[0][:,2];dw=h.W_jammer.to_numpy()-l.W_jammer.to_numpy()
        cat=category(dq,de);cache[suite,variant,mode]=(l,h)
        masks={'all':np.ones(len(l),bool)}
        masks.update({f'group_{g}':l.group.eq(g).to_numpy() for g in [-1,1]} if suite=='H' else {g:l.role.eq(g).to_numpy() for g in sorted(l.role.unique())})
        def total(t):
            a={k:x.sum(1) for k,x in t.items()}
            a['entry_exit']=a['entry']+a['exit'];a['weight_entry_exit']=a['weight']+a['entry_exit']
            a['abs_content_minus_abs_weight']=abs(a['content'])-abs(a['weight'])
            return a
        A=total(T)
        for group,m in masks.items():
            meta=[suite,variant,mode,group]
            for k,x in T.items():
                for j,c in enumerate(C):estimates.append([*meta,c,k,*stats(x[m,j],s[m])])
            for k,x in A.items():
                if k=='abs_content_minus_abs_weight':continue
                estimates.append([*meta,'all_sources',k,*stats(x[m],s[m])])
            # Difference in absolute aggregated component means, not mean of individual differences.
            den=np.bincount(s[m]-31001,minlength=100)
            n=lambda x:np.bincount(s[m]-31001,weights=x[m],minlength=100)
            a,bw=n(A['content']),n(A['weight_entry_exit']);bs=(abs(B@a)-abs(B@bw))/(B@den)
            estimates.append([*meta,'all_sources','abs_content_minus_abs_weight_entry_exit', (abs(a.sum())-abs(bw.sum()))/den.sum(),*np.quantile(bs,[.025,.975])])
            for shift in [-4.,-2.,2.,4.]:
                R=total(terms(L,H,shift))
                refs.append([*meta,shift,*[(R[k][m]-A[k][m]).mean() for k in ['content','weight','entry_exit','weight_entry_exit','total_change']],abs(R['weight_entry_exit'][m]-A['weight_entry_exit'][m]).max()])
            for measure,delta in [('Q',dq),('W',dw)]:
                cats=category(delta,de)
                for c in ['less_Q_higher_error','more_Q_lower_error','more_Q_higher_error','less_Q_lower_error','zero']:
                    z=cats==c
                    signs.append([*meta,measure,c,z[m].mean(),(delta*z)[m].mean(),(de*z)[m].mean(),(delta*z)[m].sum()/delta[m].sum() if delta[m].sum()!=0 else np.nan,(abs(delta)*z)[m].sum()/abs(delta[m]).sum() if abs(delta[m]).sum()!=0 else np.nan])
            if suite=='H' and variant=='baseline' and mode=='adaptive':
                lp=l.promotion.notna().to_numpy();hp=h.promotion.notna().to_numpy()
                changed=np.array([2-len(set(x)&set(y)) for x,y in zip(l.peer_ids,h.peer_ids)])
                partitions={'promotion':np.select([~lp&~hp,lp&~hp,~lp&hp],['neither','low_only','high_only'],default='both'),'sign':cat,'changed_peers':changed.astype(str)}
                for name,labels in partitions.items():
                    for label in sorted(set(labels)):
                        z=labels==label
                        for k,x in {'prevalence':np.ones(len(l)),'delta_Q':dq,'delta_MSE':de,**{k:A[k] for k in ['content','weight','entry_exit','weight_entry_exit']}}.items():
                            parts.append([*meta,name,label,k,*stats((x*z)[m],s[m])])
        if (variant,mode) in [('baseline','adaptive'),('uniform_tail','adaptive'),('neutral','adaptive')]:
            for r,ii in enumerate(pairs[suite]):
                HH=tuple(x[ii] for x in H);P=terms(L,HH);AA=total(P)
                e=HH[-1]**2-L[-1]**2;q=HH[0][:,2]-L[0][:,2];cc=category(q,e)
                mis=abs(h.initial_mu.to_numpy()[ii]-l.initial_mu.to_numpy())
                for group,m in masks.items():
                    for x,y in [(e,de),(q,dq),*[(P['old_signed_mse'][:,j],T['old_signed_mse'][:,j]) for j in range(4)]]:
                        np.testing.assert_allclose(np.bincount(s[m]-31001,weights=x[m],minlength=100),np.bincount(s[m]-31001,weights=y[m],minlength=100),atol=1e-8,rtol=1e-10)
                    for k in ['content','weight','entry_exit','weight_entry_exit','total_change']:
                        rep_rows.append([suite,variant,mode,group,r,'all_sources',k,AA[k][m].mean(),mis[m].mean()])
                    for j,c in enumerate(C):
                        for k in ['content','weight','total_change']:
                            rep_rows.append([suite,variant,mode,group,r,c,k,P[k][m,j].mean(),mis[m].mean()])
                    if variant=='baseline':
                        for c in ['less_Q_higher_error','more_Q_lower_error','more_Q_higher_error','less_Q_lower_error','zero']:
                            zz=m&(cc==c);s59.append([suite,group,r,c,zz.sum()/m.sum(),q[zz].sum()/m.sum(),e[zz].sum()/m.sum()])
    for lev,j in [('low',0),('high',1)]:
        a=cache['G','baseline','adaptive'][j];n=cache['G','neutral','adaptive'][j]
        for c in ['seed','citizen_pos','role','initial_mu']:assert a[c].equals(n[c])
        for role in ['all','gateway','non_gateway']:
            m=np.ones(len(a),bool) if role=='all' else a.role.eq(role).to_numpy()
            for k,x in [('adaptive',a.mu**2),('neutral',n.mu**2),('adaptive_minus_neutral',a.mu**2-n.mu**2)]:
                roles.append([lev,role,k,*stats(x.to_numpy()[m],a.seed.to_numpy()[m])])
    meta=['suite','variant','mode','group'];stat=['mean','lo','hi']
    for name,rows,cols in [('accounting',estimates,meta+['source','metric']+stat),('pairing',rep_rows,meta+['rep','source','metric','mean','initial_mismatch']),('reference',refs,meta+['shift','delta_content','delta_weight','delta_entry_exit','delta_weight_entry_exit','delta_total','max_combined_residual']),('partitions',parts,meta+['partition','category','metric']+stat),('sign_shares',signs,meta+['measure','category','prevalence','dependence_contribution','mse_contribution','net_share','gross_share']),('s59_replay',s59,['suite','group','rep','category','prevalence','Q_contribution','MSE_contribution']),('G_roles',roles,['level','role','metric']+stat)]:
        pd.DataFrame(rows,columns=cols).to_csv(out/(name+'.csv'),index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    names=['baseline_H_individuals','baseline_G_adaptive_terminal','baseline_G_frozen_terminal','neutral_H_adaptive_terminal','neutral_G_adaptive_terminal','uniform_tail_H_adaptive_terminal']
    def locate(n):
        p=a.root/(n+'.json.gz')
        return p if p.exists() else a.root/(n+'.json(1).gz')
    run(pd.DataFrame([r for n in names for r in json.load(gzip.open(locate(n),'rt'))]),a.out)

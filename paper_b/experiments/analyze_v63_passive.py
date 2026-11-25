import gzip,json,hashlib,pathlib,zipfile,argparse
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('/mnt/data'));p.add_argument('--extra',nargs='*',type=pathlib.Path,default=[]);p.add_argument('--out',type=pathlib.Path,default=pathlib.Path(__file__).parent);a=p.parse_args();root=a.root;out=a.out;out.mkdir(exist_ok=True,parents=True)
paths=[root/'baseline_H_individuals.json.gz',*a.extra];d=pd.DataFrame([r for p in paths for r in json.load(gzip.open(p,'rt'))]);keys=['seed','suite','variant','mode','level','citizen_pos'];assert not d.duplicated(keys).any() and (d.groupby(keys[:-1]).size()==100).all();assert d.floor_events.eq(0).all();seeds=np.arange(31001,31101);assert np.array_equal(np.sort(d.seed.unique()),seeds)
C=['prior','expert','jammer','peer'];check=lambda x,y:np.testing.assert_allclose(x,y,atol=1e-9,rtol=1e-10)
for c in C:
 d['q_'+c]=d.initial_phi if c=='prior' else d.Q.map(lambda x:x[c]);d['b_'+c]=d.B_raw.map(lambda x:x[c]);d['w_'+c]=d['q_'+c]/d.phi;d['z_'+c]=d['b_'+c]/d.phi;d['m_'+c]=d['b_'+c]/d['q_'+c].where(d['q_'+c]>0);d['sm_'+c]=d.signed_mse.map(lambda x:x[c])
check(d[[f'q_{c}' for c in C]].sum(axis=1),d.phi);check(d[[f'z_{c}' for c in C]].sum(axis=1),d.mu);check(d[[f'sm_{c}' for c in C]].sum(axis=1),d.mu**2);check(d.b_prior,d.initial_phi*d.initial_mu)
v=pd.read_csv(root/'validation_all_runs.csv.gz');assert len(v)==3400 and v.floor_events.eq(0).all() and not v.duplicated(keys[:-1]).any()
rng=np.random.default_rng(20261125);idx=rng.integers(0,100,(10000,100));B=np.array([np.bincount(x,minlength=100) for x in idx]);
def ss(x,seed):return np.bincount(seed-31001,weights=np.asarray(x,float),minlength=100)
def st(x,seed,den=None):
 n=ss(x,seed);q=ss(np.ones(len(x)) if den is None else den,seed);N=B@n;Q=B@q;z=np.divide(N,Q,out=np.full(10000,np.nan),where=Q!=0);return dict(mean=n.sum()/q.sum() if q.sum()!=0 else np.nan,lo=np.nanquantile(z,.025) if np.isfinite(z).any() else np.nan,hi=np.nanquantile(z,.975) if np.isfinite(z).any() else np.nan)
def save(n,r):pd.DataFrame(r).to_csv(out/(n+'.csv'),index=False)
def category(q,e):return np.select([(q<0)&(e>0),(q>0)&(e<0),(q>0)&(e>0),(q<0)&(e<0)],['less_Q_higher_error','more_Q_lower_error','more_Q_higher_error','less_Q_lower_error'],default='zero')
weights=[];levels=[];dose=[];sign=[];zero=[];risk=[];promo=[];pair=[];res=[];character=[]
# Precompute re-pairings in canonical seed -> suite -> group/role -> block order.
# For this partial H-only analysis, the stream includes H blocks only; rerun with all
# requested records for the frozen joint H/G sequence (H draws may then change).
blocks_by_suite={}; perm_by_suite={}; ordered_blocks=[]
for ssuite,bb in d[(d.variant=='baseline')&(d['mode']=='adaptive')&(d.level=='low')].groupby('suite',sort=True):
 ll=bb.sort_values(['seed','citizen_pos']).reset_index(drop=True);perm_by_suite[ssuite]=np.tile(np.arange(len(ll)),(101,1));stratum='group' if ssuite=='H' else 'role'
 for (seed,grp),xx in ll.groupby(['seed',stratum],sort=True):
  jj=xx.sort_values(['initial_mu','citizen_pos']).index.to_numpy()
  for j in range(0,len(jj),5):ordered_blocks.append((seed,ssuite,grp,j,jj[j:j+5]))
ordered_blocks.sort(key=lambda x:x[:4]);prng=np.random.default_rng(20261125)
for rep in range(1,101):
 for seed,ssuite,grp,j,jj in ordered_blocks:perm_by_suite[ssuite][rep,jj]=prng.permutation(jj)
for (suite,variant,mode),b in d.groupby(['suite','variant','mode']):
 l=b[b.level=='low'].sort_values(['seed','citizen_pos']).reset_index(drop=True);h=b[b.level=='high'].sort_values(['seed','citizen_pos']).reset_index(drop=True);assert len(l)==len(h)==10000
 for c in ['seed','citizen_pos','group','role','initial_mu','initial_phi']:assert l[c].equals(h[c]),c
 sd=l.seed.to_numpy();dq=(h.q_jammer-l.q_jammer).to_numpy();de=(h.mu**2-l.mu**2).to_numpy();cat=category(dq,de);fac=(h.mu+l.mu).to_numpy();sums=np.zeros(len(l));terms={}
 for c in C:
  ql,qh=l['q_'+c].to_numpy(),h['q_'+c].to_numpy();wl,wh=l['w_'+c].to_numpy(),h['w_'+c].to_numpy();ml,mh=l['m_'+c].to_numpy(),h['m_'+c].to_numpy();z=(h['z_'+c]-l['z_'+c]).to_numpy();both=(ql>0)&(qh>0);en=(ql==0)&(qh>0);ex=(ql>0)&(qh==0);ct=np.zeros(len(l));wt=ct.copy();ct[both]=(wl[both]+wh[both])/2*(mh[both]-ml[both]);wt[both]=(ml[both]+mh[both])/2*(wh[both]-wl[both]);ent=np.where(en,z,0);ext=np.where(ex,z,0);check(ct+wt+ent+ext,z);sums+=fac*z;terms[c]={'content':ct*fac,'weight':wt*fac,'entry':ent*fac,'exit':ext*fac,'total_change':z*fac,'old_signed_mse':(h['sm_'+c]-l['sm_'+c]).to_numpy(),'entry_rate':en,'exit_rate':ex}
 check(sums,de);res.append(dict(suite=suite,variant=variant,mode=mode,max_mse_residual=abs(sums-de).max()))
 masks={'all':np.ones(len(l),bool)};masks.update({f'group_{g}':l.group.eq(g).to_numpy() for g in [-1,1]} if suite=='H' else {r:l.role.eq(r).to_numpy() for r in sorted(l.role.unique())})
 for group,m in masks.items():
  meta=dict(suite=suite,variant=variant,mode=mode,group=group);S=lambda x,den=None:st(np.asarray(x)[m],sd[m],None if den is None else np.asarray(den)[m]);
  for c in C:
   for k,x in terms[c].items():weights.append(dict(**meta,source=c,metric=k,**S(x)))
  for lev,y in [('low',l),('high',h)]:
   for k,x in {'mse':y.mu**2,'mean':y.mu,'jammer_side':y.mu>2,'near_truth':abs(y.mu)<=.5,'phi':y.phi,**{f'{k}_{c}':y[f'{k}_{c}'] for k in ['q','w'] for c in C}}.items():levels.append(dict(**meta,level=lev,metric=k,**S(x)))
   med=y[m].groupby('seed').mu.median().reindex(seeds);levels.append(dict(**meta,level=lev,metric='seed_median_mu',**st(med,seeds)))
   for c in C:levels.append(dict(**meta,level=lev,metric='conditional_m_'+c,**S(y['m_'+c].fillna(0),y['q_'+c]>0)))
  for c in ['less_Q_higher_error','more_Q_lower_error','more_Q_higher_error','less_Q_lower_error','zero']:
   z=cat==c
   for k,x in [('prevalence',z),('Q_contribution',dq*z),('MSE_contribution',de*z)]:sign.append(dict(**meta,category=c,metric=k,**S(x)))
   for k,x in [('initial_mu',l.initial_mu),('initial_rank_low',l.initial_rank),('initial_rank_high',h.initial_rank)]:character.append(dict(**meta,category=c,metric=k,**S(np.asarray(x)*z,z)))
  for c,z in [('Q_only',(dq==0)&(de!=0)),('error_only',(dq!=0)&(de==0)),('both',(dq==0)&(de==0)),('neither',(dq!=0)&(de!=0))]:
   for k,x in [('prevalence',z),('Q_contribution',dq*z),('MSE_contribution',de*z)]:zero.append(dict(**meta,category=c,metric=k,**S(x)))
  if suite=='H' and variant=='baseline' and mode=='adaptive':
   changed=np.array([2-len(set(x)&set(y)) for x,y in zip(l.peer_ids,h.peer_ids)]);same=(h.same_peer_count-l.same_peer_count).to_numpy()
   for j in range(3):
    z=changed==j
    for k,x,den in [('prevalence',z,None),('Q_contribution',dq*z,None),('MSE_contribution',de*z,None),('mean_delta_Q',dq*z,z),('mean_delta_MSE',de*z,z),('mean_delta_same_count',same*z,z),('initial_mu',l.initial_mu*z,z),('initial_rank_low',l.initial_rank*z,z),('initial_rank_high',h.initial_rank*z,z)]:dose.append(dict(**meta,dose=j,category='all',metric=k,**S(x,den)))
    for c in np.unique(cat):
     zz=z&(cat==c)
     for k,x in [('prevalence',zz),('Q_contribution',dq*zz),('MSE_contribution',de*zz)]:dose.append(dict(**meta,dose=j,category=c,metric=k,**S(x)))
   for ev in ['sequence','strict_sequence']:
    el=l.initial_rank.ge(3).to_numpy();eh=h.initial_rank.ge(3).to_numpy();sl=l[ev].astype(bool).to_numpy();sh=h[ev].astype(bool).to_numpy();assert np.all(~sl|el) and np.all(~sh|eh)
    nl=ss(el[m],sd[m]);nh=ss(eh[m],sd[m]);xl=ss(sl[m],sd[m]);xh=ss(sh[m],sd[m]);nn=ss(np.ones(m.sum()),sd[m]);pl,ph=nl.sum()/nn.sum(),nh.sum()/nn.sum();cl,ch=xl.sum()/nl.sum(),xh.sum()/nh.sum();PL,PH=(B@nl)/(B@nn),(B@nh)/(B@nn);CL,CH=(B@xl)/(B@nl),(B@xh)/(B@nh)
    for k,x,z in [('eligibility',(ph-pl)*(ch+cl)/2,(PH-PL)*(CH+CL)/2),('completion',(ch-cl)*(ph+pl)/2,(CH-CL)*(PH+PL)/2),('total',ph*ch-pl*cl,PH*CH-PL*CL)]:risk.append(dict(**meta,event=ev,metric=k,mean=x,lo=np.quantile(z,.025),hi=np.quantile(z,.975),p_low=pl,p_high=ph,c_low=cl,c_high=ch))
   lp=l.promotion.notna().to_numpy();hp=h.promotion.notna().to_numpy()
   for c,z in [('neither',~lp&~hp),('low_only',lp&~hp),('high_only',~lp&hp),('both',lp&hp)]:
    for k,x in [('prevalence',z),('Q_contribution',dq*z),('MSE_contribution',de*z)]:promo.append(dict(**meta,category=c,metric=k,**S(x)))
 if variant=='baseline' and mode=='adaptive':
  for rep in range(101):
   ii=perm_by_suite[suite][rep]
   q=h.q_jammer.to_numpy()[ii]-l.q_jammer.to_numpy();e=h.mu.to_numpy()[ii]**2-l.mu.to_numpy()**2;cc=category(q,e);mis=abs(h.initial_mu.to_numpy()[ii]-l.initial_mu.to_numpy())
   for group,m in masks.items():
    check(ss(q[m],sd[m]),ss(dq[m],sd[m]));check(ss(e[m],sd[m]),ss(de[m],sd[m]));
    for c in ['less_Q_higher_error','more_Q_lower_error','more_Q_higher_error','less_Q_lower_error','zero']:
     z=m&(cc==c);pair.append(dict(suite=suite,group=group,rep=rep,category=c,prevalence=z.sum()/m.sum(),Q_contribution=q[z].sum()/m.sum(),MSE_contribution=e[z].sum()/m.sum(),Q_net_share=q[z].sum()/q[m].sum(),MSE_net_share=e[z].sum()/e[m].sum(),mean_initial_mismatch=mis[m].mean(),max_initial_mismatch=mis[m].max()))
for n,r in [('weight_content',weights),('terminal_levels',levels),('peer_exposure',dose),('sign_categories',sign),('zero_categories',zero),('risk_set',risk),('promotion_partition',promo),('pairing_sensitivity',pair),('identity_checks',res),('category_characteristics',character)]:save(n,r)
pol=[];ana=[]
for L in [3,4]:
 for name,e in [('baseline',.05),('epsilon35',.35),('uniform_tail',.05),('neutral',0)]:
  pp=[(1-e)*e**i for i in range(L-1)]+[e**(L-1)] if name not in ['uniform_tail','neutral'] else [.95]+[.05/(L-1)]*(L-1) if name=='uniform_tail' else [1/L]*L
  for k,p in enumerate(pp,1):pol.append(dict(L=L,policy=name,rank=k,p=p,inclusion=1-(1-p)**20))
u3=1-(2/3)**20;u4=1-.75**20
for k,lev in [(1,'low'),(2,'high')]:
 g,n=np.linalg.solve([[10+2*u4,-2*u4],[-k*u3,20/3+k*u3]],[20,80/3]);ana.append(dict(suite='G',level=lev,gateway=g,non_gateway=n,mse=.1*g*g+.9*n*n));ana.append(dict(suite='H',level=lev,mse=4))
save('policy_inclusion',pol);save('analytical_benchmark',ana)
z=zipfile.ZipFile(root/'v62_7e2e4f6724e3_primary_001_of_008.zip');summary=pd.concat([pd.read_csv(gzip.GzipFile(fileobj=z.open(n)),low_memory=False) for n in z.namelist() if n.startswith('contrast_summary_')]);print('summary',len(summary),'metrics', [x for x in summary.metric.unique() if any(k in x.lower() for k in ['hhi','top5','median','near_truth','jammer_side','phi','w_peer','w_expert'])]);sel=summary[summary.variant.isin(['baseline','neutral','uniform_tail','epsilon35','constant_audit'])&summary.kind.isin(['level_low','level_high','high_minus_low','adaptive_minus_neutral'])&summary.metric.str.contains(r'^(?:Q_expert|Q_jammer|Q_peer|W_jammer|mse|mean|variance|terminal_sd|signed_mse_.*|.*hhi.*|.*top5.*|.*median.*|.*near_truth.*|.*jammer_side.*|.*W_peer.*|.*W_expert.*)$',case=False,regex=True)];sel.to_csv(out/'archived_summary_selection.csv',index=False)
lev=pd.DataFrame(levels)
for (s,var,mode,g),x in lev[lev.metric=='mse'].groupby(['suite','variant','mode','group']):
 y=summary[(summary.suite==s)&(summary.variant==var)&(summary['mode']==mode)&(summary.group==g)&(summary.kind=='high_minus_low')&(summary.metric=='mse')];assert len(y)==1;check(x.set_index('level').loc['high','mean']-x.set_index('level').loc['low','mean'],y['mean'].iloc[0])
(out/'validation.json').write_text(json.dumps(dict(records=len(d),runs=int(d.groupby(keys[:-1]).ngroups),all_run_validation_records=len(v),floor_events=int(v.floor_events.sum()),max_phi_residual=float(abs(d[[f'q_{c}' for c in C]].sum(axis=1)-d.phi).max()),max_signed_residual=float(abs(d[[f'z_{c}' for c in C]].sum(axis=1)-d.mu).max()),summary_rows=len(summary),input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,root/'validation_all_runs.csv.gz',root/'v62_7e2e4f6724e3_primary_001_of_008.zip',root/'V63_FINAL_REVISION_PROTOCOL.md']}),indent=2));print('done',len(d),len(weights),len(pair))

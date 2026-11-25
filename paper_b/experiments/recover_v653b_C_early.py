"""Reconstruct Package C's non-terminal-dependent seed statistics; keep full recovery pending A01."""
import argparse,gzip,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def main(root,out):
 handoff=zipfile.ZipFile(root/'PaperB_V6.3_Author_Handoff(1).zip'); old=zipfile.ZipFile(io.BytesIO(handoff.read('history/v61/V61_Package_C_Analysis_Results.zip')))
 (out/'original_analyze_package_C.py').write_bytes(old.read('analyze_package_C.py'))
 oldsum=pd.read_csv(old.open('summaries.csv'));reg={x['filename']:x for x in json.loads((root/'PaperB_V6.5.3A_Local_Inventory.json').read_text())['archives']};frames={}
 for i in range(1,4):
  stem=f'V61_Small_C_{i:02d}_of_03';p=root/(stem+'(1).zip');assert hashlib.sha256(p.read_bytes()).hexdigest() in [x['sha256'] for x in reg[stem+'.zip']['checked'] if x['match']]
  with zipfile.ZipFile(p) as z:
   m=json.loads(z.read('MANIFEST.json'));assert m['source_sha256']=='dad2311e9c72fb5947f66df7484f1039c6978c95dfafe75a53330ec81062af7e'
   for x in m['members']:
    raw=z.read(x['name']);assert hashlib.sha256(raw).hexdigest()==x['sha256'] and len(raw)==x['bytes'];d=pd.read_csv(io.BytesIO(raw),compression='gzip',low_memory=False);assert len(d)==x['rows'] and list(d.columns)==x['columns'];frames.setdefault(x['dataset'],[]).append(d)
 e,j,r=[pd.concat(frames[k],ignore_index=True) for k in ['sender_events.csv.gz','rank_unification_checkpoints.csv.gz','expert_rank_checkpoints.csv.gz']];del frames
 keys=['seed','level','reliance_mode','citizen_group'];ek=keys+['citizen_pos'];S=list(range(20001,20501))
 for d in (e,j,r):assert set(d.seed)==set(S) and d.run_id.nunique()==2000 and d.design_id.eq('e86f6ff3be12').all() and d.sender_regime.eq('fixed_biased').all() and d.prior.eq('high').all()
 assert not e.duplicated(ek+['period']).any() and not j.duplicated(keys+['period']).any() and not r.duplicated(keys[:-1]+['period']).any();assert e.groupby(keys+['period']).size().eq(50).all();assert e.groupby(['seed','citizen_pos','period']).initial_state_fingerprint.nunique().eq(1).all();assert e[e.period==0].learning_target.eq('audit').all();assert e[['jammer_request_count','expert_request_count','peer_request_count']].sum(axis=1).eq(20).all()
 a=e[e.period==0];b=e[e.period==1].copy();ab=a.merge(b,on=ek,suffixes=('_0','_1'),validate='one_to_one');assert ab.jammer_rank_post_0.eq(ab.jammer_rank_pre_1).all();assert np.allclose(a.theta_mu_pre,a.theta_mu_post) and np.allclose(a.theta_sd_pre,a.theta_sd_post) and np.allclose(ab.theta_mu_pre_0,ab.theta_mu_pre_1);del ab,a
 q=e.pivot(index=['seed','level','citizen_pos','citizen_group','period'],columns='reliance_mode',values=['theta_mu_post','theta_sd_post','jammer_rank_pre','jammer_request_count','expert_request_count','peer_request_count','learning_target']);assert all(q[k].adaptive.eq(q[k].frozen).all() for k in q.columns.levels[0]);del q
 ck=e.assign(top=e.jammer_rank_pre.eq(1)).groupby(keys+['period']).top.mean();assert np.allclose(ck.sort_index(),j.set_index(keys+['period']).jammer_rank_1_share.reindex(ck.index))
 d=b.assign(top=b.jammer_rank_pre.eq(1),post_top=b.jammer_rank_post.eq(1),theta=b.learning_target.eq('theta'),rank_up=b.jammer_rank_post.lt(b.jammer_rank_pre),rank_down=b.jammer_rank_post.gt(b.jammer_rank_pre),enter_top=b.jammer_rank_pre.ne(1)&b.jammer_rank_post.eq(1),leave_top=b.jammer_rank_pre.eq(1)&b.jammer_rank_post.ne(1),e1=b.theta_mu_post.pow(2),sd1=b.theta_sd_post,initial_mu=b.theta_mu_pre)
 for rank in range(1,5):
  d[f'initial_rank{rank}_mass']=d.jammer_rank_pre.eq(rank)
  for dest in range(1,5):d[f'rank{rank}_to_{dest}']=d.jammer_rank_pre.eq(rank)&d.jammer_rank_post.eq(dest)
 cols=['top','post_top','rank_up','rank_down','enter_top','leave_top','theta','e1','sd1','initial_mu']+[c for c in d if c.startswith('initial_rank') or (c.startswith('rank') and '_to_' in c)]
 jcols=['jammer_rank_1_share','mean_jammer_rank','mean_jammer_acquisition_probability'];rcols=['expert_rank_1_share','mean_expert_rank','mean_expert_acquisition_probability']
 jlong=j[keys+['horizon_step']+jcols].melt(id_vars=keys+['horizon_step'],var_name='metric').assign(dataset='jammer_checkpoint');rlong=r[keys[:-1]+['horizon_step']+rcols].assign(citizen_group=0).melt(id_vars=keys+['horizon_step'],var_name='metric').assign(dataset='expert_checkpoint');ev=d.groupby(keys)[cols].mean().reset_index().assign(horizon_step=2).melt(id_vars=keys+['horizon_step'],var_name='metric').assign(dataset='event_p1');theta=d[d.theta].groupby(keys)[['e1','sd1','jammer_request_count','expert_request_count','peer_request_count']].mean().reset_index().assign(horizon_step=2).melt(id_vars=keys+['horizon_step'],var_name='metric').assign(dataset='event_p1_theta_only');long=pd.concat([jlong,rlong,ev,theta],ignore_index=True)
 idx=np.random.default_rng(20261124).integers(0,500,size=(10000,500));counts=np.zeros((10000,500));np.add.at(counts,(np.arange(10000)[:,None],idx),1)
 K=['dataset','metric','citizen_group','reliance_mode','horizon_step'];rows=[]
 for k,g in long.groupby(K):
  p=g.pivot(index='seed',columns='level',values='value').reindex(S);assert p.notna().all().all()
  for label,x in [('low',p.low),('high',p.high),('high_minus_low',p.high-p.low)]:
   x=x.to_numpy(float);lo,hi=np.quantile(counts@x/500,[.025,.975]);rows.append((*k,label,x.mean(),x.std(ddof=1)/np.sqrt(500),lo,hi))
 s=pd.DataFrame(rows,columns=K+['contrast','mean','mcse','lo','hi']);m=s.merge(oldsum,on=K+['contrast'],suffixes=('_new','_old'),validate='one_to_one');assert len(m)==len(s);res={}
 for k in ['mean','mcse','lo','hi']:
  np.testing.assert_allclose(m[k+'_new'],m[k+'_old'],atol=1e-10,rtol=1e-10);res[k]=float(abs(m[k+'_new']-m[k+'_old']).max())
 long.to_csv(out/'C_partial_seed_statistics.csv.gz',index=False,compression={'method':'gzip','mtime':0});s.to_csv(out/'C_partial_summaries.csv',index=False)
 missing=oldsum.merge(s[K+['contrast']],on=K+['contrast'],how='left',indicator=True);missing=missing[missing._merge=='left_only'].drop(columns='_merge');missing.to_csv(out/'C_pending_terminal_summary_rows.csv',index=False)
 report=dict(status='PARTIAL_PASS',early_events=len(e),jammer_checkpoint_rows=len(j),expert_checkpoint_rows=len(r),recovered_seed_rows=len(long),reproduced_summary_rows=len(s),pending_summary_rows=len(missing),pending_metrics=sorted(missing.metric.unique()),max_residual=res,required_input='V61_Small_A_01_of_03.zip',full_recovery=False,new_simulations=0)
 (out/'validation_C_partial.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)

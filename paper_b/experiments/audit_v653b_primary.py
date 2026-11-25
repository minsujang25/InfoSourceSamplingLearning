"""Read-only archive audit and primary seed reproduction (V6.5.3B)."""
import argparse, gzip, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
KEY=['kind','variant','suite','mode','group','metric']
def sha(b):return hashlib.sha256(b).hexdigest()
def csvread(b):return pd.read_csv(io.BytesIO(b),compression='gzip',float_precision='round_trip')
def main(root,out):
 out.mkdir(parents=True,exist_ok=True)
 reg={x['filename']:x for x in json.loads((root/'PaperB_V6.5.3A_Local_Inventory.json').read_text())['archives']}
 a=zipfile.ZipFile(root/'PaperB_V6.4.9.1_Final_Editorial_Consolidation_Package.zip');src='manuscript/analysis/v62_sources/'
 index=pd.read_csv(a.open('manuscript/analysis/v62_exhibit_source_rows.csv'))
 audit=[];members=[];tables={};structural={};seeds={};seen=set();C={}
 for category,n,template in [('primary',8,'v62_7e2e4f6724e3_primary_{i:03d}_of_008'),('C',3,'V61_Small_C_{i:02d}_of_03')]:
  for i in range(1,n+1):
   stem=template.format(i=i);p=root/(stem+'(1).zip');sh=sha(p.read_bytes());assert sh in [v['sha256'] for v in reg[stem+'.zip']['checked'] if v['match']]
   with zipfile.ZipFile(p) as z:
    assert z.testzip() is None;m=json.loads(z.read('PART_MANIFEST.json' if category=='primary' else 'MANIFEST.json'))
    if category=='primary':assert (m['design_id'],m['category'],m['part'],m['parts'])==('7e2e4f6724e3','primary',i,n)
    else:assert (m['design_id'],m['package'],m['upload_part'],m['upload_parts'])==('e86f6ff3be12','C',i,n)
    paths=[v['path'] if category=='primary' else v['name'] for v in m['members']];assert set(z.namelist())==set(paths)|{('PART_MANIFEST.json' if category=='primary' else 'MANIFEST.json')}
    for v,name in zip(m['members'],paths):
     b=z.read(name);assert len(b)==v['bytes'] and sha(b)==v['sha256'];rows=None
     if category=='primary':assert name not in seen;seen.add(name)
     if name.startswith('seed_tables/'):
      o=json.loads(gzip.decompress(b));s=v['coverage']['seeds'][0];assert len(o)==v['coverage']['rows'] and all(r['seed']==s for r in o) and s not in seeds
      keys=[tuple(r[k] for k in KEY) for r in o];vals=np.array([np.nan if r['value'] is None else r['value'] for r in o],float)
      if not seeds:canonical=keys;lookup={k:j for j,k in enumerate(keys)};assert len(lookup)==len(keys)
      if keys!=canonical:
       assert len(set(keys))==len(keys) and set(keys)==set(canonical);ix=np.array([lookup[k] for k in keys]);tmp=np.empty_like(vals);tmp[ix]=vals;vals=tmp
      seeds[s]=vals;rows=len(o)
     elif name.endswith('.csv.gz'):
      d=csvread(b);cov=v['coverage'] if category=='primary' else v;assert len(d)==cov['rows'] and list(d.columns)==cov['columns'];rows=len(d)
      if category=='primary':
       if name.startswith('contrast_summary'):tables[name]=d;assert b==a.read(src+name)
       else:structural[name]=d
      else:C[name]=rows
     elif name=='MANIFEST.json':manifest=json.loads(b);(out/'V62_MANIFEST.json').write_bytes(b)
     elif name in ('EXECUTION_SUMMARY.json','ARCHIVE_INVENTORY.json'):(out/name).write_bytes(b)
     members.append(dict(category=category,part=i,name=name,sha256=sha(b),bytes=len(b),rows=rows))
    audit.append(dict(category=category,part=i,filename=p.name,sha256=sh,bytes=p.stat().st_size,member_count=len(paths),status='PASS'))
   print('verified',stem,flush=True)
 pd.DataFrame(audit).to_csv(out/'archive_audit.csv',index=False);pd.DataFrame(members).to_csv(out/'member_audit.csv',index=False)
 assert sorted(seeds)==list(range(31001,31101)) and manifest['expected_runs']==3400 and len(manifest['grid'])==34 and manifest['bootstrap_draws']==10000
 x=np.stack([seeds[s] for s in sorted(seeds)],axis=1);old=pd.concat(tables.values(),ignore_index=True);assert not old.duplicated(KEY).any() and len(old)==len(canonical)
 old=old.set_index(KEY).reindex(pd.MultiIndex.from_tuples(canonical,names=KEY)).reset_index();assert old.n_expected.eq(100).all()
 valid=np.isfinite(x);n=valid.sum(1);z=np.where(valid,x,0.);mean=np.divide(z.sum(1),n,out=np.full(len(n),np.nan),where=n>0);mcse=np.sqrt(np.divide(np.where(valid,(x-mean[:,None])**2,0).sum(1),n*(n-1),out=np.full(len(n),np.nan),where=n>1));residual={}
 def compare(name,u,v):
  u=np.asarray(u,float);v=np.asarray(v,float);assert np.array_equal(np.isnan(u),np.isnan(v)),name;mask=np.isfinite(u);np.testing.assert_allclose(u[mask],v[mask],rtol=1e-10,atol=1e-10,err_msg=name);residual[name]=float(np.max(abs(u[mask]-v[mask]))) if mask.any() else 0.
 compare('n_valid',n,old.n_valid);compare('mean',mean,old['mean']);compare('mcse',mcse,old.mcse)
 for name,mask in [('positive',x>0),('negative',x<0),('zero',x==0)]:compare(name,np.divide((mask&valid).sum(1),n,out=np.full(len(n),np.nan),where=n>0),old[name])
 focus=index[index.file.isin(tables)];chosen=sorted({lookup[tuple(tables[r.file].iloc[int(r.data_row)-1][k] for k in KEY)] for r in focus.itertuples()})
 draw=np.random.default_rng(manifest['bootstrap_seed']).integers(0,100,(10000,100));counts=np.zeros((10000,100));np.add.at(counts,(np.arange(10000)[:,None],draw),1)
 result=[]
 for start in range(0,len(chosen),128):
  ids=chosen[start:start+128];v=valid[ids].T;zz=z[ids].T;den=counts@v.astype(float);bs=np.divide(counts@zz,den,out=np.full(den.shape,np.nan),where=den>0)
  for j,k in enumerate(ids):
   b=bs[:,j];b=b[np.isfinite(b)];lo,hi=np.quantile(b,[.025,.975]) if len(b) else (np.nan,np.nan);p=old.iloc[k];row={**dict(zip(KEY,canonical[k])),'mean':mean[k],'lo':lo,'hi':hi,'archived_lo':p.lo,'archived_hi':p.hi}
   if pd.notna(p.pooled_denominator):
    metric=canonical[k][-1];candidates=[key for key in lookup if key[:-1]==canonical[k][:-1] and key[-1].endswith('_count') and metric.startswith(key[-1][:-6]+'_')];assert len(candidates)==1,metric
    weights=x[lookup[candidates[0]]];w=np.where(valid[k]&np.isfinite(weights),weights,0.);y=z[k]*w;bd=counts@w;bn=counts@y;pool=bn[bd>0]/bd[bd>0];plo,phi=np.quantile(pool,[.025,.975]);row.update(pooled_mean=y.sum()/w.sum(),pooled_lo=plo,pooled_hi=phi,pooled_denominator=w.sum(),archived_pooled_mean=p.pooled_mean,archived_pooled_lo=p.pooled_lo,archived_pooled_hi=p.pooled_hi,archived_pooled_denominator=p.pooled_denominator)
   result.append(row)
 res=pd.DataFrame(result)
 for k in ['lo','hi','pooled_mean','pooled_lo','pooled_hi','pooled_denominator']:
  if k in res:compare('selected_'+k,res[k],res['archived_'+k])
 res.to_csv(out/'reproduced_exhibit_seed_intervals.csv',index=False)
 assert structural['structural_summary.csv.gz'].equals(csvread(a.read(src+'primary_structural_summary.csv.gz')))
 report=dict(status='PASS',archive_count=len(audit),member_count=len(members),primary_seed_tables=len(seeds),seed_statistics=int(x.size),summary_rows=len(old),exhibit_selector_rows=len(focus),unique_exhibit_keys=len(chosen),pooled_exhibit_keys=int(res.pooled_mean.notna().sum()) if 'pooled_mean' in res else 0,structural_summary_rows=len(structural['structural_summary.csv.gz']),structural_stored_summary_matches=True,residuals=residual,C_input_counts=C,C_missing_dependency='V61_Small_A_01_of_03.zip',new_simulations=0)
 (out/'validation_primary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)

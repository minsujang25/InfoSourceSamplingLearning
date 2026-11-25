"""Independent W2 archive audit and seed bootstrap; no simulation."""
import argparse, gzip, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

KEYS=['kind','variant','suite','mode','group','metric']
SEEDS=list(range(31001,31101))
def sha(b): return hashlib.sha256(b).hexdigest()
def require(x,msg):
    if not x: raise RuntimeError(msg)
def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    idx=json.loads((root/'INDEX.json').read_text()); require(idx==json.loads((root/'TRANSFER_INDEX(1).json').read_text()),'Index mismatch')
    part,=idx['parts']; p=root/part['file']; require(p.stat().st_size==part['bytes'] and sha(p.read_bytes())==part['sha256'],'ZIP checksum')
    with zipfile.ZipFile(p) as z:
        pm=json.loads(z.read('PART_MANIFEST.json')); require(pm['design_id']==idx['design_id'] and pm['category']==idx['category'] and pm['part']==pm['parts']==1,'Part identity')
        require(len(pm['members'])==part['members']==105 and len(z.namelist())==106 and set(z.namelist())=={'PART_MANIFEST.json'}|{r['path'] for r in pm['members']},'Members')
        for r in pm['members']:
            b=z.read(r['path']); require(len(b)==r['bytes'] and sha(b)==r['sha256'],'Member '+r['path'])
        m=json.loads(z.read('MANIFEST.json')); e=json.loads(z.read('EXECUTION_SUMMARY.json')); inv=json.loads(z.read('ARCHIVE_INVENTORY.json'))
        require(m['design_id']==e['design_id']==idx['design_id'] and m['seeds']==SEEDS and m['tau_social']==2 and m['reference_tau']==1 and m['horizon']==400 and m['expected_runs']==800,'Design')
        require(e['status']=='validated_scientific_grid' and e['technical_gate_pass'] and e['scientific_runs']==e['reference_runs']==800 and e['short_preflight']==e['tau1_replays']==e['full_parity']==8,'Execution')
        require(not inv['candidate_records'] and not inv['scan_errors'],'Inventory')
        v=pd.read_csv(z.open('run_validation.csv.gz'),compression='gzip'); cols=['seed','variant','suite','level','mode']
        require(set(map(tuple,v[cols].values))=={(s,*c) for s in SEEDS for c in m['grid']} and len(v)==800,'Coverage')
        require(v.periods.eq(40000).all() and v.floor_events.eq(0).all() and v.parity_pass.notna().sum()==8 and v.loc[v.parity_pass.notna(),'seed'].eq(31001).all() and v.parity_pass.dropna().eq(True).all(),'Validation')
        require(v.reference_sha256.nunique()==v.shard_sha256.nunique()==800,'Shard hashes')
        orig=pd.read_csv(z.open('contrast_summary.csv.gz'),compression='gzip'); require(len(orig)==5304,'Summary coverage')
        rows=[]
        for s in SEEDS:
            r=json.loads(gzip.decompress(z.read(f'seed_tables/seed_{s}.json.gz'))); require(len(r)==5304 and all(x['seed']==s for x in r),'Seed coverage'); rows.extend(r)
    oldfiles=sorted(root.glob('v62_7e2e4f6724e3_primary_*.zip')); require(len(oldfiles)==8,'Reference archives')
    inv2=pd.read_csv(root/'shard_inventory.csv.gz'); j=v.merge(inv2,on=cols,validate='one_to_one'); require(len(j)==800 and j.reference_sha256.eq(j.sha256).all(),'Reference shard provenance')
    df=pd.DataFrame(rows); a=df.pivot(index='seed',columns=KEYS,values='value').sort_index().sort_index(axis=1); require(a.shape==(100,5304) and a.index.tolist()==SEEDS,'Seed matrix')
    # Rebuild all derived contrasts from the two levels within each seed.
    diffs=[]
    for k in a.columns:
        kind,t,s,p,g,x=k
        def get(k,t=t,p=p): return a[(k,t,s,p,g,x)].to_numpy()
        if kind=='high_minus_low': pred=get('level_high')-get('level_low')
        elif kind=='tau2_minus_tau1': pred=get('high_minus_low')-get('high_minus_low','tau1')
        elif kind=='adaptive_minus_neutral': pred=get('high_minus_low',p='baseline')-get('high_minus_low',p='neutral')
        elif kind=='AN_tau2_minus_tau1': pred=get('adaptive_minus_neutral')-get('adaptive_minus_neutral','tau1')
        else: continue
        actual=a[k].to_numpy(); require(np.allclose(actual,pred,rtol=1e-12,atol=1e-12,equal_nan=True),'Contrast '+str(k)); diffs.append(float(np.nanmax(np.abs(actual-pred))) if np.isfinite(actual-pred).any() else 0.)
    print("Seed contrasts complete",len(diffs)*100,flush=True)
    # Compare every shared tau1 level statistic with the original V6.2 seed data.
    compare_count=0; max_ref=0.; seen=set(); ref_manifest=None
    ref_keys={k:i for i,k in enumerate(a.columns) if k[1]=="tau1" and k[0].startswith("level_")}; arr=a.to_numpy()
    for path in oldfiles:
        with zipfile.ZipFile(path) as z:
            if 'MANIFEST.json' in z.namelist(): ref_manifest=z.read('MANIFEST.json')
            for name in z.namelist():
                if not name.startswith('seed_tables/seed_') or not name.endswith('.json.gz'): continue
                s=int(Path(name).name.split('_')[1].split('.')[0]); require(s not in seen,'Duplicate baseline seed'); seen.add(s)
                for r in json.loads(gzip.decompress(z.read(name))):
                    if r['kind'] not in ('level_low','level_high') or r['variant'] not in ('baseline','neutral') or r['mode']!='adaptive': continue
                    k=(r['kind'],'tau1',r['suite'],r['variant'],r['group'],r['metric'])
                    if k not in ref_keys: continue
                    x=arr[s-SEEDS[0],ref_keys[k]]; y=r['value']; require((np.isnan(x) if y is None else abs(x-y)<=1e-12+1e-12*abs(y)),'Reference statistic '+str(k)); compare_count+=1
                    if y is not None: max_ref=max(max_ref,abs(x-y))
    print("Reference comparison complete",compare_count,flush=True)
    require(seen==set(SEEDS) and sha(ref_manifest)==m['source_manifest_sha256'],'Reference manifest/seed coverage')
    require(json.loads(ref_manifest)['source_hashes']==m['dependencies'],'Frozen dependency provenance')
    # Reproduce reported bootstrap, using shared whole-seed resampling counts.
    x=a.to_numpy(); draws=np.random.default_rng(m['bootstrap_seed']).integers(0,100,size=(m['bootstrap_draws'],100)); counts=np.zeros((len(draws),100)); np.add.at(counts,(np.arange(len(draws))[:,None],draws),1)
    results=[]
    for i in range(0,x.shape[1],128):
        y=x[:,i:i+128]; valid=np.isfinite(y); n=counts@np.where(valid,y,0.); d=counts@valid.astype(float); bs=np.divide(n,d,out=np.full_like(n,np.nan),where=d>0)
        for j,k in enumerate(a.columns[i:i+128]):
            u=y[valid[:,j],j]; b=bs[np.isfinite(bs[:,j]),j]; results.append(dict(zip(KEYS,k),n_valid=len(u),mean=u.mean() if len(u) else np.nan,lo=np.quantile(b,.025) if len(b) else np.nan,hi=np.quantile(b,.975) if len(b) else np.nan,mcse=u.std(ddof=1)/np.sqrt(len(u)) if len(u)>1 else np.nan,positive=np.mean(u>0) if len(u) else np.nan,negative=np.mean(u<0) if len(u) else np.nan,zero=np.mean(u==0) if len(u) else np.nan))
    rebuilt=pd.DataFrame(results); q=orig.merge(rebuilt,on=KEYS,validate='one_to_one',suffixes=('_stored','_recomputed')); require(len(q)==5304,'Summary key coverage')
    residual={}
    for c in ['mean','lo','hi','mcse','n_valid','positive','negative','zero']:
        u=q[c+'_stored']; w=q[c+'_recomputed']; require(np.allclose(u,w,rtol=1e-12,atol=1e-10,equal_nan=True),'Summary '+c); residual[c]=float((u-w).abs().max())
    rebuilt.to_csv(out/'recomputed_summary.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    focal=rebuilt[((rebuilt.suite=='H')&(rebuilt.group=='group_1'))|((rebuilt.suite=='G')&(rebuilt.group=='all'))]; focal.to_csv(out/'focal_results.csv',index=False)
    audit=dict(design=m['design_id'],zip_sha256=part['sha256'],members=105,validated_runs=800,reference_hash_matches=800,seed_statistics=len(df),summary_rows=len(q),reference_level_matches=compare_count,max_reference_residual=max_ref,derived_seed_identities=len(diffs)*100,max_identity_residual=max(diffs),bootstrap_draws=len(draws),summary_residual=residual,precision_residual=float(v.max_precision_residual.max()),state_residual=float(v.max_state_residual.max()),floor_events=int(v.floor_events.sum()),local_technical_gates=e,inventory_numeric_tau_records=len(inv['numeric_tau_records']),limitations='No raw W2 shards or independent simulation replay; checks of transferred seed statistics, reference provenance and author-reported runtime gates.')
    (out/'VALIDATION.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); main(a.root,a.out)

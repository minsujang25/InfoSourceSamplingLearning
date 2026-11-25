"""Verify complete Package C recovery without reading the damaged seed stream."""
import argparse, gzip, hashlib, io, json, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED = {
    'V61_Small_A_01_of_03': '97ec1681d552035cd00711640b557ab4e7c4463b2b07288a2ec5ae0b2d40a8a9',
    'V61_Small_C_01_of_03': '70519c9993576c3b7ee6f576c4a466bde1596c3f264b9550f26400523ec089cd',
    'V61_Small_C_02_of_03': 'a1cbac6d6f0f66eeacda8b301700612619679269e419c73b24eff40aeed548d8',
    'V61_Small_C_03_of_03': 'bcbb1a48c3566b4332235e036c4eae019ae7e8adef80a5aab76c985527e9ba0f',
}
KEY=['dataset','metric','citizen_group','reliance_mode','horizon_step','contrast']
SEED_KEY=KEY[:-1]+['seed','level']

def sha(b): return hashlib.sha256(b).hexdigest()

def main(root, out, partial):
    inputs={}; members=[]
    for stem, expected in EXPECTED.items():
        paths=[p for p in [root/(stem+'.zip'),root/(stem+'(1).zip')] if p.is_file()]
        assert paths and all(sha(p.read_bytes())==expected for p in paths),stem
        p=paths[0]; inputs[p.name]=expected
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            m=json.loads(z.read('MANIFEST.json'))
            assert m['source_sha256']=='dad2311e9c72fb5947f66df7484f1039c6978c95dfafe75a53330ec81062af7e'
            assert set(m['seeds'])==set(range(20001,20501))
            assert set(z.namelist())=={'MANIFEST.json'}|{x['name'] for x in m['members']}
            for x in m['members']:
                b=z.read(x['name']); assert sha(b)==x['sha256'] and len(b)==x['bytes']
                d=pd.read_csv(io.BytesIO(b),compression='gzip',low_memory=False)
                assert len(d)==x['rows'] and list(d.columns)==x['columns']
                members.append(dict(archive=p.name,member=x['name'],sha256=x['sha256'],rows=len(d)))
    with zipfile.ZipFile(root/'PaperB_V6.3_Author_Handoff(1).zip') as h:
        raw=h.read('history/v61/V61_Package_C_Analysis_Results.zip')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        original=z.read('analyze_package_C.py'); old=pd.read_csv(z.open('summaries.csv')); cap=pd.read_csv(z.open('initial_rank_capture.csv')); old_audit=json.loads(z.read('audit.json'))
        try: gzip.decompress(z.read('seed_statistics.csv.gz'))
        except EOFError: damaged=True
        else: damaged=False
    assert damaged
    (out/'analyze_package_C.py').write_bytes(original)
    rec=out/'recovered'; new=pd.read_csv(rec/'summaries.csv'); newcap=pd.read_csv(rec/'initial_rank_capture.csv'); seed=pd.read_csv(rec/'seed_statistics.csv.gz'); audit=json.loads((rec/'audit.json').read_text())
    assert audit==old_audit
    residuals={}
    def compare(label, left, right, keys):
        assert not left.duplicated(keys).any() and not right.duplicated(keys).any()
        cols=[k for k in left if k not in keys]; assert set(left)==set(right)
        a=left.sort_values(keys).reset_index(drop=True); b=right.sort_values(keys).reset_index(drop=True)
        assert a[keys].equals(b[keys]),label
        for k in cols:
            x=a[k].to_numpy(float); y=b[k].to_numpy(float)
            np.testing.assert_allclose(x,y,rtol=1e-10,atol=1e-10)
            residuals[label+'_'+k]=float(np.max(np.abs(x-y)))
    compare('summary',new,old,KEY)
    compare('conditional',newcap,cap,['level','reliance_mode','citizen_group','initial_rank'])
    assert len(new)==1152 and len(newcap)==32 and len(seed)==384000
    assert not seed.duplicated(SEED_KEY).any() and seed.groupby(SEED_KEY[:-2]).size().eq(1000).all()
    assert seed['value'].notna().all()
    prev=pd.read_csv(partial/'C_partial_seed_statistics.csv.gz'); prevsum=pd.read_csv(partial/'C_partial_summaries.csv')
    assert len(prev)==356000 and len(prevsum)==1068
    matched=seed.merge(prev[SEED_KEY],on=SEED_KEY,validate='one_to_one'); compare('partial_seed',matched,prev,SEED_KEY)
    matched_sum=new.merge(prevsum[KEY],on=KEY,validate='one_to_one'); compare('partial_summary',matched_sum,prevsum,KEY)
    missing=new.merge(prevsum[KEY],on=KEY,how='left',indicator=True); missing=missing[missing._merge=='left_only'].drop(columns='_merge')
    assert len(missing)==84
    missing.to_csv(out/'C_recovered_remaining_84.csv',index=False)
    pd.DataFrame(members).to_csv(out/'input_member_audit.csv',index=False)
    report=dict(status='PASS',input_hashes=inputs,original_package_sha256=sha(raw),original_analysis_sha256=sha(original),damaged_original_seed_stream=True,matched_terminal_rows=audit['matched_terminal_rows'],seed_statistics=len(seed),summary_rows=len(new),conditional_capture_rows=len(newcap),previous_partial_seed_rows=len(prev),recovered_additional_seed_rows=len(seed)-len(prev),additional_summary_rows=len(missing),max_abs_residual=residuals,software=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),new_simulations=0,output_hashes={p.name:sha(p.read_bytes()) for p in rec.iterdir() if p.is_file()})
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--partial',type=Path,required=True); a=p.parse_args(); main(a.root,a.out,a.partial)

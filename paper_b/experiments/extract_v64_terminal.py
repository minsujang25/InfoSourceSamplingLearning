"""Read-only V6.4.1 extension: extract RA/RB H terminal records from frozen production."""
import argparse,gzip,hashlib,json,math
from pathlib import Path

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path.home()/'Research/paperb_v6_results/v62_mechanism_closure/v62_7e2e4f6724e3')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    root=a.root;out=a.out or root/'v64_passive_transfer';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'MANIFEST.json').read_text());seeds=list(range(31001,31101))
    assert manifest['design_id']=='7e2e4f6724e3' and manifest['seeds']==seeds and manifest['horizon']==400 and manifest['expected_runs']==3400
    grid={tuple(x) for x in manifest['grid']};files=[];shards=[]
    fields=('citizen_pos','group','role','initial_mu','initial_sd','initial_phi','Q','B_raw','phi','mu','W_jammer','floor_events','signed_mse')
    for variant in ('RA','RB'):
        for mode in ('adaptive','frozen'):
            rows=[]
            for seed in seeds:
                pair={}
                for level in ('low','high'):
                    cell=(variant,'H',level,mode);assert cell in grid
                    stem='__'.join([str(seed),*cell]);path=root/'production'/(stem+'.json.gz')
                    digest=sha(path);assert (root/'production'/(stem+'.sha256')).read_text().strip()==digest,path
                    with gzip.open(path,'rt') as f:d=json.load(f)
                    assert (d['design_id'],d['phase'],d['seed'],d['variant'],d['suite'],d['level'],d['mode'],d['horizon'])==('7e2e4f6724e3','production',seed,*cell,400),path
                    assert d['validation']['periods']==40000 and d['validation']['floor_events']==0 and len(d['terminal'])==100
                    citizens={x['citizen_pos']:x for x in d['terminal']};assert set(citizens)==set(range(2,102))
                    for x in citizens.values():
                        assert x['floor_events']==0 and all(k in x for k in fields)
                        assert math.isclose(sum(x['Q'].values())+x['initial_phi'],x['phi'],rel_tol=1e-10,abs_tol=1e-9)
                        assert math.isclose(sum(x['B_raw'].values())/x['phi'],x['mu'],rel_tol=1e-10,abs_tol=1e-9)
                        rows.append(dict(seed=seed,variant=variant,suite='H',level=level,mode=mode,**{k:x[k] for k in fields}))
                    pair[level]=(citizens,d['run']['initial_state_fingerprint'])
                    shards.append(dict(file=path.name,sha256=digest))
                l,lf=pair['low'];h,hf=pair['high'];assert lf==hf
                for i in l:
                    assert all(l[i][k]==h[i][k] for k in ('group','role','initial_mu','initial_sd','initial_phi'))
            assert len(rows)==20000
            dest=out/f'{variant}_H_{mode}_terminal.json.gz'
            with gzip.open(dest,'wt',encoding='utf-8') as f:json.dump(rows,f,allow_nan=False,separators=(',',':'))
            files.append(dict(name=dest.name,bytes=dest.stat().st_size,sha256=sha(dest)))
    (out/'TRANSFER_V64.json').write_text(json.dumps(dict(design_id=manifest['design_id'],source_manifest_sha256=sha(root/'MANIFEST.json'),records=80000,files=files,shards=shards),indent=2))
    print('Upload folder:',out)

if __name__=='__main__':main()

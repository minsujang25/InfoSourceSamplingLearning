"""Read-only chronology reconstruction from frozen baseline H period histories."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import tempfile
import zipfile
from pathlib import Path

DESIGN = '7e2e4f6724e3'
MANIFEST = '3a83e3b7ff4af3ceeeaeda34bbd47633c9df07228089e25f6d57a56394cfff61'
INVENTORY = '356e6aaab87ce85bea879930962e6c8b1f9ba18bd9397b025273f9f2cf88a061'
SOURCE = '55264824e35c02284174e206d86955269d2b0bbc'
HEADER = 'citizen_pos period audit mu_pre sd_pre mu_post sd_post rank_pre rank_post latent_rank_pre latent_rank_post Q_expert_step Q_jammer_step Q_peer_step B_expert_step B_jammer_step B_peer_step peer_signed_update Q_jammer_cumulative'.split()
EVENTS = 'first_drift first_strict_drift first_direct promotion promotion_pre_mu promotion_pre_sd promotion_jammer_Q promotion_drift promotion_strict_drift postpromo_first_uptake postpromo_last_mu'.split()
TYPES = ('expert', 'jammer', 'peer')
CHECKPOINTS = (0,1,2,3,6,11,26,51,100,200,400)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-8 + 1e-10*max(abs(a),abs(b))


def replay(rows, terminal, checkpoints, mode, horizon=400):
    """Reconstruct events from chronological rows, without importing observer/model."""
    states = {}; maxima = dict(precision=0., posterior=0., peer_move=0., cumulative_jammer=0.)
    def check(k, a, b):
        require(equal(a,b), f'{k}: {a} vs {b}')
        maxima[k] = max(maxima[k], abs(a-b))
    def cp(pos, h, mu, sd, rank, latent):
        x = checkpoints[pos,h]
        require(equal(mu,x['mu']) and equal(sd,x['sd']) and (rank,latent)==(x['rank'],x['latent_rank']), 'Checkpoint mismatch')
    for row in rows:
        pos,t,a = map(int,row[:3]); require(pos in terminal and a in (0,1), 'Citizen/audit')
        pre,sd,post,psd = map(float,row[3:7]); r0,r1,l0,l1 = map(int,row[7:11]); q = list(map(float,row[11:14])); b = list(map(float,row[14:17])); move,cum = map(float,row[17:19])
        require(all(math.isfinite(x) for x in [pre,sd,post,psd,*q,*b,move,cum]) and min(sd,psd)>0 and min(q)>=0 and all(1<=x<=4 for x in (r0,r1,l0,l1)), 'Invalid numerical row')
        if t==0:
            require(pos not in states and a==1, 'Initial audit')
            cp(pos,0,pre,sd,r0,l0)
            s = {k:None for k in EVENTS}; s.update(initial_rank=r1, initial_latent_rank=l1, initial_mu=pre, initial_sd=sd, initial_phi=1/sd**2, audit_count=0, theta_count=0, postpromo_opportunities=0, postpromo_Q=0., Q=[0.]*3, B=[0.]*3, next_t=0)
            states[pos]=s
        s=states[pos]; require(t==s['next_t'] and t<horizon, 'Missing/duplicate/out-of-order citizen period')
        if t:
            require(equal(pre,s['mu']) and equal(sd,s['sd']) and (r0,l0)==(s['rank'],s['latent_rank']), 'State continuity')
        phi=1/sd**2; newphi=1/psd**2; prior_j=s['Q'][1]
        require(all(x!=0 or y==0 for x,y in zip(q,b)), 'Zero precision with nonzero numerator')
        if a:
            require(q==[0.,0.,0.] and b==[0.,0.,0.] and move==0 and equal(pre,post) and equal(sd,psd), 'Audit changed belief/evidence')
            s['audit_count']+=1
            if t and s['initial_rank']>=3 and s['promotion'] is None and r0!=1 and r1==1:
                s.update(promotion=t,promotion_pre_mu=pre,promotion_pre_sd=sd,promotion_jammer_Q=prior_j,promotion_drift=s['first_drift'],promotion_strict_drift=s['first_strict_drift'])
        else:
            require(t>0 and sum(q)>0 and (r0,l0)==(r1,l1), 'Substantive period')
            s['theta_count']+=1
            rawphi=phi+math.fsum(q); require(rawphi<=1e16,'Variance floor not supported')
            check('precision',newphi,rawphi); check('posterior',post,(phi*pre+math.fsum(b))/rawphi)
            check('peer_move',move,(b[2]-q[2]*pre)/rawphi)
            if q[1]>0 and s['first_direct'] is None: s['first_direct']=t
            # Use saved full-precision movement for the exact strict inequalities.
            if s['initial_rank']>=3 and s['promotion'] is None and q[1]==0 and (4-pre)*move>0 and abs(post-4)<abs(pre-4):
                if s['first_drift'] is None: s['first_drift']=t
                if prior_j==0 and s['first_strict_drift'] is None: s['first_strict_drift']=t
            if s['promotion'] is not None and t>s['promotion'] and s['postpromo_opportunities']<5:
                s['postpromo_opportunities']+=1; s['postpromo_Q']+=q[1]; s['postpromo_last_mu']=post
                if q[1]>0 and s['postpromo_first_uptake'] is None: s['postpromo_first_uptake']=t
            s['Q']=[x+y for x,y in zip(s['Q'],q)]; s['B']=[x+y for x,y in zip(s['B'],b)]
        require(r1==l1 if mode=='adaptive' else r1==s['initial_rank'], 'Behavioral rank policy')
        check('cumulative_jammer',cum,s['Q'][1]); check('precision',newphi,s['initial_phi']+math.fsum(s['Q']))
        check('posterior',post,(s['initial_phi']*s['initial_mu']+math.fsum(s['B']))/newphi)
        if t==1: s.update(early_audit=a,early_mu=post,early_sd=psd,early_precision=newphi,early_prior_weight=s['initial_phi']/newphi)
        s.update(mu=post,sd=psd,rank=r1,latent_rank=l1,next_t=t+1)
        if t+1 in CHECKPOINTS or t+1==horizon: cp(pos,t+1,post,psd,r1,l1)
    require(set(states)==set(terminal), 'Incomplete citizens')
    result=[]
    for pos,s in sorted(states.items()):
        require(s.pop('next_t')==horizon, 'Incomplete history')
        s['sequence']=int(s['promotion_drift'] is not None and s['postpromo_first_uptake'] is not None)
        s['strict_sequence']=int(s['sequence'] and s['promotion_strict_drift'] is not None and s['promotion_jammer_Q']==0)
        s['censored']=int(s['promotion'] is not None and s['postpromo_opportunities']<5)
        old=terminal[pos]; require(old['floor_events']==0, 'Archived floor')
        for k in EVENTS+['initial_rank','initial_latent_rank','initial_mu','initial_sd','initial_phi','audit_count','theta_count','postpromo_opportunities','postpromo_Q','mu','sd','rank','latent_rank','sequence','strict_sequence','censored']:
            require(equal(s[k],old[k]), f'Terminal mismatch citizen {pos}: {k}')
        for k,q,b in zip(TYPES,s['Q'],s['B']):
            require(equal(q,old['Q'][k]) and equal(b,old['B_raw'][k]), 'Terminal Q/B mismatch')
        s['Q']=dict(zip(TYPES,s['Q'])); s['B']=dict(zip(TYPES,s['B'])); s.update(citizen_pos=pos,group=old['group'])
        result.append(s)
    return result,maxima


def main(root, inventory, out):
    root,out=root.resolve(),out.resolve()
    require(out.suffix=='.zip' and not out.exists() and not out.is_relative_to(root), 'New output ZIP outside archive required')
    require(sha(root/'MANIFEST.json')==MANIFEST and sha(inventory)==INVENTORY, 'Manifest/inventory hash')
    with gzip.open(inventory,'rt',newline='') as f: inv=list(csv.DictReader(f))
    keys={(int(x['seed']),x['level'],x['mode']) for x in inv}
    require(len(inv)==400 and keys=={(s,l,m) for s in range(31001,31101) for l in ('low','high') for m in ('adaptive','frozen')}, 'Grid incomplete')
    out.parent.mkdir(parents=True,exist_ok=True)
    all_results=[]; records=[]; maxima={}; examples=[]; example_strict=False
    for x in sorted(inv,key=lambda x:(int(x['seed']),x['level'],x['mode'])):
        seed=int(x['seed']); level=x['level']; mode=x['mode']; stem=f'{seed}__baseline__H__{level}__{mode}'
        require(x['suite']=='H' and x['variant']=='baseline' and x['file']==stem+'.json.gz', 'Inventory identity')
        p=root/'production'/x['file']; h=p.with_name(stem+'.history.csv.gz'); digest=sha(p); hd=sha(h)
        require(digest==x['sha256']==p.with_name(stem+'.sha256').read_text().strip(), 'Shard checksum '+stem)
        with gzip.open(p,'rt') as f: obj=json.load(f)
        require((obj['design_id'],obj['phase'],obj['seed'],obj['variant'],obj['suite'],obj['level'],obj['mode'],obj['horizon'])==(DESIGN,'production',seed,'baseline','H',level,mode,400), 'Shard metadata')
        require(hd==obj['history_sha256'] and obj['validation']['periods']==40000 and obj['validation']['floor_events']==0,'History checksum/coverage '+stem)
        t={r['citizen_pos']:r for r in obj['terminal']}; c={(r['citizen_pos'],r['h']):r for r in obj['checkpoints']}
        require(len(t)==len(obj['terminal'])==100 and set(t)==set(range(2,102)) and len(c)==len(obj['checkpoints'])==1100 and set(c)=={(p,j) for p in t for j in CHECKPOINTS},'Shard records')
        require(sum(r['group']==1 for r in t.values())==sum(r['group']==-1 for r in t.values())==50,'Group counts')
        with gzip.open(h,'rt',newline='') as f:
            reader=csv.reader(f); require(next(reader)==HEADER,'History header'); result,err=replay(reader,t,c,mode)
        for k,v in err.items(): maxima[k]=max(maxima.get(k,0.),v)
        meta=dict(seed=seed,level=level,mode=mode)
        all_results.extend(dict(meta,**r) for r in result)
        records.append(dict(meta,shard_sha256=digest,history_sha256=hd,history_bytes=h.stat().st_size,rows=40000))
        # Fixed first seed/citizen examples plus lexicographically first strict sequence.
        selected={2} if seed==31001 else set()
        if not example_strict:
            strict=[r['citizen_pos'] for r in result if r['strict_sequence']]
            if strict: selected.add(min(strict)); example_strict=True
        if selected:
            with gzip.open(h,'rt',newline='') as f:
                for row in csv.DictReader(f):
                    if int(row['citizen_pos']) in selected: examples.append(dict(meta,**row))
        if len(records)%20==0: print(f'{len(records)}/400 verified',flush=True)
    require(len(all_results)==40000 and sum(r['rows'] for r in records)==16000000,'Incomplete replay')
    with tempfile.TemporaryDirectory(dir=out.parent,prefix='.v652c-') as td:
        d=Path(td); names=['replayed_citizens.json.gz','history_inventory.json','example_histories.csv.gz']
        with gzip.open(d/names[0],'wt') as f: json.dump(all_results,f,separators=(',',':'),allow_nan=False)
        (d/names[1]).write_text(json.dumps(records,indent=2)+'\n')
        with gzip.open(d/names[2],'wt',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['seed','level','mode']+HEADER); w.writeheader(); w.writerows(examples)
        transfer=dict(design_id=DESIGN,source_commit=SOURCE,manifest_sha256=MANIFEST,inventory_sha256=INVENTORY,script_sha256=sha(Path(__file__)),runs=400,citizens=40000,history_rows=16000000,examples=len(examples),max_residual=maxima,new_simulations=0,files=[dict(name=n,bytes=(d/n).stat().st_size,sha256=sha(d/n)) for n in names])
        (d/'TRANSFER_V652C.json').write_text(json.dumps(transfer,indent=2)+'\n')
        temp=d/'transfer.zip'
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for n in names+['TRANSFER_V652C.json']: z.write(d/n,n)
        with zipfile.ZipFile(temp) as z: require(z.testzip() is None,'ZIP integrity')
        require(not out.exists(),'Output exists'); temp.rename(out)
    print(f'{out}\nSHA256: {sha(out)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True,type=Path); p.add_argument('--inventory',required=True,type=Path); p.add_argument('--out',required=True,type=Path); a=p.parse_args(); main(a.root,a.inventory,a.out)

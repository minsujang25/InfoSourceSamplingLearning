"""V6.2 seed-level diagnostics and independent ZIP packaging."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from paper_b.experiments.v62_observer import require, COMPONENTS
from paper_b.experiments.v62_archive import digest
from paper_b.experiments.v62_policies import SEEDS, cells

LIMIT=20*1024**2
BOOT=20261125

def write_json(path,obj):
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False))
    tmp.replace(path)

def write_gz(path,obj):
    tmp=path.with_name(path.name+".tmp")
    with gzip.open(tmp,"wt",encoding="utf-8") as f:
        json.dump(obj,f,separators=(",",":"),allow_nan=False)
    tmp.replace(path)

def read_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        return json.load(f)

def ratio(n,d):
    return n/d if d>0 else None

def groups(rows):
    yield "all",rows
    for g in sorted({r["group"] for r in rows}):
        yield "group_"+str(g),[r for r in rows if r["group"]==g]
    for role in sorted({r["role"] for r in rows if r["role"]}):
        yield role,[r for r in rows if r["role"]==role]

def level_rows(o):
    result={}
    term=o["terminal"]
    for label,rr in groups(term):
        n=len(rr); ids={r["citizen_pos"] for r in rr}
        mus=np.array([r["mu"] for r in rr])
        def mean(fn): return sum(fn(r) for r in rr)/n
        vals=dict(n=n,mse=float(np.mean(mus**2)),mean=float(mus.mean()),variance=float(mus.var()),
                  rmse=float(np.sqrt(np.mean(mus**2))),bias2=float(mus.mean()**2),
                  Q_jammer=mean(lambda r:r["Q"]["jammer"]),W_jammer=mean(lambda r:r["W_jammer"]),
                  Q_expert=mean(lambda r:r["Q"]["expert"]),Q_peer=mean(lambda r:r["Q"]["peer"]),
                  near_truth=mean(lambda r:abs(r["mu"])<=.5),jammer_side=mean(lambda r:r["mu"]>2),
                  near_jammer=mean(lambda r:abs(r["mu"]-4)<=.5),
                  initial_rank1=mean(lambda r:r["initial_rank"]==1),
                  initial_policy=mean(lambda r:r["initial_policy"]),
                  initial_rank34=mean(lambda r:r["initial_rank"]>=3),
                  sequence=mean(lambda r:r["sequence"]),strict_sequence=mean(lambda r:r["strict_sequence"]),
                  promotion=mean(lambda r:r["promotion"] is not None),
                  censored=mean(lambda r:r["censored"]),
                  direct_first=mean(lambda r:r["first_direct"] is not None and
                       (r["first_drift"] is None or r["first_direct"]<r["first_drift"])),
                  terminal_sd=mean(lambda r:r["sd"]),audit_count=mean(lambda r:r["audit_count"]),
                  theta_count=mean(lambda r:r["theta_count"]))
        vals["sequence_conditional"]=ratio(sum(r["sequence"] for r in rr),sum(r["initial_rank"]>=3 for r in rr))
        for k in COMPONENTS:
            vals["signed_"+k]=mean(lambda r:r["signed"][k])
            vals["signed_mse_"+k]=mean(lambda r:r["signed_mse"][k])
        for w in ("early","middle","late","overall"):
            window_rows=[x for x in o["windows"] if x["citizen_pos"] in ids and (w=="overall" or x["window"]==w)]
            for kind in ("expert","jammer","peer","same_peer","other_peer","gateway_peer","non_gateway_peer"):
                xs=[x for x in window_rows if x["source_class"]==kind or (kind=="peer" and x["source_class"].endswith("_peer"))]
                if not xs:
                    continue
                totals={k:math.fsum(x[k] for x in xs) for k in (
                    "opportunities","expected_requests","requests","delivered","Q","B","Lnum","snapshot_B",
                    "snapshot_Lnum","snapshot_Q","observations","gain","signed_update","raw_draw_Lnum","prior_precision")}
                prefix=w+"_"+kind+"_"
                vals.update({prefix+k:v/n for k,v in totals.items()})
                vals[prefix+"weighted_mean"]=ratio(totals["B"],totals["Q"])
                vals[prefix+"weighted_error"]=ratio(totals["Lnum"],totals["Q"])
                vals[prefix+"snapshot_mean"]=ratio(totals["snapshot_B"],totals["snapshot_Q"])
                vals[prefix+"snapshot_error"]=ratio(totals["snapshot_Lnum"],totals["snapshot_Q"])
        result[label]=vals
    if o["suite"]=="H":
        result["all"]["group_gap"]=abs(result["group_1"]["mean"]-result["group_-1"]["mean"])
    return result

def sign_category(q,e):
    if q==0 or e==0: return "zero"
    return ("more" if q>0 else "less")+"_Q_"+("higher" if e>0 else "lower")+"_error"

CATEGORIES=("zero","more_Q_higher_error","more_Q_lower_error","less_Q_higher_error","less_Q_lower_error")

def individual_rows(low,high):
    high_by={r["citizen_pos"]:r for r in high["terminal"]}
    result={}
    for label,rr in groups(low["terminal"]):
        n=len(rr); by=defaultdict(list)
        for a in rr:
            b=high_by[a["citizen_pos"]]
            require(a["group"]==b["group"] and a["role"]==b["role"] and a["initial_mu"]==b["initial_mu"],"Citizen pairing")
            dq=b["Q"]["jammer"]-a["Q"]["jammer"]; de=b["mu"]**2-a["mu"]**2
            by[sign_category(dq,de)].append((a,b,dq,de))
        vals={}
        for cat in CATEGORIES:
            xs=by[cat]; count=len(xs); p=cat+"_"
            vals[p+"count"]=count
            vals[p+"share"]=count/n
            vals[p+"Q_contribution"]=sum(x[2] for x in xs)/n
            vals[p+"MSE_contribution"]=sum(x[3] for x in xs)/n
            vals[p+"Q_per_citizen"]=ratio(sum(x[2] for x in xs),count)
            vals[p+"MSE_per_citizen"]=ratio(sum(x[3] for x in xs),count)
            for lev,i in (("low",0),("high",1)):
                for k,fn in (("initial_rank",lambda r:r["initial_rank"]),("initial_mu",lambda r:r["initial_mu"]),
                             ("same_peers",lambda r:r["same_peer_count"]),("sequence",lambda r:r["sequence"]),
                             ("strict_sequence",lambda r:r["strict_sequence"]),("terminal_sd",lambda r:r["sd"]),
                             ("initial_rank34",lambda r:r["initial_rank"]>=3),("promotion",lambda r:r["promotion"] is not None)):
                    vals[p+lev+"_"+k]=ratio(sum(fn(x[i]) for x in xs),count)
            vals[p+"zero_Q"]=sum(x[2]==0 for x in xs)
            vals[p+"zero_error"]=sum(x[3]==0 for x in xs)
        result[label]=vals
    return result

def difference(a,b):
    return {k:(a[k]-b[k] if a[k] is not None and b[k] is not None else None)
            for k in a.keys()&b.keys() if k!="n"}

def build_seed(seed,outputs):
    idx={(o["variant"],o["suite"],o["level"],o["mode"]):o for o in outputs}
    require(set(idx)==set(cells()) and len(outputs)==34,"Incomplete seed")
    levels={k:level_rows(o) for k,o in idx.items()}
    deltas={}; cats={}; rows=[]
    def emit(kind,variant,suite,mode,group,vals):
        rows.extend(dict(seed=seed,kind=kind,variant=variant,suite=suite,mode=mode,
                         group=group,metric=k,value=v) for k,v in sorted(vals.items()))
    for (v,s,l,m),lev in levels.items():
        for g,vals in lev.items():
            emit("level_"+l,v,s,m,g,vals)
        if l!="high": continue
        low=levels[v,s,"low",m]
        cats[v,s,m]=individual_rows(idx[v,s,"low",m],idx[v,s,"high",m])
        for g,vals in lev.items():
            d=difference(vals,low[g])
            deltas[v,s,m,g]=d
            emit("high_minus_low",v,s,m,g,d)
            emit("sign_category",v,s,m,g,cats[v,s,m][g])
    for (v,s,m,g),d in list(deltas.items()):
        if m!="adaptive": continue
        if v!="baseline":
            emit("variant_minus_baseline",v,s,m,g,difference(d,deltas["baseline",s,m,g]))
        if v in ("baseline","RA","RB"):
            af=difference(d,deltas[v,s,"frozen",g])
            emit("adaptive_minus_frozen",v,s,m,g,af)
            if v!="baseline":
                af0=difference(deltas["baseline",s,m,g],deltas["baseline",s,"frozen",g])
                emit("AF_minus_baseline_AF",v,s,m,g,difference(af,af0))
        if v=="neutral":
            emit("adaptive_minus_neutral","baseline",s,m,g,difference(deltas["baseline",s,m,g],d))
    return rows

def summarize(rows,seeds):
    keyfields=("kind","variant","suite","mode","group","metric")
    series=defaultdict(dict)
    for r in rows:
        key=tuple(r[k] for k in keyfields)
        require(r["seed"] not in series[key],"Duplicate seed statistic")
        series[key][r["seed"]]=r["value"]
    n=len(seeds); draws=np.random.default_rng(BOOT).integers(0,n,size=(10000,n))
    result=[]
    for key,values in sorted(series.items()):
        x=np.array([values.get(s,np.nan) if values.get(s) is not None else np.nan for s in seeds],dtype=float)
        valid=np.isfinite(x); count=int(valid.sum())
        if count:
            sims=x[draws]
            den=np.isfinite(sims).sum(axis=1)
            bs=np.divide(np.nansum(sims,axis=1),den,out=np.full(len(den),np.nan),where=den>0)
            bs=bs[np.isfinite(bs)]
            lo,hi=np.quantile(bs,[.025,.975])
            result.append(dict(zip(keyfields,key),n_valid=count,n_expected=n,
                mean=float(np.mean(x[valid])),lo=float(lo),hi=float(hi),
                mcse=float(np.std(x[valid],ddof=1)/np.sqrt(count)) if count>1 else None,
                positive=float(np.mean(x[valid]>0)),negative=float(np.mean(x[valid]<0)),
                zero=float(np.mean(x[valid]==0))))
        else:
            result.append(dict(zip(keyfields,key),n_valid=0,n_expected=n,
                               mean=None,lo=None,hi=None,mcse=None,positive=None,negative=None,zero=None))
    return result

def csv_gz(path,rows):
    require(bool(rows),"Empty CSV")
    tmp=path.with_name(path.name+".tmp")
    with gzip.open(tmp,"wt",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)

def package(root,files,folder,design_id):
    """Independent, size-checked ZIPs with SHA-256 member manifests.
    Each member is a complete seed/run table or file, never a byte fragment."""
    dest=root/"shareable"/folder; dest.mkdir(parents=True,exist_ok=True)
    meta=[]
    for f in sorted(files):
        rel=str(f.relative_to(root))
        size=f.stat().st_size
        require(size<LIMIT-512*1024,"Oversize logical member: "+rel)
        meta.append(dict(path=rel,bytes=size,sha256=digest(f)))
    batches=[]; batch=[]; size=0
    for r in meta:
        estimated=r["bytes"]+512+2*len(r["path"])
        if batch and size+estimated>LIMIT-512*1024:
            batches.append(batch); batch=[]; size=0
        batch.append(r); size+=estimated
    if batch: batches.append(batch)
    parts=[]
    for i,batch in enumerate(batches,1):
        name=f"v62_{design_id}_{folder}_{i:03d}_of_{len(batches):03d}.zip"
        path=dest/name; tmp=path.with_suffix(".tmp")
        man=dict(design_id=design_id,category=folder,part=i,parts=len(batches),members=batch)
        with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_STORED) as z:
            for r in batch: z.write(root/r["path"],r["path"])
            z.writestr("PART_MANIFEST.json",json.dumps(man,indent=2))
        require(tmp.stat().st_size<=LIMIT,"ZIP exceeds 20 MiB")
        tmp.replace(path)
        parts.append(dict(file=name,bytes=path.stat().st_size,sha256=digest(path),members=len(batch)))
    index=dict(design_id=design_id,category=folder,parts=parts)
    write_json(dest/"INDEX.json",index)
    verify_package(dest)
    return index

def verify_package(folder):
    folder=Path(folder)
    idx=json.loads((folder/"INDEX.json").read_text())
    seen=set()
    for i,p in enumerate(idx["parts"],1):
        f=folder/p["file"]
        require(f.stat().st_size==p["bytes"]<=LIMIT and digest(f)==p["sha256"],"Part integrity")
        with zipfile.ZipFile(f) as z:
            man=json.loads(z.read("PART_MANIFEST.json"))
            require(man["design_id"]==idx["design_id"] and man["part"]==i and man["parts"]==len(idx["parts"]),"Part identity")
            require(set(z.namelist())=={"PART_MANIFEST.json"}|{r["path"] for r in man["members"]},"Member set")
            for r in man["members"]:
                require(r["path"] not in seen,"Duplicate member")
                seen.add(r["path"])
                data=z.read(r["path"])
                require(len(data)==r["bytes"] and hashlib.sha256(data).hexdigest()==r["sha256"],"Member integrity")
    return len(seen)

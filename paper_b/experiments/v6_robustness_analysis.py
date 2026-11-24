"""Frozen H/G robustness contrasts; compare variants against matched archive."""
from collections import defaultdict
from paper_b.experiments.v6_scientific_analysis import METRICS, summarize

EXTRA = ("requests_jammer","requests_expert","requests_same_peer","requests_other_peer",
         "request_share_jammer","request_share_expert","request_share_same_peer","request_share_other_peer")
MEASURES = (*METRICS,*EXTRA)


def terminal_index(groups, requests, horizon):
    q = {}
    for r in requests:
        if int(r["horizon_step"]) == horizon and r["learning_target"]=="theta":
            k=(r["run_id"],r["group_type"],str(r["group"]))
            if k in q:
                raise ValueError("Duplicate theta request row")
            q[k]=r
    index={}
    for r in groups:
        k=(r["cell"],r["sender_regime"],r["reliance_mode"],r["group_type"],str(r["group"]))
        if k in index:
            raise ValueError("Duplicate terminal group")
        rr=q[(r["run_id"],r["group_type"],str(r["group"]))]
        index[k]={m:float(r[m]) for m in METRICS}
        index[k].update({m:float(rr[m]) for m in EXTRA})
    return index


def contrasts(groups,requests,horizon,seed,suite,variant):
    idx=terminal_index(groups,requests,horizon)
    low,high=("H_low_S_high","H_high_S_high") if suite=="H" else ("G_low","G_high")
    prior="high" if suite=="H" else "flat"
    rows=[]
    for receiver in ("adaptive","frozen"):
        cats=sorted((k[3],k[4]) for k in idx if k[:3]==(low,"null",receiver))
        if not cats:
            raise ValueError("No groups")
        for kind,group in cats:
            def emit(name,sender,terms):
                r=dict(seed=seed,suite=suite,variant=variant,contrast=name,prior=prior,
                       sender_regime=sender,reliance_mode=receiver,group_type=kind,group=group)
                for m in MEASURES:
                    r["delta_"+m]=sum(w*idx[(cell,s,receiver,kind,group)][m] for cell,s,w in terms)
                if name=="high_minus_low":
                    for m in ("W","Q"):
                        r["divergence_"+m]=int(r["delta_"+m+"_jammer"]<0 and r["delta_mse_truth"]>0)
                rows.append(r)
            for sender in ("null","fixed_biased"):
                emit("high_minus_low",sender,[(high,sender,1),(low,sender,-1)])
            for level,cell in (("low",low),("high",high)):
                emit("damage_"+level,"fixed_biased",[(cell,"fixed_biased",1),(cell,"null",-1)])
            emit("damage_high_minus_low","fixed_biased",[(high,"fixed_biased",1),(high,"null",-1),
                                                        (low,"fixed_biased",-1),(low,"null",1)])
    return rows


def change_rows(rows,baseline):
    keys=("seed","suite","contrast","prior","sender_regime","reliance_mode","group_type","group")
    idx={tuple(r[k] for k in keys):r for r in baseline}
    if len(idx)!=len(baseline):
        raise ValueError("Duplicate baseline contrast")
    out=[]
    for r in rows:
        b=idx[tuple(r[k] for k in keys)]
        rr={k:r[k] for k in keys}
        rr["variant"]=r["variant"]+"_minus_baseline"
        rr.update({k:float(v)-float(b[k]) for k,v in r.items() if k.startswith("delta_")})
        out.append(rr)
    return out


def summarize_variants(rows,seeds):
    blocks=defaultdict(list)
    for r in rows:
        blocks[r["variant"]].append(r)
    return [dict(variant=v,**r) for v,block in sorted(blocks.items()) for r in summarize(block,seeds)]

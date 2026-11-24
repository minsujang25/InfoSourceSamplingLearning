"""Read-only final-production summaries from validated scientific shards."""
from collections import defaultdict
import math
import statistics
from unittest.mock import patch

import networkx as nx
from paper_b.experiments import v6_scientific_analysis as sci

CLASSES=("expert","jammer","same_peer","other_peer","gateway_peer")
REQUEST_METRICS=tuple(f"{prefix}_{c}" for prefix in ("requests","request_share") for c in CLASSES)
EXTRA=(*REQUEST_METRICS,"I_expert","I_same_peer","I_other_peer")
DIMENSIONS=("suite","contrast","prior","sender_regime","reliance_mode","group_type","group")


def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)


def extended_contrasts(outputs,horizon):
    enriched=[]
    for o in outputs:
        q={(r["group_type"],str(r["group"])):r for r in o["group_request_checkpoints"]
           if int(r["horizon_step"])==horizon and r["learning_target"]=="theta"}
        g=[]
        for r in o["group_terminal"]:
            rr=q[(r["group_type"],str(r["group"]))]
            require(rr["total_requests"]>0,"Undefined terminal substantive request share")
            g.append(dict(r,**{k:float(rr[k]) for k in REQUEST_METRICS}))
        enriched.append({"run":o["run"],"group_terminal":g})
    # Scope changes only the analysis list, not any model or old runner.
    with patch.object(sci,"METRICS",(*sci.METRICS,*EXTRA)):
        rows=sci.seed_contrasts(enriched)
    index={tuple(r[k] for k in DIMENSIONS):r for r in rows}
    require(len(index)==len(rows),"Duplicate contrast key")
    more=[]
    for r in rows:
        if r["reliance_mode"]!="adaptive":
            continue
        k=tuple("frozen" if field=="reliance_mode" else r[field] for field in DIMENSIONS)
        f=index[k]
        out={k:r[k] for k in ("seed",*DIMENSIONS)}
        out["reliance_mode"]="adaptive_minus_frozen"
        out.update({k:r[k]-f[k] for k in r if k.startswith("delta_")})
        more.append(out)
    for r in rows:
        if r["contrast"]=="high_minus_low":
            r["divergence_X"]=int(r["delta_requests_jammer"]<0 and r["delta_mse_truth"]>0)
            for v in ("X","Q","W"):
                metric={"X":"requests_jammer","Q":"Q_jammer","W":"W_jammer"}[v]
                r["inverse_divergence_"+v]=int(r["delta_"+metric]>0 and r["delta_mse_truth"]<0)
    return rows+more


def group_gap_rows(outputs):
    rows=[]
    for o in outputs:
        r=o["run"]
        if r["suite"]!="H":
            continue
        by_step=defaultdict(dict)
        for b in o["group_belief_checkpoints"]:
            if b["group_type"]=="citizen_group":
                by_step[int(b["horizon_step"])][str(b["group"])]=float(b["mean_belief"])
        for t,values in sorted(by_step.items()):
            require(set(values)=={"1","-1"},"Missing H group")
            rows.append(dict({k:r[k] for k in ("run_id","seed","suite","cell","prior","level",
                                               "sender_regime","reliance_mode")},
                             horizon_step=t,gap=abs(values["1"]-values["-1"])))
    return rows


def gap_contrasts(rows,horizon):
    idx={(r["level"],r["prior"],r["sender_regime"],r["reliance_mode"]):r
         for r in rows if r["horizon_step"]==horizon}
    out=[]
    if not idx:
        return out
    seed=rows[0]["seed"]
    for sender in ("null","fixed_biased"):
        for receiver in ("adaptive","frozen"):
            ds={}
            for prior in ("low","high"):
                ds[prior]=idx[("high",prior,sender,receiver)]["gap"]-idx[("low",prior,sender,receiver)]["gap"]
                out.append(dict(seed=seed,suite="H",contrast="high_minus_low",prior=prior,
                                sender_regime=sender,reliance_mode=receiver,group_type="all",group="all",
                                delta_group_gap=ds[prior]))
            out.append(dict(seed=seed,suite="H",contrast="H_by_S",prior="interaction",
                            sender_regime=sender,reliance_mode=receiver,group_type="all",group="all",
                            delta_group_gap=ds["high"]-ds["low"]))
    return out


def topology_rows(outputs):
    """Peer opportunity graph only; ego -> sampled source, not message flow."""
    by_level={}
    for o in outputs:
        r=o["run"]
        level=r["level"]
        nodes={int(b["citizen_pos"]):int(b["citizen_group"]) for b in o["beliefs"]}
        edges=[(int(e["ego"]),int(e["source"])) for e in o["edges"]
               if int(e["period"])==0 and e["source_type"]=="citizen"]
        require(len(nodes)==100 and len(edges)==200 and len(set(edges))==200
                and all(a!=b and a in nodes and b in nodes for a,b in edges),"Invalid peer topology")
        identity=(r["structural_fingerprint"],set(edges),nodes)
        if level in by_level:
            require(identity==by_level[level][0],"Peer topology varies within level")
            continue
        g=nx.DiGraph()
        g.add_nodes_from(nodes);g.add_edges_from(edges)
        outdeg=[v for _,v in g.out_degree()]
        indeg=[v for _,v in g.in_degree()]
        require(set(outdeg)=={2},"Peer outdegree not fixed")
        same=sum(nodes[a]==nodes[b] for a,b in edges)/len(edges)
        row=dict(seed=r["seed"],suite=r["suite"],level=level,
                 structural_fingerprint=r["structural_fingerprint"],n_citizens=len(nodes),
                 peer_edges=len(edges),peer_outdegree=2,peer_indegree_mean=statistics.mean(indeg),
                 peer_indegree_sd=statistics.pstdev(indeg),directed_reciprocity=float(nx.reciprocity(g)),
                 undirected_transitivity=float(nx.transitivity(g.to_undirected())),
                 same_group_peer_share=same,ei_index=1-2*same)
        by_level[level]=(identity,row)
    require(set(by_level)=={"low","high"},"Missing topology level")
    return [by_level[k][1] for k in sorted(by_level)]

"""V6.2 read-only archive/reconstruction gates."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import networkx as nx
import model.InfoSourceSamplingLearning as m
from paper_b.experiments import run_v6_pilot as base, run_v6_scientific as sci
from paper_b.experiments import run_matched_pilot as up, run_v6_final_production as final
from paper_b.experiments.v62_policies import SEEDS
from paper_b.experiments.v62_observer import require, close

ARCHIVES={
    "production":("paper_b_v6_final_production_e86f6ff3be12_shareable.zip","0fd01381ddc86f6828fda5aab2e7b3a9d83b358508bb901d21acae76ecf27ce0","e86f6ff3be12"),
    "pilot":("paper_b_v6_scientific_85faac503d4a_shareable.zip","084f1558b9f5c360861c75e1889467be9439b9bc84505d0f4a9fe3ce577925a3","85faac503d4a"),
    "RA":("paper_b_v61_RA_robustness_cde64cbb5c54_shareable.zip","ea341fa55b227de3959e4b5b9533fdaae459ec3b0c6a7764f957d3be309cf1cb","cde64cbb5c54"),
    "RB":("paper_b_v61_RB_robustness_c7708a68be6e_shareable.zip","26a5a12a8b6ef2267b1f2564204380729e7ac6f99198b86f00d3ad96652c5604","c7708a68be6e"),
    "behavioral":("paper_b_v6_robustness_7247f5ad0d7a_behavioral_shareable.zip","271268dd0b29faa1ce0563d4110de8cc1b12d863437a8405f6b1fc838fce9e84","7247f5ad0d7a"),
}
KNOWN_OLD=set(range(20001,20501))|set(range(6001,6021))|set(range(7501,7601))|{19999,8001,8002}

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def read_rows(z, table, predicate=lambda r: True):
    with z.open(table+".csv.gz") as f:
        with gzip.GzipFile(fileobj=f) as g:
            return [r for r in csv.DictReader(io.TextIOWrapper(g,encoding="utf-8",newline="")) if predicate(r)]

def locate(roots,explicit):
    paths={}
    for tag,(name,sha,_) in ARCHIVES.items():
        candidates=[Path(explicit[tag]).expanduser()] if explicit.get(tag) else [
            p for root in roots if root.is_dir() for p in root.rglob(name)]
        if not candidates:
            require(tag not in ("production","pilot"),f"Missing required archive {name}; supply --{tag}-zip or --archive-root")
            paths[tag]=None
            continue
        good=[p for p in candidates if p.is_file() and digest(p)==sha]
        require(bool(good),f"Archive SHA mismatch: {tag}")
        paths[tag]=good[0].resolve()
    return paths

def load(paths,roots):
    require(not (set(SEEDS)&KNOWN_OLD),"Known seed overlap")
    data={}; inventory={}
    for tag,p in paths.items():
        if p is None:
            inventory[tag]={"status":"not_found_optional"}
            continue
        _,sha,did=ARCHIVES[tag]
        with zipfile.ZipFile(p) as z:
            prefix="v61" if tag in ("RA","RB") else "v6"
            man=json.loads(z.read(prefix+"_manifest.json"))
            gate=json.loads(z.read(prefix+"_gate.json"))
            require(man["design_id"]==did and gate["design_id"]==did and gate["pass"] is True,"Archive gate "+tag)
            require(not set(SEEDS)&set(man["seeds"]),"Seed overlap "+tag)
            require(man["software_versions"]==up.software_versions(),"Software mismatch "+tag)
            d=dict(man); d.pop("design_id")
            require(up.canonical_hash(d)[:12]==did,"Archive canonical identity "+tag)
            require(man.get("upstream_fingerprint",man.get("model_fingerprint"))==up.code_fingerprint()==sci.EXPECTED_UPSTREAM,"Upstream fingerprint "+tag)
            require(man.get("scientific_runner_fingerprint",man.get("scientific_fingerprint"))==sci.fingerprint(),"Scientific fingerprint "+tag)
            if tag=="production":
                require(man["final_runner_fingerprint"]==final.fingerprint(),"Production fingerprint")
                require(man["seeds"]==list(range(20001,20501)) and gate["grid_runs"]==12000,"Production sample")
            pred=lambda r: r.get("sender_regime")=="fixed_biased" and (r.get("suite")=="G" or r.get("prior")=="high")
            seed=20001 if tag=="production" else 6001
            pred_old=lambda r: pred(r) and int(r["seed"])==seed
            tables={}
            for name in ("runs","group_terminal","beliefs","channel_citizens","capture","group_belief_checkpoints"):
                if name+".csv.gz" in z.namelist():
                    tables[name]=read_rows(z,name,(lambda r: pred(r) and (int(r["seed"])==seed or r["suite"]=="G")) if tag=="production" and name=="runs" else pred_old)
            if tag=="production":
                tables["peer_topology"]=read_rows(z,"peer_topology",lambda r:r["suite"]=="G")
        data[tag]=tables
        inventory[tag]=dict(status="verified",sha256=sha,design_id=did,seeds=man["seeds"])
    # Check seed declarations of other accessible archived ZIP manifests.
    scanned=[]
    for p in sorted({p.resolve() for root in roots if root.is_dir() for p in root.rglob("*.zip")}):
        if p in paths.values():
            continue
        try:
            with zipfile.ZipFile(p) as z:
                for n in ("v6_manifest.json","v61_manifest.json","pilot_manifest.json"):
                    if n not in z.namelist():
                        continue
                    obj=json.loads(z.read(n))
                    seeds=obj.get("seeds")
                    if isinstance(seeds,list) and all(isinstance(x,int) for x in seeds):
                        require(not set(SEEDS)&set(seeds),"Seed overlap in "+str(p))
                        scanned.append(dict(file=p.name,manifest=n,n_seeds=len(seeds)))
        except zipfile.BadZipFile:
            # Unrelated damaged ZIPs cannot establish seed provenance.
            scanned.append(dict(file=p.name,status="unreadable"))
    inventory["other_manifest_scan"]=scanned
    return data,inventory

def compare_replay(out,tag,tables,variant):
    run=out["run"]; rid=run["run_id"]
    def matching(row):
        return row["run_id"]==rid and (tag!="behavioral" or row.get("variant")==variant)
    old=[r for r in tables["runs"] if matching(r)]
    require(len(old)==1,"Missing archived replay run "+tag+" "+rid)
    old=old[0]
    for k in ("structural_fingerprint","initial_state_fingerprint"):
        require(old[k]==run[k],"Replay fingerprint "+k)
    fields=("mse_truth","mean_belief","Q_jammer_channel","Q_expert_channel","W_jammer_channel",
            "W_peer_incoming_hhi","W_peer_incoming_top5_share")
    for k in fields:
        require(close(float(old[k]),float(run[k])),"Replay "+tag+" "+rid+" "+k)
    coverage={"runs":1}
    for table,keys,metrics in [
        ("group_terminal",("group_type","group"),("mse_truth","mean_belief","Q_jammer","Q_expert","W_jammer")),
        ("beliefs",("citizen_pos",),("terminal_mu_theta","terminal_sd_theta")),
        ("channel_citizens",("citizen_pos",),("Q_jammer","Q_expert","W_jammer")),
        ("group_belief_checkpoints",("group_type","group","horizon_step"),("mse_truth","mean_belief"))]:
        if table not in tables:
            continue
        ref={tuple(str(r[k]) for k in keys):r for r in tables[table] if matching(r)}
        require(bool(ref),"Missing replay table "+table)
        actual={tuple(str(r[k]) for k in keys):r for r in out[table]}
        require(set(ref)==set(actual),"Replay table keys "+table)
        for key,r in ref.items():
            for k in metrics:
                require(close(float(r[k]),float(actual[key][k])),"Replay table mismatch "+table+" "+k)
        coverage[table]=len(ref)
    return dict(tag=tag,variant=variant,run_id=rid,coverage=coverage,passed=True)

def structural_benchmark(tables):
    refs={(int(r["seed"]),r["level"]):r for r in tables["peer_topology"]}
    runs={(int(r["seed"]),r["level"],r["reliance_mode"]):r for r in tables["runs"] if r["suite"]=="G"}
    require(len(refs)==1000 and len(runs)==2000,"Incomplete production G reference")
    rows=[]
    for seed in range(20001,20501):
        for level in ("low","high"):
            cfg,env,_,gateways=base.config_for(seed,"G",level,"flat",400)
            cfg.update(network_environment=env,peer_signal_mode="live",reliance_mode="adaptive",
                jammer_active=False,jammer_regime="fixed_biased",surveil_ability=1,
                stop_on_convergence=False,peer_evidence_mode="source_posterior",frozen_ranking_mode="first_audit",
                initial_audit_mode="centroid",tau_social=1.,jammer_credibility_penalty=1.)
            model=m.InfoSampleModel(model_attribute=cfg,rng=seed)
            r=refs[seed,level]
            require(up.structural_fingerprint(model)==r["structural_fingerprint"],"G reconstruction fingerprint")
            graph=nx.DiGraph(); graph.add_nodes_from(int(c.pos) for c in model.citizens)
            mass={int(c.pos):0. for c in model.citizens}
            for c in model.citizens:
                for s in c.info_source:
                    if s.type_of_agent=="citizen":
                        graph.add_edge(int(c.pos),int(s.pos))
                        mass[int(s.pos)]+=1/len(c.info_source)
            require(graph.number_of_edges()==200 and all(d==2 for _,d in graph.out_degree()),"G peer degree")
            indegree_sd=float(np.std([d for _,d in graph.in_degree()]))
            transitivity=float(nx.transitivity(graph.to_undirected()))
            require(close(indegree_sd,float(r["peer_indegree_sd"])) and close(transitivity,float(r["undirected_transitivity"])),"G topology mismatch")
            vals=np.array(list(mass.values())); vals/=vals.sum()
            hhi=float(np.sum(vals**2)); top5=float(np.sort(vals)[-5:].sum())
            for mode in ("adaptive","frozen"):
                old=runs[seed,level,mode]
                require(old["structural_fingerprint"]==r["structural_fingerprint"],"G archived fingerprint")
                wh=float(old["W_peer_incoming_hhi"]); wt=float(old["W_peer_incoming_top5_share"])
                rows.append(dict(seed=seed,level=level,mode=mode,U_hhi=hhi,U_top5=top5,
                    W_hhi=wh,W_top5=wt,excess_hhi=wh-hhi,excess_top5=wt-top5,
                    indegree_sd=indegree_sd,transitivity=transitivity,
                    gateway_opportunities=sum(b in gateways for a,b in graph.edges())/200,
                    structural_fingerprint=r["structural_fingerprint"]))
    return rows

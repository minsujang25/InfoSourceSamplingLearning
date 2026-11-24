"""Bounded strategic-sender scope diagnostic. No archived code modified."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from unittest.mock import patch
import zipfile

import model.InfoSourceSamplingLearning as m
from paper_b.experiments import run_v6_pilot as base
from paper_b.experiments import run_v6_scientific as sci
from paper_b.experiments import run_matched_pilot as up
from paper_b.experiments import v6_channel_bridge_analysis as archive_api
from paper_b.experiments.run_v6_robustness import audit_t0
from paper_b.experiments.v6_channel_bridge_analysis import summarize

CHANNELS=("elite_only","extended_live")
SENDERS=("null","adaptive","credibility_aware")
RECEIVERS=("adaptive","frozen")
SETTINGS={"baseline":(1,5),"segmented4":(4,5),"refresh1":(1,1)}
VARIANTS=("segmented4","refresh1")
PHASES={"preflight":((7501,),26,tuple(SETTINGS)),
        "strategic":(tuple(range(7501,7521)),400,VARIANTS)}
REP=("extended_live","extended_live","flat","credibility_aware","adaptive")
FIELDS=sci.FIELDS
ORIGINAL_RUN=base.run_condition


def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)


def cells():
    return [c for c in base.cells("C") if c[0] in CHANNELS and c[3] in SENDERS]


def set_floor():
    m.MIN_SD=1e-8
    m.MIN_VAR=m.MIN_SD**2


@contextmanager
def variant_scope(variant):
    k,interval=SETTINGS[variant]
    def run(**kw):
        kw["k"]=k
        kw["base_config"]=dict(kw["base_config"],surveillance_interval=interval)
        return ORIGINAL_RUN(**kw)
    # Native model uses the same K for initial credibility establishment
    # and all subsequent surveillance refreshes.
    with patch.object(base,"run_condition",run):
        yield


def fingerprint():
    paths=("paper_b/experiments/run_v6_strategic_scope.py",
           "paper_b/experiments/v6_channel_bridge_analysis.py",
           "paper_b/experiments/run_v6_robustness.py",
           "paper_b/V6_STRATEGIC_SCOPE_PROTOCOL.md")
    return up.canonical_hash({"scientific":sci.fingerprint(),
       "files":{p:hashlib.sha256((base.ROOT/p).read_bytes()).hexdigest() for p in paths}})


def validate_strategy(out,horizon,variant):
    k,interval=SETTINGS[variant]
    r=out["run"]; sender=r["sender_regime"]
    require(int(r["K"])==k and r["variant"]==variant
            and r["initial_audit_K"]==k and r["strategic_K"]==k
            and r["surveillance_interval"]==interval,"Incorrect strategic settings")
    periods=defaultdict(list)
    for row in out["jammer_strategy"]:
        periods[int(row["period"])].append(row)
    require(set(periods)==({0} if sender=="null" else set(range(horizon))),
            "Incomplete sender strategy trajectory")
    init=periods[0]
    require(len(init)==k and len({int(x["cluster"]) for x in init})==k
            and sum(int(x["cluster_size"]) for x in init)==base.N
            and all(int(x["cluster_size"])>0 and x["strategy_phase"]=="initial_audit"
                    and x["refresh"] is True and float(x["message_sd"])==1.
                    and math.isfinite(float(x["segment_mean"]))
                    and float(x["message_mean"])==float(x["segment_mean"]) for x in init),
            "Native K centroid audit failed")
    if sender=="null":
        return
    prior=None
    for t in range(1,horizon):
        rows=periods[t]
        refresh=(t-1)%interval==0
        require(1<=len(rows)<=k and sum(int(x["cluster_size"]) for x in rows)==base.N,
                "Invalid audience segmentation")
        require(len({int(x["cluster"]) for x in rows})==len(rows),"Duplicate segment")
        if t==1:
            require(len(rows)==k,"First strategic refresh did not use requested K")
        require(all(x["refresh"] is refresh and x["strategy_phase"]=="adaptive_disruption"
                    and x["optimization_rule"]==sender and float(x["message_sd"])==1.
                    and math.isfinite(float(x["message_mean"]))
                    and float(x["objective_denominator"])>0 for x in rows),
                "Invalid strategic refresh or objective")
        # State stored by original model, held unchanged between refreshes.
        current={int(x["cluster"]):(float(x["message_mean"]),int(x["cluster_size"])) for x in rows}
        if not refresh:
            require(current==prior,"Strategy changed between refreshes")
        prior=current


def validate_within_k_audit(outputs):
    """Match audit exposure and credibility within K and environment.

    Null is forced last by design, so its behavioral rank is not required
    to match active senders. Active myopic/aware and adaptive/frozen share
    the same initial credibility and behavioral rank.
    """
    signatures=audit_t0(outputs)
    for env in CHANNELS:
        subset=[o for o in outputs if o["run"]["cell"]==env]
        require(len(subset)==6,"Incomplete within-K audit block")
        common=None
        active=None
        clusters=None
        for o in subset:
            r=o["run"]
            rows=signatures[r["run_id"]]
            require(len(rows)==base.N,"Incomplete initial audit events")
            # audit_t0's final field is jammer_rank_post.
            raw={pos:values[:-1] for pos,values in rows.items()}
            if common is None:
                common=raw
            else:
                require(raw==common,"Within-K audit message or credibility mismatch")
            if r["sender_regime"]!="null":
                if active is None:
                    active=rows
                else:
                    require(rows==active,"Active sender/receiver initial rank mismatch")
            else:
                events=[e for e in o["sender_events"] if int(e["period"])==0]
                require(all(int(e["jammer_rank_post"])==int(e["source_count"]) for e in events),
                        "Null slot not ranked last")
            partition=sorted((int(x["cluster"]),int(x["cluster_size"]),float(x["segment_mean"]))
                             for x in o["jammer_strategy"] if int(x["period"])==0)
            if clusters is None:
                clusters=partition
            else:
                require(partition==clusters,"Within-K centroid partition mismatch")


def validate(outputs,seed,horizon,variant,did):
    expected={(c[0],c[3],c[4]) for c in cells()}
    actual={(o["run"]["cell"],o["run"]["sender_regime"],o["run"]["reliance_mode"]) for o in outputs}
    require(len(outputs)==12 and expected==actual,"Incomplete strategic grid")
    base.check_block(outputs,"strategic_generic")
    validate_within_k_audit(outputs)
    for o in outputs:
        r=o["run"]
        require(r["seed"]==seed and r["design_id"]==did and r["suite"]=="C","Wrong run identity")
        base.check_run(o,horizon)
        validate_strategy(o,horizon,variant)
        groups={(x["group_type"],x["group"]):int(x["n_citizens"]) for x in o["group_terminal"]}
        require(groups[("all","all")]==base.N and len(groups)==len(o["group_terminal"]),"Missing groups")
        steps={0,horizon,*(int(x["horizon_step"]) for x in o["belief_checkpoints"])}
        b=o["group_belief_checkpoints"]; q=o["group_request_checkpoints"]
        eb={(s,k,g) for s in steps for k,g in groups}
        eq={(s,k,g,t) for s,k,g in eb for t in ("all","theta","credibility")}
        require(len(b)==len(eb) and {(x["horizon_step"],x["group_type"],x["group"]) for x in b}==eb,
                "Missing belief checkpoints")
        require(len(q)==len(eq) and {(x["horizon_step"],x["group_type"],x["group"],x["learning_target"]) for x in q}==eq,
                "Missing request checkpoints")
        for x in q:
            require(x["total_requests"]==x["citizen_periods"]*base.CREDIT,"Request budget mismatch")
            if x["learning_target"]=="all":
                require(x["citizen_periods"]==x["n_citizens"]*x["horizon_step"],"Incomplete request history")


def worker(seed,variant,horizon,did,root):
    set_floor()
    with variant_scope(variant):
        outputs=[sci.run_instrumented(seed,"C",c,horizon,did) for c in cells()]
        chosen=next(o for o in outputs if o["run"]["cell"]==REP[0]
                    and o["run"]["sender_regime"]==REP[3] and o["run"]["reliance_mode"]==REP[4])
        plain,_=base.run_one(seed,"C",REP,horizon,did,instrument=False)
        for key in ("beliefs","belief_checkpoints","lambda_checkpoints","jammer_strategy"):
            require(base.parity_payload(chosen[key])==base.parity_payload(plain[key]),"Logger parity failed: "+key)
    k,interval=SETTINGS[variant]
    for o in outputs:
        meta=dict(variant=variant,initial_audit_K=k,strategic_K=k,surveillance_interval=interval)
        o["run"].update(meta)
        for key in (*base.KEYS,"group_terminal","group_acquisition","group_belief_checkpoints","group_request_checkpoints"):
            for row in o[key]:
                row.update(meta)
    validate(outputs,seed,horizon,variant,did)
    path=Path(root)/"shards"/f"{variant}__{seed}.json.gz"
    tmp=path.with_suffix(".tmp")
    with gzip.open(tmp,"wt",encoding="utf-8") as f:
        json.dump(dict(design_id=did,seed=seed,variant=variant,parity_pass=True,outputs=outputs),f,allow_nan=True)
    os.replace(tmp,path)
    return seed,variant


def read_shard(root,seed,variant,horizon,did):
    with gzip.open(root/"shards"/f"{variant}__{seed}.json.gz","rt",encoding="utf-8") as f:
        shard=json.load(f)
    require((shard["design_id"],shard["seed"],shard["variant"],shard["parity_pass"])==
            (did,seed,variant,True),"Shard identity mismatch")
    validate(shard["outputs"],seed,horizon,variant,did)
    return shard["outputs"]


def compare_block(outputs,a,horizon):
    for o in outputs:
        r=o["run"]
        old=a["runs"][(int(r["seed"]),r["network_environment"],"null",r["reliance_mode"])]
        for k in ("initial_state_fingerprint","structural_fingerprint"):
            require(r[k]==old[k],"Archive structure mismatch")
        # K4 changes the initial audit even for the null sender.
        if horizon==400 and int(r["initial_audit_K"])==1:
            archive_api.compare_archive(o,a)


def baseline_replay(a,did):
    set_floor()
    chosen=[c for c in cells() if c[3]=="null"]
    chosen += [("extended_live","extended_live","flat",s,"frozen") for s in SENDERS[1:]]
    rows=[]
    for cell in chosen:
        with variant_scope("baseline"):
            o=sci.run_instrumented(7501,"C",cell,400,did)
        archive_api.compare_archive(o,a,replay=True)
        rows.append(dict(run_id=o["run"]["run_id"],pass_=True))
    return rows


def contrasts(seed,variant,paths):
    # paths keyed by seed, environment, sender, receiver
    steps=sorted(next(iter(paths.values())))
    result=[]
    for receiver in RECEIVERS:
        for sender in SENDERS[1:]:
            for t in steps:
                en,ln=(paths[(seed,c,"null",receiver)][t] for c in CHANNELS)
                es,ls=(paths[(seed,c,sender,receiver)][t] for c in CHANNELS)
                de,dl=es-en,ls-ln
                result.append(dict(seed=seed,variant=variant,reliance_mode=receiver,sender_regime=sender,
                    horizon_step=t,D_E=de,D_L=dl,D_L_minus_E=dl-de,
                    sender_MSE_E=es,sender_MSE_L=ls,sender_MSE_L_minus_E=ls-es,
                    null_MSE_E=en,null_MSE_L=ln,null_MSE_L_minus_E=ln-en))
    return result


def differences(rows,baseline):
    fields=("seed","reliance_mode","sender_regime","horizon_step")
    index={tuple(r[k] for k in fields):r for r in baseline}
    require(len(index)==len(baseline),"Duplicate baseline contrasts")
    return [dict({k:r[k] for k in fields},variant=r["variant"]+"_minus_baseline",
                 **{k:r[k]-index[tuple(r[f] for f in fields)][k] for k in r
                    if k.startswith(("D_","sender_MSE_","null_MSE_"))}) for r in rows]


def records(out,key):
    return [out["run"]] if key=="run" else out[key]


def consolidate(root,d,a):
    seeds=d["seeds"];h=d["horizon"];did=d["design_id"]
    columns=defaultdict(set);counts=defaultdict(int);all_rows=[]; audit_rows=[]
    for seed in seeds:
        old=None
        if h==400:
            paths={k:v for k,v in a["checkpoint"].items() if k[0]==seed and k[1] in CHANNELS}
            old=contrasts(seed,"archived_baseline",paths)
            all_rows.extend(old)
        sig=None
        for variant in d["variants"]:
            outputs=read_shard(root,seed,variant,h,did)
            compare_block(outputs,a,h)
            if SETTINGS[variant][0]==1:
                now=audit_t0(outputs)
                if sig is None:
                    sig=now
                else:
                    require(now==sig,"K1 audit changed with surveillance interval")
            # Initial credibility and rank are outcomes of K. Refresh1
            # supplies a K1 comparator in every full-run seed.
            for o in outputs:
                r=o["run"]
                events=[e for e in o["sender_events"] if int(e["period"])==0]
                audit_rows.append(dict(seed=seed,variant=variant,network_environment=r["network_environment"],
                     sender_regime=r["sender_regime"],reliance_mode=r["reliance_mode"],
                     initial_audit_K=r["initial_audit_K"],n_citizens=len(events),
                     mean_abs_jammer_delta=sum(abs(float(e["jammer_delta_mu_post"])) for e in events)/len(events),
                     mean_jammer_rank=sum(int(e["jammer_rank_post"]) for e in events)/len(events),
                     top_jammer_share=sum(int(e["jammer_rank_post"])==1 for e in events)/len(events)))
            paths={archive_api.key(o["run"]):{int(r["horizon_step"]):float(r["mse_truth"])
                   for r in o["belief_checkpoints"]} for o in outputs}
            rows=contrasts(seed,variant,paths);all_rows.extend(rows)
            if old is not None:
                all_rows.extend(differences(rows,old))
            for o in outputs:
                for key in FIELDS:
                    for row in records(o,key):
                        columns[key].update(row);counts[key]+=1
    require(counts["run"]==d["grid_runs"],"Incomplete strategic grid")
    summary=summarize(all_rows,seeds,("variant","sender_regime","reliance_mode","horizon_step"))
    with ExitStack() as stack:
        writers={}
        for key in FIELDS:
            if columns[key]:
                name="runs" if key=="run" else key
                f=stack.enter_context(gzip.open(root/f"{name}.csv.gz","wt",encoding="utf-8",newline=""))
                w=csv.DictWriter(f,fieldnames=sorted(columns[key]));w.writeheader();writers[key]=w
        for seed in seeds:
            for variant in d["variants"]:
                for o in read_shard(root,seed,variant,h,did):
                    for key,w in writers.items():
                        w.writerows(records(o,key))
    base.csv_gz(root/"initial_audit_seed_metrics.csv.gz",audit_rows)
    base.csv_gz(root/"strategic_seed_contrasts.csv.gz",all_rows)
    base.csv_gz(root/"strategic_contrast_summary.csv.gz",summary)
    return dict(counts),len(all_rows)


def design_for(phase):
    seeds,horizon,variants=PHASES[phase]
    d=dict(phase=phase,seeds=list(seeds),horizon=horizon,variants=list(variants),
           N=100,credit=20,epsilon=.05,tau_social=1.,initial_audit="centroid",
           frozen_ranking="first_audit",peer_evidence="source_posterior",cells=cells(),
           settings={v:{"initial_audit_K":SETTINGS[v][0],"strategic_K":SETTINGS[v][0],
                        "surveillance_interval":SETTINGS[v][1]} for v in variants},
           grid_runs=len(seeds)*len(variants)*12,parity_runs=len(seeds)*len(variants),
           archive_replay_runs=6 if phase=="preflight" else 0,
           archive_id=archive_api.ARCHIVE_ID,archive_sha256=archive_api.ARCHIVE_SHA,
           upstream_fingerprint=up.code_fingerprint(),scientific_runner_fingerprint=sci.fingerprint(),
           strategic_fingerprint=fingerprint(),software_versions=up.software_versions(),
           production_authorized=False)
    d["design_id"]=up.canonical_hash(d)[:12]
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=tuple(PHASES),required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--gate0e-zip",required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    require(args.workers>0,"Workers must be positive")
    require(up.code_fingerprint()==sci.EXPECTED_UPSTREAM,"Scientific fingerprint changed")
    a=archive_api.load_archive(Path(args.gate0e_zip).expanduser().resolve(),
                              up.code_fingerprint(),up.software_versions())
    d=design_for(args.phase);did=d["design_id"];h=d["horizon"]
    parent=Path(args.output_dir).expanduser().resolve()
    root=parent/f"v6_strategic_{args.phase}_{did}"
    print(json.dumps(d,indent=2),flush=True);print("Output:",root,flush=True)
    if args.dry_run:
        return
    if args.phase=="strategic":
        pre=design_for("preflight")
        gpath=parent/f"v6_strategic_preflight_{pre['design_id']}"/"v6_gate.json"
        require(gpath.exists(),"Complete matching preflight first")
        g=json.loads(gpath.read_text())
        require(g.get("pass") is True and g.get("design_id")==pre["design_id"]
                and g.get("grid_runs")==36 and g.get("parity_runs")==3
                and g.get("archive_replay_pass") is True and g.get("matched_t0_within_k") is True
                and g.get("matched_t0_k1_preflight") is True,
                "Matching strategic preflight not passed")
    require(not root.exists() or args.resume,"Output exists: use --resume with identical design")
    root.mkdir(parents=True,exist_ok=True);(root/"shards").mkdir(exist_ok=True)
    manifest=root/"v6_manifest.json"
    if manifest.exists():
        require(json.loads(manifest.read_text())==json.loads(json.dumps(d)),"Manifest mismatch")
    manifest.write_text(json.dumps(d,indent=2),encoding="utf-8")
    gp=root/"v6_gate.json";gp.write_text(json.dumps({"pass":False,"status":"incomplete","design_id":did}))
    tasks=[]
    for seed in d["seeds"]:
        for v in d["variants"]:
            path=root/"shards"/f"{v}__{seed}.json.gz"
            if args.resume and path.exists():
                compare_block(read_shard(root,seed,v,h,did),a,h)
            else:
                tasks.append((seed,v,h,did,str(root)))
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as pool:
            futures=[pool.submit(worker,*t) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                seed,v=f.result()
                compare_block(read_shard(root,seed,v,h,did),a,h)
                print(f"[{i}/{len(futures)}] {seed} {v} validated",flush=True)
    counts,n=consolidate(root,d,a)
    replay=baseline_replay(a,did) if args.phase=="preflight" else []
    gate={"pass":True,"status":args.phase+"_only","design_id":did,
          "grid_runs":d["grid_runs"],"parity_runs":d["parity_runs"],
          "matched_t0_within_k":True,"matched_t0_k1_preflight":True if h==26 else None,
          "archive_replay_pass":True if replay else None,"archive_replay":replay,
          "archive_null_matches":len(d["seeds"])*sum(SETTINGS[v][0]==1 for v in d["variants"])*4 if h==400 else 0,
          "new_k4_null_runs":len(d["seeds"])*sum(SETTINGS[v][0]==4 for v in d["variants"])*4,
          "row_counts":counts,"seed_contrast_rows":n,"production_authorized":False}
    gp.write_text(json.dumps(gate,indent=2),encoding="utf-8")
    bundle=root/f"paper_b_v6_strategic_{args.phase}_{did}_shareable.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.glob("*.json"))+sorted(root.glob("*.csv.gz")):
            z.write(path,path.name)
    print(json.dumps(gate,indent=2),flush=True);print("Shareable ZIP:",bundle,flush=True)


if __name__=="__main__":
    main()

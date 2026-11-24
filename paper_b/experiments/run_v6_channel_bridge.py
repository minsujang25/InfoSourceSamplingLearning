"""V6 fixed-biased peer-channel bridge; separate from frozen Gate 0E2."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import zipfile

import model.InfoSourceSamplingLearning as m
from paper_b.experiments import run_v6_pilot as base
from paper_b.experiments import run_v6_scientific as sci
from paper_b.experiments import run_matched_pilot as up
from paper_b.experiments import v6_channel_bridge_analysis as a

FIELDS=sci.FIELDS
PHASES={"preflight":((7501,),26),"bridge":(a.SEEDS,400)}
REPS=(("extended_live","extended_live","flat","fixed_biased","frozen"),
      ("extended_muted","extended_muted","flat","null","adaptive"))


def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)


def cells():
    return [c for c in base.cells("C") if c[3] in ("null","fixed_biased")]


def set_floor():
    m.MIN_SD=1e-8
    m.MIN_VAR=m.MIN_SD**2


def fingerprint():
    paths=("paper_b/experiments/run_v6_channel_bridge.py",
           "paper_b/experiments/v6_channel_bridge_analysis.py",
           "paper_b/V6_CHANNEL_BRIDGE_PROTOCOL.md")
    return up.canonical_hash({"scientific":sci.fingerprint(),
           "extra":{p:hashlib.sha256((base.ROOT/p).read_bytes()).hexdigest() for p in paths}})


def validate(outputs,seed,horizon,did):
    expected={(c[0],c[3],c[4]) for c in cells()}
    actual={(o["run"]["cell"],o["run"]["sender_regime"],o["run"]["reliance_mode"]) for o in outputs}
    require(len(outputs)==16 and actual==expected,"Incomplete or duplicate C grid")
    base.check_block(outputs,"C")
    for o in outputs:
        r=o["run"]
        require(r["seed"]==seed and r["design_id"]==did and r["suite"]=="C","Wrong bridge run identity")
        require(r["cell"]==r["network_environment"] and r["prior"]=="flat","Unexpected channel or prior")
        base.check_run(o,horizon)
        groups={(x["group_type"],x["group"]):int(x["n_citizens"]) for x in o["group_terminal"]}
        require(groups[("all","all")]==100 and len(groups)==len(o["group_terminal"]),
                "Invalid group terminal summaries")
        steps={0,horizon,*(int(x["horizon_step"]) for x in o["belief_checkpoints"])}
        b=o["group_belief_checkpoints"]; q=o["group_request_checkpoints"]
        eb={(s,k,g) for s in steps for k,g in groups}
        eq={(s,k,g,t) for s,k,g in eb for t in ("all","theta","credibility")}
        require(len(b)==len(eb) and {(x["horizon_step"],x["group_type"],x["group"]) for x in b}==eb,
                "Missing group belief checkpoint")
        require(len(q)==len(eq) and {(x["horizon_step"],x["group_type"],x["group"],x["learning_target"]) for x in q}==eq,
                "Missing group request checkpoint")
        for x in q:
            require(x["total_requests"]==x["citizen_periods"]*base.CREDIT,"Request budget mismatch")
            if x["learning_target"]=="all":
                require(x["citizen_periods"]==x["horizon_step"]*x["n_citizens"],"Request history incomplete")
    # Existing C validator assumes four sender regimes. Its M/S/L comparison
    # logic is reused via an explicit two-sender validation below instead.


def validate_c(outputs,seed,horizon,did):
    # base.check_block's C-specific count is hardcoded for four senders;
    # run its generic matching checks, then verify exactly this 2-sender grid.
    from unittest.mock import patch
    original=base.check_block
    def generic(rows,suite):
        original(rows,"bridge_generic")
    with patch.object(base,"check_block",generic):
        validate(outputs,seed,horizon,did)
    extended=[o for o in outputs if o["run"]["cell"]!="elite_only"]
    require(len({o["run"]["structural_fingerprint"] for o in extended})==1,"M/S/L graph mismatch")
    sig=defaultdict(dict)
    fields=("learning_target","peer_audit_signature","peer_initial_means","peer_outgoing_sds",
            "source_count","peer_request_count","peer_delivered_count","expert_request_count",
            "expert_delivered_count","jammer_request_count","jammer_delivered_count","jammer_message_mean",
            "theta_mu_pre","theta_mu_post","theta_sd_pre","theta_sd_post",
            "jammer_delta_mu_post","jammer_delta_sd_post","jammer_rank_post")
    for o in extended:
        r=o["run"]
        for e in o["sender_events"]:
            if int(e["period"])==0:
                k=(r["sender_regime"],r["reliance_mode"],int(e["citizen_pos"]))
                ch=r["cell"]
                require(ch not in sig[k],"Duplicate audit event")
                sig[k][ch]=tuple(e[f] for f in fields)
    require(len(sig)==2*2*100,"Missing matched audit events")
    require(all(set(v)==set(a.CHANNELS[1:]) and len(set(v.values()))==1 for v in sig.values()),
            "M/S/L t0 audit mismatch")


def worker(seed,horizon,did,root):
    set_floor()
    outputs=[sci.run_instrumented(seed,"C",c,horizon,did) for c in cells()]
    validate_c(outputs,seed,horizon,did)
    for rep in REPS:
        chosen=next(o for o in outputs if o["run"]["cell"]==rep[0]
                    and o["run"]["sender_regime"]==rep[3] and o["run"]["reliance_mode"]==rep[4])
        plain,_=base.run_one(seed,"C",rep,horizon,did,instrument=False)
        for k in ("beliefs","belief_checkpoints","lambda_checkpoints","jammer_strategy"):
            require(base.parity_payload(chosen[k])==base.parity_payload(plain[k]),"Bridge logger parity failed: "+k)
    dest=Path(root)/"shards"/f"C__{seed}.json.gz"
    tmp=dest.with_suffix(".tmp")
    with gzip.open(tmp,"wt",encoding="utf-8") as f:
        json.dump(dict(design_id=did,seed=seed,parity_pass=True,outputs=outputs),f,allow_nan=True)
    os.replace(tmp,dest)
    return seed


def read_shard(root,seed,horizon,did):
    with gzip.open(root/"shards"/f"C__{seed}.json.gz","rt",encoding="utf-8") as f:
        s=json.load(f)
    require((s["design_id"],s["seed"],s["parity_pass"])==(did,seed,True),"Shard manifest mismatch")
    validate_c(s["outputs"],seed,horizon,did)
    return s["outputs"]


def replay_archive(archive,did):
    set_floor()
    cells_to_replay=[c for c in cells() if c[3]=="null"]
    cells_to_replay += [("extended_live","extended_live","flat",sender,"frozen")
                        for sender in ("adaptive","credibility_aware")]
    result=[]
    for c in cells_to_replay:
        out=sci.run_instrumented(7501,"C",c,400,did)
        a.compare_archive(out,archive,replay=True)
        result.append({"run_id":out["run"]["run_id"],"pass":True})
        print("Archived T400 replay PASS:",out["run"]["run_id"],flush=True)
    return result


def compare_block(outputs,archive,horizon):
    for o in outputs:
        r=o["run"]
        if horizon==400:
            a.compare_archive(o,archive)
        else:
            old=archive["runs"][(int(r["seed"]),r["network_environment"],"null",r["reliance_mode"])]
            for k in ("initial_state_fingerprint","structural_fingerprint"):
                require(r[k]==old[k],"Preflight archive initial/graph mismatch")


def consolidate(root,design,archive):
    horizon=design["horizon"]; did=design["design_id"]; seeds=design["seeds"]
    counts=defaultdict(int); columns=defaultdict(set); contrasts=[]
    for seed in seeds:
        outputs=read_shard(root,seed,horizon,did)
        compare_block(outputs,archive,horizon)
        contrasts.extend(a.contrast_rows(outputs,archive if horizon==400 else None))
        for o in outputs:
            for key in FIELDS:
                for r in sci.records(o,key):
                    columns[key].update(r); counts[key]+=1
    require(counts["run"]==design["grid_runs"],"Incomplete bridge grid")
    interactions=a.interaction_rows(contrasts)
    summaries=a.summarize(contrasts,seeds,("source","sender_regime","reliance_mode","horizon_step"))
    inter_summary=a.summarize(interactions,seeds,("dimension","contrast","sender_regime","reliance_mode","horizon_step"))
    with ExitStack() as stack:
        writers={}
        for key in FIELDS:
            if not columns[key]:
                continue
            filename="runs" if key=="run" else key
            f=stack.enter_context(gzip.open(root/f"{filename}.csv.gz","wt",encoding="utf-8",newline=""))
            w=csv.DictWriter(f,fieldnames=sorted(columns[key]));w.writeheader();writers[key]=w
        for seed in seeds:
            for o in read_shard(root,seed,horizon,did):
                for key,w in writers.items():
                    w.writerows(sci.records(o,key))
    for name,rows in (("channel_seed_contrasts",contrasts),("channel_contrast_summary",summaries),
                      ("channel_interactions",interactions),("channel_interaction_summary",inter_summary)):
        base.csv_gz(root/f"{name}.csv.gz",rows)
    return dict(counts),len(contrasts),len(interactions)


def design_for(phase):
    seeds,horizon=PHASES[phase]
    d=dict(phase=phase,seeds=list(seeds),horizon=horizon,N=100,cells=cells(),credit=20,
           epsilon=.05,K=1,surveillance=5,tau_social=1.,initial_audit="centroid",
           frozen_ranking="first_audit",peer_evidence="source_posterior",sender_event_period_limit=2,
           grid_runs=len(seeds)*16,parity_runs=len(seeds)*2,
           archive_replay_runs=10 if phase=="preflight" else 0,
           archive_id=a.ARCHIVE_ID,archive_sha256=a.ARCHIVE_SHA,
           upstream_fingerprint=up.code_fingerprint(),scientific_runner_fingerprint=sci.fingerprint(),
           bridge_fingerprint=fingerprint(),software_versions=up.software_versions(),
           production_authorized=False)
    d["design_id"]=up.canonical_hash(d)[:12]
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=tuple(PHASES),required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--gate0e-zip",required=True)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    require(args.workers>0,"workers must be positive")
    require(up.code_fingerprint()==sci.EXPECTED_UPSTREAM,"Scientific model fingerprint changed")
    archive=a.load_archive(Path(args.gate0e_zip).expanduser().resolve(),up.code_fingerprint(),up.software_versions())
    d=design_for(args.phase);did=d["design_id"];horizon=d["horizon"]
    parent=Path(args.output_dir).expanduser().resolve()
    root=parent/f"v6_channel_{args.phase}_{did}"
    print(json.dumps(d,indent=2),flush=True);print("Output:",root,flush=True)
    if args.dry_run:
        return
    if args.phase=="bridge":
        pre=design_for("preflight")
        path=parent/f"v6_channel_preflight_{pre['design_id']}"/"v6_gate.json"
        require(path.exists(),"Complete matching bridge preflight first")
        g=json.loads(path.read_text())
        require(g.get("pass") is True and g.get("design_id")==pre["design_id"]
                and g.get("archive_replay_pass") is True and g.get("grid_runs")==16
                and g.get("parity_runs")==2 and len(g.get("archive_replay",[]))==10,
                "Matching preflight not passed")
    require(not root.exists() or args.resume,"Output exists; use --resume for same design")
    root.mkdir(parents=True,exist_ok=True);(root/"shards").mkdir(exist_ok=True)
    manifest=root/"v6_manifest.json"
    if manifest.exists():
        require(json.loads(manifest.read_text())==json.loads(json.dumps(d)),"Manifest mismatch")
    manifest.write_text(json.dumps(d,indent=2),encoding="utf-8")
    gp=root/"v6_gate.json";gp.write_text(json.dumps({"pass":False,"status":"incomplete","design_id":did}))
    tasks=[]
    for seed in d["seeds"]:
        shard=root/"shards"/f"C__{seed}.json.gz"
        if args.resume and shard.exists():
            compare_block(read_shard(root,seed,horizon,did),archive,horizon)
        else:
            tasks.append((seed,horizon,did,str(root)))
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as pool:
            futures=[pool.submit(worker,*t) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                seed=f.result()
                compare_block(read_shard(root,seed,horizon,did),archive,horizon)
                print(f"[{i}/{len(futures)}] seed {seed} validated",flush=True)
    counts,n,ni=consolidate(root,d,archive)
    replay=replay_archive(archive,did) if args.phase=="preflight" else []
    gate={"pass":True,"status":args.phase+"_only","design_id":did,"grid_runs":d["grid_runs"],
          "parity_runs":d["parity_runs"],"blocks":len(d["seeds"]),"full_horizon":horizon,
          "matched_t0_audit":True,"archive_replay_pass":True if replay else None,
          "archive_replay":replay,"archive_null_matches":len(d["seeds"])*8 if horizon==400 else 0,
          "row_counts":counts,"channel_contrast_rows":n,"interaction_rows":ni,
          "production_authorized":False}
    bundle=root/f"paper_b_v6_channel_{args.phase}_{did}_shareable.zip"
    gp.write_text(json.dumps(gate,indent=2),encoding="utf-8")
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.glob("*.json"))+sorted(root.glob("*.csv.gz")):
            z.write(path,path.name)
    print(json.dumps(gate,indent=2),flush=True)
    print("Shareable ZIP:",bundle,flush=True)
    print("Keep raw shards:",root/"shards",flush=True)


if __name__=="__main__":
    main()

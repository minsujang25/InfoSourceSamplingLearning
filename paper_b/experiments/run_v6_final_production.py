"""Frozen V6 final structural production; unchanged scientific model/worker."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import zipfile

from paper_b.experiments import run_v6_pilot as base
from paper_b.experiments import run_v6_scientific as sci
from paper_b.experiments import run_matched_pilot as up
from paper_b.experiments import run_v6_robustness as prior
from paper_b.experiments.v6_scientific_analysis import summarize
from paper_b.experiments.v6_final_production_analysis import (
    extended_contrasts, group_gap_rows, gap_contrasts, topology_rows,
)

SEEDS=tuple(range(20001,20501))
PREFLIGHT_SEEDS=(19999,)
HORIZON=400
EXPECTED_SCI="3e217225e6552887d611478c7b3e44457628aea152bbf275ffcbed8d28202e89"
PHASES={"preflight":PREFLIGHT_SEEDS,"production":SEEDS}
COMPACT=("runs","group_terminal","seed_contrasts","contrast_summary",
         "final_seed_contrasts","final_contrast_summary","group_gap_checkpoints",
         "group_gap_seed_contrasts","group_gap_summary","peer_topology","group_belief_checkpoints")
FILES=("paper_b/experiments/run_v6_final_production.py",
       "paper_b/experiments/v6_final_production_analysis.py",
       "paper_b/experiments/run_v6_robustness.py",
       "paper_b/V6_FINAL_PRODUCTION_PROTOCOL.md")


def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)


def fingerprint():
    return up.canonical_hash({"scientific":sci.fingerprint(),
        "files":{p:hashlib.sha256((base.ROOT/p).read_bytes()).hexdigest() for p in FILES}})


def design_for(phase):
    seeds=PHASES[phase]
    require(up.code_fingerprint()==sci.EXPECTED_UPSTREAM,"Upstream fingerprint changed")
    require(sci.fingerprint()==EXPECTED_SCI,"Scientific runner fingerprint changed")
    d=dict(phase=phase,seeds=list(seeds),horizon=HORIZON,N=base.N,
           cells={s:base.cells(s) for s in sci.SUITES},epsilon=.05,credit=base.CREDIT,
           K=1,surveillance=5,tau_social=1.,initial_audit="centroid",
           frozen_ranking="first_audit",peer_evidence="source_posterior",
           event_period_limit=2,grid_runs=len(seeds)*24,parity_runs=len(seeds)*2,
           archive_replay_runs=2 if phase=="preflight" else 0,
           baseline_archive_id=prior.ARCHIVE_ID,baseline_archive_sha256=prior.ARCHIVE_SHA,
           upstream_fingerprint=up.code_fingerprint(),scientific_runner_fingerprint=sci.fingerprint(),
           final_runner_fingerprint=fingerprint(),software_versions=up.software_versions(),
           production_authorized=phase=="production")
    d["design_id"]=up.canonical_hash(d)[:12]
    return d


def shard_path(root,seed,suite):
    return root/"shards"/f"{suite}__{seed}.json.gz"


def consolidate(root,d):
    seeds=d["seeds"];did=d["design_id"];h=d["horizon"]
    counts=defaultdict(int);columns=defaultdict(set)
    core=[];extended=[];gaps=[];gap_effects=[];topology=[]
    for seed in seeds:
        for suite in sci.SUITES:
            outputs=sci.read_shard(shard_path(root,seed,suite),did,suite,seed,h)
            cr=sci.seed_contrasts(outputs)
            ex=extended_contrasts(outputs,h)
            gp=group_gap_rows(outputs)
            require(len(cr)==(48 if suite=="H" else 30),"Unexpected scientific contrasts")
            require(len(ex)==len(cr)*3//2,"Missing receiver contrast")
            core.extend(cr);extended.extend(ex);gaps.extend(gp)
            gap_effects.extend(gap_contrasts(gp,h))
            topology.extend(topology_rows(outputs))
            for o in outputs:
                for key in sci.FIELDS:
                    rr=sci.records(o,key)
                    counts[key]+=len(rr)
                    for row in rr:
                        columns[key].update(row)
    require(counts["run"]==d["grid_runs"],"Incomplete final grid")
    require(len(topology)==len(seeds)*4 and len(gap_effects)==len(seeds)*12,
            "Incomplete topology or gap contrasts")
    # Separate original analysis and final supplementary additions. All seeds,
    # all conditions and all outcome directions are retained.
    derived={
        "seed_contrasts":core,"contrast_summary":summarize(core,seeds),
        "final_seed_contrasts":extended,"final_contrast_summary":summarize(extended,seeds),
        "group_gap_checkpoints":gaps,"group_gap_seed_contrasts":gap_effects,
        "group_gap_summary":summarize(gap_effects,seeds),"peer_topology":topology,
    }
    with ExitStack() as stack:
        writers={}
        for key in sci.FIELDS:
            if not columns[key]:
                continue
            name="runs" if key=="run" else key
            f=stack.enter_context(gzip.open(root/f"{name}.csv.gz","wt",encoding="utf-8",newline=""))
            w=csv.DictWriter(f,fieldnames=sorted(columns[key]));w.writeheader();writers[key]=w
        for seed in seeds:
            for suite in sci.SUITES:
                outputs=sci.read_shard(shard_path(root,seed,suite),did,suite,seed,h)
                for o in outputs:
                    for key,w in writers.items():
                        w.writerows(sci.records(o,key))
    for name,rows in derived.items():
        base.csv_gz(root/f"{name}.csv.gz",rows)
    return dict(counts),{k:len(v) for k,v in derived.items()}


def write_bundle(root,bundle,names=None):
    temp=bundle.with_suffix(".tmp")
    with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in sorted(root.glob("*.json"))+sorted(root.glob("*.csv.gz")):
            if names is None or path.suffix==".json" or path.name.removesuffix(".csv.gz") in names:
                z.write(path,path.name)
    os.replace(temp,bundle)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=tuple(PHASES),required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--baseline-zip")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    require(args.workers>=1,"workers must be positive")
    parent=Path(args.output_dir).expanduser().resolve()
    path=Path(args.baseline_zip).expanduser().resolve() if args.baseline_zip else (
        parent/f"v6_scientific_{prior.ARCHIVE_ID}"/f"paper_b_v6_scientific_{prior.ARCHIVE_ID}_shareable.zip")
    archive=prior.archive(path)
    d=design_for(args.phase);did=d["design_id"];seeds=d["seeds"]
    root=parent/f"v6_final_{args.phase}_{did}"
    print(json.dumps(d,indent=2),flush=True);print("Output:",root,flush=True)
    if args.dry_run:
        return
    if args.phase=="production":
        pre=design_for("preflight")
        gp=parent/f"v6_final_preflight_{pre['design_id']}"/"v6_gate.json"
        require(gp.exists(),"Complete matching full-horizon final preflight first")
        g=json.loads(gp.read_text())
        require(g.get("pass") is True and g.get("design_id")==pre["design_id"]
                and g.get("status")=="preflight_only" and g.get("grid_runs")==24
                and g.get("parity_runs")==2 and g.get("archive_replay_pass") is True
                and len(g.get("archive_replay",[]))==2,"Matching final preflight not passed")
    require(not root.exists() or args.resume,"Output exists: use --resume for same design")
    root.mkdir(parents=True,exist_ok=True);(root/"shards").mkdir(exist_ok=True)
    manifest=root/"v6_manifest.json"
    if manifest.exists():
        require(json.loads(manifest.read_text())==json.loads(json.dumps(d)),"Manifest mismatch")
    manifest.write_text(json.dumps(d,indent=2),encoding="utf-8")
    gp=root/"v6_gate.json"
    gp.write_text(json.dumps({"pass":False,"status":"incomplete","design_id":did}))
    tasks=[]
    for seed in seeds:
        for suite in sci.SUITES:
            shard=shard_path(root,seed,suite)
            if args.resume and shard.exists():
                sci.read_shard(shard,did,suite,seed,HORIZON)
            else:
                tasks.append((seed,suite,HORIZON,did,str(root)))
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as pool:
            futures=[pool.submit(sci.worker,*t) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                print(f"[{i}/{len(futures)}] {f.result()}",flush=True)
    counts,derived=consolidate(root,d)
    replay=prior.baseline_replay(archive,did) if args.phase=="preflight" else []
    gate={"pass":True,"status":args.phase+"_only","design_id":did,
          "grid_runs":d["grid_runs"],"parity_runs":d["parity_runs"],
          "blocks":len(seeds)*2,"full_horizon":HORIZON,"row_counts":counts,
          "derived_row_counts":derived,"archive_replay_pass":True if replay else None,
          "archive_replay":replay,"production_authorized":d["production_authorized"]}
    # A gate is published only after consolidation and optional replay.
    gp.write_text(json.dumps(gate,indent=2),encoding="utf-8")
    full=root/f"paper_b_v6_final_{args.phase}_{did}_full.zip"
    compact=root/f"paper_b_v6_final_{args.phase}_{did}_shareable.zip"
    write_bundle(root,full)
    write_bundle(root,compact,COMPACT)
    print(json.dumps(gate,indent=2),flush=True)
    print("Shareable ZIP:",compact,flush=True);print("Full ZIP:",full,flush=True)
    print("Keep raw shards:",root/"shards",flush=True)


if __name__=="__main__":
    main()

"""V6.2 master runner. One command: gates, fixed grid, summaries, split archives.

Run from the repository root in the original paper-b environment. No archived
source files or original outputs are modified. This runner never adjusts the
grid to results, and no scientific sign is an execution gate.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import csv
import fcntl
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import traceback

# Bound native numerical threads before importing NumPy / sklearn.
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
import model.InfoSourceSamplingLearning as m
from paper_b.experiments import run_v6_pilot as base, run_v6_scientific as sci
from paper_b.experiments import run_matched_pilot as up
from paper_b.experiments.v62_policies import (VARIANTS, SEEDS, PARITY_SEEDS, cells, native_cell, scope)
from paper_b.experiments.v62_observer import Observer, observe, require, close
from paper_b.experiments import v62_archive as archive, v62_reporting as report

ROOT=base.ROOT
HISTORY_HEADER=("citizen_pos","period","audit","mu_pre","sd_pre","mu_post","sd_post","rank_pre",
                "rank_post","latent_rank_pre","latent_rank_post","Q_expert_step","Q_jammer_step",
                "Q_peer_step","B_expert_step","B_jammer_step","B_peer_step","peer_signed_update",
                "Q_jammer_cumulative")
RUN_FIELDS=("run_id","seed","suite","cell","level","prior","sender_regime","reliance_mode",
            "initial_state_fingerprint","structural_fingerprint","mse_truth","mean_belief","belief_variance",
            "Q_jammer_channel","Q_expert_channel","W_jammer_channel","W_peer_incoming_hhi","W_peer_incoming_top5_share")
PARITY_FIELDS=("beliefs","belief_checkpoints","lambda_checkpoints","jammer_strategy",
               "sender_events","channel_citizens","group_terminal","group_belief_checkpoints",
               "group_request_checkpoints","rank_unification_checkpoints")

def scientific_design(inventory):
    # All code/protocol dependencies and installed package versions are frozen
    # in the design identity, not the path or worker count.
    files={}
    for directory in (ROOT/"model",ROOT/"paper_b"):
        for p in sorted(directory.rglob("*.py")):
            files[str(p.relative_to(ROOT))]=archive.digest(p)
    for p in sorted((ROOT/"paper_b").glob("*.md")):
        files[str(p.relative_to(ROOT))]=archive.digest(p)
    files["environment.yml"]=archive.digest(ROOT/"environment.yml")
    d=dict(schema=1,horizon=400,seeds=list(SEEDS),grid=[list(x) for x in cells()],
           expected_runs=3400,short_preflight_horizon=26,parity_seeds=list(PARITY_SEEDS),
           expected_full_parity=55,bootstrap_seed=report.BOOT,bootstrap_draws=10000,
           source_hashes=files,software=up.software_versions(),python=sys.version,
           archives={k:{x:v[x] for x in ("status","sha256","design_id") if x in v}
                     for k,v in inventory.items() if isinstance(v,dict)})
    d["design_id"]=up.canonical_hash(d)[:12]
    return d

def job_key(seed,cell):
    return "__".join([str(seed),*cell])

def paths(root,phase,seed,cell):
    folder=Path(root)/phase
    stem=job_key(seed,cell)
    return folder/(stem+".json.gz"),folder/(stem+".sha256"),folder/(stem+".history.csv.gz")

def parity_required(phase,seed,cell):
    v,s,level,mode=cell
    return phase=="preflight" or (phase=="production" and seed in PARITY_SEEDS and level=="low" and mode=="adaptive")

def replay_variant(phase):
    return phase.startswith("replay_")

def run_native(seed,cell,horizon,did,observer=None):
    v,s,lev,mode=cell
    m.MIN_SD=1e-8; m.MIN_VAR=m.MIN_SD**2
    with scope(v):
        with observe(observer) if observer is not None else nullcontext():
            return sci.run_instrumented(seed,s,native_cell(s,lev,mode),horizon,did)

def validate_saved(obj,phase,seed,cell,horizon,did):
    v,s,lev,mode=cell
    require((obj["design_id"],obj["phase"],obj["seed"],obj["variant"],obj["suite"],obj["level"],obj["mode"],obj["horizon"])==
            (did,phase,seed,v,s,lev,mode,horizon),"Shard identity")
    require(obj["validation"]["periods"]==100*horizon and len(obj["terminal"])==100,"Shard coverage")
    require(len({x["citizen_pos"] for x in obj["terminal"]})==100,"Duplicate citizens")
    require(obj["parity_pass"] is True if parity_required(phase,seed,cell) else obj["parity_pass"] is None,"Parity provenance")
    for r in obj["terminal"]:
        require(close(sum(r["signed"].values()),r["mu"]) and close(sum(r["signed_mse"].values()),r["mu"]**2),"Stored accounting")
    if replay_variant(phase):
        count=sum(x["I_corrective_crosscut_any"] is None for x in obj["channel_citizens"])
        require(obj["replay_missingness"]=={"channel_citizens.I_corrective_crosscut_any":count},
                "Replay missingness accounting")
    require(obj["run"]["seed"]==seed,"Native seed")
    return obj

def saved(root,phase,seed,cell,horizon,did):
    path,sha,history=paths(root,phase,seed,cell)
    if not path.exists() and not sha.exists():
        return None
    require(path.is_file() and sha.is_file(),"Interrupted or incomplete shard; preserve and inspect "+str(path))
    obj=validate_saved(report.read_gz(path),phase,seed,cell,horizon,did)
    require(sha.read_text().strip()==archive.digest(path),"Shard checksum")
    if obj["history_sha256"] is not None:
        require(history.is_file() and archive.digest(history)==obj["history_sha256"],"History checksum")
    return obj

def worker(root,phase,seed,cell,horizon,did):
    path,sha,hist=paths(root,phase,seed,cell)
    # The checksum is the commit marker. A process killed between atomic
    # file renames has not committed a scientific shard. Preserve the orphan
    # and retry exactly the same job. A committed checksum mismatch is fatal.
    if path.exists() and not sha.exists():
        quarantine=path.parent/"interrupted"; quarantine.mkdir(exist_ok=True)
        for f in (path,hist):
            if f.exists():
                dest=quarantine/(f.name+"."+str(os.getpid()))
                require(not dest.exists(),"Existing interrupted artifact")
                f.replace(dest)
    if saved(root,phase,seed,cell,horizon,did) is not None:
        return job_key(seed,cell)
    v,s,lev,mode=cell
    _,_,_,gateways=base.config_for(seed,s,lev,"high" if s=="H" else "flat",horizon)
    keep_history=phase=="production" and v=="baseline" and s=="H"
    htmp=hist.with_name(hist.name+".tmp")
    with gzip.open(htmp,"wt",encoding="utf-8",newline="") if keep_history else nullcontext(None) as f:
        w=csv.writer(f) if f is not None else None
        if w is not None: w.writerow(HISTORY_HEADER)
        observer=Observer(v,gateways,w)
        out=run_native(seed,cell,horizon,did,observer)
    obs=observer.finish(out,horizon)
    parity=None
    if parity_required(phase,seed,cell):
        plain=run_native(seed,cell,horizon,did)
        for field in PARITY_FIELDS:
            require(base.parity_payload(out[field])==base.parity_payload(plain[field]),"Observer parity "+field)
        parity=True
    hsha=None
    if keep_history:
        htmp.replace(hist); hsha=archive.digest(hist)
    obj=dict(design_id=did,phase=phase,seed=seed,variant=v,suite=s,level=lev,mode=mode,horizon=horizon,
             run={k:out["run"][k] for k in RUN_FIELDS},parity_pass=parity,history_sha256=hsha,**obs)
    # Keep replay tables only for technical archive comparisons.
    if replay_variant(phase):
        obj.update(report.replay_tables(out))
    validate_saved(obj,phase,seed,cell,horizon,did)
    report.write_gz(path,obj)
    tmp=sha.with_suffix(".tmp"); tmp.write_text(archive.digest(path)+"\n"); tmp.replace(sha)
    return job_key(seed,cell)

def run_jobs(pool,root,phase,jobs,horizon,did):
    folder=root/phase; folder.mkdir(exist_ok=True)
    failed={}
    futures={pool.submit(worker,str(root),phase,seed,cell,horizon,did):(seed,cell) for seed,cell in jobs}
    for i,f in enumerate(as_completed(futures),1):
        seed,cell=futures[f]
        key=job_key(seed,cell)
        try:
            f.result()
            print(f"{phase} [{i}/{len(futures)}] {key} OK",flush=True)
        except Exception:
            failed[key]=traceback.format_exc()
            (folder/(key+".error.txt")).write_text(failed[key])
            print(f"{phase} [{i}/{len(futures)}] {key} FAILED",flush=True)
    return failed

def matching(outputs,require_all=True):
    by=defaultdict(list)
    for o in outputs:
        by[o["suite"],o["level"]].append(o)
    for key,rr in by.items():
        require(len({x["run"]["structural_fingerprint"] for x in rr})==1,"Opportunity mismatch "+str(key))
        require(len({x["run"]["initial_state_fingerprint"] for x in rr})==1,"Initial state mismatch "+str(key))
    for suite in ("H","G"):
        rr=[o for o in outputs if o["suite"]==suite]
        if rr: require(len({o["run"]["initial_state_fingerprint"] for o in rr})==1,"Low/high initial mismatch "+suite)
    reference={(o["suite"],o["level"],o["mode"]):o for o in outputs if o["variant"]=="baseline"}
    failures={}
    for o in outputs:
        if o["variant"]=="baseline": continue
        key=(o["suite"],o["level"],o["mode"])
        ref=reference.get(key)
        if ref is None:
            if require_all: raise RuntimeError("Missing reference "+str(key))
            continue
        def signature(x,ra):
            return sorted([dict(r) if not ra else {k:v for k,v in r.items() if k!="rank_post"}
                           for r in x["initial_audit"] if not ra or r["source_class"]!="jammer"],
                          key=lambda r:(r["citizen_pos"],r["source_pos"]))
        ra=o["variant"]=="RA"
        if signature(o,ra)!=signature(ref,ra):
            failures[o["variant"]]="Matched t0 audit mismatch "+str(key)
    return failures

def archive_jobs(data):
    for tag in ("production","pilot","RA","RB","behavioral"):
        if tag not in data: continue
        if tag in ("production","pilot"):
            selected=[c for c in cells() if c[0]=="baseline"]
        elif tag in ("RA","RB"):
            selected=[c for c in cells() if c[0]==tag]
        else:
            selected=[c for c in cells() if c[0] in ("uniform_tail","constant_audit")]
        yield tag,[(20001 if tag=="production" else 6001,c) for c in selected]

def run_main(args):
    roots=[Path(p).expanduser().resolve() for p in args.archive_root]
    explicit={k:getattr(args,k.lower()+"_zip") for k in archive.ARCHIVES}
    paths_=archive.locate(roots,explicit)
    data,inventory=archive.load(paths_,roots)
    design=scientific_design(inventory); did=design["design_id"]
    root=Path(args.output_dir).expanduser().resolve()/("v62_"+did)
    print("Design:",did,"Output:",root,"Grid:",len(cells())*len(SEEDS),flush=True)
    if args.dry_run:
        print(json.dumps({"archives":inventory,"grid":design["grid"]},indent=2))
        return
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/"MANIFEST.json"
    if manifest.exists():
        require(json.loads(manifest.read_text())==design,"Resume design mismatch")
    else: report.write_json(manifest,design)
    report.write_json(root/"ARCHIVE_INVENTORY.json",inventory)
    execution=dict(design_id=did,status="incomplete",preflight={},archive_replay=[],production_failed={})
    report.write_json(root/"EXECUTION_SUMMARY.json",execution)
    subprocess.run([sys.executable,"-m","unittest","paper_b.test_v62_mechanism_closure","paper_b.test_v61_assumption_robustness","-v"],check=True,cwd=ROOT)
    structural=archive.structural_benchmark(data["production"])
    report.csv_gz(root/"structural_benchmark.csv.gz",structural)
    structural_seed,structural_stats=report.structural_summary(structural)
    report.csv_gz(root/"structural_seed_contrasts.csv.gz",structural_seed)
    report.csv_gz(root/"structural_summary.csv.gz",structural_stats)
    allowed=set(VARIANTS)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as pool:
        for tag,jobs in archive_jobs(data):
            phase="replay_"+tag
            failed=run_jobs(pool,root,phase,jobs,400,did)
            for seed,cell in jobs:
                key=job_key(seed,cell)
                try:
                    require(key not in failed,"Replay execution failed")
                    o=saved(root,phase,seed,cell,400,did)
                    execution["archive_replay"].append(archive.compare_replay(o,tag,data[tag],cell[0]))
                except Exception:
                    execution["archive_replay"].append(dict(tag=tag,cell=list(cell),passed=False,error=traceback.format_exc()))
                    allowed.discard(cell[0])
            report.write_json(root/"EXECUTION_SUMMARY.json",execution)
        failed=run_jobs(pool,root,"preflight",[(SEEDS[0],c) for c in cells()],26,did)
        good=[]
        for cell in cells():
            key=job_key(SEEDS[0],cell)
            if key in failed:
                allowed.discard(cell[0])
            else:
                good.append(saved(root,"preflight",SEEDS[0],cell,26,did))
        failures=matching(good,require_all=False)
        allowed-=set(failures)
        if "baseline" not in allowed:
            allowed=set()
        for v in VARIANTS:
            execution["preflight"][v]=dict(passed=v in allowed,error=failures.get(v),
                failed_cells=[k for k in failed if "__"+v+"__" in k])
        report.write_json(root/"EXECUTION_SUMMARY.json",execution)
        if args.preflight_only:
            print(json.dumps(execution,indent=2))
            require(allowed==set(VARIANTS),"Preflight incomplete")
            return
        jobs=[(seed,c) for seed in SEEDS for c in cells() if c[0] in allowed]
        failures=run_jobs(pool,root,"production",jobs,400,did) if jobs else {}
        execution["production_failed"]=failures
        execution["eligible_blocks"]=sorted(allowed)
    execution["completed_runs"]=len(jobs)-len(failures)
    report.write_json(root/"EXECUTION_SUMMARY.json",execution)
    require(allowed==set(VARIANTS) and not failures and execution["completed_runs"]==3400,
            "Grid incomplete; no scientific consolidation. Inspect EXECUTION_SUMMARY.json.")
    seed_dir=root/"seed_tables"; seed_dir.mkdir(exist_ok=True)
    detail=[]; histories=[]; parity=0
    for seed in SEEDS:
        outputs=[]
        for c in cells():
            obj=saved(root,"production",seed,c,400,did)
            require(obj is not None,"Missing production shard")
            outputs.append(obj)
            parity+=int(obj["parity_pass"] is True)
            p,sha,h=paths(root,"production",seed,c)
            detail.extend((p,sha))
            if obj["history_sha256"]: histories.append(h)
        require(not matching(outputs),"Production matched audit failure")
        report.write_gz(seed_dir/f"seed_{seed}.json.gz",report.build_seed(seed,outputs))
        print(f"Consolidated seed {seed}",flush=True)
    require(parity==55 and len(histories)==400,"Full parity/history coverage")
    # Summaries stream seed files; keep scientific cell definitions fixed.
    rows=(r for seed in SEEDS for r in report.read_gz(seed_dir/f"seed_{seed}.json.gz"))
    summary=report.summarize(rows,list(SEEDS))
    summary_paths=[]
    for start in range(0,len(summary),25000):
        p=root/f"contrast_summary_{start//25000+1:03d}.csv.gz"
        report.csv_gz(p,summary[start:start+25000]);summary_paths.append(p)
    execution.update(status="validated_scientific_grid",technical_gate_pass=True,parity_runs=parity,
                     history_runs=len(histories),scientific_runs=3400,summary_rows=len(summary))
    report.write_json(root/"EXECUTION_SUMMARY.json",execution)
    primary=[manifest,root/"ARCHIVE_INVENTORY.json",root/"EXECUTION_SUMMARY.json",
             root/"structural_benchmark.csv.gz",root/"structural_seed_contrasts.csv.gz",root/"structural_summary.csv.gz",*summary_paths,*seed_dir.glob("seed_*.json.gz")]
    indices={folder:report.package(root,files,folder,did) for folder,files in
             (("primary",primary),("individual_detail",detail),("histories",histories))}
    report.write_json(root/"shareable"/"TRANSFER_INDEX.json",indices)
    print("COMPLETE:",root/"shareable",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--output-dir",default="~/Research/paperb_v6_results/v62_mechanism_closure")
    p.add_argument("--archive-root",action="append",default=None)
    for k in archive.ARCHIVES:
        p.add_argument("--"+k.lower()+"-zip")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--verify-package",type=Path)
    args=p.parse_args()
    if args.verify_package:
        print("Verified members:",report.verify_package(args.verify_package));return
    require(args.workers>=1,"workers must be positive")
    if args.archive_root is None:
        args.archive_root=["~/Research/paperb_v6_results",str(ROOT/"production_results"),str(ROOT/"local_results")]
    parent=Path(args.output_dir).expanduser().resolve()
    parent.mkdir(parents=True,exist_ok=True)
    # macOS/Linux advisory lock automatically releases on interruption.
    with (parent/".v62.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_main(args)

if __name__=="__main__":
    main()

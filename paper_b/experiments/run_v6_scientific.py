"""V6 structural scientific pilot; no production or adaptive design expansion."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from unittest.mock import patch
import zipfile

import model.InfoSourceSamplingLearning as model_module
from paper_b.experiments import run_matched_pilot as upstream
from paper_b.experiments import run_v6_pilot as base
from paper_b.experiments.v6_scientific_measurement import summarize_model
from paper_b.experiments.v6_scientific_analysis import seed_contrasts, summarize

SUITES = ("H","G")
PHASES = {"preflight": ((8001,),26), "scientific": (tuple(range(6001,6021)),400)}
EXPECTED_UPSTREAM = "754d7c373ff4b9f6eb24e1b0044917d559f69397531a053d5ab64d314cd67774"
FIELDS = ("run", *(key for key in base.KEYS if key!="edges"), "group_terminal",
          "group_acquisition", "group_belief_checkpoints", "group_request_checkpoints")
REPRESENTATIVE = {
    "H": ("H_high_S_high","high","high","fixed_biased","adaptive"),
    "G": ("G_high","high","flat","fixed_biased","frozen"),
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def fingerprint():
    paths = ("paper_b/experiments/run_v6_scientific.py",
             "paper_b/experiments/v6_scientific_measurement.py",
             "paper_b/experiments/v6_scientific_analysis.py",
             "paper_b/V6_SCIENTIFIC_PILOT_PROTOCOL.md")
    return upstream.canonical_hash({"base": base.fingerprint(),
            "files": {p: hashlib.sha256((base.ROOT/p).read_bytes()).hexdigest() for p in paths}})


def run_instrumented(seed,suite,cell,horizon,design_id):
    # Capture the instance created by the unchanged upstream runner.
    # No subclass or step interception, no mutation of the model or its RNG.
    instances = []
    factory = upstream.InfoSampleModel
    def capture(*args, **kwargs):
        instance = factory(*args, **kwargs)
        instances.append(instance)
        return instance
    with patch.object(upstream, "InfoSampleModel", capture):
        out, gateways = base.run_one(seed,suite,cell,horizon,design_id)
    require(len(instances)==1, "Expected one model instance")
    base.check_run(out,horizon)
    out["group_terminal"] = base.group_outputs(out,gateways)
    out["group_acquisition"] = base.edge_groups(out,gateways)
    model = instances.pop()
    steps = sorted({0,horizon,*(int(r["horizon_step"]) for r in out["belief_checkpoints"])})
    beliefs,requests = summarize_model(model,gateways,steps,base.CREDIT)
    # Keep exact post-period indexing and independent consistency checks.
    common = {k:out["run"][k] for k in
              ("run_id","design_id","suite","cell","seed","sender_regime","reliance_mode","prior")}
    out["group_belief_checkpoints"] = [dict(common,**r) for r in beliefs]
    out["group_request_checkpoints"] = [dict(common,**r) for r in requests]
    pop = {r["horizon_step"]:r for r in beliefs if r["group_type"]=="all"}
    for r in out["belief_checkpoints"]:
        require(math.isclose(float(r["mse_truth"]),pop[int(r["horizon_step"])]["mse_truth"],
                             rel_tol=1e-10,abs_tol=1e-10), "Population checkpoint MSE mismatch")
    terminal = {(r["group_type"],r["group"]):r for r in beliefs if r["horizon_step"]==horizon}
    for r in out["group_terminal"]:
        t = terminal[(r["group_type"],r["group"])]
        for k in ("mse_truth","mean_belief","belief_variance","squared_bias"):
            require(math.isclose(float(r[k]),float(t[k]),rel_tol=1e-10,abs_tol=1e-10),
                    "Terminal subgroup mismatch")
    require(len(model.reliance_history)==horizon, "Incomplete history")
    # Existing base validator requires checkpoint edges. Validate here and again
    # upon reading local shards; do not export raw checkpoint edges to CSV.
    return out


def validate_outputs(outputs, suite, seed, horizon, design_id):
    expected = {(c[0],c[3],c[4]) for c in base.cells(suite)}
    actual = {(o["run"]["cell"],o["run"]["sender_regime"],o["run"]["reliance_mode"]) for o in outputs}
    require(expected==actual and len(outputs)==len(expected), "Missing or duplicate cells")
    base.check_block(outputs,suite)
    # Topology and groups must match across the intended comparisons.
    # H maps depend on H not S. G roles are fixed by the structural helper.
    if suite=="H":
        for level in ("low","high"):
            require(len({o["run"]["structural_fingerprint"] for o in outputs
                         if o["run"]["level"]==level})==1, "H map varies with S")
    for o in outputs:
        r = o["run"]
        require(r["design_id"]==design_id and r["seed"]==seed and r["suite"]==suite, "Wrong shard identity")
        base.check_run(o,horizon)
        group_rows = o["group_terminal"]
        steps = {0,horizon,*(int(x["horizon_step"]) for x in o["belief_checkpoints"])}
        groups = {(x["group_type"],x["group"]):int(x["n_citizens"]) for x in group_rows}
        require(len(groups)==len(group_rows) and groups[("all","all")]==base.N,"Missing group summaries")
        if suite=="H":
            require(groups[("citizen_group","1")]==groups[("citizen_group","-1")]==50,
                    "Unbalanced fixed groups")
        else:
            require(groups[("gateway_role","gateway")]==10 and groups[("gateway_role","non_gateway")]==90,
                    "Gateway role count mismatch")
        b = o["group_belief_checkpoints"]
        q = o["group_request_checkpoints"]
        expected_b = {(s,k,g) for s in steps for k,g in groups}
        expected_q = {(s,k,g,t) for s,k,g in expected_b for t in ("all","theta","credibility")}
        require(len(b)==len(expected_b) and {(x["horizon_step"],x["group_type"],x["group"]) for x in b}==expected_b,
                "Incomplete group belief checkpoints")
        require(len(q)==len(expected_q) and {(x["horizon_step"],x["group_type"],x["group"],x["learning_target"]) for x in q}==expected_q,
                "Incomplete group request checkpoints")
        for x in q:
            require(x["total_requests"]==x["citizen_periods"]*base.CREDIT, "Budget mismatch")
            if x["learning_target"]=="all":
                require(x["citizen_periods"]==x["n_citizens"]*x["horizon_step"], "Request horizon mismatch")


def worker(seed,suite,horizon,design_id,root):
    model_module.MIN_SD=1e-8
    model_module.MIN_VAR=model_module.MIN_SD**2
    outputs = [run_instrumented(seed,suite,c,horizon,design_id) for c in base.cells(suite)]
    validate_outputs(outputs,suite,seed,horizon,design_id)
    cell=REPRESENTATIVE[suite]
    chosen=next(o for o in outputs if o["run"]["cell"]==cell[0]
                and o["run"]["sender_regime"]==cell[3] and o["run"]["reliance_mode"]==cell[4])
    plain,_=base.run_one(seed,suite,cell,horizon,design_id,instrument=False)
    for key in ("beliefs","belief_checkpoints","lambda_checkpoints","jammer_strategy"):
        require(base.parity_payload(chosen[key])==base.parity_payload(plain[key]),
                f"Scientific logger parity failed: {key}")
    dest=Path(root)/"shards"/f"{suite}__{seed}.json.gz"
    temp=dest.with_suffix(".tmp")
    with gzip.open(temp,"wt",encoding="utf-8") as f:
        json.dump(dict(design_id=design_id,suite=suite,seed=seed,parity_pass=True,
                       outputs=outputs),f,allow_nan=True)
    os.replace(temp,dest)
    return str(dest)


def read_shard(path,design_id,suite,seed,horizon):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        shard=json.load(f)
    require((shard["design_id"],shard["suite"],shard["seed"],shard["parity_pass"])==
            (design_id,suite,seed,True),"Invalid shard manifest")
    outputs=shard["outputs"]
    validate_outputs(outputs,suite,seed,horizon,design_id)
    return outputs


def records(out,key):
    return [out["run"]] if key=="run" else out[key]


def consolidate(root,design,seeds,horizon):
    # Two streaming passes; bounded by one seed/suite shard, not all 480 runs.
    design_id=design["design_id"]
    columns=defaultdict(set)
    counts=defaultdict(int)
    contrasts=[]
    for seed in seeds:
        for suite in SUITES:
            outputs=read_shard(root/"shards"/f"{suite}__{seed}.json.gz",design_id,suite,seed,horizon)
            contrasts.extend(seed_contrasts(outputs))
            for o in outputs:
                for key in FIELDS:
                    rr=records(o,key)
                    counts[key]+=len(rr)
                    for r in rr:
                        columns[key].update(r)
    require(counts["run"]==design["grid_runs"],"Incomplete grid")
    summaries=summarize(contrasts,seeds)
    with ExitStack() as stack:
        writers={}
        for key in FIELDS:
            if not columns[key]:
                continue
            name="runs" if key=="run" else key
            f=stack.enter_context(gzip.open(root/f"{name}.csv.gz","wt",encoding="utf-8",newline=""))
            w=csv.DictWriter(f,fieldnames=sorted(columns[key]))
            w.writeheader()
            writers[key]=w
        for seed in seeds:
            for suite in SUITES:
                outputs=read_shard(root/"shards"/f"{suite}__{seed}.json.gz",design_id,suite,seed,horizon)
                for o in outputs:
                    for key,w in writers.items():
                        w.writerows(records(o,key))
    base.csv_gz(root/"seed_contrasts.csv.gz",contrasts)
    base.csv_gz(root/"contrast_summary.csv.gz",summaries)
    return dict(counts),len(contrasts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=tuple(PHASES),required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    require(args.workers>=1,"workers must be positive")
    seeds,horizon=PHASES[args.phase]
    upstream_fp=upstream.code_fingerprint()
    require(upstream_fp==EXPECTED_UPSTREAM,"Upstream scientific fingerprint changed; inspect before execution")
    design=dict(phase=args.phase,seeds=list(seeds),horizon=horizon,N=base.N,
                cells={s:base.cells(s) for s in SUITES},epsilon=.05,credit=base.CREDIT,
                K=1,surveillance=5,tau_social=1.,initial_audit="centroid",
                frozen_ranking="first_audit",peer_evidence="source_posterior",
                event_period_limit=2,grid_runs=len(seeds)*24,parity_runs=len(seeds)*2,
                upstream_fingerprint=upstream_fp,scientific_runner_fingerprint=fingerprint(),
                software_versions=upstream.software_versions(),production_authorized=False)
    design_id=upstream.canonical_hash(design)[:12]
    design["design_id"]=design_id
    parent=Path(args.output_dir).expanduser().resolve()
    root=parent/f"v6_{args.phase}_{design_id}"
    print(json.dumps(design,indent=2),flush=True)
    print(f"Output: {root}",flush=True)
    if args.dry_run:
        return
    if args.phase=="scientific":
        # Require a completed preflight of this exact runner and environment.
        pre=dict(design)
        pre.pop("design_id")
        pre.update(phase="preflight",seeds=list(PHASES["preflight"][0]),horizon=26,
                   grid_runs=24,parity_runs=2)
        pre_id=upstream.canonical_hash(pre)[:12]
        pre_root=parent/f"v6_preflight_{pre_id}"
        gate_path=pre_root/"v6_gate.json"
        require(gate_path.exists(),"Run matching preflight first")
        gate=json.loads(gate_path.read_text())
        require(gate.get("pass") is True and gate.get("design_id")==pre_id and
                gate.get("status")=="preflight_only" and gate.get("grid_runs")==24 and
                gate.get("parity_runs")==2,"Matching preflight not passed")
    require(not root.exists() or args.resume,"Output exists: use identical command plus --resume")
    root.mkdir(parents=True,exist_ok=True)
    (root/"shards").mkdir(exist_ok=True)
    manifest=root/"v6_manifest.json"
    if manifest.exists():
        require(json.loads(manifest.read_text())==json.loads(json.dumps(design)),"Manifest mismatch")
    manifest.write_text(json.dumps(design,indent=2),encoding="utf-8")
    gate_path=root/"v6_gate.json"
    gate_path.write_text(json.dumps(dict(pass_=False,status="incomplete",design_id=design_id)).replace('"pass_"','"pass"'),encoding="utf-8")
    tasks=[]
    for seed in seeds:
        for suite in SUITES:
            shard=root/"shards"/f"{suite}__{seed}.json.gz"
            if args.resume and shard.exists():
                read_shard(shard,design_id,suite,seed,horizon)
            else:
                tasks.append((seed,suite,horizon,design_id,str(root)))
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as pool:
            futures=[pool.submit(worker,*task) for task in tasks]
            for i,f in enumerate(as_completed(futures),1):
                print(f"[{i}/{len(futures)}] {f.result()}",flush=True)
    counts,n_contrasts=consolidate(root,design,seeds,horizon)
    gate={"pass":True,"status":args.phase+"_only","design_id":design_id,
          "grid_runs":design["grid_runs"],"parity_runs":design["parity_runs"],
          "blocks":len(seeds)*2,"full_horizon":horizon,"row_counts":counts,
          "seed_contrast_rows":n_contrasts,"production_authorized":False}
    gate_path.write_text(json.dumps(gate,indent=2),encoding="utf-8")
    bundle=root/f"paper_b_v6_{args.phase}_{design_id}_shareable.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.glob("*.json"))+sorted(root.glob("*.csv.gz")):
            z.write(path,path.name)
    print(json.dumps(gate,indent=2),flush=True)
    print(f"Shareable ZIP: {bundle}",flush=True)
    print(f"Raw validated shards retained locally: {root/'shards'}",flush=True)


if __name__=="__main__":
    main()

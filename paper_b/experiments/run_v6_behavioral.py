"""V6 H/G behavioral-rule sensitivity; isolated from all previous runners."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
import csv
import gzip
import hashlib
import io
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
from paper_b.experiments.v6_behavioral_rules import (
    AUDIT_P, REFERENCE_HORIZON, uniform_tail_rank_probabilities, constant_audit_choice,
)
from paper_b.experiments import run_v6_robustness as previous
from paper_b.experiments.v6_robustness_analysis import contrasts, change_rows, summarize_variants

SUITES=("H","G")
VARIANTS=("uniform_tail","constant_audit")
ARCHIVE_ID="85faac503d4a"
ARCHIVE_SHA="084f1558b9f5c360861c75e1889467be9439b9bc84505d0f4a9fe3ce577925a3"
ARCHIVE_FP="3e217225e6552887d611478c7b3e44457628aea152bbf275ffcbed8d28202e89"
ORIGINAL_CELLS=base.cells
ORIGINAL_CONFIG=base.config_for
FIELDS=sci.FIELDS
PHASES={"preflight":((6001,),26,("baseline",*VARIANTS)),
        "robustness":(tuple(range(6001,6021)),400,VARIANTS)}


def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)


def cells(suite):
    return [c for c in ORIGINAL_CELLS(suite) if suite=="G" or c[2]=="high"]


@contextmanager
def variant_scope(variant):
    require(variant in ("baseline",*VARIANTS),"Unknown variant")
    with ExitStack() as stack:
        if variant=="uniform_tail":
            # Both actual acquisition and diagnostic probability recomputation
            # must use the same function; metrics read actual history.
            stack.enter_context(patch.object(m,"recursive_rank_probabilities",uniform_tail_rank_probabilities))
            stack.enter_context(patch.object(up,"recursive_rank_probabilities",uniform_tail_rank_probabilities))
        if variant=="constant_audit":
            stack.enter_context(patch.object(m.Citizen,"learn_theta_or_delta",constant_audit_choice))
        yield


def fingerprint():
    paths=("paper_b/experiments/run_v6_behavioral.py",
           "paper_b/experiments/v6_behavioral_rules.py",
           "paper_b/experiments/v6_robustness_analysis.py",
           "paper_b/experiments/run_v6_robustness.py",
           "paper_b/V6_BEHAVIORAL_ROBUSTNESS_PROTOCOL.md")
    return up.canonical_hash({"scientific":sci.fingerprint(),
        "extra":{p:hashlib.sha256((base.ROOT/p).read_bytes()).hexdigest() for p in paths}})


def expected_audits(variant,horizon):
    if variant=="constant_audit":
        return 1.0+(horizon-1)*AUDIT_P
    return math.fsum(1.0/(t+1) for t in range(horizon))


def set_floor():
    m.MIN_SD=1e-8
    m.MIN_VAR=m.MIN_SD**2


def validate(outputs,seed,suite,variant,horizon,design_id):
    with patch.object(base,"cells",cells):
        sci.validate_outputs(outputs,suite,seed,horizon,design_id)
    for o in outputs:
        r=o["run"]
        require(r["variant"]==variant,"Variant mismatch")
        require(r["epsilon_requested"]==.05 and r["external_precision_cap"] is None,
                "Unexpected epsilon or external precision")
        require(r["rank_rule"]==("uniform_tail" if variant=="uniform_tail" else "recursive")
                and r["audit_rule"]==("constant_after_t0" if variant=="constant_audit" else "harmonic"),
                "Behavioral rule label mismatch")
        require(math.isclose(r["expected_audits_per_citizen"],expected_audits(variant,horizon),
                             rel_tol=0,abs_tol=1e-12),"Audit expectation mismatch")
        by_ego=defaultdict(list)
        for e in o["edges"]:
            by_ego[(e["period"],e["ego"])].append(e)
        require(bool(by_ego),"No rank audit records")
        for edges in by_ego.values():
            probs=[float(e["expected_reliance"]) for e in edges]
            d=len(probs)
            if variant=="uniform_tail":
                expected=sorted(uniform_tail_rank_probabilities(d,.05))
            else:
                expected=sorted(m.recursive_rank_probabilities(d,.05))
            require(all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
                        for a,b in zip(sorted(probs),expected)),"Actual rank policy mismatch")
            if r["sender_regime"]=="null":
                jammer=[float(e["expected_reliance"]) for e in edges if e["source_type"]=="disruptivejammer"]
                require(len(jammer)==1 and math.isclose(jammer[0],min(expected),abs_tol=1e-12),
                        "Null slot not at tail probability")
        pop={int(x["horizon_step"]):x for x in o["group_request_checkpoints"]
             if x["group_type"]=="all" and x["learning_target"]=="credibility"}
        require(pop[1]["citizen_periods"]==base.N,"Mandatory first audit missing")
        total=int(pop[horizon]["citizen_periods"])
        require(base.N<=total<=base.N*horizon,"Invalid realized audit count")
        require(math.isclose(float(r["actual_audits_per_citizen"]),total/base.N,abs_tol=1e-12),
                "Audit count metadata mismatch")
        state_periods=sum(int(x["state_learning_periods"]) for x in o["channel_citizens"])
        require(state_periods+total==base.N*horizon,"Audit/substantive period mismatch")


def worker(seed,suite,variant,horizon,design_id,root):
    set_floor()
    with variant_scope(variant):
        outputs=[sci.run_instrumented(seed,suite,c,horizon,design_id) for c in cells(suite)]
        rep=sci.REPRESENTATIVE[suite]
        chosen=next(o for o in outputs if o["run"]["cell"]==rep[0]
                    and o["run"]["sender_regime"]==rep[3] and o["run"]["reliance_mode"]==rep[4])
        plain,_=base.run_one(seed,suite,rep,horizon,design_id,instrument=False)
        for key in ("beliefs","belief_checkpoints","lambda_checkpoints","jammer_strategy"):
            require(base.parity_payload(chosen[key])==base.parity_payload(plain[key]),
                    "Variant logger parity failed: "+key)
    for o in outputs:
        audit=next(x for x in o["group_request_checkpoints"] if x["group_type"]=="all"
                   and x["learning_target"]=="credibility" and x["horizon_step"]==horizon)
        meta=dict(variant=variant,epsilon_requested=.05,external_precision_cap=None,
                  rank_rule="uniform_tail" if variant=="uniform_tail" else "recursive",
                  audit_rule="constant_after_t0" if variant=="constant_audit" else "harmonic",
                  constant_post_t0_p=AUDIT_P if variant=="constant_audit" else None,
                  audit_reference_horizon=REFERENCE_HORIZON,
                  expected_audits_per_citizen=expected_audits(variant,horizon),
                  actual_audits_per_citizen=int(audit["citizen_periods"])/base.N)
        o["run"].update(meta)
        for key in (*base.KEYS,"group_terminal","group_acquisition","group_belief_checkpoints","group_request_checkpoints"):
            for row in o[key]:
                row.update(meta)
    validate(outputs,seed,suite,variant,horizon,design_id)
    dest=Path(root)/"shards"/f"{variant}__{suite}__{seed}.json.gz"
    tmp=dest.with_suffix(".tmp")
    with gzip.open(tmp,"wt",encoding="utf-8") as f:
        json.dump(dict(design_id=design_id,seed=seed,suite=suite,variant=variant,
                       parity_pass=True,outputs=outputs),f,allow_nan=True)
    os.replace(tmp,dest)
    return str(dest)


def read_shard(root,seed,suite,variant,horizon,design_id):
    with gzip.open(root/"shards"/f"{variant}__{suite}__{seed}.json.gz","rt",encoding="utf-8") as f:
        s=json.load(f)
    require((s["design_id"],s["seed"],s["suite"],s["variant"],s["parity_pass"])==
            (design_id,seed,suite,variant,True),"Shard identity mismatch")
    validate(s["outputs"],seed,suite,variant,horizon,design_id)
    return s["outputs"]


def zip_csv(z,name):
    with z.open(name+".csv.gz") as f:
        with gzip.GzipFile(fileobj=f) as g:
            return list(csv.DictReader(io.TextIOWrapper(g,encoding="utf-8",newline="")))


def archive(path):
    require(path.is_file(),"Missing original scientific shareable ZIP: "+str(path))
    with path.open("rb") as f:
        sha=hashlib.file_digest(f,"sha256").hexdigest()
    require(sha==ARCHIVE_SHA,"Original scientific ZIP SHA256 mismatch; do not mix archives")
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read("v6_manifest.json"))
        gate=json.loads(z.read("v6_gate.json"))
        d=dict(manifest)
        require(d.pop("design_id")==ARCHIVE_ID and up.canonical_hash(d)[:12]==ARCHIVE_ID,"Archive design mismatch")
        require(gate["pass"] is True and gate["design_id"]==ARCHIVE_ID and gate["grid_runs"]==480,
                "Archive gate mismatch")
        require(manifest["scientific_runner_fingerprint"]==ARCHIVE_FP==sci.fingerprint(),
                "Scientific runner fingerprint mismatch")
        require(manifest["upstream_fingerprint"]==up.code_fingerprint()==sci.EXPECTED_UPSTREAM,
                "Scientific model fingerprint mismatch")
        require(manifest["software_versions"]==up.software_versions(),"Software version mismatch")
        runs=zip_csv(z,"runs")
        groups=zip_csv(z,"group_terminal")
        requests=[r for r in zip_csv(z,"group_request_checkpoints")
                  if int(r["horizon_step"])==400 and r["learning_target"]=="theta"]
        beliefs=zip_csv(z,"beliefs")
    require(len(runs)==480 and len({r["run_id"] for r in runs})==480,"Incomplete archive grid")
    return dict(runs={r["run_id"]:r for r in runs},groups=groups,requests=requests,beliefs=beliefs)


def check_archive_structure(o,a):
    r=o["run"]
    b=a["runs"][r["run_id"]]
    for k in ("initial_state_fingerprint","structural_fingerprint","cell","sender_regime","reliance_mode"):
        require(str(r[k])==str(b[k]),"Archived matched structure mismatch: "+k)


def baseline_replay(a,design_id):
    """Two preselected T400 runs compared with original archive, not the new variant."""
    set_floor()
    report=[]
    for suite in SUITES:
        rep=sci.REPRESENTATIVE[suite]
        with variant_scope("baseline"):
            o=sci.run_instrumented(6001,suite,rep,400,design_id)
        check_archive_structure(o,a)
        rid=o["run"]["run_id"]
        def compare(got,old,keys,metrics):
            idx={tuple(str(r[k]) for k in keys):r for r in old if r["run_id"]==rid}
            require(len(got)==len(idx),"Archived baseline row count mismatch")
            for r in got:
                b=idx[tuple(str(r[k]) for k in keys)]
                for k in metrics:
                    require(math.isclose(float(r[k]),float(b[k]),rel_tol=1e-12,abs_tol=1e-12),
                            "Archived baseline numeric mismatch: "+k)
        compare(o["beliefs"],a["beliefs"],("citizen_pos",),("terminal_mu_theta","terminal_sd_theta"))
        compare(o["group_terminal"],a["groups"],("group_type","group"),
                ("mse_truth","mean_belief","squared_bias","belief_variance","Q_jammer","Q_expert","W_jammer","W_same_peer"))
        compare([r for r in o["group_request_checkpoints"] if r["horizon_step"]==400 and r["learning_target"]=="theta"],
                a["requests"],("group_type","group"),("total_requests","requests_jammer","requests_expert",
                                                     "requests_same_peer","requests_other_peer","requests_gateway_peer"))
        report.append(dict(run_id=rid,pass_=True))
    return report


def audit_t0(outputs):
    fields=("learning_target","peer_audit_signature","peer_initial_means","peer_outgoing_sds",
            "source_count","peer_request_count","peer_delivered_count","expert_request_count",
            "expert_delivered_count","jammer_request_count","jammer_delivered_count","jammer_message_mean",
            "theta_mu_pre","theta_mu_post","theta_sd_pre","theta_sd_post",
            "jammer_delta_mu_post","jammer_delta_sd_post","jammer_rank_post")
    return {o["run"]["run_id"]:{int(r["citizen_pos"]):tuple(r[k] for k in fields)
                            for r in o["sender_events"] if int(r["period"])==0} for o in outputs}


def rows_for(outputs,key):
    for o in outputs:
        yield from ([o["run"]] if key=="run" else o[key])


def consolidate(root,design,a):
    seeds=design["seeds"]; horizon=design["horizon"]; variants=design["variants"]; did=design["design_id"]
    columns=defaultdict(set); counts=defaultdict(int); all_contrasts=[]
    for seed in seeds:
        for suite in SUITES:
            old=None
            if horizon==400:
                ids={f"{suite}__{seed}__{c[0]}__{c[3]}__{c[4]}" for c in cells(suite)}
                old=contrasts([r for r in a["groups"] if r["run_id"] in ids],
                              [r for r in a["requests"] if r["run_id"] in ids],
                              horizon,seed,suite,"baseline")
                all_contrasts.extend(old)
            sig=None
            for variant in variants:
                outputs=read_shard(root,seed,suite,variant,horizon,did)
                for o in outputs:
                    check_archive_structure(o,a)
                s=audit_t0(outputs)
                if sig is None:
                    sig=s
                else:
                    require(s==sig,"Initial audit changed across sensitivity variants")
                cr=contrasts(list(rows_for(outputs,"group_terminal")),
                             list(rows_for(outputs,"group_request_checkpoints")),horizon,seed,suite,variant)
                all_contrasts.extend(cr)
                if old is not None:
                    all_contrasts.extend(change_rows(cr,old))
                for key in FIELDS:
                    for row in rows_for(outputs,key):
                        columns[key].update(row)
                        counts[key]+=1
    require(counts["run"]==design["grid_runs"],"Incomplete robustness grid")
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
                for variant in variants:
                    outputs=read_shard(root,seed,suite,variant,horizon,did)
                    for key,w in writers.items():
                        w.writerows(rows_for(outputs,key))
    base.csv_gz(root/"seed_contrasts.csv.gz",all_contrasts)
    base.csv_gz(root/"contrast_summary.csv.gz",summarize_variants(all_contrasts,seeds))
    return dict(counts),len(all_contrasts)


def design_for(phase):
    seeds,horizon,variants=PHASES[phase]
    d=dict(phase=phase,seeds=list(seeds),horizon=horizon,variants=list(variants),N=100,
           cells={s:cells(s) for s in SUITES},credit=20,K=1,surveillance=5,tau_social=1.,
           initial_audit="centroid",frozen_ranking="first_audit",peer_evidence="source_posterior",
           variant_settings={"baseline":{"epsilon":.05,"rank_rule":"recursive","audit":"harmonic"},
                             "uniform_tail":{"epsilon":.05,"rank_rule":"uniform_tail","audit":"harmonic"},
                             "constant_audit":{"epsilon":.05,"rank_rule":"recursive","audit":"constant_after_t0",
                                               "post_t0_p":AUDIT_P,"reference_horizon":REFERENCE_HORIZON}},
           grid_runs=len(seeds)*16*len(variants),parity_runs=len(seeds)*2*len(variants),
           archive_replay_runs=2 if phase=="preflight" else 0,
           archive_id=ARCHIVE_ID,archive_sha256=ARCHIVE_SHA,upstream_fingerprint=up.code_fingerprint(),
           scientific_runner_fingerprint=sci.fingerprint(),behavioral_fingerprint=fingerprint(),
           software_versions=up.software_versions(),production_authorized=False)
    d["design_id"]=up.canonical_hash(d)[:12]
    return d


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
    archive_path=Path(args.baseline_zip).expanduser().resolve() if args.baseline_zip else (
        parent/f"v6_scientific_{ARCHIVE_ID}"/f"paper_b_v6_scientific_{ARCHIVE_ID}_shareable.zip")
    a=archive(archive_path)
    d=design_for(args.phase)
    did=d["design_id"]
    root=parent/f"v6_{args.phase}_{did}_behavioral"
    print(json.dumps(d,indent=2),flush=True)
    print("Output:",root,flush=True)
    if args.dry_run:
        return
    if args.phase=="robustness":
        pre=design_for("preflight")
        pp=parent/f"v6_preflight_{pre['design_id']}_behavioral"/"v6_gate.json"
        require(pp.exists(),"Complete robustness preflight first")
        g=json.loads(pp.read_text())
        require(g.get("pass") is True and g.get("design_id")==pre["design_id"] and
                g.get("archive_replay_pass") is True and g.get("matched_t0_audit") is True,
                "Matching robustness preflight not passed")
    require(not root.exists() or args.resume,"Output exists: use --resume with same design")
    root.mkdir(parents=True,exist_ok=True); (root/"shards").mkdir(exist_ok=True)
    manifest=root/"v6_manifest.json"
    if manifest.exists():
        require(json.loads(manifest.read_text())==json.loads(json.dumps(d)),"Manifest mismatch")
    manifest.write_text(json.dumps(d,indent=2))
    gp=root/"v6_gate.json"
    gp.write_text(json.dumps({"pass":False,"status":"incomplete","design_id":did}))
    tasks=[]
    for seed in d["seeds"]:
        for suite in SUITES:
            for v in d["variants"]:
                path=root/"shards"/f"{v}__{suite}__{seed}.json.gz"
                if args.resume and path.exists():
                    read_shard(root,seed,suite,v,d["horizon"],did)
                else:
                    tasks.append((seed,suite,v,d["horizon"],did,str(root)))
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as pool:
            futures=[pool.submit(worker,*t) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                print(f"[{i}/{len(futures)}] {f.result()}",flush=True)
    counts,n=consolidate(root,d,a)
    replay=baseline_replay(a,did) if args.phase=="preflight" else None
    gate={"pass":True,"status":args.phase+"_only","design_id":did,
          "grid_runs":d["grid_runs"],"parity_runs":d["parity_runs"],"row_counts":counts,
          "seed_contrast_rows":n,"archive_replay_pass":True if replay is not None else None,
          "archive_replay":replay,"matched_t0_audit":True,
          "production_authorized":False}
    gp.write_text(json.dumps(gate,indent=2))
    bundle=root/f"paper_b_v6_{args.phase}_{did}_behavioral_shareable.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.glob("*.json"))+sorted(root.glob("*.csv.gz")):
            z.write(path,path.name)
    print(json.dumps(gate,indent=2),flush=True)
    print("Shareable ZIP:",bundle,flush=True)


if __name__=="__main__":
    main()

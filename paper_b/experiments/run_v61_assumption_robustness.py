"""V6.1 isolated assumption diagnostics. Run from repository root."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
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

import numpy as np
import model.InfoSourceSamplingLearning as m
from paper_b.experiments import run_matched_pilot as up
from paper_b.experiments import run_v6_pilot as base
from paper_b.experiments import run_v6_scientific as sci
from paper_b.experiments import run_v6_robustness as hg
from paper_b.experiments import v6_channel_bridge_analysis as ca
from paper_b.experiments import v6_robustness_analysis as ha
from paper_b.experiments.v61_assumption_interventions import intervention, VARIANTS, BOUND

ORIGINAL_CELLS = base.cells
EARLY = 11
BOOT_SEED = 20261124
DRAWS = 10000
C_REP = ("extended_live","extended_live","flat","credibility_aware","adaptive")
REP = dict(sci.REPRESENTATIVE, C=C_REP)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def cells(variant, suite):
    if variant == "RC":
        require(suite == "C", "RC is C only")
        return [c for c in ORIGINAL_CELLS("C")
                if c[3] in ("adaptive","credibility_aware")]
    require(variant in ("RA","RB") and suite in ("H","G"), "Invalid H/G variant")
    return [c for c in ORIGINAL_CELLS(suite) if suite == "G" or c[2] == "high"]


def identity(row):
    return (row["cell"],row["sender_regime"],row["reliance_mode"])


def floor():
    m.MIN_SD = 1e-8
    m.MIN_VAR = m.MIN_SD**2


@contextmanager
def early_observer(records, gateways):
    """Record immediate source draws and assigned precision, never upstream provenance.

    After compute_pending_update, before synchronous commit. The next period's
    pre-rank records the committed behavioral rank. No RNG or state writes.
    """
    original = m.Citizen.compute_pending_update
    def observed(self):
        original(self)
        t = int(self.model.period)
        if t >= EARLY:
            return
        rank = {source:i+1 for i,source in enumerate(self._behavioral_ranking())}
        for j,source in enumerate(self._sample_order):
            messages = self.sampled_msgs[j]
            n = len(messages)
            typ = source.type_of_agent
            cls = ("expert" if typ == "infoprovider" else
                   "jammer" if typ == "disruptivejammer" else
                   "same_peer" if source.group_id == self.group_id else "other_peer")
            records.append(dict(period=t, citizen_pos=int(self.pos),
                citizen_group=int(self.group_id), source_pos=int(source.pos),
                source_class=cls, source_type=typ,
                gateway_ego=None if gateways is None else int(self.pos in gateways),
                gateway_source=None if gateways is None or typ!="citizen" else int(source.pos in gateways),
                learning_target="audit" if self.theta_or_delta==1 else "theta",
                rank_pre=rank[source], requested=int(self.num_request[j]), delivered=n,
                message_mean=float(sum(messages)/n) if n else None,
                delivered_mean_sq=float(sum(float(x)**2 for x in messages)/n) if n else None,
                assigned_precision=float(self._pending_evidence_precision.get(source,0.0)),
                theta_mu_pre=float(self.mu_theta),theta_sd_pre=float(self.sd_theta),
                theta_mu_pending=float(self._pending_mu_theta),
                theta_sd_pending=float(self._pending_sd_theta)))
    with patch.object(m.Citizen,"compute_pending_update",observed):
        yield


def run_observed(seed,suite,cell,horizon,did):
    # Determine roles using the same structural helper, not a new random stream.
    _,_,_,gateways = base.config_for(seed,suite,cell[1],cell[2],horizon)
    trace=[]
    with early_observer(trace,gateways):
        out=sci.run_instrumented(seed,suite,cell,horizon,did)
    out["early_sources"]=trace
    return out


def validate_one(o,variant,seed,suite,h,did):
    r=o["run"]
    require((r["seed"],r["suite"],r["design_id"])==(seed,suite,did),"Run identity")
    base.check_run(o,h)
    trace=o["early_sources"]
    by=defaultdict(list)
    for e in trace:
        by[(e["period"],e["citizen_pos"])].append(e)
        require(0<=e["delivered"]<=e["requested"]<=base.CREDIT,"Delivery accounting")
        require(math.isfinite(e["assigned_precision"]) and e["assigned_precision"]>=0,"Precision accounting")
        if e["learning_target"]=="audit":
            require(e["assigned_precision"]==0,"Audit added state precision")
        if variant=="RB" and e["source_class"] in ("expert","jammer"):
            require(e["assigned_precision"]<=1+1e-10,"RB external precision cap")
    require(len(by)==base.N*min(EARLY,h),"Incomplete early event grid")
    for rows in by.values():
        require(sum(x["requested"] for x in rows)==base.CREDIT,"Early budget")
        require(sorted(x["rank_pre"] for x in rows)==list(range(1,len(rows)+1)),"Rank permutation")
        require(len({x["source_pos"] for x in rows})==len(rows),"Duplicate source")
    if variant=="RA":
        for x in o["jammer_strategy"]:
            if int(x["period"])==0 and r["sender_regime"]=="fixed_biased":
                require(x["message_mean"]==4.0 and x["strategy_phase"]=="initial_audit_fixed_biased",
                        "RA initial strategy not applied")
    if variant=="RB":
        for c in o["channel_citizens"]:
            for cls in ("expert","jammer"):
                require(float(c["Q_"+cls])<=int(c["state_learning_periods"])+1e-8,"RB cumulative cap")
    if variant=="RC":
        rows=o["jammer_strategy"]
        require({int(x["period"]) for x in rows}==set(range(h)) and len(rows)==h,
                "Incomplete K=1 sender trajectory")
        for x in rows:
            if int(x["period"])==0:
                require(x["message_mean"]==x["segment_mean"],"RC changed initial audit")
            else:
                u=float(x["unconstrained_message_mean"])
                require(float(x["message_mean"])==max(-BOUND,min(BOUND,u)),"RC optimizer mismatch")
                require(float(x["objective_denominator"])>0,"RC nonconcave objective")
                require(bool(x["bound_active"])==(float(x["message_mean"])!=u),"RC bound flag")
    groups={(x["group_type"],str(x["group"])):int(x["n_citizens"]) for x in o["group_terminal"]}
    steps={0,h,*(int(x["horizon_step"]) for x in o["belief_checkpoints"])}
    expected={(t,k,g) for t in steps for k,g in groups}
    b=o["group_belief_checkpoints"]; q=o["group_request_checkpoints"]
    require(len(b)==len(expected) and {(int(x["horizon_step"]),x["group_type"],str(x["group"])) for x in b}==expected,
            "Incomplete belief checkpoint grid")
    eq={(t,k,g,z) for t,k,g in expected for z in ("all","theta","credibility")}
    require(len(q)==len(eq) and {(int(x["horizon_step"]),x["group_type"],str(x["group"]),x["learning_target"]) for x in q}==eq,
            "Incomplete request checkpoint grid")
    require(all(x["total_requests"]==x["citizen_periods"]*base.CREDIT for x in q),"Request budgets")


def validate_block(outputs,variant,seed,suite,h,did):
    expected={(c[0],c[3],c[4]) for c in cells(variant,suite)}
    require(len(outputs)==len(expected) and {identity(o["run"]) for o in outputs}==expected,"Incomplete block")
    base.check_block(outputs,"generic")
    if suite in ("H","G"):
        with patch.object(base,"cells",lambda s: cells(variant,s)):
            sci.validate_outputs(outputs,suite,seed,h,did)
    else:
        ext=[o for o in outputs if o["run"]["cell"]!="elite_only"]
        require(len({o["run"]["structural_fingerprint"] for o in ext})==1,"M/S/L graphs differ")
        sig=hg.audit_t0(outputs)
        for sender in ("adaptive","credibility_aware"):
            for mode in ("adaptive","frozen"):
                ids=[o["run"]["run_id"] for o in ext if identity(o["run"])[1:]==(sender,mode)]
                require(len(ids)==3 and all(sig[i]==sig[ids[0]] for i in ids),"M/S/L audit mismatch")
    for o in outputs:
        validate_one(o,variant,seed,suite,h,did)


def shard(root,seed,suite):
    return Path(root)/"shards"/f"{suite}__{seed}.json.gz"


def worker(variant,seed,suite,h,did,root):
    floor()
    with intervention(variant):
        outputs=[run_observed(seed,suite,c,h,did) for c in cells(variant,suite)]
        rep=REP[suite]
        chosen=next(o for o in outputs if identity(o["run"])==(rep[0],rep[3],rep[4]))
        plain,_=base.run_one(seed,suite,rep,h,did,instrument=False)
        for k in ("beliefs","belief_checkpoints","lambda_checkpoints","jammer_strategy"):
            require(base.parity_payload(chosen[k])==base.parity_payload(plain[k]),"Logger parity: "+k)
    validate_block(outputs,variant,seed,suite,h,did)
    dest=shard(root,seed,suite); tmp=dest.with_suffix(".tmp")
    with gzip.open(tmp,"wt",encoding="utf-8") as f:
        json.dump(dict(design_id=did,variant=variant,seed=seed,suite=suite,
                       parity_pass=True,outputs=outputs),f,allow_nan=True)
    os.replace(tmp,dest)
    return str(dest)


def read_block(root,variant,seed,suite,h,did):
    with gzip.open(shard(root,seed,suite),"rt",encoding="utf-8") as f:
        data=json.load(f)
    require((data["design_id"],data["variant"],data["seed"],data["suite"],data["parity_pass"])
            ==(did,variant,seed,suite,True),"Shard provenance")
    outputs=data["outputs"]
    validate_block(outputs,variant,seed,suite,h,did)
    return outputs


def compare_hg_null(o,a):
    rid=o["run"]["run_id"]
    old={int(x["citizen_pos"]):x for x in a["beliefs"] if x["run_id"]==rid}
    require(len(old)==base.N,"Missing archived null citizens")
    for x in o["beliefs"]:
        for k in ("terminal_mu_theta","terminal_sd_theta"):
            require(math.isclose(float(x[k]),float(old[int(x["citizen_pos"])][k]),
                                 abs_tol=1e-12,rel_tol=1e-12),"RA null changed: "+k)


def preflight_archive(variant,a,did):
    floor()
    if variant in ("RA","RB"):
        return hg.baseline_replay(a,did)
    results=[]
    replay=[c for c in ORIGINAL_CELLS("C") if c[3]=="null"]
    replay.extend(c for c in ORIGINAL_CELLS("C")
                  if c[0]=="extended_live" and c[3] in ("adaptive","credibility_aware") and c[4]=="frozen")
    for cell in replay:
        o=sci.run_instrumented(7501,"C",cell,400,did)
        ca.compare_archive(o,a,replay=True)
        results.append(dict(run_id=o["run"]["run_id"],pass_=True))
    # Also verify the RC patch itself leaves null unchanged, not merely
    # the unpatched archived replay. One predetermined representative.
    cell=("extended_live","extended_live","flat","null","adaptive")
    with intervention("RC"):
        o=sci.run_instrumented(7501,"C",cell,400,did)
    ca.compare_archive(o,a,replay=True)
    results.append(dict(run_id=o["run"]["run_id"],pass_=True,patched_null=True))
    return results


def preflight_initial(root,variant,d):
    floor()
    for suite in d["suites"]:
        seed=d["seeds"][0]; h=d["horizon"]; did=d["design_id"]
        out=read_block(root,variant,seed,suite,h,did)
        rep=REP[suite]
        new=next(o for o in out if identity(o["run"])==(rep[0],rep[3],rep[4]))
        old=sci.run_instrumented(seed,suite,rep,h,did)
        sig_old=hg.audit_t0([old])[old["run"]["run_id"]]
        sig_new=hg.audit_t0([new])[new["run"]["run_id"]]
        if variant!="RA":
            require(sig_old==sig_new,"Unintended initial audit change")
        else:
            # The audit sample count, peers and initial substantive states
            # must be identical; only Jammer draws and resulting audit
            # credibility/rank may differ.
            eo={int(x["citizen_pos"]):x for x in old["sender_events"] if int(x["period"])==0}
            en={int(x["citizen_pos"]):x for x in new["sender_events"] if int(x["period"])==0}
            shift=4.0-float(old["jammer_strategy"][0]["message_mean"])
            fields=("learning_target","peer_audit_signature","peer_initial_means",
                    "peer_outgoing_sds","source_count","peer_request_count","peer_delivered_count",
                    "expert_request_count","expert_delivered_count","jammer_request_count",
                    "jammer_delivered_count","theta_mu_pre","theta_mu_post","theta_sd_pre","theta_sd_post")
            require(set(eo)==set(en),"RA initial citizen IDs")
            for pos,x in eo.items():
                y=en[pos]
                require(all(x[k]==y[k] for k in fields),"RA unintended audit change")
                if int(x["jammer_delivered_count"]):
                    require(math.isclose(float(y["jammer_message_mean"])-float(x["jammer_message_mean"]),
                                         shift,abs_tol=1e-12,rel_tol=1e-12),"RA audit mean shift")
    return True


def run_meta(o):
    r=o["run"]
    return {k:r[k] for k in ("suite","seed","cell","sender_regime","reliance_mode","run_id")}


def early_summary(o):
    meta=run_meta(o)
    grouped=defaultdict(lambda:[0,0,0,0,0.,0.,0.,0.,0.,0.])
    for x in o["early_sources"]:
        roles=[("all","all"),("citizen_group",str(x["citizen_group"]))]
        if x["gateway_ego"] is not None:
            roles.append(("gateway_role","gateway" if x["gateway_ego"] else "non_gateway"))
        for kind,g in roles:
            k=(x["period"],kind,g,x["learning_target"],x["source_class"])
            a=grouped[k]; a[0]+=1; a[1]+=int(x["rank_pre"]==1)
            a[2]+=x["requested"]; a[3]+=x["delivered"]
            a[4]+=x["assigned_precision"]
            if x["delivered"]:
                a[5]+=x["message_mean"]*x["delivered"]
                a[6]+=x["delivered_mean_sq"]*x["delivered"]
                a[7]+=x["message_mean"]*x["assigned_precision"]
                a[8]+=x["message_mean"]**2*x["assigned_precision"]
            a[9]+=x["rank_pre"]
    fields=("source_slots","top_rank_slots","requests","delivered","precision",
            "sum_delivered_messages","sum_squared_delivered_messages",
            "precision_times_message_mean","precision_times_squared_message_mean","sum_rank")
    return [dict(meta,period=k[0],group_type=k[1],group=k[2],learning_target=k[3],source_class=k[4],
                 **dict(zip(fields,v))) for k,v in sorted(grouped.items())]


def capture_rows(o):
    meta=run_meta(o); groups=defaultdict(list)
    for b in o["beliefs"]:
        v=float(b["terminal_mu_theta"])
        groups[("all","all")].append(v)
        groups[("citizen_group",str(b["citizen_group"]))].append(v)
    rows=[]
    for (kind,g),vals in sorted(groups.items()):
        x=np.asarray(vals)
        rows.append(dict(meta,group_type=kind,group=g,n=len(x),mse=float(np.mean(x*x)),
                         jammer_side=float(np.mean(x>2)),near_jammer=float(np.mean(abs(x-4)<=.5)),
                         near_truth=float(np.mean(abs(x)<=.5)),abs_over_16=float(np.mean(abs(x)>16)),
                         over_40=float(np.mean(x>40))))
    return rows


def write_csv(path,rows):
    require(bool(rows),"Empty table: "+str(path))
    fields=sorted({k for r in rows for k in r})
    opener=gzip.open if path.name.endswith(".gz") else open
    with opener(path,"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def c_contrasts(outputs,a,variant):
    seed=int(outputs[0]["run"]["seed"])
    paths={ca.key(o["run"]):{int(x["horizon_step"]):float(x["mse_truth"])
                             for x in o["belief_checkpoints"]} for o in outputs}
    result=[]
    for mode in ca.RECEIVERS:
        for sender in ("adaptive","credibility_aware"):
            for step in ca.STEPS:
                null=[a["checkpoint"][(seed,c,"null",mode)][step] for c in ca.CHANNELS]
                original=[a["checkpoint"][(seed,c,sender,mode)][step] for c in ca.CHANNELS]
                bounded=[paths[(seed,c,sender,mode)][step] for c in ca.CHANNELS]
                old=ca.channel_contrasts(seed,mode,sender,step,null,original,"baseline")
                new=ca.channel_contrasts(seed,mode,sender,step,null,bounded,variant)
                change={k:new[k]-old[k] if k.startswith(("D_","null_MSE_","sender_MSE_")) else new[k] for k in new}
                change["source"]=variant+"_minus_baseline"
                result.extend((old,new,change))
    return result


def summarize(rows,seeds):
    """Common whole-seed bootstrap draws for every complete scalar contrast."""
    rng=np.random.default_rng(BOOT_SEED)
    draws=rng.integers(0,len(seeds),size=(DRAWS,len(seeds)))
    grouped=defaultdict(dict)
    for r in rows:
        labels={k:v for k,v in r.items() if k!="seed" and not k.startswith(("delta_","divergence_","D_","sender_MSE_","null_MSE_"))}
        key=tuple(sorted(labels.items()))
        seed=int(r["seed"])
        require(seed not in grouped[key],"Duplicate contrast seed")
        grouped[key][seed]=r
    summaries=[]
    for key,block in grouped.items():
        require(set(block)==set(seeds),"Incomplete contrast seed grid")
        for metric in sorted(k for k in block[seeds[0]] if k.startswith(("delta_","divergence_","D_","sender_MSE_","null_MSE_"))):
            x=np.array([float(block[s][metric]) for s in seeds])
            require(np.isfinite(x).all(),"Nonfinite contrast")
            lo,hi=np.quantile(x[draws].mean(axis=1),[.025,.975])
            summaries.append(dict(dict(key),metric=metric,n=len(x),mean=float(x.mean()),
                                  mcse=float(x.std(ddof=1)/np.sqrt(len(x))),lo=float(lo),hi=float(hi),
                                  positive_share=float(np.mean(x>0)),negative_share=float(np.mean(x<0))))
    return summaries


def consolidate(root,variant,d,a):
    h=d["horizon"]; did=d["design_id"]
    tables=defaultdict(list); contrasts=[]
    for seed in d["seeds"]:
        for suite in d["suites"]:
            outputs=read_block(root,variant,seed,suite,h,did)
            for o in outputs:
                if suite=="C":
                    ca.compare_archive(o,a)
                else:
                    hg.check_archive_structure(o,a)
                    if h==400 and variant=="RA" and o["run"]["sender_regime"]=="null":
                        compare_hg_null(o,a)
                # Compact results, while raw early sources/citizen/event/edge records
                # remain in local gzipped JSON shards.
                tables["runs"].append(o["run"])
                tables["capture"].extend(capture_rows(o))
                tables["early_source_summary"].extend(early_summary(o))
                for key in ("group_terminal","group_belief_checkpoints","group_request_checkpoints","group_acquisition",
                            "rank_unification_checkpoints","expert_rank_checkpoints","jammer_strategy"):
                    tables[key].extend(o[key])
            if h==400:
                if suite=="C":
                    contrasts.extend(c_contrasts(outputs,a,variant))
                else:
                    ids={o["run"]["run_id"] for o in outputs}
                    old=ha.contrasts([r for r in a["groups"] if r["run_id"] in ids],
                                     [r for r in a["requests"] if r["run_id"] in ids],h,seed,suite,"baseline")
                    new=ha.contrasts(list(hg.rows_for(outputs,"group_terminal")),
                                     list(hg.rows_for(outputs,"group_request_checkpoints")),h,seed,suite,variant)
                    contrasts.extend(old+new+ha.change_rows(new,old))
    require(len(tables["runs"])==d["grid_runs"],"Incomplete grid")
    if contrasts:
        tables["seed_contrasts"]=contrasts
        tables["contrast_summary"]=summarize(contrasts,d["seeds"])
    for name,rows in tables.items():
        if rows:
            write_csv(root/(name+".csv.gz"),rows)
    return {name:len(rows) for name,rows in tables.items()}


def design(variant,phase):
    suites=["C"] if variant=="RC" else ["H","G"]
    first=7501 if variant=="RC" else 6001
    seeds=[first] if phase=="preflight" else list(range(first,first+20))
    paths=("paper_b/experiments/v61_assumption_interventions.py",
           "paper_b/experiments/run_v61_assumption_robustness.py",
           "paper_b/V61_ASSUMPTION_PROTOCOL.md")
    files={p:hashlib.sha256((base.ROOT/p).read_bytes()).hexdigest() for p in paths}
    d=dict(variant=variant,phase=phase,horizon=26 if phase=="preflight" else 400,
           suites=suites,seeds=seeds,grid={s:cells(variant,s) for s in suites},
           grid_runs=len(seeds)*sum(len(cells(variant,s)) for s in suites),
           parity_runs=len(seeds)*len(suites),early_periods=EARLY,
           model_fingerprint=up.code_fingerprint(),scientific_fingerprint=sci.fingerprint(),
           software_versions=up.software_versions(),file_sha256=files,
           archive_sha256=ca.ARCHIVE_SHA if variant=="RC" else hg.ARCHIVE_SHA,
           audit_mean=4.0 if variant=="RA" else None,external_cap=1 if variant=="RB" else None,
           strategic_bound=BOUND if variant=="RC" else None,bootstrap_seed=BOOT_SEED,bootstrap_draws=DRAWS)
    # Normalize tuples for stable equality with JSON-loaded manifests.
    d=json.loads(json.dumps(d))
    d["design_id"]=up.canonical_hash(d)[:12]
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",choices=VARIANTS,required=True)
    p.add_argument("--phase",choices=("preflight","robustness"),required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--baseline-zip",required=True,help="Exact scientific H/G or original Gate 0E2 ZIP")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    require(args.workers>0,"Positive worker count required")
    require(up.code_fingerprint()==sci.EXPECTED_UPSTREAM,"Archived upstream fingerprint changed")
    require(sci.fingerprint()==hg.ARCHIVE_FP,"Archived scientific fingerprint changed")
    variant=args.experiment; phase=args.phase
    arc=Path(args.baseline_zip).expanduser().resolve()
    a=ca.load_archive(arc,up.code_fingerprint(),up.software_versions()) if variant=="RC" else hg.archive(arc)
    d=design(variant,phase); did=d["design_id"]; h=d["horizon"]
    parent=Path(args.output_dir).expanduser().resolve()
    root=parent/f"v61_{variant}_{phase}_{did}"
    print(json.dumps(d,indent=2),flush=True); print("Output:",root,flush=True)
    if args.dry_run:
        return
    if phase=="robustness":
        pre=design(variant,"preflight")
        gpath=parent/f"v61_{variant}_preflight_{pre['design_id']}"/"v61_gate.json"
        require(gpath.is_file(),"Run matching preflight first")
        g=json.loads(gpath.read_text())
        require(g.get("pass") is True and g.get("design_id")==pre["design_id"]
                and g.get("grid_runs")==pre["grid_runs"] and g.get("parity_runs")==pre["parity_runs"]
                and g.get("archive_replay_pass") is True and g.get("initial_audit_pass") is True,
                "Matching preflight has not passed")
    require(not root.exists() or args.resume,"Output exists; use --resume only for identical design")
    (root/"shards").mkdir(parents=True,exist_ok=True)
    manifest=root/"v61_manifest.json"
    if manifest.exists():
        require(json.loads(manifest.read_text())==d,"Manifest mismatch")
    manifest.write_text(json.dumps(d,indent=2))
    gate=root/"v61_gate.json"
    gate.write_text(json.dumps(dict(pass_=False,design_id=did,status="incomplete")))
    tasks=[]
    for seed in d["seeds"]:
        for suite in d["suites"]:
            if args.resume and shard(root,seed,suite).exists():
                read_block(root,variant,seed,suite,h,did)
            else:
                tasks.append((variant,seed,suite,h,did,str(root)))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as pool:
        fs=[pool.submit(worker,*t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):
            print(f"[{i}/{len(fs)}]",f.result(),flush=True)
    replay=preflight_archive(variant,a,did) if phase=="preflight" else []
    init=preflight_initial(root,variant,d) if phase=="preflight" else True
    counts=consolidate(root,variant,d,a)
    report=dict(design_id=did,variant=variant,phase=phase,grid_runs=d["grid_runs"],
                parity_runs=d["parity_runs"],archive_replay_pass=bool(replay) if phase=="preflight" else True,
                archive_replays=replay,initial_audit_pass=init,counts=counts)
    report["pass"]=True
    gate.write_text(json.dumps(report,indent=2))
    # Share lightweight seed contrasts and checks first; retain all other
    # local CSVs/raw shards for follow-up diagnostic extracts.
    out=root/f"paper_b_v61_{variant}_{phase}_{did}_shareable.zip"
    include=["v61_manifest.json","v61_gate.json","runs.csv.gz","capture.csv.gz",
             "seed_contrasts.csv.gz","contrast_summary.csv.gz"]
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name in include:
            path=root/name
            if path.exists():
                z.write(path,name)
    print(json.dumps(report,indent=2),flush=True); print("Shareable ZIP:",out,flush=True)


if __name__=="__main__":
    main()

"""W2: frozen 8-cell peer-weight sensitivity; separate from V6.2 outputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
import fcntl
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch
import zipfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from paper_b.experiments import run_v62_mechanism_closure as old
from paper_b.experiments import v62_reporting as report
from paper_b.experiments.v62_archive import digest
from paper_b.experiments.v62_observer import Observer, observe, require, close
from paper_b.experiments.v62_policies import SEEDS, native_cell, scope
from paper_b.experiments import run_v6_pilot as base, run_v6_scientific as sci
from paper_b.experiments import run_matched_pilot as up
import model.InfoSourceSamplingLearning as m

SOURCE = "7e2e4f6724e3"
SOURCE_COMMIT = "55264824e35c02284174e206d86955269d2b0bbc"
SOURCE_MANIFEST_SHA = "3a83e3b7ff4af3ceeeaeda34bbd47633c9df07228089e25f6d57a56394cfff61"
GRID = tuple((v, s, l, "adaptive") for v in ("baseline", "neutral")
             for s in ("H", "G") for l in ("low", "high"))
GATE_SEED = SEEDS[0]
DEFAULT_REFERENCE = Path.home() / "Research/paperb_v6_results/v62_mechanism_closure/v62_7e2e4f6724e3"
DEFAULT_OUTPUT = Path.home() / "Research/paperb_v6_results/v63_w2"

def checked_tau(tau):
    require(type(tau) in (int, float) and math.isfinite(tau) and tau in (1., 2.),
            "Only pre-specified tau=1 replay and tau=2 sensitivity are allowed")
    return float(tau)

@contextmanager
def tau_scope(tau):
    # base.run_one explicitly passes tau_social=1, so a config-only patch
    # would be ineffective. Intercept exactly that named argument.
    tau = checked_tau(tau)
    native = base.run_condition
    validator = base.check_run
    def condition(*args, **kwargs):
        require(kwargs["tau_social"] == 1., "Unexpected native tau argument")
        kwargs["tau_social"] = tau
        out = native(*args, **kwargs)
        require(float(out["run"]["tau_social"]) == tau, "Actual tau mismatch")
        return out
    def validate(out, horizon):
        require(float(out["run"]["tau_social"]) == tau, "Actual validator tau mismatch")
        # The original validator hardcodes tau=1. Check the real tau above,
        # then invoke ALL remaining original checks on a temporary metadata
        # view. Never change the actual output, model, or archived record.
        view = {**out, "run": {**out["run"], "tau_social": 1.}}
        validator(view, horizon)
    with patch.object(base, "run_condition", condition), patch.object(base, "check_run", validate):
        yield

def run_native(seed, cell, horizon, did, tau, observer=None):
    v, s, lev, mode = cell
    m.MIN_SD = 1e-8
    m.MIN_VAR = m.MIN_SD ** 2
    with scope(v), tau_scope(tau), (observe(observer) if observer is not None else nullcontext()):
        return sci.run_instrumented(seed, s, native_cell(s, lev, mode), horizon, did)

def audit_signature(obj):
    return sorted(obj["initial_audit"], key=lambda r: (r["citizen_pos"], r["source_pos"]))

def compare_tree(a, b, path="root"):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), "Replay keys: "+path)
        for k in a:
            compare_tree(a[k], b[k], path+"."+str(k))
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), "Replay length: "+path)
        for i, (x, y) in enumerate(zip(a, b)):
            compare_tree(x, y, path+"["+str(i)+"]")
    elif isinstance(a, float):
        require(isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12),
                "Replay numeric mismatch: "+path)
    else:
        require(a == b, "Replay exact mismatch: "+path)

def reference(root, seed, cell):
    path, sha, _ = old.paths(root, "production", seed, cell)
    require(path.is_file() and sha.is_file(), "Missing archived reference "+str(path))
    require(digest(path) == sha.read_text().strip(), "Archived reference checksum "+str(path))
    obj = old.validate_saved(report.read_gz(path), "production", seed, cell, 400, SOURCE)
    return obj, sha.read_text().strip()

def check_match(obj, ref):
    for key in ("initial_state_fingerprint", "structural_fingerprint"):
        require(obj["run"][key] == ref["run"][key], "Reference fingerprint: "+key)
    require(audit_signature(obj) == audit_signature(ref), "First-audit mismatch")
    left = {r["citizen_pos"]: r for r in obj["terminal"]}
    right = {r["citizen_pos"]: r for r in ref["terminal"]}
    require(left.keys() == right.keys(), "Reference IDs")
    for i, row in left.items():
        require(all(row[k] == right[i][k] for k in
                    ("group","role","initial_mu","initial_sd","initial_phi","peer_ids","initial_rank",
                     "initial_latent_rank")), "Reference initial citizen attributes")

def validate(obj, seed, cell, horizon, did, tau, phase):
    v,s,lev,mode = cell
    require((obj["design_id"],obj["phase"],obj["seed"],obj["variant"],obj["suite"],
             obj["level"],obj["mode"],obj["horizon"],obj["tau_social"]) ==
            (did,phase,seed,v,s,lev,mode,horizon,tau), "W2 shard identity")
    require(obj["run"]["seed"] == seed and float(obj["run"]["tau_social"]) == tau,
            "Native identity/tau")
    term = obj["terminal"]
    require(len(term) == 100 and {r["citizen_pos"] for r in term} == set(range(2,102)),
            "Citizen coverage")
    require(obj["validation"]["periods"] == 100*horizon, "Observer coverage")
    require(obj["validation"]["floor_events"] == sum(r["floor_events"] for r in term),
            "Floor accounting")
    for r in term:
        require(close(sum(r["signed"].values()),r["mu"]) and
                close(sum(r["signed_mse"].values()),r["mu"]**2), "Terminal accounting")
        if not r["floor_events"]:
            require(close(r["initial_phi"]+sum(r["Q"].values()),r["phi"]) and
                    close(sum(r["B_raw"].values())/r["phi"],r["mu"]), "Raw accounting")
    require(close(sum(r["mu"]**2 for r in term)/100,obj["run"]["mse_truth"]), "Population MSE")
    expected_parity = phase == "preflight" or (phase == "production" and seed == GATE_SEED)
    require(obj["parity_pass"] is (True if expected_parity else None), "Parity provenance")
    json.dumps(obj, allow_nan=False)
    return obj

def saved(root, seed, cell, horizon, did, tau, phase):
    path, sha, _ = old.paths(root,phase,seed,cell)
    if not path.exists() and not sha.exists():
        return None
    require(path.is_file() and sha.is_file(), "Incomplete shard "+str(path))
    require(digest(path)==sha.read_text().strip(), "W2 shard checksum "+str(path))
    return validate(report.read_gz(path),seed,cell,horizon,did,tau,phase)

def worker(root, refroot, seed, cell, horizon, did, tau, phase):
    root, refroot = Path(root), Path(refroot)
    path, sha, _ = old.paths(root,phase,seed,cell)
    ref, ref_sha = reference(refroot,seed,cell)
    if path.exists() and not sha.exists():
        q = path.parent/"interrupted"
        q.mkdir(exist_ok=True)
        dest=q/(path.name+"."+str(os.getpid()))
        require(not dest.exists(),"Existing interrupted artifact")
        path.replace(dest)
    existing = saved(root,seed,cell,horizon,did,tau,phase)
    if existing is not None:
        require(existing["reference_sha256"] == ref_sha, "Reference checksum changed")
        check_match(existing,ref)
        if phase=="replay":
            verify_replay(existing,ref)
        return old.job_key(seed,cell)
    v,s,lev,mode=cell
    _,_,_,gateways=base.config_for(seed,s,lev,"high" if s=="H" else "flat",horizon)
    observer=Observer(v,gateways,None)
    out=run_native(seed,cell,horizon,did,tau,observer)
    obs=observer.finish(out,horizon)
    parity=None
    if phase=="preflight" or (phase=="production" and seed==GATE_SEED):
        plain=run_native(seed,cell,horizon,did,tau)
        for field in old.PARITY_FIELDS:
            require(base.parity_payload(out[field])==base.parity_payload(plain[field]),
                    "Observer parity "+field)
        parity=True
    obj=dict(design_id=did,phase=phase,seed=seed,variant=v,suite=s,level=lev,mode=mode,
             horizon=horizon,tau_social=tau,reference_sha256=ref_sha,parity_pass=parity,
             run={k:out["run"][k] for k in (*old.RUN_FIELDS,"tau_social")},**obs)
    validate(obj,seed,cell,horizon,did,tau,phase)
    check_match(obj,ref)
    if phase=="replay":
        verify_replay(obj,ref)
    report.write_gz(path,obj)
    tmp=sha.with_suffix(".tmp")
    tmp.write_text(digest(path)+"\n")
    tmp.replace(sha)
    return old.job_key(seed,cell)

def verify_replay(obj, ref):
    for k in old.RUN_FIELDS:
        compare_tree(obj["run"][k],ref["run"][k],"run."+k)
    for k in ("terminal","initial_audit","checkpoints","windows","validation"):
        compare_tree(obj[k],ref[k],k)

def inventory(roots, excluded):
    # Only metadata, not raw shard data. Numeric non-baseline tau values
    # flag candidates for manual scientific comparability assessment.
    rows=[]; errors=[]
    def visit(obj, where):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k.lower() in ("tau_social","peer_residual_sd") and isinstance(v,(int,float)) and not isinstance(v,bool):
                    rows.append(dict(path=where,key=k,value=v))
                elif isinstance(v,(dict,list)):
                    visit(v,where)
        elif isinstance(obj,list):
            for v in obj:
                if isinstance(v,(dict,list)):
                    visit(v,where)
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_relative_to(excluded):
                continue
            try:
                if path.suffix.lower()==".json" and "manifest" in path.name.lower():
                    visit(json.loads(path.read_text()),str(path))
                elif path.suffix.lower()==".zip":
                    with zipfile.ZipFile(path) as z:
                        for n in z.namelist():
                            if n.lower().endswith(".json") and "manifest" in Path(n).name.lower():
                                visit(json.loads(z.read(n)),str(path)+"::"+n)
            except (OSError,ValueError,zipfile.BadZipFile) as exc:
                errors.append(dict(path=str(path),error=str(exc)))
    # Deduplicate repeated manifest fields, retain full paths.
    unique={(r["path"],r["key"],r["value"]):r for r in rows}
    return dict(roots=[str(r) for r in roots],numeric_tau_records=list(unique.values()),
                candidate_records=[r for r in unique.values() if r["value"]!=1.],
                scan_errors=errors,
                limitation="Manifest metadata only; no claim of exhaustive archive contents")

def design(refroot):
    p=refroot/"MANIFEST.json"
    require(digest(p)==SOURCE_MANIFEST_SHA,"V6.2 manifest hash mismatch")
    ref=json.loads(p.read_text())
    require(ref["design_id"]==SOURCE and ref["seeds"]==list(SEEDS) and
            ref["expected_runs"]==3400 and ref["horizon"]==400 and
            set(GRID)<=set(map(tuple,ref["grid"])),"Reference manifest")
    for rel,sha in ref["source_hashes"].items():
        require(digest(base.ROOT/rel)==sha,"Frozen dependency changed: "+rel)
    require(ref["software"]==up.software_versions(),"Package versions differ from V6.2")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=base.ROOT,text=True).strip()
    files={str(p.relative_to(base.ROOT)):digest(p) for p in
           (Path(__file__),base.ROOT/"paper_b/test_v63_w2.py",base.ROOT/"paper_b/V63_W2_PROTOCOL.md")}
    d=dict(schema=1,source_design=SOURCE,source_commit=SOURCE_COMMIT,
           source_manifest_sha256=SOURCE_MANIFEST_SHA,checkout_commit=head,
           dependencies=ref["source_hashes"],w2_sources=files,software=up.software_versions(),
           python=sys.version,horizon=400,seeds=list(SEEDS),grid=[list(c) for c in GRID],
           expected_runs=800,tau_social=2.,reference_tau=1.,gate_seed=GATE_SEED,
           bootstrap_seed=report.BOOT,bootstrap_draws=10000)
    d["design_id"]=up.canonical_hash(d)[:12]
    return d

METRICS=("mse","mean","variance","Q_jammer","W_jammer","Q_expert","Q_peer",
         "near_truth","jammer_side","near_jammer","terminal_sd","audit_count","theta_count",
         "initial_rank1","initial_policy","initial_rank34","promotion",
         "peer_weight_raw","peer_weight_evidence","expert_weight_raw","jammer_weight_raw",
         "total_raw_precision","floor_events",
         *(f"{w}_peer_{k}" for w in ("early","middle","late","overall")
           for k in ("weighted_mean","weighted_error","snapshot_error","Q")))

def level_metrics(obj):
    full=report.level_rows(obj)
    for g,rr in report.groups(obj["terminal"]):
        def total(r): return r["initial_phi"]+sum(r["Q"].values())
        full[g].update(peer_weight_raw=sum(r["Q"]["peer"]/total(r) for r in rr)/len(rr),
                       expert_weight_raw=sum(r["Q"]["expert"]/total(r) for r in rr)/len(rr),
                       jammer_weight_raw=sum(r["Q"]["jammer"]/total(r) for r in rr)/len(rr),
                       peer_weight_evidence=sum(r["Q"]["peer"]/sum(r["Q"].values())
                                                if sum(r["Q"].values()) else 0. for r in rr)/len(rr),
                       total_raw_precision=sum(total(r) for r in rr)/len(rr),
                       floor_events=sum(r["floor_events"] for r in rr))
    return {g:{k:d.get(k) for k in METRICS} for g,d in full.items()}

def seed_rows(seed, outputs):
    levels={(tau,*cell):level_metrics(obj) for (tau,cell),obj in outputs.items()}
    rows=[]; delta={}
    def emit(kind,tau,s,policy,g,d):
        rows.extend(dict(seed=seed,kind=kind,variant=tau,suite=s,mode=policy,
                         group=g,metric=k,value=v) for k,v in sorted(d.items()))
    for (tau,v,s,l,mode),groups in levels.items():
        for g,d in groups.items():
            emit("level_"+l,tau,s,v,g,d)
            if l=="high":
                x=report.difference(d,levels[tau,v,s,"low",mode][g])
                delta[tau,v,s,g]=x
                emit("high_minus_low",tau,s,v,g,x)
    for (tau,v,s,g),d in delta.items():
        if tau=="tau2":
            emit("tau2_minus_tau1","tau2",s,v,g,
                 report.difference(d,delta["tau1",v,s,g]))
        if v=="baseline":
            an=report.difference(d,delta[tau,"neutral",s,g])
            emit("adaptive_minus_neutral",tau,s,"baseline",g,an)
            if tau=="tau2":
                an1=report.difference(delta["tau1","baseline",s,g],delta["tau1","neutral",s,g])
                emit("AN_tau2_minus_tau1","tau2",s,"baseline",g,report.difference(an,an1))
    return rows

def batch(pool,root,refroot,jobs,horizon,did,tau,phase):
    (root/phase).mkdir(exist_ok=True)
    futures={pool.submit(worker,str(root),str(refroot),seed,c,horizon,did,tau,phase):(seed,c)
             for seed,c in jobs}
    failures={}
    for i,f in enumerate(as_completed(futures),1):
        seed,c=futures[f]; key=old.job_key(seed,c)
        try:
            f.result()
            print(f"{phase} [{i}/{len(futures)}] {key} OK",flush=True)
        except Exception:
            failures[key]=traceback.format_exc()
            (root/phase/(key+".error.txt")).write_text(failures[key])
            print(f"{phase} [{i}/{len(futures)}] {key} FAILED",flush=True)
    require(not failures,f"{phase} failed: {sorted(failures)}; inspect per-cell error files")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,default=DEFAULT_REFERENCE)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--archive-root",type=Path,action="append")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--inventory-only",action="store_true")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--inventory-resolution",help="Document why flagged archives are not a comparable W2 design")
    a=p.parse_args()
    refroot=a.reference.expanduser().resolve()
    parent=a.output_dir.expanduser().resolve()
    require(a.workers>=1,"workers")
    require(not parent.is_relative_to(refroot) and not refroot.is_relative_to(parent),
            "W2 output must be separate from V6.2 reference root")
    roots=[x.expanduser().resolve() for x in (a.archive_root or
           [DEFAULT_REFERENCE.parents[1],base.ROOT/"production_results",base.ROOT/"local_results"])]
    inv=inventory(roots,parent)
    print(json.dumps(inv,indent=2),flush=True)
    if a.inventory_only:
        return
    require(not inv["scan_errors"] or a.inventory_resolution,"Inventory scan errors; inspect inventory")
    require(not inv["candidate_records"] or a.inventory_resolution,
            "Non-baseline tau metadata found; inspect comparability before execution")
    inv["resolution"]=a.inventory_resolution
    d=design(refroot); did=d["design_id"]; root=parent/("w2_"+did)
    parent.mkdir(parents=True,exist_ok=True)
    with (parent/".w2.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root.mkdir(exist_ok=True)
        manifest=root/"MANIFEST.json"
        if manifest.exists(): require(json.loads(manifest.read_text())==d,"Resume design mismatch")
        else: report.write_json(manifest,d)
        report.write_json(root/"ARCHIVE_INVENTORY.json",inv)
        status=dict(design_id=did,status="incomplete",scientific_runs=0)
        report.write_json(root/"EXECUTION_SUMMARY.json",status)
        subprocess.run([sys.executable,"-m","unittest","paper_b.test_v63_w2",
                        "paper_b.test_v62_mechanism_closure","-v"],cwd=base.ROOT,check=True)
        gate=[(GATE_SEED,c) for c in GRID]
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context("spawn")) as pool:
            batch(pool,root,refroot,gate,26,did,2.,"preflight")
            batch(pool,root,refroot,gate,400,did,1.,"replay")
            status.update(short_preflight=8,tau1_replays=8)
            report.write_json(root/"EXECUTION_SUMMARY.json",status)
            if a.preflight_only:
                status["status"]="preflight_pass"
                report.write_json(root/"EXECUTION_SUMMARY.json",status)
                print("PREFLIGHT PASS:",root,flush=True); return
            # Full-horizon W2 parity on the first seed before other seeds.
            batch(pool,root,refroot,gate,400,did,2.,"production")
            status["full_parity"]=8
            report.write_json(root/"EXECUTION_SUMMARY.json",status)
            batch(pool,root,refroot,[(seed,c) for seed in SEEDS[1:] for c in GRID],
                  400,did,2.,"production")
        seed_dir=root/"seed_tables"; seed_dir.mkdir(exist_ok=True)
        validations=[]; floors=0
        for seed in SEEDS:
            outputs={}
            for c in GRID:
                ref,sha=reference(refroot,seed,c)
                obj=saved(root,seed,c,400,did,2.,"production")
                require(obj is not None and obj["reference_sha256"]==sha,"Production coverage")
                check_match(obj,ref)
                outputs["tau1",c]=ref; outputs["tau2",c]=obj
                floors+=obj["validation"]["floor_events"]
                path,_,_=old.paths(root,"production",seed,c)
                validations.append(dict(seed=seed,variant=c[0],suite=c[1],level=c[2],mode=c[3],
                                        reference_sha256=sha,shard_sha256=digest(path),
                                        parity_pass=obj["parity_pass"],**obj["validation"]))
            report.write_gz(seed_dir/f"seed_{seed}.json.gz",seed_rows(seed,outputs))
        require(len(validations)==800 and sum(x["parity_pass"] is True for x in validations)==8,
                "Scientific/technical coverage")
        report.csv_gz(root/"run_validation.csv.gz",validations)
        rows=(r for seed in SEEDS for r in report.read_gz(seed_dir/f"seed_{seed}.json.gz"))
        summary=report.summarize(rows,list(SEEDS))
        require(all(r["n_expected"]==100 for r in summary),"Seed summary")
        report.csv_gz(root/"contrast_summary.csv.gz",summary)
        status.update(status="validated_scientific_grid",technical_gate_pass=True,
                      scientific_runs=800,reference_runs=800,floor_events=floors,summary_rows=len(summary))
        report.write_json(root/"EXECUTION_SUMMARY.json",status)
        files=[manifest,root/"ARCHIVE_INVENTORY.json",root/"EXECUTION_SUMMARY.json",
               root/"run_validation.csv.gz",root/"contrast_summary.csv.gz",*seed_dir.glob("seed_*.json.gz")]
        index=report.package(root,files,"w2_primary",did)
        report.write_json(root/"shareable"/"TRANSFER_INDEX.json",index)
        print("COMPLETE:",root/"shareable",flush=True)

if __name__=="__main__":
    main()

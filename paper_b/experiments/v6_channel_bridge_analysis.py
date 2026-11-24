"""Archive validation and matched peer-channel estimands (no model writes)."""
import csv
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict
import zipfile
import numpy as np

ARCHIVE_ID="b7a571b4818a"
ARCHIVE_SHA="7a1428f999b809079a13d16773de34713d474862e18a26b60187a58a1c35157d"
CHANNELS=("elite_only","extended_muted","extended_stale","extended_live")
RECEIVERS=("adaptive","frozen")
OLD_SENDERS=("null","adaptive","credibility_aware")
SEEDS=tuple(range(7501,7601))
PAIRS={"M_minus_E":(1,0),"S_minus_M":(2,1),"L_minus_S":(3,2),"L_minus_E":(3,0)}
STEPS=(1,2,3,6,11,26,51,100,101,151,200,300,400)


def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)


def key(r):
    return (int(r["seed"]),r["network_environment"],r["jammer_regime"],r["reliance_mode"])


def read_rows(z,name):
    with z.open(name) as raw:
        stream=gzip.GzipFile(fileobj=raw) if name.endswith(".gz") else raw
        with io.TextIOWrapper(stream,encoding="utf-8",newline="") as text:
            yield from csv.DictReader(text)


def load_archive(path,upstream_fp,software_versions):
    require(path.is_file(),"Missing Gate 0E-2 original shareable ZIP")
    with path.open("rb") as f:
        require(hashlib.file_digest(f,"sha256").hexdigest()==ARCHIVE_SHA,"Gate 0E-2 SHA256 mismatch")
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read("pilot_manifest.json"))
        gate=json.loads(z.read("pilot_gate.json"))
        require(manifest["design_id"]==ARCHIVE_ID and gate["design_id"]==ARCHIVE_ID
                and gate["pass"] is True and gate["observed_runs"]==2400,
                "Gate 0E-2 archive status mismatch")
        require(manifest["scientific_code_fingerprint"]==upstream_fp and
                manifest["software_versions"]==software_versions,"Archive code/software mismatch")
        require(tuple(manifest["seeds"])==SEEDS and tuple(manifest["network_environments"])==CHANNELS
                and set(manifest["sender_regimes"])==set(OLD_SENDERS)
                and manifest["initial_regimes"]==["flat"],"Archive grid mismatch")
        settings={"n_citizens":100,"horizon_T":400,"K":1,"credit":20,"epsilon":.05,
                  "surveillance_interval":5,"tau_social":1.,"comparison_rule":"delta_comparison",
                  "peer_evidence_mode":"source_posterior","initial_audit_mode":"centroid",
                  "frozen_ranking_mode":"first_audit","jammer_credibility_penalty":1.}
        require(all(manifest[k]==v for k,v in settings.items()),"Archive scientific specification mismatch")
        runs={}
        for r in read_rows(z,"runs.csv"):
            k=key(r)
            require(k not in runs and int(r["steps_run"])==400,"Invalid archived run")
            runs[k]=r
        expected={(s,c,j,r) for s in SEEDS for c in CHANNELS for j in OLD_SENDERS for r in RECEIVERS}
        require(set(runs)==expected,"Incomplete archive grid")
        # All null citizens, plus the two predeclared strategic replay cells.
        replay={(7501,"extended_live","adaptive","frozen"),
                (7501,"extended_live","credibility_aware","frozen")}
        selected={k for k in runs if k[2]=="null"}|replay
        terminal=defaultdict(dict)
        for r in read_rows(z,"terminal_beliefs.csv.gz"):
            k=key(r)
            if k in selected:
                pos=int(r["citizen_pos"])
                require(pos not in terminal[k],"Duplicate archive citizen")
                terminal[k][pos]=(int(r["citizen_group"]),float(r["terminal_mu_theta"]),float(r["terminal_sd_theta"]))
        require(set(terminal)==selected and all(len(v)==100 for v in terminal.values()),
                "Incomplete archive terminal data")
        checkpoint=defaultdict(dict)
        for r in read_rows(z,"belief_checkpoints.csv"):
            k=key(r); step=int(r["horizon_step"])
            require(step not in checkpoint[k],"Duplicate archive checkpoint")
            checkpoint[k][step]=float(r["mse_truth"])
        require(set(checkpoint)==expected and all(set(v)==set(STEPS) for v in checkpoint.values()),
                "Incomplete archive time paths")
    return {"runs":runs,"terminal":terminal,"checkpoint":checkpoint}


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)


def compare_archive(out,archive,replay=False):
    r=out["run"]
    k=key(r)
    # Fixed-biased is new: match its initial state and graph to archived null.
    old=archive["runs"][(k[0],k[1],"null",k[3])]
    for f in ("structural_fingerprint","initial_state_fingerprint"):
        require(r[f]==old[f],"Archive initial/graph fingerprint mismatch: "+f)
    if k[2]!="null" and not replay:
        return
    old=archive["runs"][k]
    require(close(r["mse_truth"],old["mse_truth"]),"Archive terminal MSE mismatch")
    actual={int(b["citizen_pos"]):(int(b["citizen_group"]),float(b["terminal_mu_theta"]),
                                  float(b["terminal_sd_theta"])) for b in out["beliefs"]}
    expected=archive["terminal"][k]
    require(set(actual)==set(expected),"Archive terminal citizen IDs mismatch")
    for pos,(g,mu,sd) in actual.items():
        eg,em,es=expected[pos]
        require(g==eg and close(mu,em) and close(sd,es),"Archive terminal belief/SD mismatch")
    got={int(b["horizon_step"]):float(b["mse_truth"]) for b in out["belief_checkpoints"]}
    require(set(got)==set(archive["checkpoint"][k]),"Archive checkpoint grid mismatch")
    require(all(close(v,archive["checkpoint"][k][s]) for s,v in got.items()),
            "Archive checkpoint MSE mismatch")


def channel_contrasts(seed,receiver,sender,step,null_mse,sender_mse,source):
    d=[b-a for a,b in zip(null_mse,sender_mse)]
    row=dict(seed=seed,reliance_mode=receiver,sender_regime=sender,horizon_step=step,source=source)
    for i,c in enumerate("EMSL"):
        row["D_"+c]=d[i]; row["null_MSE_"+c]=null_mse[i]; row["sender_MSE_"+c]=sender_mse[i]
    for label,(i,j) in PAIRS.items():
        row["D_"+label]=d[i]-d[j]
        row["sender_MSE_"+label]=sender_mse[i]-sender_mse[j]
        row["null_MSE_"+label]=null_mse[i]-null_mse[j]
    for metric in ("D","sender_MSE","null_MSE"):
        require(close(row[metric+"_M_minus_E"]+row[metric+"_S_minus_M"]+
                      row[metric+"_L_minus_S"],row[metric+"_L_minus_E"]),
                "Channel contrast telescope failed")
    return row


def contrast_rows(outputs,archive=None):
    seed=int(outputs[0]["run"]["seed"])
    paths={key(o["run"]):{int(r["horizon_step"]):float(r["mse_truth"])
                        for r in o["belief_checkpoints"]} for o in outputs}
    steps=sorted(next(iter(paths.values())))
    rows=[]
    for receiver in RECEIVERS:
        for step in steps:
            null=[paths[(seed,c,"null",receiver)][step] for c in CHANNELS]
            fixed=[paths[(seed,c,"fixed_biased",receiver)][step] for c in CHANNELS]
            rows.append(channel_contrasts(seed,receiver,"fixed_biased",step,null,fixed,"new_bridge"))
            if archive is not None:
                old_null=[archive["checkpoint"][(seed,c,"null",receiver)][step] for c in CHANNELS]
                require(all(close(a,b) for a,b in zip(null,old_null)),"Matched null mismatch")
                for sender in OLD_SENDERS[1:]:
                    old_sender=[archive["checkpoint"][(seed,c,sender,receiver)][step] for c in CHANNELS]
                    rows.append(channel_contrasts(seed,receiver,sender,step,old_null,old_sender,"archived_gate0e2"))
    return rows


def interaction_rows(rows):
    idx={(r["seed"],r["horizon_step"],r["reliance_mode"],r["sender_regime"]):r for r in rows}
    require(len(idx)==len(rows),"Duplicate channel contrast row")
    result=[]
    for k,r in idx.items():
        seed,step,receiver,sender=k
        metrics=[m for m in r if m.startswith(("D_","sender_MSE_","null_MSE_"))]
        def emit(label,other,dimension):
            result.append(dict(seed=seed,horizon_step=step,contrast=label,dimension=dimension,
                               reliance_mode=receiver,sender_regime=sender,
                               **{m:r[m]-other[m] for m in metrics}))
        if receiver=="adaptive":
            emit("adaptive_minus_frozen",idx[(seed,step,"frozen",sender)],"receiver")
        if sender!="fixed_biased" and (seed,step,receiver,"fixed_biased") in idx:
            emit(sender+"_minus_fixed_biased",idx[(seed,step,receiver,"fixed_biased")],"sender")
    return result


def summarize(rows,seeds,dimensions):
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in dimensions)].append(r)
    out=[]
    for k,block in sorted(groups.items()):
        block.sort(key=lambda r:r["seed"])
        require([r["seed"] for r in block]==list(seeds),"Incomplete paired analysis group")
        n=len(block)
        rng=np.random.default_rng(20261124)
        ix=rng.integers(0,n,size=(10000,n))
        for metric in [m for m in block[0] if m.startswith(("D_","sender_MSE_","null_MSE_"))]:
            x=np.array([r[metric] for r in block],dtype=float)
            require(np.isfinite(x).all(),"Nonfinite channel contrast")
            ci=np.quantile(x[ix].mean(axis=1),[.025,.975])
            out.append(dict(zip(dimensions,k),metric=metric,n_seeds=n,mean=float(x.mean()),
                            mcse=float(x.std(ddof=1)/math.sqrt(n)) if n>1 else None,
                            bootstrap_lo=float(ci[0]) if n>1 else None,
                            bootstrap_hi=float(ci[1]) if n>1 else None,
                            positive_fraction=float((x>0).mean()),negative_fraction=float((x<0).mean())))
    return out

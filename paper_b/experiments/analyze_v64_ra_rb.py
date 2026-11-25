"""V6.4.1 RA/RB passive extension. No model execution."""
import argparse, gzip, hashlib, json, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

C = ["prior", "expert", "jammer", "peer"]
SEEDS = np.arange(31001, 31101)
VARIANTS = ("RA", "RB")
MODES = ("adaptive", "frozen")
LEVELS = ("low", "high")
def sha(b):
    return hashlib.sha256(b).hexdigest()
def close(x, y):
    np.testing.assert_allclose(x, y, atol=1e-9, rtol=1e-10)
def run(root, handoff, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "TRANSFER_V64.json").read_text())
    assert manifest["design_id"] == "7e2e4f6724e3" and manifest["records"] == 80000
    expected = {f"{s}__{v}__H__{l}__{a}.json.gz" for s in SEEDS for v in VARIANTS for l in LEVELS for a in MODES}
    assert len(manifest["shards"]) == 800 and {x["file"] for x in manifest["shards"]} == expected
    assert all(re.fullmatch("[0-9a-f]{64}", x["sha256"]) for x in manifest["shards"])
    with zipfile.ZipFile(handoff) as z:
        source = z.read("inventory/V62_MANIFEST.json")
        assert sha(source) == manifest["source_manifest_sha256"]
        source_meta = json.loads(source)
        assert source_meta["design_id"] == manifest["design_id"] and source_meta["seeds"] == SEEDS.tolist()
        assert {(v,"H",l,a) for v in VARIANTS for l in LEVELS for a in MODES} <= {tuple(x) for x in source_meta["grid"]}
        frozen = pd.concat([pd.read_csv(gzip.open(z.open(n), "rt")) for n in z.namelist() if "/v62_sources/contrast_summary_" in n], ignore_index=True)
    files = {x["name"]:x for x in manifest["files"]}
    assert set(files) == {f"{v}_H_{a}_terminal.json.gz" for v in VARIANTS for a in MODES}
    records = []
    for v in VARIANTS:
        for a in MODES:
            name = f"{v}_H_{a}_terminal.json.gz"
            b = (root/name).read_bytes()
            assert len(b) == files[name]["bytes"] and sha(b) == files[name]["sha256"]
            d = pd.DataFrame(json.loads(gzip.decompress(b)))
            assert len(d)==20000 and d.variant.eq(v).all() and d["mode"].eq(a).all() and d.suite.eq("H").all()
            records.append(d)
    d = pd.concat(records, ignore_index=True)
    keys = ["seed","variant","mode","level","citizen_pos"]
    assert not d.duplicated(keys).any() and (d.groupby(keys[:-1]).size()==100).all()
    assert len(d.groupby(keys[:-1]))==800 and set(d.seed)==set(SEEDS)
    assert set(d.level)==set(LEVELS) and set(d.group)=={-1,1}
    assert (d.groupby(keys[:-1]+["group"]).size()==50).all() and d.floor_events.eq(0).all()
    assert all(set(g.citizen_pos)==set(range(2,102)) for _,g in d.groupby(keys[:-1]))
    for c in C:
        d["q_"+c] = d.initial_phi if c=="prior" else d.Q.map(lambda x:x[c])
        d["b_"+c] = d.B_raw.map(lambda x:x[c])
        d["s_"+c] = d.signed_mse.map(lambda x:x[c])
    q=d[["q_"+c for c in C]].to_numpy(); b=d[["b_"+c for c in C]].to_numpy()
    phi=d.phi.to_numpy(); mu=d.mu.to_numpy()
    assert np.isfinite(q).all() and np.isfinite(b).all() and (q>=0).all() and (phi>0).all()
    close(q.sum(1),phi);close(b.sum(1)/phi,mu);close(d.b_prior,d.initial_phi*d.initial_mu)
    close(d[["s_"+c for c in C]].to_numpy(),mu[:,None]*b/phi[:,None])
    close(d.W_jammer,d.q_jammer/(phi-d.initial_phi))
    B=np.random.default_rng(20261125).integers(0,100,(10000,100))
    B=np.array([np.bincount(x,minlength=100) for x in B],dtype=float)
    def stats(x):
        boot=B@x/100
        return [x.mean(),*np.quantile(boot,[.025,.975])]
    def arrays(x):
        q=x[["q_"+c for c in C]].to_numpy(); b=x[["b_"+c for c in C]].to_numpy()
        return q,q/x.phi.to_numpy()[:,None],b/x.phi.to_numpy()[:,None],np.divide(b,q,out=np.zeros_like(b),where=q>0),x.mu.to_numpy()
    def terms(L,H,shift=0):
        ql,wl,zl,ml,al=L;qh,wh,zh,mh,ah=H
        f=(al+ah)[:,None];both=(ql>0)&(qh>0)
        dz=zh-zl+shift*(wh-wl)
        content=np.where(both,(wl+wh)/2*(mh-ml),0)*f
        weight=np.where(both,(ml+mh+2*shift)/2*(wh-wl),0)*f
        entry=np.where((ql==0)&(qh>0),dz,0)*f
        exit=np.where((ql>0)&(qh==0),dz,0)*f
        close(content+weight+entry+exit,dz*f)
        close((dz*f).sum(1),ah**2-al**2)
        return {"content":content,"weight":weight,"entry":entry,"exit":exit,"entry_exit":entry+exit,"weight_entry_exit":weight+entry+exit,"midpoint_total":dz*f,"endpoint_total":ah[:,None]*zh-al[:,None]*zl}
    rows=[]; seed_rows=[]; refs=[]; comparison=[]; seed_cache={}; attrs=None
    def compare(v,a,g,kind,metric,x):
        old=frozen[(frozen.suite=="H")&(frozen.variant==v)&(frozen["mode"]==a)&(frozen.group==g)&(frozen.kind==kind)&(frozen.metric==metric)]
        assert len(old)==1,(v,a,g,kind,metric,len(old))
        old=old.iloc[0];new=stats(x)
        close(new[0],old["mean"])
        comparison.append([v,a,g,kind,metric,new[0],old["mean"],new[0]-old["mean"],new[1],new[2],old.lo,old.hi])
    for (v,a),g in d.groupby(["variant","mode"],sort=True):
        l,h=[g[g.level.eq(lev)].sort_values(["seed","citizen_pos"]).reset_index(drop=True) for lev in LEVELS]
        cols=["seed","citizen_pos","group","role","initial_mu","initial_sd","initial_phi"]
        assert l[cols].equals(h[cols])
        if attrs is None:attrs=l[cols].copy()
        else:assert attrs.equals(l[cols])
        L,H=arrays(l),arrays(h);T=terms(L,H)
        delta=H[-1]**2-L[-1]**2
        for name,mask in [("all",np.ones(len(l),bool)),("group_1",l.group.eq(1).to_numpy()),("group_-1",l.group.eq(-1).to_numpy())]:
            s=l.seed.to_numpy()[mask]
            den=np.bincount(s-31001,minlength=100)
            def sm(x):return np.bincount(s-31001,weights=x[mask],minlength=100)/den
            outm={}
            for metric,x in T.items():
                for j,c in enumerate(C+["all_sources"]):
                    y=sm(x[:,j] if j<4 else x.sum(1))
                    rows.append([v,a,name,c,metric,*stats(y)])
                    seed_rows.extend([v,a,name,c,metric,int(seed),float(value)] for seed,value in zip(SEEDS,y))
                    if c=="all_sources":outm[metric]=y
            y=sm(delta);close(y,outm["midpoint_total"]);close(y,outm["endpoint_total"])
            seed_cache[v,a,name]=outm
            for metric,x in [("mse",delta),("Q_expert",H[0][:,1]-L[0][:,1]),("Q_jammer",H[0][:,2]-L[0][:,2]),("Q_peer",H[0][:,3]-L[0][:,3]),("W_jammer",h.W_jammer.to_numpy()-l.W_jammer.to_numpy())]:
                compare(v,a,name,"high_minus_low",metric,sm(x))
            for shift in [-4,-2,2,4]:
                R=terms(L,H,shift)
                diff=R["weight_entry_exit"].sum(1)-T["weight_entry_exit"].sum(1)
                close(diff,0);close(R["content"],T["content"])
                refs.append([v,a,name,shift,float(np.max(abs(diff[mask])))])
            ct=outm["content"];wt=outm["weight_entry_exit"]
            boot=(abs(B@ct)-abs(B@wt))/100
            rows.append([v,a,name,"all_sources","abs_content_minus_abs_weight_entry_exit",abs(ct.mean())-abs(wt.mean()),*np.quantile(boot,[.025,.975])])
    interactions=[]
    for v in VARIANTS:
        for g in ["all","group_1","group_-1"]:
            A,F=[seed_cache[v,a,g] for a in MODES]
            for metric in A:
                x=A[metric]-F[metric]
                interactions.append([v,g,metric,*stats(x)])
            compare(v,"adaptive",g,"adaptive_minus_frozen","mse",A["midpoint_total"]-F["midpoint_total"])
    for name,data,cols in [
        ("RA_RB_accounting",rows,["variant","mode","group","source","metric","mean","lo","hi"]),
        ("RA_RB_seed_components",seed_rows,["variant","mode","group","source","metric","seed","value"]),
        ("RA_RB_reference",refs,["variant","mode","group","shift","max_combined_residual"]),
        ("RA_RB_interactions",interactions,["variant","group","metric","mean","lo","hi"]),
        ("RA_RB_frozen_comparison",comparison,["variant","mode","group","kind","metric","mean","frozen_mean","difference","lo","hi","frozen_lo","frozen_hi"])]:
        pd.DataFrame(data,columns=cols).to_csv(out/(name+".csv"),index=False)
    validation={"design_id":manifest["design_id"],"records":len(d),"runs":800,"shard_names_verified":800,"raw_shard_bytes_rechecked":False,"source_manifest_sha256":sha(source),"files":manifest["files"],"frozen_point_comparisons":len(comparison),"max_frozen_point_residual":max(abs(x[7]) for x in comparison),"max_reference_residual":max(x[-1] for x in refs),"bootstrap_seed":20261125,"bootstrap_draws":10000,"new_simulation_runs":0}
    (out/"RA_RB_validation.json").write_text(json.dumps(validation,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--handoff",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.root,a.handoff,a.out)

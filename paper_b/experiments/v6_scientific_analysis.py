"""Predeclared matched-seed contrasts. No outcome-dependent filtering."""
from collections import defaultdict
import math
import statistics
import numpy as np

METRICS = ("mse_truth", "mean_belief", "squared_bias", "belief_variance",
           "W_jammer", "Q_jammer", "I_jammer", "W_same_peer", "Q_same_peer",
           "W_expert", "Q_expert", "W_other_peer", "Q_other_peer")


def seed_contrasts(outputs):
    index = {}
    for out in outputs:
        for row in out["group_terminal"]:
            key = (row["cell"], row["sender_regime"], row["reliance_mode"],
                   row["group_type"], row["group"])
            if key in index:
                raise ValueError(f"Duplicate terminal group: {key}")
            index[key] = row
    first = outputs[0]["run"]
    suite, seed = first["suite"], first["seed"]
    rows = []

    def emit(name, prior, sender, receiver, group_type, group, terms):
        vals = [index[(cell,s,r,group_type,group)] for cell,s,r,weight in terms]
        row = dict(seed=seed, suite=suite, contrast=name, prior=prior,
                   sender_regime=sender, reliance_mode=receiver,
                   group_type=group_type, group=group)
        for metric in METRICS:
            row["delta_"+metric] = sum(weight*float(value[metric])
                                        for value,(_,_,_,weight) in zip(vals,terms))
        if name=="high_minus_low":
            row["divergence_W"] = int(row["delta_W_jammer"]<0 and row["delta_mse_truth"]>0)
            row["divergence_Q"] = int(row["delta_Q_jammer"]<0 and row["delta_mse_truth"]>0)
        rows.append(row)

    for receiver in ("adaptive","frozen"):
        priors = ("low","high") if suite=="H" else ("flat",)
        for prior in priors:
            low = f"H_low_S_{prior}" if suite=="H" else "G_low"
            high = f"H_high_S_{prior}" if suite=="H" else "G_high"
            groups = sorted((k[3],k[4]) for k in index if k[:3]==(low,"null",receiver))
            for kind,group in groups:
                for sender in ("null","fixed_biased"):
                    emit("high_minus_low",prior,sender,receiver,kind,group,
                         [(high,sender,receiver,1),(low,sender,receiver,-1)])
                emit("damage_high_minus_low",prior,"fixed_biased",receiver,kind,group,
                     [(high,"fixed_biased",receiver,1),(high,"null",receiver,-1),
                      (low,"fixed_biased",receiver,-1),(low,"null",receiver,1)])
        if suite=="H":
            groups = sorted((k[3],k[4]) for k in index if k[:3]==("H_low_S_low","null",receiver))
            for kind,group in groups:
                for sender in ("null","fixed_biased"):
                    emit("H_by_S", "interaction",sender,receiver,kind,group,
                         [("H_high_S_high",sender,receiver,1),("H_low_S_high",sender,receiver,-1),
                          ("H_high_S_low",sender,receiver,-1),("H_low_S_low",sender,receiver,1)])
    return rows


def summarize(rows, expected_seeds):
    groups = defaultdict(list)
    dimensions = ("suite","contrast","prior","sender_regime","reliance_mode","group_type","group")
    for r in rows:
        groups[tuple(r[k] for k in dimensions)].append(r)
    results = []
    for key,block in sorted(groups.items()):
        block.sort(key=lambda r:int(r["seed"]))
        if [int(r["seed"]) for r in block] != sorted(expected_seeds):
            raise ValueError(f"Incomplete matched contrast: {key}")
        # Same whole-seed resampling indices for all metrics in a contrast.
        rng = np.random.default_rng(20261124)
        n = len(block)
        idx = rng.integers(0,n,size=(10000,n))
        for metric in [k for k in block[0] if k.startswith("delta_") or k.startswith("divergence_")]:
            v = np.asarray([float(r[metric]) for r in block])
            if not np.isfinite(v).all():
                raise ValueError(f"Nonfinite matched contrast {key}/{metric}")
            ci = np.quantile(v[idx].mean(axis=1),[.025,.975])
            results.append(dict(zip(dimensions,key),metric=metric,n_seeds=n,
                                mean=float(v.mean()),mcse=float(v.std(ddof=1)/math.sqrt(n)) if n>1 else None,
                                bootstrap_lo=float(ci[0]) if n>1 else None,
                                bootstrap_hi=float(ci[1]) if n>1 else None,
                                positive_fraction=float((v>0).mean()),
                                negative_fraction=float((v<0).mean())))
    return results

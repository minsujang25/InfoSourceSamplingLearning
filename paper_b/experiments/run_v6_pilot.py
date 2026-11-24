"""V6 technical pilot, not a production runner. Execute from repo root."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from paper_b.experiments.run_matched_pilot import (
    canonical_hash, code_fingerprint, run_condition, software_versions,
)
from paper_b.experiments.run_local_diagnostic import base_model_config
from paper_b.structural_designs import (
    balanced_fixed_group_ids, exp4_initial_beliefs,
    exp4_homophily_source_maps, exp3_redundancy_source_maps,
)
import model.InfoSourceSamplingLearning as model_module

ROOT = Path(__file__).resolve().parents[2]
SENDERS = ("null", "fixed_biased")
RECEIVERS = ("adaptive", "frozen")
CHANNELS = ("elite_only", "extended_muted", "extended_stale", "extended_live")
SUITES = ("H", "G", "C")
N = 100
T = 26
CREDIT = 20
SEEDS = (8001, 8002)
KEYS = ("beliefs", "lambda_checkpoints", "belief_checkpoints",
        "evidence_checkpoints", "expert_rank_checkpoints", "channel_checkpoints",
        "channel_group_checkpoints", "channel_citizens", "rank_unification_checkpoints",
        "jammer_strategy", "sender_events", "edges")
CLASSES = ("expert", "same_peer", "other_peer", "jammer")


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def fingerprint():
    paths = ["paper_b/measurement.py", "paper_b/structural_designs.py",
             "paper_b/validation.py", "paper_b/experiments/run_v6_pilot.py",
             "paper_b/V6_MODEL_SPECIFICATION.md", "paper_b/V6_MEASUREMENT_PROTOCOL.md",
             "paper_b/V6_EXPERIMENT_MATRIX.md"]
    return canonical_hash({"upstream": code_fingerprint(),
                           "extra": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                     for p in paths}})


def cells(suite):
    if suite == "H":
        return [(f"H_{h}_S_{s}", h, s, sender, r)
                for h in ("low", "high") for s in ("low", "high")
                for sender in SENDERS for r in RECEIVERS]
    if suite == "G":
        return [(f"G_{g}", g, "flat", sender, r)
                for g in ("low", "high") for sender in SENDERS for r in RECEIVERS]
    if suite == "C":
        return [(env, env, "flat", sender, r)
                for env in CHANNELS
                for sender in (*SENDERS, "adaptive", "credibility_aware")
                for r in RECEIVERS]
    raise ValueError(suite)


def config_for(seed, suite, level, prior, horizon):
    cfg = base_model_config(regime="flat", seed=seed, n_citizens=N,
                            max_steps=horizon, k=1, epsilon=.05, credit=CREDIT,
                            comparison_rule="delta_comparison", surveillance_interval=5)
    gateways = None
    if suite == "H":
        groups = balanced_fixed_group_ids(seed=seed, n_citizens=N)
        maps = exp4_homophily_source_maps(seed=seed, n_citizens=N, group_ids=groups,
                                        peer_degree=2, low_homophily=.50, high_homophily=.90)
        cfg.update(mu_theta=exp4_initial_beliefs(seed=seed, n_citizens=N, group_ids=groups,
                                               segregation=prior, high_group_shift=3.,
                                               residual_sd=1.),
                   initial_theta_type=f"production_exp4_{prior}",
                   fixed_group_ids=groups, structural_source_map=maps[level])
        env = f"production_exp4_h_{level}"
        regime = f"segregation_{prior}"
    elif suite == "G":
        maps = exp3_redundancy_source_maps(seed=seed, n_citizens=N, peer_degree=2,
                                         expert_access_share=.10)
        cfg["structural_source_map"] = maps[level]
        gateways = set(maps["expert_gateways"])
        env, regime = f"production_exp3_r_{level}", "flat"
    else:
        env, regime = level, "flat"
    return cfg, env, regime, gateways


def run_one(seed, suite, cell, horizon, design_id, instrument=True):
    label, level, prior, sender, reliance = cell
    cfg, env, regime, gateways = config_for(seed, suite, level, prior, horizon)
    run_id = f"{suite}__{seed}__{label}__{sender}__{reliance}"
    out = run_condition(
        base_config=cfg, seed=seed, regime=regime, environment=env,
        reliance_mode=reliance, jammer_active=sender in ("adaptive", "credibility_aware"),
        jammer_regime=sender, k=1, design_id=design_id, block_id=run_id,
        save_edge_log=instrument, peer_evidence_mode="source_posterior", tau_social=1.,
        jammer_credibility_penalty=1., frozen_ranking_mode="first_audit",
        initial_audit_mode="centroid", gateway_positions=gateways,
        record_expert_rank_checkpoints=instrument,
        record_channel_decomposition=instrument,
        record_rank_unification_checkpoints=instrument,
        channel_decomposition_steps=tuple(sorted({1, 2, 3, 6, 11, 26, 51, 100, 151, 200, 300, horizon})),
        sender_event_period_limit=2 if instrument else 0,
    )
    labels = {"suite": suite, "cell": label, "level": level, "prior": prior,
              "sender_regime": sender, "run_id": run_id}
    out["run"].update(labels)
    for key in KEYS:
        for row in out[key]:
            row.update(labels)
    return out, gateways


def group_outputs(out, gateways):
    """Terminal citizen summaries; group/role conditioning is explicit."""
    b = {int(r["citizen_pos"]): r for r in out["beliefs"]}
    q = {int(r["citizen_pos"]): r for r in out["channel_citizens"]}
    require(len(b) == N and set(b) == set(q), "Missing terminal citizen match")
    groups = defaultdict(list)
    for pos, row in b.items():
        groups[("all", "all")].append(pos)
        groups[("citizen_group", str(row["citizen_group"]))].append(pos)
        if gateways is not None:
            groups[("gateway_role", "gateway" if pos in gateways else "non_gateway")].append(pos)
    common = {k: out["run"][k] for k in ("run_id", "suite", "cell", "seed",
              "sender_regime", "reliance_mode", "prior")}
    rows = []
    for (kind, group), ids in groups.items():
        x = [float(b[i]["terminal_mu_theta"]) for i in ids]
        mean = statistics.mean(x)
        var = statistics.pvariance(x)
        mse = statistics.mean(v*v for v in x)
        require(math.isclose(mse, mean*mean+var, rel_tol=1e-10, abs_tol=1e-10),
                "MSE decomposition mismatch")
        row = dict(common, group_type=kind, group=group, n_citizens=len(ids),
                   mean_belief=mean, squared_bias=mean*mean, belief_variance=var, mse_truth=mse)
        for prefix in ("Q", "W", "I"):
            for cls in CLASSES:
                key = f"{prefix}_{cls}"
                row[key] = statistics.mean(float(q[i][key]) for i in ids)
        rows.append(row)
    return rows


def edge_groups(out, gateways):
    """Group mean of citizen-level acquisition masses and inclusion probabilities."""
    by_ego = defaultdict(list)
    for row in out["edges"]:
        by_ego[(int(row["period"]), int(row["ego"]))].append(row)
    groups = defaultdict(list)
    for (period, ego), edges in by_ego.items():
        e = edges[0]
        require(len({r["source"] for r in edges}) == len(edges), "Duplicate edge")
        require(math.isclose(sum(float(r["expected_reliance"]) for r in edges), 1., abs_tol=1e-9),
                "Expected acquisition not normalized")
        values = {}
        for cls in (*CLASSES, "gateway_peer"):
            def match(r):
                st = r["source_type"]
                if cls == "expert":
                    return st == "infoprovider"
                if cls == "jammer":
                    return st == "disruptivejammer"
                if cls == "gateway_peer":
                    return st == "citizen" and gateways is not None and int(r["source"]) in gateways
                return st == "citizen" and ((r["source_group"] == r["ego_group"]) == (cls == "same_peer"))
            p = sum(float(r["expected_reliance"]) for r in edges if match(r))
            values[f"lambda_{cls}"] = p
            values[f"inclusion_{cls}"] = 1.-(1.-p)**CREDIT
            values[f"realized_{cls}"] = sum(float(r["realized_reliance"]) for r in edges if match(r))
        categories = [("all", "all"), ("citizen_group", str(e["ego_group"]))]
        if gateways is not None:
            categories.append(("gateway_role", "gateway" if ego in gateways else "non_gateway"))
        for kind, group in categories:
            groups[(period, kind, group, "all")].append(values)
            groups[(period, kind, group, str(e["learning_target"]))].append(values)
    common = {k: out["run"][k] for k in ("run_id", "suite", "cell", "seed",
              "sender_regime", "reliance_mode", "prior")}
    return [dict(common, period=p, horizon_step=p+1, group_type=k, group=g,
                 learning_target=t, n_citizens=len(v),
                 **{name: statistics.mean(row[name] for row in v) for name in v[0]})
            for (p, k, g, t), v in sorted(groups.items())]


def check_run(out, horizon):
    r = out["run"]
    require(int(r["steps_run"]) == horizon and math.isfinite(float(r["mse_truth"])), "Horizon/MSE")
    require(r["initial_audit_mode"] == "centroid" and r["frozen_ranking_mode"] == "first_audit"
            and r["peer_evidence_mode"] == "source_posterior" and float(r["tau_social"]) == 1.
            and r["jammer_regime"] == r["sender_regime"], "Unexpected scientific setting")
    require(len(out["beliefs"]) == N and len(out["sender_events"]) == 2*N, "Incomplete citizen rows")
    require(all(math.isfinite(float(b["terminal_mu_theta"])) and
                math.isfinite(float(b["terminal_sd_theta"])) and float(b["terminal_sd_theta"]) > 0
                for b in out["beliefs"]), "Nonfinite belief")
    for row in out["channel_citizens"]:
        q = [float(row[f"Q_{cls}"]) for cls in CLASSES]
        w = [float(row[f"W_{cls}"]) for cls in CLASSES]
        require(all(math.isfinite(x) and x >= 0 for x in q+w), "Invalid Q/W")
        require(math.isclose(sum(w), float(sum(q)>0), abs_tol=1e-9), "W normalization")
        if r["sender_regime"] == "null":
            require(q[3] == 0., "Null substantive precision")
        if r["cell"] == "extended_muted":
            require(q[1] == q[2] == 0., "Muted substantive precision")
    for row in out["sender_events"]:
        if int(row["period"]) == 0:
            require(row["learning_target"] == "audit" and
                    math.isclose(float(row["theta_mu_pre"]), float(row["theta_mu_post"]), abs_tol=1e-12, rel_tol=0)
                    and math.isclose(float(row["theta_sd_pre"]), float(row["theta_sd_post"]), abs_tol=1e-12, rel_tol=0),
                    "Initial audit changed substantive belief")
        if r["cell"] == "extended_muted" and int(row["period"]) >= 1:
            require(int(row["peer_delivered_count"]) == 0, "Muted delivery")
    if r["reliance_mode"] == "frozen":
        by_edge = defaultdict(list)
        for edge in out["edges"]:
            if int(edge["period"]) >= 1:
                by_edge[(edge["ego"], edge["source"])].append(float(edge["expected_reliance"]))
        require(bool(by_edge), "No frozen edge audit")
        require(all(max(v)-min(v) <= 1e-12 for v in by_edge.values()), "Frozen policy changed")


def check_block(outputs, suite):
    by_cell = defaultdict(list)
    by_prior = defaultdict(set)
    for out in outputs:
        r = out["run"]
        by_cell[r["cell"]].append(r)
        by_prior[r["prior"]].add(r["initial_state_fingerprint"])
    require(all(len(v)==1 for v in by_prior.values()), "Initial state mismatch within prior")
    for rows in by_cell.values():
        require(len({r["structural_fingerprint"] for r in rows})==1, "Graph mismatch within cell")
    if suite == "C":
        extended = [o for o in outputs if o["run"]["cell"] != "elite_only"]
        require(len({o["run"]["structural_fingerprint"] for o in extended}) == 1, "Extended graph mismatch")
        sig = defaultdict(dict)
        for o in extended:
            r = o["run"]
            for e in o["sender_events"]:
                if int(e["period"]) == 0:
                    key = (r["sender_regime"], r["reliance_mode"], int(e["citizen_pos"]))
                    sig[key][r["cell"]] = tuple(e[k] for k in (
                        "learning_target", "peer_audit_signature", "peer_initial_means",
                        "peer_outgoing_sds", "source_count", "peer_request_count",
                        "peer_delivered_count", "theta_mu_pre", "theta_mu_post", "jammer_delta_mu_post"))
        require(len(sig)==len(RECEIVERS)*4*N, "Missing matched audit signatures")
        require(all(len(v)==3 and len(set(v.values()))==1 for v in sig.values()),
                "M/S/L initial audit mismatch")


def worker(seed, suite, horizon, design_id, root):
    model_module.MIN_SD = 1e-8
    model_module.MIN_VAR = 1e-16
    outputs = []
    grid = cells(suite)
    for cell in grid:
        out, gateways = run_one(seed, suite, cell, horizon, design_id)
        check_run(out, horizon)
        out["group_terminal"] = group_outputs(out, gateways)
        out["group_acquisition"] = edge_groups(out, gateways)
        require(math.isclose(out["group_terminal"][0]["mse_truth"], float(out["run"]["mse_truth"]),
                             rel_tol=1e-9, abs_tol=1e-9), "Population MSE mismatch")
        outputs.append(out)
    check_block(outputs, suite)
    # One deliberately selected representative per suite/seed, NOT chosen by outcomes.
    representative = {"H": ("H_high_S_high", "high", "high", "fixed_biased", "adaptive"),
                      "G": ("G_high", "high", "flat", "fixed_biased", "frozen"),
                      "C": ("extended_live", "extended_live", "flat", "credibility_aware", "adaptive")}[suite]
    baseline = next(o for o in outputs if o["run"]["cell"]==representative[0]
                    and o["run"]["sender_regime"]==representative[3]
                    and o["run"]["reliance_mode"]==representative[4])
    plain, _ = run_one(seed, suite, representative, horizon, design_id, instrument=False)
    # Canonical serialization compares NaN sentinels consistently (NaN != NaN).
    for key in ("beliefs", "belief_checkpoints", "lambda_checkpoints", "jammer_strategy"):
        require(canonical_hash(baseline[key]) == canonical_hash(plain[key]),
                f"Logger parity failed: {key}")
    path = Path(root)/"shards"/f"{suite}__{seed}.json.gz"
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump({"design_id": design_id, "suite": suite, "seed": seed, "parity_pass": True,
                   "outputs": outputs}, f, allow_nan=True)
    os.replace(tmp, path)
    return str(path)


def load_shard(path, design_id, suite, seed):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        shard = json.load(f)
    require((shard["design_id"], shard["suite"], shard["seed"], shard["parity_pass"])
            == (design_id, suite, seed, True), "Shard manifest mismatch")
    outputs = shard["outputs"]
    expected = {(c[0], c[3], c[4]) for c in cells(suite)}
    actual = {(o["run"]["cell"], o["run"]["sender_regime"], o["run"]["reliance_mode"]) for o in outputs}
    require(actual==expected and len(outputs)==len(expected), "Incomplete or duplicate shard cells")
    for o in outputs:
        check_run(o, T)
    check_block(outputs, suite)
    return outputs


def csv_gz(path, rows):
    fields = sorted({k for r in rows for k in r})
    if not fields:
        return
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("technical", "scientific"), default="technical")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(args.workers >= 1, "workers must be positive")
    if args.phase == "scientific":
        require(args.dry_run, "Scientific execution locked until technical output review and production measurement freeze")
        print(json.dumps({"status": "candidate only", "seeds": list(range(6001,6021)),
                          "horizon": 400, "suites": ["H","G"], "grid_runs": 480}, indent=2))
        return
    design = {"phase": "technical", "seeds": SEEDS, "horizon": T, "N": N,
              "cells": {s: cells(s) for s in SUITES}, "credit": CREDIT, "epsilon": .05,
              "K": 1, "surveillance": 5, "tau_social": 1., "initial_audit": "centroid",
              "frozen_ranking": "first_audit", "peer_evidence": "source_posterior",
              "event_period_limit": 2, "grid_runs": 112, "parity_runs": 6,
              "upstream_fingerprint": code_fingerprint(), "v6_fingerprint": fingerprint(),
              "software_versions": software_versions()}
    design_id = canonical_hash(design)[:12]
    design["design_id"] = design_id
    root = Path(args.output_dir).expanduser().resolve()/f"v6_technical_{design_id}"
    print(json.dumps(design, indent=2), flush=True)
    print(f"Output: {root}", flush=True)
    if args.dry_run:
        return
    require(not root.exists() or args.resume, "Output exists; use identical command plus --resume")
    root.mkdir(parents=True, exist_ok=True)
    (root/"shards").mkdir(exist_ok=True)
    manifest = root/"v6_manifest.json"
    if manifest.exists():
        require(json.loads(manifest.read_text()) == json.loads(json.dumps(design)), "Manifest mismatch")
    manifest.write_text(json.dumps(design, indent=2), encoding="utf-8")
    gate_path = root/"v6_gate.json"
    gate_path.write_text(json.dumps({"pass": False, "status": "incomplete", "design_id": design_id}), encoding="utf-8")
    tasks = []
    for seed in SEEDS:
        for suite in SUITES:
            path = root/"shards"/f"{suite}__{seed}.json.gz"
            if args.resume and path.exists():
                load_shard(path, design_id, suite, seed)
            else:
                tasks.append((seed, suite, T, design_id, str(root)))
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(worker, *task) for task in tasks]
            for i, f in enumerate(as_completed(futures), 1):
                print(f"[{i}/{len(futures)}] {f.result()}", flush=True)
    outputs = []
    for seed in SEEDS:
        for suite in SUITES:
            outputs.extend(load_shard(root/"shards"/f"{suite}__{seed}.json.gz", design_id, suite, seed))
    require(len(outputs)==112 and len({o["run"]["run_id"] for o in outputs})==112, "Grid incomplete")
    csv_gz(root/"runs.csv.gz", [o["run"] for o in outputs])
    for key in (*KEYS, "group_terminal", "group_acquisition"):
        if key != "edges":
            csv_gz(root/f"{key}.csv.gz", [r for o in outputs for r in o[key]])
    gate = {"pass": True, "status": "technical_only", "design_id": design_id,
            "grid_runs": 112, "parity_runs": 6, "blocks": 6,
            "matched_t0_audits_equal_MSL": True, "frozen_checkpoint_policy_constant": True,
            "full_horizon": T, "production_authorized": False}
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    bundle = root/f"paper_b_v6_technical_{design_id}_shareable.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.glob("*.json")) + sorted(root.glob("*.csv.gz")):
            z.write(p, p.name)
        for p in sorted((root/"shards").glob("*.json.gz")):
            z.write(p, "shards/"+p.name)
    print(json.dumps(gate, indent=2), flush=True)
    print(f"Shareable ZIP: {bundle}", flush=True)


if __name__ == "__main__":
    main()

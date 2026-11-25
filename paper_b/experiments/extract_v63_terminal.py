"""Read committed V6.2 shards only; no simulation or edits to source/results.

Run from repository root:
    python -m paper_b.experiments.extract_v63_terminal [--root PATH] [--out PATH] [--include-frozen-G]
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess

DESIGN_ID = "7e2e4f6724e3"
SOURCE_COMMIT = "55264824e35c02284174e206d86955269d2b0bbc"
SEEDS = list(range(31001, 31101))
FIELDS = (
    "citizen_pos", "group", "role", "initial_mu", "initial_sd", "initial_phi",
    "initial_rank", "initial_latent_rank", "initial_policy", "same_peer_count",
    "gateway_peer_count", "peer_ids", "Q", "B_raw", "B_adjusted", "phi",
    "mu", "sd", "W_jammer", "floor_events", "audit_count", "theta_count",
    "rank", "latent_rank", "first_drift", "first_strict_drift", "first_direct",
    "promotion", "promotion_pre_mu", "promotion_pre_sd", "promotion_jammer_Q",
    "promotion_drift", "promotion_strict_drift", "postpromo_opportunities",
    "postpromo_first_uptake", "postpromo_Q", "postpromo_last_mu", "sequence",
    "strict_sequence", "censored", "signed", "signed_mse",
)
RUN_FIELDS = (
    "initial_state_fingerprint", "structural_fingerprint", "mse_truth",
    "mean_belief", "Q_jammer_channel", "Q_expert_channel", "W_jammer_channel",
    "W_peer_incoming_hhi", "W_peer_incoming_top5_share",
)
AUDIT_EQUAL_FIELDS = (
    "requested", "delivered", "message_mean", "sample_mean_variance",
    "delta_post", "delta_sd_post", "rank_post",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(str(message))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.home() /
                   "Research/paperb_v6_results/v62_mechanism_closure/v62_7e2e4f6724e3")
    p.add_argument("--out", type=Path)
    p.add_argument("--include-frozen-G", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "v63_passive_transfer").resolve()
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["design_id"] == DESIGN_ID and manifest["seeds"] == SEEDS
            and manifest["horizon"] == 400 and manifest["expected_runs"] == 3400,
            "Unexpected source manifest")
    # Refuse to overwrite committed simulation results or an existing transfer.
    require(out != root and not out.is_relative_to(root / "production"),
            "Output must not be a production directory")
    require(not out.exists() or not any(out.iterdir()),
            f"Output directory is not empty: {out}. Use --out with a fresh directory.")
    wanted = [("baseline", "G", "adaptive"), ("neutral", "H", "adaptive"),
              ("neutral", "G", "adaptive"), ("uniform_tail", "H", "adaptive")]
    if args.include_frozen_G:
        wanted.append(("baseline", "G", "frozen"))
    # Existing baseline H terminal records are already available. Reopen H
    # shards only for external first-audit evidence and run fingerprints.
    all_specs = wanted + [("baseline", "H", "adaptive"), ("baseline", "H", "frozen")]
    grid = {tuple(x) for x in manifest["grid"]}
    inventory, audit, runs, pair_checks, files = [], [], [], [], []
    out.mkdir(parents=True, exist_ok=True)
    for variant, suite, mode in all_specs:
        records = []
        for seed in SEEDS:
            lowhigh = {}
            for level in ("low", "high"):
                cell = (variant, suite, level, mode)
                require(cell in grid, f"Missing manifest cell: {cell}")
                stem = "__".join([str(seed), *cell])
                path = root / "production" / (stem + ".json.gz")
                sha = root / "production" / (stem + ".sha256")
                checksum = digest(path)
                require(sha.read_text().strip() == checksum, f"Checksum mismatch: {path}")
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    d = json.load(f)
                require((d["design_id"], d["phase"], d["seed"], d["variant"],
                         d["suite"], d["level"], d["mode"], d["horizon"]) ==
                        (DESIGN_ID, "production", seed, *cell, 400), f"Identity: {path}")
                require(d["validation"]["periods"] == 40000
                        and d["validation"]["floor_events"] == 0
                        and len(d["terminal"]) == 100, f"Coverage/floor: {path}")
                meta = {k: d[k] for k in ("seed", "variant", "suite", "level", "mode")}
                r = {x["citizen_pos"]: x for x in d["terminal"]}
                require(set(r) == set(range(2, 102)) and
                        all(x["floor_events"] == 0 for x in r.values()), f"Citizen IDs/floor: {path}")
                for x in r.values():
                    require(all(k in x for k in FIELDS), f"Missing terminal field: {path}")
                    require(close(sum(x["Q"].values()) + x["initial_phi"], x["phi"])
                            and close(sum(x["B_raw"].values()) / x["phi"], x["mu"])
                            and close(sum(x["signed_mse"].values()), x["mu"]**2),
                            f"Terminal accounting: {path}")
                    if (variant, suite, mode) in wanted:
                        records.append({**meta, **{k: x[k] for k in FIELDS},
                                        "terminal_squared_error": x["mu"]**2})
                rr = d["run"]
                runs.append({**meta, **{k: rr[k] for k in RUN_FIELDS}, **d["validation"]})
                inventory.append({**meta, "sha256": checksum, "file": path.name})
                ext = {(x["citizen_pos"], x["source_pos"]): x for x in d["initial_audit"]
                       if x["source_pos"] in (0, 1)}
                require(len(ext) == (200 if suite == "H" else 110), f"Audit coverage: {path}")
                audit.extend({**meta, **x} for x in ext.values())
                lowhigh[level] = (r, rr, ext)
            l, L, le = lowhigh["low"]
            h, H, he = lowhigh["high"]
            require(L["initial_state_fingerprint"] == H["initial_state_fingerprint"]
                    and set(le) == set(he), f"Initial pairing: {seed}, {cell}")
            for i in l:
                require(all(l[i][k] == h[i][k] for k in
                            ("group", "role", "initial_mu", "initial_sd", "initial_phi")),
                        f"Initial state mismatch: {seed}, {cell}, {i}")
            if variant == "neutral":
                require(all(l[i]["Q"][k] == h[i]["Q"][k] for i in l
                            for k in ("expert", "jammer")), f"Neutral external Q: {seed}, {cell}")
            eq = {k: sum(le[i][k] == he[i][k] for i in le) for k in AUDIT_EQUAL_FIELDS}
            pair_checks.append(dict(seed=seed, variant=variant, suite=suite, mode=mode,
                                    n_external=len(le), **{f"equal_{k}": v for k, v in eq.items()}))
        if records:
            require(len(records) == 20000, "Incomplete terminal extract")
            path = out / f"{variant}_{suite}_{mode}_terminal.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(records, f, allow_nan=False, separators=(",", ":"))
            files.append(path)
        print(f"Validated {variant}/{suite}/{mode}: 200 shards", flush=True)

    def csvgz(name, rows):
        path = out / name
        with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        files.append(path)

    require(len(inventory) == len(all_specs)*200 and
            len({x["file"] for x in inventory}) == len(inventory), "Shard inventory coverage")
    csvgz("run_metadata.csv.gz", runs)
    csvgz("first_audit_external.csv.gz", audit)
    csvgz("first_audit_pair_checks.csv.gz", pair_checks)
    csvgz("shard_inventory.csv.gz", inventory)
    try:
        extractor_commit = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        extractor_commit = None
    index = {
        "design_id": DESIGN_ID,
        "source_commit": SOURCE_COMMIT,
        "extractor_commit": extractor_commit,
        "extractor_sha256": digest(Path(__file__)),
        "required_terminal_records": 80000,
        "optional_frozen_G": args.include_frozen_G,
        "source_manifest_sha256": digest(manifest_path),
        "files": [{"name": f.name, "bytes": f.stat().st_size, "sha256": digest(f)} for f in files],
    }
    (out / "TRANSFER_V63.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))
    print("Upload folder:", out)


if __name__ == "__main__":
    main()

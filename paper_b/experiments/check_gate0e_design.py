"""Preflight-only checker for the prospective Gate 0E + Gate 1 matrix.

This script NEVER imports Mesa, runs simulations, or writes production
results. It checks the design freeze before any model intervention is coded.

    python -m paper_b.experiments.check_gate0e_design
"""
from __future__ import annotations

import json
from pathlib import Path


PLAN = Path(__file__).resolve().parents[1] / "GATE0E_GATES_MATRIX_2026-11-23.json"


def check_design(data: dict) -> dict:
    if data.get("status") != "design_only_not_yet_implemented_or_run":
        raise ValueError("Gate 0E design must not be mislabeled as an executed experiment.")
    invariant = data["invariants"]
    if invariant["audit_probability"] != "1/(t+1)":
        raise ValueError("Do not modify the citizen audit schedule.")
    if invariant["initial_audit_mode"] != "centroid":
        raise ValueError("Do not modify the non-disruptive initial audit.")
    if invariant["peer_evidence_mode"] != "source_posterior":
        raise ValueError("Peer evidence must be source-level posterior.")
    if float(invariant["tau_social"]) != 1.0:
        raise ValueError("Expected canonical tau_social=1.")
    if invariant["network_treatments"] != [
        "elite_only", "extended_muted", "extended_stale", "extended_live"
    ]:
        raise ValueError("Network treatment list or order drifted.")
    channel = data["channel_contract"]
    if channel["intervention_start_period"] != 1 or channel["matched_audit_period"] != 0:
        raise ValueError("Channel intervention must begin after t0 audit.")
    if not all(channel[k] for k in (
        "mute_consumes_draws_before_discard",
        "fixed_peer_initial_mean_and_sd",
        "stale_likelihood_uses_delivered_sd",
        "preserve_source_count_and_budget_in_extended",
        "live_matches_existing_extended_mode",
    )):
        raise ValueError("Channel intervention implementation contracts incomplete.")
    phase_runs = {}
    all_seeds = set()
    for phase in data["phases"]:
        if "seeds" in phase:
            seeds = [int(s) for s in phase["seeds"]]
        else:
            start, end = phase["seed_range_inclusive"]
            seeds = list(range(int(start), int(end) + 1))
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError(f"{phase['id']}: duplicated/empty seed set")
        overlap = all_seeds.intersection(seeds)
        if overlap:
            raise ValueError(f"{phase['id']}: seed overlap {sorted(overlap)[:5]}")
        all_seeds.update(seeds)
        if int(phase["N"]) < 4 or int(phase["T"]) < 2:
            raise ValueError(f"{phase['id']}: invalid N or T")
        if any(len(phase[key]) != len(set(phase[key]))
               for key in ("networks", "senders", "receivers", "prior_levels")):
            raise ValueError(f"{phase['id']}: duplicate treatment")
        if "null" not in phase["senders"]:
            raise ValueError(f"{phase['id']}: must include null baseline")
        count = (
            len(seeds) * len(phase["prior_levels"]) *
            len(phase["networks"]) * len(phase["senders"]) *
            len(phase["receivers"])
        )
        if count != int(phase["expected_runs"]):
            raise ValueError(f"{phase['id']}: expected_runs {phase['expected_runs']} "
                             f"but matrix contains {count}")
        phase_runs[phase["id"]] = {
            "n_seed_blocks": len(seeds),
            "n_conditions_per_seed": count // len(seeds),
            "n_runs": count,
            "N": phase["N"], "T": phase["T"],
        }
    if phase_runs != {
        "gate0e_smoke": {
            "n_seed_blocks": 2, "n_conditions_per_seed": 24, "n_runs": 48,
            "N": 24, "T": 6,
        },
        "gate0e_mechanism": {
            "n_seed_blocks": 20, "n_conditions_per_seed": 24, "n_runs": 480,
            "N": 100, "T": 200,
        },
        "gate0e_stability": {
            "n_seed_blocks": 100, "n_conditions_per_seed": 24, "n_runs": 2400,
            "N": 100, "T": 400,
        },
        "gate1_original_confirmatory": {
            "n_seed_blocks": 500, "n_conditions_per_seed": 24, "n_runs": 12000,
            "N": 100, "T": 400,
        },
    }:
        raise ValueError("Design conditions drifted from the frozen matrix.")
    gates = data["data_firewall"]
    if not all(gates.values()):
        raise ValueError("All manuscript and prior-production firewalls must be on.")
    return {
        "pass": True,
        "status": data["status"],
        "n_phases": len(phase_runs),
        "n_planned_runs_across_all_phases": sum(
            phase["n_runs"] for phase in phase_runs.values()
        ),
        "phase_summary": phase_runs,
        "warning": "This validates only the proposed experiment matrix, not ABM code, simulated data, or scientific conclusions.",
    }


def main() -> None:
    design = json.loads(PLAN.read_text(encoding="utf-8"))
    print(json.dumps(check_design(design), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

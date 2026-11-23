"""Local macOS/Linux entry point for frozen Gate 0E experiment phases.

Smoke and 20-seed mechanism phases are executable only after the Gate 0E
GitHub Actions smoke succeeds. This wrapper reads the frozen matrix and passes
all scientific options explicitly to the matched runner.

Examples:
    python -m paper_b.experiments.run_gate0e_local --phase smoke --workers 2
    python -m paper_b.experiments.run_gate0e_local --phase mechanism \\
        --workers 4 --output-dir "$HOME/Research/paperb_gate0e_results"
    python -m paper_b.experiments.run_gate0e_local --phase mechanism \\
        --workers 4 --resume

Use --dry-run to print the exact invocation without running simulations.
The 12,000-run Gate 1 confirmatory design is intentionally not exposed:
highS/lowS crosswalk and a different seed protocol are still required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from paper_b.experiments.check_gate0e_design import PLAN, check_design

PHASES = {
    "smoke": "gate0e_smoke",
    "mechanism": "gate0e_mechanism",
    "stability": "gate0e_stability",
}


def phase_command(
    *,
    phase: str,
    workers: int,
    output_dir: str,
    resume: bool = False,
    edge_log: bool = False,
    event_limit: int | None = None,
) -> tuple[list[str], Path]:
    if phase not in PHASES:
        raise ValueError(f"Unknown Gate 0E phase: {phase}")
    if workers < 1:
        raise ValueError("--workers must be positive")
    data = json.loads(PLAN.read_text(encoding="utf-8"))
    check_design(data)
    spec = next(p for p in data["phases"] if p["id"] == PHASES[phase])
    if "seeds" in spec:
        seeds = ",".join(str(x) for x in spec["seeds"])
    else:
        first,last = spec["seed_range_inclusive"]
        seeds = ",".join(str(x) for x in range(first,last+1))
    if event_limit is None:
        event_limit = 6 if phase=="smoke" else (
            26 if phase=="mechanism" else 0
        )
    if event_limit < 0:
        raise ValueError("--sender-event-period-limit cannot be negative.")
    root = Path(output_dir).expanduser().resolve() / PHASES[phase]
    cmd = [
        sys.executable, "-m", "paper_b.experiments.run_matched_pilot",
        "--seeds", seeds,
        "--initial-regimes", ",".join(spec["prior_levels"]),
        "--network-environments", ",".join(spec["networks"]),
        "--n-citizens", str(spec["N"]),
        "--horizon", str(spec["T"]),
        "--k", "1", "--epsilon", "0.05", "--credit", "20",
        "--surveillance-interval", "5",
        "--sender-regimes", ",".join(spec["senders"]),
        "--peer-evidence-mode", "source_posterior",
        "--tau-social", "1",
        "--jammer-credibility-penalty", "1",
        "--frozen-ranking-mode", "first_audit",
        "--initial-audit-mode", "centroid",
        "--sender-event-period-limit", str(event_limit),
        "--workers", str(workers),
        "--output-dir", str(root),
    ]
    cmd.append("--save-edge-log" if edge_log else "--no-edge-log")
    if resume:
        cmd.append("--resume")
    return cmd, root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen Gate 0E peer-channel experiment matrix"
    )
    parser.add_argument(
        "--phase", choices=tuple(PHASES), required=True,
        help="smoke=48, mechanism=480, stability=2400 runs",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output-dir", default="~/Research/paperb_gate0e_results",
        help="Results must be on local disk, not iCloud Drive.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-edge-log", action="store_true")
    parser.add_argument(
        "--sender-event-period-limit",type=int,default=None,
        help="0 disables citizen event trace; mechanism defaults to 26.",
    )
    args = parser.parse_args()
    cmd, root = phase_command(
        phase=args.phase,
        workers=args.workers,
        output_dir=args.output_dir,
        resume=args.resume,
        edge_log=args.save_edge_log,
        event_limit=args.sender_event_period_limit,
    )
    print(f"Gate 0E phase: {args.phase}")
    print(f"Output root: {root}")
    print("Exact matched runner command:")
    import shlex
    print(shlex.join(cmd),flush=True)
    if args.dry_run:
        print("DRY RUN ONLY — no simulations executed.")
        return
    subprocess.run(cmd,check=True)
    print("Gate 0E phase completed with pilot validation PASS.")
    print(f"Shareable ZIP appears under: {root}")


if __name__ == "__main__":
    main()

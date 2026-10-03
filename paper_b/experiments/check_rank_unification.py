"""Deterministic checks for rank-competition unification audit."""

from __future__ import annotations

from argparse import Namespace

from paper_b.experiments.run_rank_unification_audit import (
    _deterministic_reconstruction,
)


def check_deterministic_reconstruction() -> None:
    args = Namespace(
        n_citizens=20,
        peer_degree=2,
        expert_access_share=0.10,
        epsilon=0.05,
        credit=20,
        low_homophily=0.50,
        high_homophily=0.90,
        high_group_shift=3.0,
        prior_residual_sd=1.0,
    )
    rows = _deterministic_reconstruction(args, [6001, 6002])
    assert rows

    exp2 = [row for row in rows if row["audit_block"] == "exp2_gateway_rank"]
    fixed = [
        row for row in rows
        if row["audit_block"] == "fixed_biased_crowding_in"
    ]
    assert exp2
    assert fixed

    high = [row for row in exp2 if row["condition"] == "high"]
    assert all(float(row["top_target_share"]) == 1.0 for row in high)
    assert all(int(row["best_target_rank"]) == 1 for row in high)
    assert all(
        0.0 <= float(row["target_inclusion_probability"]) <= 1.0
        for row in exp2
    )

    groups = {int(row["citizen_group"]) for row in fixed}
    assert groups == {-1, 1}
    assert all(1 <= int(row["best_target_rank"]) <= 4 for row in fixed)


def main() -> None:
    check_deterministic_reconstruction()
    print("Paper B rank-unification checks: PASS")


if __name__ == "__main__":
    main()

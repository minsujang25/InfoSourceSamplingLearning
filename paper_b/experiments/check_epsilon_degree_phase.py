"""Deterministic checks for epsilon-by-degree phase design."""

from __future__ import annotations

from paper_b.experiments.run_epsilon_degree_phase import (
    _degree_nesting_check,
    _initial_rank_surface,
    _new_specs,
)


def check_specs() -> None:
    specs = _new_specs()
    assert len(specs) == 11
    labels = {spec["spec_label"] for spec in specs}
    assert "epsilon_0.02_d2" in labels
    assert "epsilon_0.30_d4" in labels
    assert "epsilon_0.05_d2" not in labels
    assert "epsilon_0.10_d2" not in labels
    assert "epsilon_0.20_d2" not in labels
    assert "epsilon_0.05_d4" not in labels


def check_degree_nesting() -> None:
    assert _degree_nesting_check(
        seeds=[6001, 6002, 6003],
        n_citizens=20,
        low_homophily=0.50,
        high_homophily=0.90,
    )


def check_rank_surface() -> None:
    rows = _initial_rank_surface(
        seeds=[6001, 6002],
        n_citizens=20,
        credit=20,
        low_homophily=0.50,
        high_homophily=0.90,
        high_group_shift=3.0,
        prior_residual_sd=1.0,
    )
    assert len(rows) == 60
    for row in rows:
        assert 1.0 <= float(row["mean_expert_rank"]) <= 5.0
        assert 0.0 <= float(
            row["mean_expert_acquisition_probability"]
        ) <= 1.0
        assert 0.0 <= float(
            row["mean_expert_inclusion_probability"]
        ) <= 1.0


def main() -> None:
    check_specs()
    check_degree_nesting()
    check_rank_surface()
    print("Paper B epsilon-degree phase checks: PASS")


if __name__ == "__main__":
    main()

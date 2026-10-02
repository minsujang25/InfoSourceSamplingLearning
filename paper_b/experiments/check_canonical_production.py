"""Deterministic design checks for Paper B canonical production."""

from __future__ import annotations

from collections import Counter

from paper_b.experiments.run_canonical_production import (
    HOMOPHILY_LEVELS,
    PRIMARY_SENDERS,
    REDUNDANCY_LEVELS,
    RELIANCE_MODES,
    SEGREGATION_LEVELS,
    TAU_SOCIAL,
    expected_conditions,
)


def main() -> None:
    assert TAU_SOCIAL == 1.0
    assert RELIANCE_MODES == ("adaptive", "frozen")
    assert HOMOPHILY_LEVELS == ("low", "high")
    assert SEGREGATION_LEVELS == ("low", "high")
    assert REDUNDANCY_LEVELS == ("low", "high")
    assert PRIMARY_SENDERS == ("null", "fixed_biased")

    iv = expected_conditions("IV")
    iii = expected_conditions("III")

    assert len(iv) == 14
    assert len(iii) == 10
    assert len(iv) + len(iii) == 24

    def key(row):
        return (
            row["production_block"],
            row["homophily_level"],
            row["segregation_level"],
            row["redundancy_level"],
            row["reliance_mode"],
            row["sender_regime"],
        )

    assert len({key(row) for row in iv}) == 14
    assert len({key(row) for row in iii}) == 10

    assert not any(
        row["sender_regime"] == "adaptive"
        and row["reliance_mode"] == "frozen"
        for row in iv + iii
    )

    iv_counts = Counter(row["production_block"] for row in iv)
    assert iv_counts == {
        "null_primary": 8,
        "fixed_biased_highS": 4,
        "adaptive_jammer_secondary": 2,
    }

    iii_counts = Counter(row["production_block"] for row in iii)
    assert iii_counts == {
        "redundancy_primary": 8,
        "adaptive_jammer_secondary": 2,
    }

    iv_fixed = [
        row for row in iv
        if row["production_block"] == "fixed_biased_highS"
    ]
    assert all(row["segregation_level"] == "high" for row in iv_fixed)

    secondary = [
        row for row in iv + iii
        if row["production_block"] == "adaptive_jammer_secondary"
    ]
    assert secondary
    assert all(row["reliance_mode"] == "adaptive" for row in secondary)
    assert all(row["sender_regime"] == "adaptive" for row in secondary)

    print("Paper B canonical-production design checks: PASS")


if __name__ == "__main__":
    main()

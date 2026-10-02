"""Deterministic checks for the frozen final Paper B calibration design."""

from __future__ import annotations

from collections import Counter

from paper_b.experiments.run_final_calibration import (
    CALIBRATION_SENDERS,
    RELIANCE_MODES,
    TAU_SOCIAL,
    expected_conditions_per_seed,
)


def main() -> None:
    conditions = expected_conditions_per_seed()
    assert TAU_SOCIAL == 1.0
    assert RELIANCE_MODES == ("adaptive", "frozen")
    assert CALIBRATION_SENDERS == ("null", "fixed_biased")
    assert len(conditions) == 20

    keys = {
        (
            row["experiment"],
            row["diagnostic_block"],
            row["homophily_level"],
            row["segregation_level"],
            row["redundancy_level"],
            row["reliance_mode"],
            row["sender_regime"],
        )
        for row in conditions
    }
    assert len(keys) == len(conditions)

    counts = Counter(row["experiment"] for row in conditions)
    assert counts == {"IV": 12, "III": 8}

    assert not any(
        row["sender_regime"] == "adaptive"
        for row in conditions
    )

    iv_fixed = [
        row for row in conditions
        if row["experiment"] == "IV"
        and row["sender_regime"] == "fixed_biased"
    ]
    assert len(iv_fixed) == 4
    assert all(row["segregation_level"] == "high" for row in iv_fixed)

    iv_null = [
        row for row in conditions
        if row["experiment"] == "IV"
        and row["sender_regime"] == "null"
    ]
    assert len(iv_null) == 8

    iii = [row for row in conditions if row["experiment"] == "III"]
    assert len(iii) == 8
    assert {row["sender_regime"] for row in iii} == {
        "null",
        "fixed_biased",
    }

    for experiment in ("III", "IV"):
        modes = {
            row["reliance_mode"]
            for row in conditions
            if row["experiment"] == experiment
        }
        assert modes == {"adaptive", "frozen"}

    print("Paper B final-calibration design checks: PASS")


if __name__ == "__main__":
    main()

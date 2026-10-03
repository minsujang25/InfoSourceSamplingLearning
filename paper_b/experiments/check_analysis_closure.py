"""Deterministic checks for the final Paper B analysis-closure audit."""

from __future__ import annotations

from paper_b.experiments.run_analysis_closure import _a_peer_sna_seed_rows


def main() -> None:
    rows = _a_peer_sna_seed_rows(
        seeds=[6001, 6002, 6003],
        n_citizens=20,
        peer_degree=2,
        low_homophily=0.50,
        high_homophily=0.90,
    )
    assert len(rows) == 6
    assert {row["homophily_level"] for row in rows} == {"low", "high"}
    for row in rows:
        assert row["n_peer_edges"] == 40
        assert row["peer_outdegree_min"] == 2
        assert row["peer_outdegree_max"] == 2
        assert 0.0 <= row["peer_reciprocity"] <= 1.0
        assert 0.0 <= row["peer_undirected_transitivity"] <= 1.0
        assert 0.0 <= row["peer_same_group_share"] <= 1.0
        assert -1.0 <= row["peer_ei_index"] <= 1.0
    print("Paper B analysis-closure checks: PASS")


if __name__ == "__main__":
    main()

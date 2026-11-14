"""Behavioral checks for truth-blind progressive clique search."""

from collections import Counter

import numpy as np
import pytest
from scipy import sparse

from qhyper.metrics import candidate_recall, evaluate_hyperedge_sets
from qhyper.projection import project_hypergraph
from qhyper.revision_candidates import (
    _neighbors,
    is_clique_from_neighbors,
    propose_progressive_candidates,
)


def test_small_complete_graph_enumerates_all_low_order_cliques():
    A = project_hypergraph([(0, 1, 2, 3)], 4, mode="binary")
    proposed, audit = propose_progressive_candidates(
        A, budget=20, max_k=3, beam_width=20, seed=7
    )
    assert len(proposed) == 10
    assert Counter(map(len, proposed)) == {2: 6, 3: 4}
    assert audit["per_size_visited"] == {"2": 6, "3": 4}
    assert not any(audit["per_size_truncated"].values())


def test_nonmaximal_truth_can_be_returned_and_expanded():
    A = project_hypergraph([(0, 1, 2, 3), (2, 3, 4)], 5, mode="binary")
    truth = [(0, 1, 2), (0, 1, 2, 3)]
    proposed, audit = propose_progressive_candidates(
        A, budget=100, max_k=4, beam_width=100, seed=4
    )
    assert set(truth).issubset(proposed)
    assert audit["returned_count"] <= audit["budget"]
    assert all(is_clique_from_neighbors(_neighbors(A), e) for e in proposed)
    coverage = candidate_recall(proposed, truth)
    assert coverage["recall"] == 1.0
    selected = [truth[0]]
    exact = evaluate_hyperedge_sets(selected, truth)
    assert exact["recall"] <= coverage["recall"]


def test_budget_quotas_and_search_cost_are_separate():
    A = project_hypergraph([tuple(range(6))], 6, mode="binary")
    proposed, audit = propose_progressive_candidates(
        A,
        budget=5,
        max_k=4,
        per_size_budget={2: 1, 3: 2, 4: 2},
        beam_width=100,
        seed=9,
    )
    assert len(proposed) == 5
    assert Counter(map(len, proposed)) == {2: 1, 3: 2, 4: 2}
    assert audit["returned_count"] == 5
    assert audit["visited_count"] > 5
    assert audit["per_size_budget"] == {"2": 1, "3": 2, "4": 2}


def test_reproducible_and_explicitly_truncated():
    A = project_hypergraph([tuple(range(7))], 7, mode="binary")
    args = dict(
        budget=5, max_k=5, beam_width=2, max_visited=3,
        exploration_fraction=0.25, seed=13,
    )
    first, audit_a = propose_progressive_candidates(A, **args)
    second, audit_b = propose_progressive_candidates(A, **args)
    assert first == second
    assert audit_a == audit_b
    assert audit_a["per_size_truncated"]["2"]
    assert audit_a["per_size_truncated"]["3"]
    assert len(first) <= 5
    assert len(first) == len(set(first))
    assert all(is_clique_from_neighbors(_neighbors(A), e) for e in first)


def test_empty_observation_and_invalid_inputs():
    A = sparse.csr_matrix((5, 5), dtype=float)
    empty_budget, empty_audit = propose_progressive_candidates(
        A, budget=0, max_k=5
    )
    assert empty_budget == []
    assert empty_audit["visited_count"] == 0
    proposed, audit = propose_progressive_candidates(A, budget=10, max_k=5)
    assert proposed == []
    assert audit["visited_count"] == 0
    with pytest.raises(ValueError, match="per_size_budget sum"):
        propose_progressive_candidates(
            A, budget=3, max_k=5, per_size_budget={2: 4}
        )
    bad = np.zeros((5, 5))
    bad[0, 1] = bad[1, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        propose_progressive_candidates(sparse.csr_matrix(bad), budget=5)


def test_observation_only_callback_changes_proposal_priority():
    A = project_hypergraph([(0, 1, 2, 3)], 4, mode="binary")
    target = (1, 2, 3)
    proposed, audit = propose_progressive_candidates(
        A,
        budget=1,
        min_k=3,
        max_k=3,
        score_fn=lambda edge: 1.0 if edge == target else 0.0,
        seed=2,
    )
    assert proposed == [target]
    assert audit["score_policy"] == "callback"

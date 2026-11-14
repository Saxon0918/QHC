"""Selection tests emphasize the gap between graph coverage and true events."""

from itertools import combinations

import numpy as np
import pytest

from qhyper.projection import project_hypergraph
from qhyper.revision_selection import (
    coverage_value, gamma_code_length, select_greedy_coverage,
    select_mdl_greedy, two_part_description_length,
    select_exact_projection_map, exact_posterior_small, select_top_m,
)


def test_weighted_coverage_has_diminishing_returns() -> None:
    A = project_hypergraph([(0, 1, 2), (1, 2, 3)], 4, mode="binary")
    e = (1, 2, 3)
    small = [(0, 1)]
    large = small + [(0, 1, 2)]
    gain_small = coverage_value(A, small + [e]) - coverage_value(A, small)
    gain_large = coverage_value(A, large + [e]) - coverage_value(A, large)
    assert gain_small >= gain_large >= 0


def test_coverage_can_prefer_false_triad_with_same_projection() -> None:
    # The three pairs are truth, but the observed K3 also admits one triad.
    pairs = list(combinations(range(3), 2))
    A = project_hypergraph(pairs, 3, mode="binary")
    selected = select_greedy_coverage(A, pairs + [(0, 1, 2)], max_selected=1)
    assert selected == [(0, 1, 2)]
    assert coverage_value(A, selected) == 3.0
    assert set(selected) != set(pairs)


def test_two_part_code_is_finite_and_does_not_claim_truth() -> None:
    pairs = list(combinations(range(3), 2))
    A = project_hypergraph(pairs, 3, mode="binary")
    assert gamma_code_length(1) == 1
    assert gamma_code_length(4) == 5
    assert two_part_description_length(A, [(0, 1, 2)]) < two_part_description_length(A, pairs)
    assert select_mdl_greedy(A, pairs + [(0, 1, 2)]) == [(0, 1, 2)]


def test_conditional_reward_can_select_projection_redundant_edge() -> None:
    A = project_hypergraph([(0, 1, 2, 3)], 4, mode="binary")
    edges = [(0, 1, 2, 3), (0, 1, 2)]
    pure = select_mdl_greedy(A, edges)
    conditional = select_mdl_greedy(A, edges, scores=[0.999, 0.999], score_weight=100.0)
    assert len(pure) == 1
    assert set(conditional) == set(edges)


def test_selection_rejects_nonclique_and_respects_budget() -> None:
    A = project_hypergraph([(0, 1), (1, 2)], 3, mode="binary")
    with pytest.raises(ValueError, match="clique"):
        select_greedy_coverage(A, [(0, 1, 2)], max_selected=1)
    assert len(select_greedy_coverage(A, [(0, 1), (1, 2)], max_selected=1)) == 1
    with pytest.raises(ValueError):
        select_mdl_greedy(A, [(0, 1)], scores=[np.nan])


def test_empty_single_node_graph_has_finite_code() -> None:
    assert two_part_description_length(np.zeros((1, 1)), []) == 2


def test_top_m_respects_scores_and_ties() -> None:
    assert select_top_m([(1, 2), (0, 1), (0, 2)], [0.8, 0.8, 0.1], 2) == [
        (0, 1), (1, 2)]


def test_exact_projection_map_uses_prior_and_reports_ambiguity() -> None:
    pairs = list(combinations(range(3), 2))
    A = project_hypergraph(pairs, 3, mode="binary")
    universe = [(0, 1, 2), *pairs]
    assert select_exact_projection_map(A, universe, [0.9, 0.1, 0.1, 0.1]) == [(0, 1, 2)]
    assert set(select_exact_projection_map(A, universe, [0.01, 0.8, 0.8, 0.8])) == set(pairs)
    posterior = exact_posterior_small(A, universe, [0.2] * 4)
    assert posterior["feasible_structure_count"] > 1
    assert posterior["posterior_entropy_nats"] > 0
    assert all(0 < p < 1 for p in posterior["marginal_probability"])


def test_exact_projection_requires_counted_dyad_support() -> None:
    A = project_hypergraph([(0, 1), (1, 2)], 3, mode="binary")
    with pytest.raises(ValueError, match="does not cover"):
        select_exact_projection_map(A, [(0, 1)], [0.2])


def test_exact_posterior_entropy_handles_underflowing_structures() -> None:
    pairs = list(combinations(range(3), 2))
    A = project_hypergraph(pairs, 3, mode="binary")
    universe = [(0, 1, 2), *pairs]
    posterior = exact_posterior_small(A, universe, [0.5, 1e-300, 1e-300, 1e-300])
    assert posterior["feasible_structure_count"] > 1
    assert np.isfinite(posterior["posterior_entropy_nats"])
    assert posterior["posterior_entropy_nats"] >= 0

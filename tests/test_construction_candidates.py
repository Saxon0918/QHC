"""Topology-only local proposals for latent, possibly nonclique groups."""

import inspect

import numpy as np
import pytest

from qhyper.construction_candidates import propose_groups
from qhyper.hamiltonian import feature_similarity_adjacency


def test_star_closed_neighborhood_proposes_nonclique() -> None:
    # Leaves 1 and 2 are not adjacent, but {0,1,2} lies in N[0].
    A = np.array([
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ], dtype=float)
    groups = propose_groups(A, max_size=4, max_candidates=100)
    assert groups == tuple(sorted(set(groups)))
    assert (0, 1, 2) in groups
    assert (0, 1, 2, 3) in groups
    assert (1, 2) not in groups  # pair proposals are observed graph edges
    assert (0, 1) in groups


def test_budget_size_and_pair_controls_are_deterministic() -> None:
    A = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=float)
    first = propose_groups(A, max_candidates=7)
    second = propose_groups(A, max_candidates=7)
    assert first == second
    assert len(first) == 7
    assert len(set(first)) == len(first)
    assert {len(group) for group in first} >= {2, 3}
    no_pairs = propose_groups(A, max_candidates=50, include_pairs=False)
    assert all(len(group) >= 3 for group in no_pairs)
    assert propose_groups(A, max_candidates=0) == ()


def test_full_local_pool_is_permutation_equivariant() -> None:
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=float)
    groups = propose_groups(A, max_candidates=1000, neighborhood_cap=5)
    permutation = np.array([3, 0, 4, 1, 2])
    inverse = np.argsort(permutation)
    permuted = propose_groups(A[np.ix_(permutation, permutation)],
                              max_candidates=1000, neighborhood_cap=5)
    mapped = tuple(sorted(tuple(sorted(int(inverse[u]) for u in group))
                          for group in groups))
    assert mapped == permuted


def test_input_is_observation_only_and_validated() -> None:
    names = inspect.signature(propose_groups).parameters
    assert "labels" not in names and "hyperedges_truth" not in names
    with pytest.raises(ValueError, match="symmetric"):
        propose_groups(np.array([[0, 1], [0, 0]], dtype=float))
    with pytest.raises(ValueError, match="min_size"):
        propose_groups(np.eye(2), min_size=1)


def test_optional_feature_union_adds_label_blind_nonclique_group() -> None:
    # Observed A has only 0--1. Feature mutual-kNN adds 0--2, yielding a
    # nonclique group {0,1,2} in the union closed neighborhood of node 0.
    A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.98, 0.2]])
    feature_graph = feature_similarity_adjacency(X, k=1)
    assert (feature_graph != feature_graph.T).nnz == 0
    assert np.all(feature_graph.data >= 0)
    without = propose_groups(A, max_size=3, max_candidates=20)
    with_features = propose_groups(A, features=X, feature_k=1,
                                   max_size=3, max_candidates=20)
    assert (0, 1, 2) not in without
    assert (0, 1, 2) in with_features
    assert (0, 2) in with_features
    assert (1, 2) not in with_features
    assert with_features == propose_groups(A, features=X, feature_k=1,
                                           max_size=3, max_candidates=20)
    assert without == propose_groups(A, features=None, feature_k=1,
                                     max_size=3, max_candidates=20)


def test_feature_protocol_validates_alignment_and_neighbor_count() -> None:
    A = np.zeros((3, 3))
    with pytest.raises(ValueError, match="aligned"):
        propose_groups(A, features=np.ones((2, 2)))
    with pytest.raises(ValueError, match="feature_k"):
        propose_groups(A, features=np.ones((3, 2)), feature_k=0)

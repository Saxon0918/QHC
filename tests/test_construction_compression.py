"""Mathematical checks for normalized hard-hypergraph compression."""

import numpy as np
import pytest
from scipy import sparse

from qhyper.construction_compression import (
    _trial_error_change,
    compress_kernel,
    normalized_incidence_operator,
)


def _target_with_triple(n: int, group: tuple[int, ...], weight: float) -> np.ndarray:
    columns = [sparse.csr_matrix(np.eye(n)),
               sparse.csr_matrix((np.ones(len(group)),
                                  (np.asarray(group), np.zeros(len(group), dtype=int))),
                                 shape=(n, 1))]
    B = sparse.hstack(columns, format="csr")
    theta, _ = normalized_incidence_operator(B, [1.0] * n + [weight])
    return theta


def test_exact_normalization_and_positive_degrees() -> None:
    B = sparse.csr_matrix([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=float)
    theta, degree = normalized_incidence_operator(B, [1, 1, 1, 3])
    np.testing.assert_allclose(degree, [4, 4, 4])
    np.testing.assert_allclose(theta, (np.eye(3) + np.ones((3, 3))) / 4)
    np.testing.assert_allclose(theta @ np.sqrt(degree), np.sqrt(degree))
    assert np.linalg.eigvalsh(theta).min() >= -1e-12
    with pytest.raises(ValueError, match="positive weighted degree"):
        normalized_incidence_operator(np.array([[1], [0]]), [1])


def test_k3_uniform_kernel_cannot_be_exact_with_nonnegative_pairs() -> None:
    # Every pair contributes exactly half of its incident weight to a node's
    # unnormalized diagonal. For pair atoms alone, Theta_ii = 1/2 whenever
    # degrees are positive. Unit singleton atoms only increase this value.
    # The uniform 3-way kernel J/3 has diagonal 1/3, so nonnegative pair
    # atoms cannot reproduce it, even before the singleton requirement.
    target = np.ones((3, 3)) / 3
    pair_only = sparse.csr_matrix([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=float)
    theta_pairs, _ = normalized_incidence_operator(pair_only, [0.1, 2, 4])
    np.testing.assert_allclose(np.diag(theta_pairs), 0.5)
    B = sparse.csr_matrix([
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1],
    ], dtype=float)
    for pair_weights in ([0, 0, 0], [0.1, 2, 4], [100, 100, 100]):
        theta, _ = normalized_incidence_operator(B, [1, 1, 1, *pair_weights])
        assert np.all(np.diag(theta) >= 0.5)
        assert np.linalg.norm(theta - target, "fro") >= np.sqrt(3) / 6 - 1e-12
    result = compress_kernel(target, ((0, 1), (0, 2), (1, 2)),
                             budget=3, weight_grid=(0.5, 1, 4, 100))
    assert result.error_fro >= np.sqrt(3) / 6 - 1e-12


def test_greedy_exactly_recovers_grid_representable_kernel() -> None:
    target = _target_with_triple(4, (0, 2, 3), 2.0)
    result = compress_kernel(target, ((0, 1), (0, 2, 3), (1, 2, 3)),
                             budget=2, weight_grid=(0.5, 1, 2))
    assert result.selected_groups == ((0, 2, 3),)
    assert result.weights[-1] == 2.0
    assert result.error_fro < 1e-12
    assert result.status == "no_improvement"
    assert len(result.all_groups) == 5  # four unit singletons plus one triple
    exact, degree = normalized_incidence_operator(result.incidence, result.weights)
    np.testing.assert_allclose(result.theta, exact, atol=1e-14)
    assert np.all(degree > 0)


def test_incremental_trial_change_matches_full_recomputation() -> None:
    target = _target_with_triple(5, (0, 1, 3), 1.5)
    theta = _target_with_triple(5, (0, 2, 4), 0.7)
    degree = np.ones(5)
    degree[[0, 2, 4]] += 0.7
    nodes = np.array([1, 2, 3])
    weight = 2.0
    change = _trial_error_change(target, theta, degree, nodes, weight)
    B = sparse.csr_matrix([
        [1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0],
    ], dtype=float)
    updated, _ = normalized_incidence_operator(B, [1] * 5 + [0.7, 2.0])
    expected = np.sum((target - updated)**2) - np.sum((target - theta)**2)
    np.testing.assert_allclose(change, expected, atol=1e-13)


def test_tie_free_selection_commutes_with_node_permutation() -> None:
    target = _target_with_triple(5, (0, 2, 4), 3.0)
    candidates = ((0, 2, 4), (0, 1), (1, 2, 3))
    first = compress_kernel(target, candidates, budget=1, weight_grid=(0.5, 1, 3))
    permutation = np.array([3, 0, 4, 1, 2])  # new index -> old index
    inverse = np.argsort(permutation)
    mapped_candidates = tuple(tuple(sorted(int(inverse[u]) for u in group))
                              for group in candidates)
    second = compress_kernel(target[np.ix_(permutation, permutation)],
                             mapped_candidates, budget=1,
                             weight_grid=(0.5, 1, 3))
    expected_group = tuple(sorted(int(inverse[u]) for u in first.selected_groups[0]))
    assert second.selected_groups == (expected_group,)
    np.testing.assert_allclose(second.theta,
                               first.theta[np.ix_(permutation, permutation)])


def test_no_improvement_and_input_validation() -> None:
    identity = np.eye(3)
    result = compress_kernel(identity, ((0, 1, 2),), budget=2)
    assert result.selected_groups == ()
    assert result.status == "no_improvement"
    assert result.error_fro == 0
    with pytest.raises(ValueError, match="symmetric"):
        compress_kernel(np.array([[1, 1], [0, 1]]), [(0, 1)], budget=1)
    with pytest.raises(ValueError, match="repeated"):
        compress_kernel(np.eye(3), [(0, 0)], budget=1)

"""C0-C3/Q1 controls use only observed graph information."""

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import expm

from qhyper.classic_features import (
    classical_candidate_features,
    classical_candidate_matrix,
    classical_feature_names,
    dynamic_family_matrix,
    dynamic_family_features,
    family_feature_names,
    short_time_coefficient,
)


def graph():
    # Candidate (0,1,2) is a clique, with nodes 3 and 4 outside.
    A = np.zeros((5, 5), dtype=float)
    for a, b in [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (1, 4), (2, 4)]:
        A[a, b] = A[b, a] = 1
    return sparse.csr_matrix(A)


def test_classical_features_and_exact_short_time_term():
    A = graph()
    edge = (0, 1, 2)
    vector = classical_candidate_features(A, edge)
    names = classical_feature_names()
    assert len(vector) == len(names)
    values = dict(zip(names, vector))
    assert values["size"] == 3
    assert values["internal_edges"] == 3
    assert values["internal_density"] == 1
    assert values["boundary_weight"] == 4
    assert values["external_neighbor_count"] == 2
    assert values["external_common_neighbor_sum"] == 2
    assert values["internal_triangle_count"] == 1
    assert values["short_time_c2_coefficient"] == short_time_coefficient(A, edge)
    assert values["short_time_c2_coefficient"] < 0

    degrees = np.asarray(A.sum(axis=1)).ravel()
    inv = degrees ** -0.5
    H = 0.5 * (np.eye(5) - np.diag(inv) @ A.toarray() @ np.diag(inv))
    external_columns = H[np.ix_([3, 4], [0, 1, 2])]
    gram = external_columns.T @ external_columns
    expected = -(gram.sum() - np.trace(gram)) / 3
    assert values["short_time_c2_coefficient"] == pytest.approx(expected)


def test_attributes_and_permutation_invariance():
    A = graph()
    X = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [2, 0]], dtype=float)
    baseline = classical_candidate_features(A, (0, 1, 2), X)
    permutation = np.array([2, 4, 0, 1, 3])
    old_at_new = np.argsort(permutation)
    moved = classical_candidate_features(
        A[old_at_new, :][:, old_at_new], tuple(permutation[[0, 1, 2]]), X[old_at_new]
    )
    np.testing.assert_allclose(baseline, moved)
    assert np.all(np.isfinite(baseline))


def test_proposal_source_is_fixed_vocabulary():
    A = graph()
    names = classical_feature_names()
    local = classical_candidate_features(A, (0, 1, 2), source="local_clique")
    unknown = classical_candidate_features(A, (0, 1, 2), source="unlisted_method")
    assert local[names.index("source_local_clique")] == 1
    assert unknown[names.index("source_other")] == 1
    batch = classical_candidate_matrix(A, [(0, 1, 2), (0, 1)], sources=["local_clique", "edge"])
    np.testing.assert_allclose(batch[0], local)
    assert batch.shape == (2, len(names))


def test_candidate_can_have_multiple_proposal_sources():
    A = graph()
    names = classical_feature_names()
    multi = classical_candidate_features(
        A, (0, 1, 2), source=["local_clique", "maximal_clique", "unknown_proposal"]
    )
    assert multi[names.index("source_local_clique")] == 1
    assert multi[names.index("source_maximal_clique")] == 1
    assert multi[names.index("source_other")] == 1
    assert sum(multi[names.index("source_local_clique") :]) == 3
    batch = classical_candidate_matrix(
        A, [(0, 1, 2)], sources=[("local_clique", "maximal_clique")]
    )
    assert batch.shape == (1, len(names))
    assert batch[0, names.index("source_local_clique")] == 1
    assert batch[0, names.index("source_maximal_clique")] == 1


@pytest.mark.parametrize("family", ["heat", "wave", "spectral", "quantum_pair"])
def test_dynamics_controls_have_documented_shapes(family):
    A = graph()
    degree = np.asarray(A.sum(axis=1)).ravel()
    inv = degree ** -0.5
    H = sparse.csr_matrix(0.5 * (np.eye(5) - np.diag(inv) @ A.toarray() @ np.diag(inv)))
    times = [0.0, 0.5]
    features = dynamic_family_features(A, H, (0, 1, 2), times, family)
    assert len(features) == len(family_feature_names(family, times))
    assert np.all(np.isfinite(features))
    if family in {"heat", "wave"}:
        np.testing.assert_allclose(features[:3], [1, 1, 0], atol=1e-10)
    if family == "quantum_pair":
        np.testing.assert_allclose(features[:5], [0, 0, 0, 1, 1 / 3], atol=1e-10)


def test_bad_graph_rejected():
    with pytest.raises(ValueError):
        classical_candidate_features(np.array([[0, 1], [0, 0]]), (0, 1))


def test_classical_dynamics_against_dense_oracle():
    A = graph()
    dense = A.toarray()
    degree = dense.sum(axis=1)
    L = np.eye(5) - np.diag(degree ** -0.5) @ dense @ np.diag(degree ** -0.5)
    H = sparse.csr_matrix(0.5 * L)
    edge = (0, 1, 2)
    states = np.eye(5)[:, list(edge)]
    coherent = states.sum(axis=1, keepdims=True) / np.sqrt(len(edge))
    inputs = np.column_stack((coherent, states))
    t = 0.7
    eigval, eigvec = np.linalg.eigh(L)
    kernels = {
        "heat": expm(-t * L),
        "wave": eigvec @ np.diag(np.cos(t * np.sqrt(np.maximum(eigval, 0)))) @ eigvec.T,
    }
    for family, kernel in kernels.items():
        acted = kernel @ inputs
        localization = np.sum(acted[list(edge)] ** 2, axis=0) / np.sum(acted ** 2, axis=0)
        expected = [localization[0], np.mean(localization[1:]), localization[0] - np.mean(localization[1:])]
        actual = dynamic_family_features(A, H, edge, [t], family)
        np.testing.assert_allclose(actual, expected, atol=1e-11)

    U = expm(-1j * t * H.toarray())
    block = np.abs(U[np.ix_(edge, edge)]) ** 2
    offdiag = block[~np.eye(len(edge), dtype=bool)]
    expected_pair = [np.mean(offdiag), np.max(offdiag), np.std(offdiag), np.mean(np.diag(block)), np.mean(block)]
    actual_pair = dynamic_family_features(A, H, edge, [t], "quantum_pair")
    np.testing.assert_allclose(actual_pair, expected_pair, atol=1e-11)


@pytest.mark.parametrize("family", ["heat", "wave", "spectral", "quantum_pair"])
def test_batched_dynamics_equal_candidate_wise(family):
    A = graph()
    degree = np.asarray(A.sum(axis=1)).ravel()
    inv = degree ** -0.5
    H = sparse.csr_matrix(0.5 * (np.eye(5) - np.diag(inv) @ A.toarray() @ np.diag(inv)))
    candidates = [(0, 1, 2), (0, 1), (1, 2, 4), (1, 3), (0, 1, 2)]
    times = [0.0, 0.25, 1.0]
    expected = np.vstack(
        [dynamic_family_features(A, H, edge, times, family) for edge in candidates]
    )
    actual = dynamic_family_matrix(A, H, candidates, times, family, max_state_columns=7)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_batch_column_limit_must_fit_candidate():
    A = graph()
    with pytest.raises(ValueError):
        dynamic_family_matrix(A, None, [(0, 1, 2)], [1.0], "heat", max_state_columns=3)

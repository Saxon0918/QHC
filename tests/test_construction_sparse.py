"""Certified operator approximation and sparse-action regression tests."""

from itertools import combinations

import numpy as np
import pytest
from scipy import sparse

from qhyper.construction_kernels import matched_markov, quantum_transport
from qhyper.construction_sparse import (
    selected_transport_columns,
    sparse_transport_hypergraph,
)
from qhyper.hamiltonian import build_hamiltonian


def _example_transport() -> tuple[sparse.csr_matrix, np.ndarray]:
    adjacency = np.array(
        [[0, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 1, 0, 1, 0],
         [0, 1, 1, 0, 1], [0, 0, 0, 1, 0]], dtype=float
    )
    h = build_hamiltonian(adjacency)
    return h, quantum_transport(h, [1.4])["transport"][0]


def _operator_from_incidence(incidence: sparse.csr_matrix,
                             weights: np.ndarray) -> np.ndarray:
    edge_degrees = np.asarray(incidence.sum(axis=0)).ravel()
    inverse_edge = np.divide(1.0, edge_degrees,
                             out=np.zeros_like(edge_degrees), where=edge_degrees > 0)
    node_degrees = np.asarray(incidence @ weights).ravel()
    middle = (incidence @ sparse.diags(weights * inverse_edge) @ incidence.T).toarray()
    return middle / np.sqrt(node_degrees[:, None] * node_degrees[None, :])


def test_full_top_k_reproduces_exact_transport_kernel():
    _, p = _example_transport()
    result = sparse_transport_hypergraph(p, top_k=p.shape[0])
    kernel = p @ p.T
    np.testing.assert_allclose(result.normalized_theta.toarray(), kernel, atol=1e-13)
    np.testing.assert_allclose(result.residual_degrees, 0, atol=1e-13)
    np.testing.assert_allclose(
        _operator_from_incidence(result.incidence, result.edge_weights), kernel,
        atol=1e-13,
    )
    assert result.trace_error_exact == pytest.approx(0.0, abs=1e-12)
    assert result.trace_bound == pytest.approx(0.0, abs=1e-12)
    assert result.incidence.shape == (p.shape[0], 2 * p.shape[0])


@pytest.mark.parametrize("landmark_ids,top_k", [([0, 2], 2), ([0, 1, 3], 3),
                                                 ([0, 1, 2, 3, 4], 2)])
def test_partial_and_top_k_operator_is_certified(landmark_ids, top_k):
    _, p = _example_transport()
    result = sparse_transport_hypergraph(
        p[:, landmark_ids], top_k=top_k, landmark_ids=landmark_ids
    )
    theta = result.normalized_theta.toarray()
    kernel = p @ p.T
    difference = theta - kernel
    np.testing.assert_allclose(theta, theta.T, atol=1e-13)
    np.testing.assert_allclose(theta.sum(axis=1), 1, atol=1e-12)
    assert np.min(theta) >= -1e-13
    assert np.linalg.eigvalsh(theta).min() >= -1e-12
    assert np.linalg.eigvalsh(difference).min() >= -1e-12
    assert np.linalg.norm(difference, ord=2) <= result.trace_bound + 1e-12
    assert np.trace(difference) <= result.trace_bound + 1e-12
    np.testing.assert_allclose(
        _operator_from_incidence(result.incidence, result.edge_weights), theta,
        atol=1e-12,
    )
    if len(landmark_ids) == p.shape[0]:
        assert result.trace_error_exact == pytest.approx(np.trace(difference), abs=1e-12)
    else:
        assert result.trace_error_exact is None


def test_zero_time_and_no_landmark_degenerate_to_identity():
    p = np.eye(4)
    for columns, ids in ((p, [0, 1, 2, 3]), (p[:, [0, 2]], [0, 2]),
                         (p[:, :0], [])):
        result = sparse_transport_hypergraph(columns, top_k=1, landmark_ids=ids)
        np.testing.assert_allclose(result.normalized_theta.toarray(), np.eye(4))
        np.testing.assert_allclose(
            np.asarray(result.incidence.sum(axis=1)).ravel(), 1
        )


def test_uniform_transport_has_global_group_when_untruncated():
    p = np.full((5, 5), 1 / 5)
    exact = sparse_transport_hypergraph(p, top_k=5)
    np.testing.assert_allclose(exact.normalized_theta.toarray(), p)
    truncated = sparse_transport_hypergraph(p, top_k=2)
    delta = truncated.normalized_theta.toarray() - p
    assert np.linalg.eigvalsh(delta).min() >= -1e-12
    assert truncated.trace_error_exact == pytest.approx(np.trace(delta))
    assert all(len(support) == 2 for support in truncated.supports)


def test_top_k_maximizes_single_column_trace_gain():
    p = np.array([0.40, 0.26, 0.19, 0.15])
    for k in (2, 3):
        result = sparse_transport_hypergraph(p[:, None], top_k=k)
        selected = result.supports[0]
        assert selected == tuple(range(k))

        def gain(indices):
            b = p[list(indices)]
            return b.sum() - float(b @ b) / b.sum()

        assert gain(selected) == pytest.approx(
            max(gain(indices) for indices in combinations(range(4), k))
        )


def test_sparse_expm_actions_match_dense_references():
    h, _ = _example_transport()
    ids = [3, 0]
    times = [0.0, 0.4, 1.2]
    for family, builder in (("quantum", quantum_transport),
                            ("matched_markov", matched_markov)):
        selected = selected_transport_columns(h, times, ids, family=family,
                                              batch_size=1)
        full = builder(h, times)
        assert len(selected) == len(times)
        for columns, transport in zip(selected, full["transport"]):
            np.testing.assert_allclose(columns, transport[:, ids], atol=1e-12)
            np.testing.assert_allclose(columns.sum(axis=0), 1, atol=1e-12)


def test_sparse_action_accepts_graph_beyond_dense_oracle_cap():
    h = sparse.csr_matrix((129, 129), dtype=float)
    for family in ("quantum", "matched_markov"):
        columns = selected_transport_columns(h, [0.0, 2.0], [2, 128],
                                             family=family, batch_size=1)
        assert len(columns) == 2
        for column_matrix in columns:
            np.testing.assert_allclose(column_matrix[:, 0], np.eye(129)[:, 2])
            np.testing.assert_allclose(column_matrix[:, 1], np.eye(129)[:, 128])


def test_invalid_selected_columns_and_parameters_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        sparse_transport_hypergraph(np.array([[0.2], [0.2]]), top_k=1)
    with pytest.raises(ValueError, match="unit mass"):
        sparse_transport_hypergraph(np.array([[1, 1], [0, 0]]), top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        sparse_transport_hypergraph(np.eye(3), top_k=0)
    with pytest.raises(ValueError, match="distinct"):
        sparse_transport_hypergraph(np.eye(3)[:, :2], top_k=1,
                                    landmark_ids=[0, 0])
    with pytest.raises(ValueError, match="family"):
        selected_transport_columns(sparse.eye(3), [1], [0], family="heat")

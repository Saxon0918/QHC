import numpy as np
import pytest
from scipy import sparse

from qhyper.hamiltonian import (
    build_hamiltonian,
    checked_adjacency,
    feature_similarity_adjacency,
    normalized_laplacian,
)
from qhyper.quantum_action import quantum_action, reference_unitary


def test_normalized_laplacian_and_isolated_vertex_convention():
    a = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )
    expected = np.array(
        [[0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
    )
    h = build_hamiltonian(a)
    assert sparse.isspmatrix_csr(h)
    np.testing.assert_allclose(h.toarray(), expected)
    np.testing.assert_allclose(normalized_laplacian(a).toarray(), 2 * expected)
    np.testing.assert_allclose(np.linalg.eigvalsh(h.toarray()), [0.0, 0.0, 1.0])
    assert h[2, 2] == 0.0


def test_weighted_graph_spectrum_and_unitarity():
    a = np.array(
        [
            [0, 2, 0, 0],
            [2, 0, 3, 1],
            [0, 3, 0, 4],
            [0, 1, 4, 0],
        ],
        dtype=float,
    )
    h = build_hamiltonian(a)
    eigenvalues = np.linalg.eigvalsh(h.toarray())
    assert eigenvalues.min() >= -1e-12
    assert eigenvalues.max() <= 1 + 1e-12
    unitary = reference_unitary(h, 2.0)
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(4), atol=1e-12)
    p = np.abs(unitary) ** 2
    np.testing.assert_allclose(p.sum(axis=0), np.ones(4), atol=1e-12)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(4), atol=1e-12)


def test_sparse_action_matches_dense_reference_without_full_propagator():
    a = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    h = build_hamiltonian(a)
    states = np.eye(4)[:, :3]
    results = list(quantum_action(h, states, [0.0, 0.3, 1.7], batch_size=2))
    assert len(results) == 6
    for time, columns, evolved in results:
        expected = reference_unitary(h, time) @ states[:, columns]
        np.testing.assert_allclose(evolved, expected, atol=1e-12)
        assert evolved.shape[1] <= 2


def test_invalid_graph_inputs_are_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        checked_adjacency(np.array([[0, 1], [0, 0]]))
    with pytest.raises(ValueError, match="negative"):
        checked_adjacency(np.array([[0, -1], [-1, 0]]))
    with pytest.raises(ValueError, match="self-loops"):
        checked_adjacency(np.eye(2))
    with pytest.raises(ValueError, match="required"):
        build_hamiltonian(np.zeros((2, 2)), lam=0.5)


def test_feature_graph_is_optional_and_symmetric():
    a = np.zeros((4, 4))
    x = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.0, 0.0]])
    ax = feature_similarity_adjacency(x, k=1)
    np.testing.assert_allclose(ax.toarray(), ax.T.toarray())
    assert ax[3].nnz == 0
    np.testing.assert_allclose(build_hamiltonian(a, x, lam=0).toarray(), np.zeros((4, 4)))
    h_feature = build_hamiltonian(a, x, lam=1, feature_k=1)
    assert h_feature[0, 1] < 0

"""Covariance identity and Ward compression tests for full transport matrices."""

import numpy as np
import pytest

from qhyper.construction_cluster import cluster_transport_hypergraph
from qhyper.construction_kernels import quantum_transport


def _irregular_transport() -> np.ndarray:
    rng = np.random.default_rng(20261115)
    h = rng.uniform(-0.9, 0.7, size=(5, 5))
    h = (h + h.T) / 2
    return quantum_transport(h, [1.25])["transport"][0]


def _within_cluster_covariance(p, clusters):
    difference = np.zeros((len(p), len(p)))
    for group in clusters:
        columns = p[:, group]
        mean = columns.mean(axis=1)
        deviation = columns - mean[:, None]
        difference += deviation @ deviation.T
    return difference


def test_clustered_operator_and_covariance_certificate():
    p = _irregular_transport()
    result = cluster_transport_hypergraph(p, partition=[[0, 2], [1, 3, 4]])
    incidence = result.incidence.toarray()
    theta = result.normalized_theta.toarray()
    kernel = p.T @ p
    np.testing.assert_allclose(incidence, np.column_stack((p[:, [0, 2]].sum(axis=1),
                                                           p[:, [1, 3, 4]].sum(axis=1))))
    np.testing.assert_allclose(incidence.sum(axis=0), [2, 3], atol=1e-12)
    np.testing.assert_allclose(incidence.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(theta, incidence @ np.diag([1 / 2, 1 / 3])
                               @ incidence.T, atol=1e-12)
    np.testing.assert_allclose(theta.sum(axis=1), 1, atol=1e-12)
    assert np.linalg.eigvalsh(theta).min() >= -1e-12
    difference = kernel - theta
    np.testing.assert_allclose(difference,
                               _within_cluster_covariance(p, result.clusters),
                               atol=1e-12)
    assert np.linalg.eigvalsh(difference).min() >= -1e-12
    assert result.spectral_error <= result.trace_error + 1e-12
    assert result.trace_error == pytest.approx(np.trace(difference), abs=1e-12)
    assert result.spectral_error == pytest.approx(
        np.linalg.norm(difference, ord=2), abs=1e-12
    )
    assert result.membership_nnz == result.incidence.nnz


def test_ward_extremes_exact_and_uniform():
    p = _irregular_transport()
    exact = cluster_transport_hypergraph(p, n_clusters=len(p))
    np.testing.assert_allclose(exact.normalized_theta.toarray(), p.T @ p,
                               atol=1e-12)
    assert exact.clusters == tuple((index,) for index in range(len(p)))
    assert exact.trace_error == pytest.approx(0, abs=1e-12)
    assert exact.spectral_error == pytest.approx(0, abs=1e-12)

    one = cluster_transport_hypergraph(p, n_clusters=1)
    assert one.clusters == (tuple(range(len(p))),)
    np.testing.assert_allclose(one.incidence.toarray(), np.ones((len(p), 1)),
                               atol=1e-12)
    np.testing.assert_allclose(one.normalized_theta.toarray(),
                               np.ones_like(p) / len(p), atol=1e-12)
    assert one.trace_error == pytest.approx(np.trace(p.T @ p) - 1, abs=1e-12)


def test_tie_free_ward_is_permutation_equivariant():
    p = _irregular_transport()
    permutation = [2, 0, 4, 1, 3]
    original = cluster_transport_hypergraph(p, n_clusters=3)
    permuted = cluster_transport_hypergraph(
        p[np.ix_(permutation, permutation)], n_clusters=3
    )
    mapped = {frozenset(permutation[index] for index in group)
              for group in permuted.clusters}
    expected = {frozenset(group) for group in original.clusters}
    assert mapped == expected
    np.testing.assert_allclose(
        permuted.normalized_theta.toarray(),
        original.normalized_theta.toarray()[np.ix_(permutation, permutation)],
        atol=1e-12,
    )
    assert permuted.trace_error == pytest.approx(original.trace_error, abs=1e-12)


def test_disconnected_transport_and_deterministic_ties():
    p = np.eye(4)
    first = cluster_transport_hypergraph(p, n_clusters=2)
    second = cluster_transport_hypergraph(p, n_clusters=2)
    assert first.clusters == second.clusters
    np.testing.assert_allclose(first.normalized_theta.toarray(),
                               second.normalized_theta.toarray())
    assert first.trace_error > 0


def test_invalid_transport_partition_and_budget_are_rejected():
    p = np.eye(3)
    with pytest.raises(ValueError, match="exactly one"):
        cluster_transport_hypergraph(p)
    with pytest.raises(ValueError, match="exactly one"):
        cluster_transport_hypergraph(p, n_clusters=2, partition=[[0], [1, 2]])
    with pytest.raises(ValueError, match="n_clusters"):
        cluster_transport_hypergraph(p, n_clusters=0)
    with pytest.raises(ValueError, match="cover every"):
        cluster_transport_hypergraph(p, partition=[[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="nonempty"):
        cluster_transport_hypergraph(p, partition=[[0, 1, 2], []])
    with pytest.raises(ValueError, match="symmetric"):
        cycle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        cluster_transport_hypergraph(cycle, n_clusters=2)
    with pytest.raises(ValueError, match="max_nodes"):
        cluster_transport_hypergraph(p, n_clusters=2, max_nodes=2)

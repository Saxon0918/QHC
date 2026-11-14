"""The revised spectral features are algebraic diagnostics, not truth oracles."""

import numpy as np
import pytest

from qhyper.group_response import extract_group_quantum_features
from qhyper.quantum_action import reference_unitary
from qhyper.revision_spectrum import (
    classical_boundary_gram_matrix,
    group_gram_from_operator,
    matched_classical_spectral_feature_matrix,
    operator_spectral_statistics,
    q3_from_retention,
    quantum_group_feature_matrices,
    quantum_spectral_feature_matrix,
)


def _hamiltonian() -> np.ndarray:
    adjacency = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ], dtype=float)
    degree = adjacency.sum(axis=1)
    scale = 1 / np.sqrt(degree)
    return (np.eye(5) - scale[:, None] * adjacency * scale[None, :]) / 2


def test_unitary_gram_q3_identity_and_complement_determinant() -> None:
    h = _hamiltonian()
    U = reference_unitary(h, 1.4)
    M, G = group_gram_from_operator(U, (0, 1, 2))
    np.testing.assert_allclose(M + G, np.eye(3), atol=1e-11)
    assert np.linalg.eigvalsh(M).min() >= -1e-11
    assert np.linalg.eigvalsh(M).max() <= 1 + 1e-11
    block = U[np.ix_([0, 1, 2], [0, 1, 2])]
    np.testing.assert_allclose(np.linalg.det(M), abs(np.linalg.det(block)) ** 2,
                               atol=1e-11)
    coherent, incoherent, contrast = q3_from_retention(M)
    np.testing.assert_allclose(coherent, np.sum(abs(block @ np.ones(3) / np.sqrt(3)) ** 2))
    np.testing.assert_allclose(incoherent, np.sum(abs(block) ** 2) / 3)
    np.testing.assert_allclose(contrast, coherent - incoherent)

    # Jacobi's complementary-minor identity: Qvol's raw determinant alone
    # cannot distinguish a candidate from its complement.
    complement = U[np.ix_([3, 4], [3, 4])]
    np.testing.assert_allclose(abs(np.linalg.det(block)) ** 2,
                               abs(np.linalg.det(complement)) ** 2, atol=1e-11)


def test_qvol_qcorr_short_time_match_direct_classical_gram() -> None:
    h = _hamiltonian()
    group = [(1, 2, 3)]
    classical, names = classical_boundary_gram_matrix(h, group)
    assert names[0] == "Cgram_boundary_energy_per_node"
    assert classical.shape == (1, 4)
    assert classical[0, 0] > 0
    assert classical[0, 1] > 0
    eps = 1e-6
    times = (0.05, 0.02, 0.01)
    features, labels = quantum_spectral_feature_matrix(h, group, times, eps=eps)
    assert features.shape == (1, 12)
    assert len(labels) == 12 and len(set(labels)) == 12
    for t_index, t in enumerate(times):
        expected_vol = (1 - eps) * classical[0, 0]
        expected_corr = (1 - eps) ** 2 * classical[0, 1]
        np.testing.assert_allclose(features[0, 4 * t_index] / t**2,
                                   expected_vol, rtol=0.025)
        np.testing.assert_allclose(features[0, 4 * t_index + 1] / t**4,
                                   expected_corr, rtol=0.04)


def test_zero_time_and_shared_dense_sparse_action_agree() -> None:
    h = _hamiltonian()
    groups = [(0, 1), (0, 1, 2)]
    zero, _ = quantum_spectral_feature_matrix(h, groups, (0.0,))
    np.testing.assert_allclose(zero, 0, atol=1e-12)
    dense, _ = quantum_spectral_feature_matrix(h, groups, (0.25, 1.0))
    action, _ = quantum_spectral_feature_matrix(h, groups, (0.25, 1.0),
                                                 dense_limit=2)
    np.testing.assert_allclose(dense, action, rtol=1e-9, atol=1e-10)


def test_shared_q3_columns_match_existing_exact_reference() -> None:
    h = _hamiltonian()
    groups = [(0, 1), (1, 2, 3)]
    times = (0.25, 1.0)
    new, names = quantum_group_feature_matrices(h, groups, times)
    old, _ = extract_group_quantum_features(h, groups, times, exact_incoherent=True)
    assert len(names["Q3"]) == 3 * len(times)
    np.testing.assert_allclose(new["Q3"], old, rtol=1e-9, atol=1e-10)


def test_matched_classical_uses_same_statistics_without_leakage_interpretation() -> None:
    h = _hamiltonian()
    groups = [(0, 1, 2)]
    for kind in ("heat", "wave"):
        values, names = matched_classical_spectral_feature_matrix(
            h, groups, (0.0, 0.7), kind=kind)
        assert values.shape == (1, 8)
        assert all(name.startswith("Cmatched_") for name in names)
        np.testing.assert_allclose(values[0, :4], 0, atol=1e-12)
        assert np.isfinite(values).all()
        assert values[0, 4] >= 0


def test_qcorr_nonzero_with_only_one_correlated_pair() -> None:
    # This is non-additivity in a three-node matrix, but not an irreducible
    # three-node relation: node 2 is completely independent of the pair.
    offdiag = 0.1
    M = np.array([[0.8, offdiag, 0.0], [offdiag, 0.8, 0.0], [0.0, 0.0, 1.0]])
    omega = operator_spectral_statistics(M)[1]
    assert omega > 0
    diagonal = (1 - 1e-6) * 0.8 + 1e-6
    rho_regularized = (1 - 1e-6) * offdiag / diagonal
    np.testing.assert_allclose(omega, -np.log(1 - rho_regularized**2))


def test_invalid_operator_and_retention_are_rejected() -> None:
    with pytest.raises(ValueError, match="unitary"):
        group_gram_from_operator(np.eye(3) * 0.5, (0, 1))
    with pytest.raises(ValueError, match="PSD contraction"):
        operator_spectral_statistics(np.eye(2) * 2)


def test_regularization_keeps_singular_retention_finite() -> None:
    values = operator_spectral_statistics(np.diag([1.0, 0.0]), eps=1e-5)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[0], -np.log(1e-5) / 2)
    np.testing.assert_allclose(values[1], 0, atol=1e-12)

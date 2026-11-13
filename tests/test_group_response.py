import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from qhyper.group_response import (
    canonical_candidate,
    coherent_group_response,
    extract_group_quantum_features,
    short_time_coefficient,
)
from qhyper.hamiltonian import build_hamiltonian


def _weighted_test_hamiltonian():
    rng = np.random.default_rng(2026)
    a = rng.uniform(0, 1, size=(7, 7))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0)
    return build_hamiltonian(a)


def _dense_response(h, nodes, time):
    nodes = np.asarray(nodes, dtype=int)
    k = len(nodes)
    u = expm(-1j * time * h.toarray())
    state = np.zeros(h.shape[0], dtype=complex)
    state[nodes] = 1 / np.sqrt(k)
    coherent = np.sum(np.abs((u @ state)[nodes]) ** 2).real
    incoherent = np.sum(np.abs(u[np.ix_(nodes, nodes)]) ** 2).real / k
    return coherent, incoherent, coherent - incoherent, u


def test_direct_response_cross_interference_and_random_phase_identity():
    h = _weighted_test_hamiltonian()
    nodes = (0, 2, 4)
    grid = [0.0, 0.1, 0.5, 1.0, 2.0]
    response = coherent_group_response(h, nodes, grid)
    assert response["features"].shape == (3 * len(grid),)
    for index, time in enumerate(grid):
        coherent, incoherent, contrast, u = _dense_response(h, nodes, time)
        np.testing.assert_allclose(response["R_coh"][index], coherent, atol=1e-12)
        np.testing.assert_allclose(response["R_inc"][index], incoherent, atol=1e-12)
        np.testing.assert_allclose(response["C"][index], contrast, atol=1e-12)
        assert -1e-12 <= coherent <= 1 + 1e-12
        assert -1e-12 <= incoherent <= 1 + 1e-12
        assert -1 - 1e-12 <= contrast <= 1 + 1e-12

        # Formula (1): interference is the sum of pairwise amplitude crosses.
        crosses = sum(
            2 / len(nodes)
            * np.real(np.sum(u[list(nodes), i] * np.conj(u[list(nodes), j])))
            for i, j in itertools.combinations(nodes, 2)
        )
        np.testing.assert_allclose(contrast, crosses, atol=1e-12)

        # Formula (2): exhaustive Rademacher average equals the mixture.
        phase_mean = 0.0
        for signs in itertools.product([-1, 1], repeat=len(nodes)):
            state = np.zeros(h.shape[0], dtype=complex)
            state[list(nodes)] = np.asarray(signs) / np.sqrt(len(nodes))
            phase_mean += np.sum(np.abs((u @ state)[list(nodes)]) ** 2) / (2 ** len(nodes))
        np.testing.assert_allclose(phase_mean, incoherent, atol=1e-12)


def test_short_time_coefficient_and_evenness():
    h = _weighted_test_hamiltonian()
    nodes = (0, 2, 4)
    coefficient = short_time_coefficient(h, nodes)
    assert coefficient <= 0
    for time in [1e-3, 2e-3]:
        response = coherent_group_response(h, nodes, [time, -time])
        np.testing.assert_allclose(response["C"][0], response["C"][1], atol=1e-12)
        assert abs(response["C"][0] / time**2 - coefficient) < 1e-5
    # Formula (3) has fourth-order remainder in the real-symmetric case.
    errors = []
    for time in [0.04, 0.02]:
        contrast = coherent_group_response(h, nodes, [time])["C"][0]
        errors.append(abs(contrast - coefficient * time**2))
    assert errors[0] > errors[1]
    assert 10 < errors[0] / errors[1] < 25


def test_degenerate_cases_and_nonzero_component_boundary():
    a = np.zeros((5, 5))
    a[0, 1] = a[1, 0] = 1
    a[1, 2] = a[2, 1] = 1
    a[3, 4] = a[4, 3] = 1
    h = build_hamiltonian(a)
    at_zero = coherent_group_response(h, (0, 1), [0.0])
    np.testing.assert_allclose([at_zero["R_coh"][0], at_zero["R_inc"][0]], [1, 1])
    np.testing.assert_allclose(at_zero["C"][0], 0, atol=1e-14)
    all_vertices = coherent_group_response(h, range(5), [0.5, 2.0])
    np.testing.assert_allclose(all_vertices["C"], 0, atol=1e-12)
    component = coherent_group_response(h, (0, 1, 2), [0.5, 2.0])
    np.testing.assert_allclose(component["R_coh"], [1, 1], atol=1e-12)
    np.testing.assert_allclose(component["R_inc"], [1, 1], atol=1e-12)
    np.testing.assert_allclose(component["C"], 0, atol=1e-12)


def test_batched_feature_matrix_matches_individual_calls():
    h = _weighted_test_hamiltonian()
    groups = [(0, 2, 4), (1, 3), (2, 3, 5, 6)]
    times = [0.25, 1.0, 2.0]
    matrix, diagnostics = extract_group_quantum_features(
        h, groups, times, max_state_columns=10
    )
    assert matrix.shape == (3, 9)
    for index, group in enumerate(groups):
        expected = coherent_group_response(h, group, times)
        np.testing.assert_allclose(matrix[index], expected["features"], atol=1e-12)
        np.testing.assert_allclose(diagnostics["C"][index], expected["C"], atol=1e-12)


def test_phase_estimator_reports_variance_and_is_reproducible():
    h = _weighted_test_hamiltonian()
    arguments = dict(exact_incoherent=False, n_phase_samples=8, seed=731)
    first = coherent_group_response(h, (0, 2, 4), [0.5, 1.0], **arguments)
    second = coherent_group_response(h, (0, 2, 4), [0.5, 1.0], **arguments)
    np.testing.assert_array_equal(first["features"], second["features"])
    assert first["incoherent_reference"] == "rademacher"
    assert np.all(first["phase_sample_variance"] >= 0)
    np.testing.assert_allclose(
        first["phase_estimator_variance"], first["phase_sample_variance"] / 8
    )


def test_perturbation_bound_for_fixed_group_and_hamiltonian():
    h = _weighted_test_hamiltonian()
    rng = np.random.default_rng(44)
    delta = rng.normal(size=h.shape)
    delta = 0.01 * (delta + delta.T) / 2
    perturbed = h + delta
    norm = np.linalg.norm(delta, ord=2)
    time = 1.2
    before = coherent_group_response(h, (0, 2, 4), [time])
    after = coherent_group_response(perturbed, (0, 2, 4), [time])
    assert abs(after["R_coh"][0] - before["R_coh"][0]) <= 2 * time * norm + 1e-12
    assert abs(after["R_inc"][0] - before["R_inc"][0]) <= 2 * time * norm + 1e-12
    assert abs(after["C"][0] - before["C"][0]) <= 4 * time * norm + 1e-12


def test_invalid_group_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        canonical_candidate([1, 1], 5)
    with pytest.raises(ValueError, match="at least|between"):
        canonical_candidate([1], 5)
    with pytest.raises(ValueError, match="outside"):
        canonical_candidate([0, 5], 5)
